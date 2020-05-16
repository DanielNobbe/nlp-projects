import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from torch.distributions import kl_divergence
from torch.distributions.normal import Normal

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.optim import Adam
from torch.nn.functional import cross_entropy

from data import padded_collate, get_datasets


class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, latent_size, num_layers):
        super(Encoder, self).__init__()
        gru_args = dict(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,  # this is also the default
            batch_first=True,
            dropout=0,  # dropout on the outputs of each GRU layer except last with dropout prob given here
            bidirectional=True,
        )

        self.rnn = nn.GRU(**gru_args)
        num_directions = 2  # bidiractional
        self.h2m = nn.Linear(hidden_size * num_layers * num_directions, latent_size)
        self.h2v = nn.Linear(hidden_size * num_layers * num_directions, latent_size)

    def forward(self, input):
        encoder_output, hn = self.rnn(input)
        num_layers, batch_size, hidden_size = hn.size()
        hn = hn.transpose(0, 1).reshape(batch_size, -1)
        mean = self.h2m(hn)
        logvar = self.h2v(hn)
        std = torch.exp(logvar / 2)  # TODO Understand this magic
        return mean, std


class Sampler(nn.Module):
    def __init__(self, latent_size):
        super(Sampler, self).__init__()
        self.latent_size = latent_size  # TODO not used

    def forward(self, mean, std, batch_size):
        m = Normal(mean, std)
        sample = m.rsample()
        return sample


class WordDropout(nn.Module):
    def __init__(self, dropout_probability, unk_token_idx):
        super(WordDropout, self).__init__()
        self.dropout_probability = dropout_probability
        self.unk_token_idx = unk_token_idx

    def forward(self, padded_sentences):
        x = padded_sentences.clone()
        dropout_mask = (
            torch.rand_like(padded_sentences, dtype=torch.float32)
            >= self.dropout_probability
        )
        x[dropout_mask] = self.unk_token_idx
        return x


class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, latent_size, num_layers):
        # TODO this is code duplication, get rid of it
        super(Decoder, self).__init__()
        gru_args = dict(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,  # this is also the default
            batch_first=True,
            dropout=0,  # dropout on the outputs of each GRU layer except last with dropout prob given here
            bidirectional=False,
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.l2h = nn.Linear(latent_size, num_layers * hidden_size)
        self.rnn = nn.GRU(**gru_args)

    def forward(self, z, input):
        h = self.l2h(z)
        batch_size = h.size(0)
        h = h.reshape(batch_size, self.num_layers, self.hidden_size)
        h = h.transpose(0, 1)

        output, hn = self.rnn(input, h)

        return output


class SentenceVAE(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        latent_size,
        num_layers,
        unk_token_idx,
        word_dropout_probability=0.0,
    ):
        super(SentenceVAE, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_size
        )

        self.encoder = Encoder(
            embedding_size, hidden_size, latent_size, num_layers=num_layers
        )
        self.sampler = Sampler(latent_size)
        self.word_dropout = WordDropout(word_dropout_probability, unk_token_idx)
        self.decoder = Decoder(
            embedding_size, hidden_size, latent_size, num_layers=num_layers
        )
        self.decoded2vocab = nn.Linear(
            hidden_size, vocab_size
        )  # TODO should this be in the decoder

    def _embed_and_pack(self, input, lengths):
        embedded = self.embedding(input)
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        return packed

    def forward(self, input, lengths):
        batch_size = input.size(0)

        packed = self._embed_and_pack(input, lengths)

        mean, std = self.encoder(packed)

        z = self.sampler(mean, std, batch_size)

        decoder_input = self.word_dropout(input)
        packed = self._embed_and_pack(decoder_input, lengths)

        decoded = self.decoder(z, packed)

        unpacked, lengths = pad_packed_sequence(decoded, batch_first=True)

        out = self.decoded2vocab(unpacked)
        return out, mean, std


def train_one_epoch(model, optimizer, data_loader, device):
    prior = Normal(0.0, 1.0)
    model.train()
    for bx, by, bl in data_loader:
        logp, mean, std = model(bx.to(device), bl)

        b, l, c = logp.shape
        pred = logp.transpose(1, 2)  # pred shape: (batch_size, vocab_size, seq_length)
        target = by.to(device)  # target shape: (batch_size, seq_length)

        # TODO Is this fixed now? What kind of values are we supposed to get here?
        # TODO ignore index is hardcoded here
        nll = cross_entropy(pred, target, ignore_index=0, reduction="none")
        nll = nll.sum(-1)

        q = Normal(mean, std)
        kl = kl_divergence(q, prior)
        kl = kl.sum(-1)

        # elbo = log-likelihood - D_kl
        # max elbo <-> min -elbo
        # -elbo = -log-likelihood + D_kl
        loss = (nll + kl).mean()

        print(
            "nll mean: {} \t kl mean: {} \t loss: {}".format(
                nll.mean().item(), kl.mean().item(), loss.item()
            )
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train(
    data_path,
    device,
    num_epochs,
    batch_size_train,
    batch_size_valid,
    learning_rate,
    num_layers,
    embedding_size,
    hidden_size,
    latent_size,
    word_dropout,
    print_every,
    tensorboard_logging,
    logdir,
    model_save_path
):

    train_data, val_data, test_data = get_datasets(data_path)
    device = torch.device(device)
    vocab_size = train_data.tokenizer.vocab_size

    model = SentenceVAE(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_layers=num_layers,
        word_dropout_probability=word_dropout,
        unk_token_idx=train_data.tokenizer.unk_token_id,
    )
    model.to(device)

    train_loader = DataLoader(
        train_data, batch_size=batch_size_train, shuffle=True, collate_fn=padded_collate
    )

    val_loader = DataLoader(
        val_data, batch_size=batch_size_valid, shuffle=False, collate_fn=padded_collate
    )

    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device)

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='../Data/Dataset')
    parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cuda', 'cpu'])

    parser.add_argument('-ne', '--num_epochs', type=int, default=10)
    parser.add_argument('-sbt', '--batch_size_train', type=int, default=32)
    parser.add_argument('-sbv', '--batch_size_valid', type=int, default=256)

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-se', '--embedding_size', type=int, default=300)
    parser.add_argument('-sh', '--hidden_size', type=int, default=256)
    parser.add_argument('-sl', '--latent_size', type=int, default=16)

    parser.add_argument('-wd', '--word_dropout', type=float, default=0.0)
    # parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-v','--print_every', type=int, default=50)
    parser.add_argument('-tb','--tensorboard_logging', action='store_true')
    parser.add_argument('-log','--logdir', type=str, default='logs')
    parser.add_argument('-m','--model_save_path', type=str, default='models')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    args = vars(args)
    train(**args)
