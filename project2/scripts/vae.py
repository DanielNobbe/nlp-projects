import torch
from torch import nn
from torch.utils.data import DataLoader

from torch.distributions import kl_divergence
from torch.distributions.normal import Normal

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data import padded_collate


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
        self.latent_size = latent_size

    def forward(self, mean, std, batch_size):
        m = Normal(mean, std)
        sample = m.rsample()
        return sample


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

        __import__("pdb").set_trace()
        output, hn = self.rnn(input, h)

        return output


class SentenceVAE(nn.Module):
    def __init__(
        self, vocab_size, embedding_size, hidden_size, latent_size, num_layers
    ):
        super(SentenceVAE, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_size
        )

        self.encoder = Encoder(
            embedding_size, hidden_size, latent_size, num_layers=num_layers
        )
        self.sampler = Sampler(latent_size)
        self.decoder = Decoder(
            embedding_size, hidden_size, latent_size, num_layers=num_layers
        )
        self.decoded2vocab = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, lengths):
        batch_size = input.size(0)
        embedded = self.embedding(input)
        # TODO here should ignore those that are not actual input, since the sentence lengths are variable
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        __import__("pdb").set_trace()
        mean, std = self.encoder(packed)
        z = self.sampler(mean, std, batch_size)
        decoded = self.decoder(z, packed)
        unpacked, lengths = pad_packed_sequence(decoded, batch_first=True)

        out = self.decoded2vocab(unpacked)
        return out


def train():
    from data import get_datasets

    train_data, val_data, test_data = get_datasets()
    vocab_size = train_data.tokenizer.vocab_size
    model = SentenceVAE(
        vocab_size=vocab_size,
        embedding_size=256,
        hidden_size=128,
        latent_size=32,
        num_layers=2,
    )

    train_loader = DataLoader(
        train_data, batch_size=64, shuffle=True, collate_fn=padded_collate
    )
    val_loader = DataLoader(
        train_data, batch_size=256, shuffle=False, collate_fn=padded_collate
    )

    model.train()
    for input_sentences_batch, target_sentences_batch, lengths in train_loader:

        out = model(input_sentences_batch, lengths)

        optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()


if __name__ == "__main__":
    train()
