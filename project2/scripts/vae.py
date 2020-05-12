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

        mean, std = self.encoder(packed)
        z = self.sampler(mean, std, batch_size)
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
        pred = logp.view(b * l, c)
        target = by.view(b * l)

        nll = cross_entropy(pred, target, ignore_index=0, reduction="sum") / b

        q = Normal(mean, std)
        kl = torch.sum(kl_divergence(prior, q))

        loss = nll + kl

        print(nll, kl)
        print(loss)
        print("loss {:.3f} accuracy {:.3f}".format(epoch + 1, valid_loss, valid_acc))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def train(
    epochs,
    num_layers=2,
    embedding_size=256,
    hidden_size=128,
    latent_size=32,
    batch_size_train=64,
    batch_size_valid=256,
    learning_rate=0.001,
    device=torch.device("cpu"),
):

    train_data, val_data, test_data = get_datasets()
    vocab_size = train_data.tokenizer.vocab_size

    model = SentenceVAE(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_layers=num_layers,
    )
    model.to(device)

    train_loader = DataLoader(
        train_data, batch_size=batch_size_train, shuffle=True, collate_fn=padded_collate
    )

    val_loader = DataLoader(
        val_data, batch_size=batch_size_valid, shuffle=False, collate_fn=padded_collate
    )

    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_one_epoch(model, optimizer, train_loader, device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    epochs = 50
    train(epochs, device=device)
