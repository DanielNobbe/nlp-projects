import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from torch.distributions import kl_divergence
from torch.distributions.normal import Normal

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.optim import Adam, RMSprop
from torch.nn.functional import cross_entropy

from tqdm import tqdm, trange

from torch import logsumexp
import numpy as np

import pdb

from data import padded_collate, get_datasets
import pickle


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
        self.softplus = nn.Softplus()

    def forward(self, input):
        encoder_output, hn = self.rnn(input)
        num_layers, batch_size, hidden_size = hn.size()
        hn = hn.transpose(0, 1).reshape(batch_size, -1)
        mean = self.h2m(hn)
        logvar = self.h2v(hn)
        # std = torch.exp(logvar / 2)
        std = self.softplus(logvar)
        return mean, std


class Sampler(nn.Module):
    def __init__(self, latent_size):
        super(Sampler, self).__init__()
        self.latent_size = latent_size  # TODO not used

    def forward(self, mean, std):
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
        if self.training:
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


class Lagrangian(nn.Module):
    def __init__(self, minimum_desired_rate):
        super(Lagrangian, self).__init__()
        self.lagrangian_multiplier = torch.nn.Parameter(torch.tensor([1.01])) # TODO: Maybe this should be a vector?
        self.minimum_desired_rate = minimum_desired_rate

    def forward(self, kl):
        return self.lagrangian_multiplier * (self.minimum_desired_rate - kl.mean())


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
        model_save_path='models',
        freebits=None,
    ):
        super(SentenceVAE, self).__init__()
        self.model_save_path = model_save_path
        self.saved_model_files = []
        Path(self.model_save_path).mkdir(exist_ok=True, parents=True)
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
        self.freebits = freebits

        # This is a bit of a hack to track running means and standard deviations
        self.tracked_means = nn.BatchNorm1d(num_features=latent_size, momentum=None)
        self.tracked_stds = nn.BatchNorm1d(num_features=latent_size, momentum=None)


    def _embed_and_pack(self, input, lengths):
        embedded = self.embedding(input)
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        return packed


    def encode(self, input, lengths):
        batch_size = input.size(0)

        packed = self._embed_and_pack(input, lengths)
        mean, std = self.encoder(packed)
        return mean, std
        

    def decode(self, input, z, lengths):
        decoder_input = self.word_dropout(input)
        packed = self._embed_and_pack(decoder_input, lengths)

        decoded = self.decoder(z, packed)

        unpacked, lengths = pad_packed_sequence(decoded, batch_first=True)

        out = self.decoded2vocab(unpacked)  # shape: (batch_size, seq_length, vocab_size)
        out = out.transpose(1, 2)  # pred shape: (batch_size, vocab_size, seq_length)
        return out


    def forward(self, input, lengths):
        mean, std = self.encode(input, lengths)

        z = self.sampler(mean, std)

        out = self.decode(input, z, lengths)

        # Throw away the results of these modules,
        # we only use them to track running averages
        self.tracked_means(mean)
        self.tracked_stds(std)

        return out, mean, std


    def save_model(self, filename):
        save_file_path = Path(self.model_save_path) / filename
        torch.save(self.state_dict(), save_file_path)
        self.saved_model_files.append(filename)
        torch.save(self.tracked_means.state_dict(), save_file_path.with_suffix('.means'))
        torch.save(self.tracked_stds.state_dict(), save_file_path.with_suffix('.stds'))
        self.tracked_means.reset_running_stats()
        self.tracked_stds.reset_running_stats()

    def load_from(self, save_file_path):
        self.load_state_dict(torch.load(save_file_path))




def standard_vae_loss_terms(pred, target, mean, std, ignore_index=0, prior=Normal(0.0, 1.0), 
    print_loss=True, loss_lists=None):

    nll = cross_entropy(pred, target, ignore_index=ignore_index, reduction="none")
    nll = nll.sum(-1)

    q = Normal(mean, std)
    kl = kl_divergence(q, prior)
    kl = kl.sum(-1)

    # elbo = log-likelihood - D_kl
    # max elbo <-> min -elbo
    # -elbo = -log-likelihood + D_kl

    if print_loss:
        # print(
        tqdm.write(
            "nll mean: {} \t kl mean: {} \t loss mean: {}".format(
                nll.mean().item(), kl.mean().item(), (nll + kl).mean().item()
            )
        )

    if loss_lists is not None:
        loss_lists[0].append(nll.mean().item()) # store NLL in list
        loss_lists[1].append(kl.mean().item()) # store KL in list

    return nll, kl

def standard_vae_loss(pred, target, mean, std, ignore_index=0, print_loss=True, loss_lists=None):
    nll, kl = standard_vae_loss_terms(pred, target, mean, std, ignore_index=ignore_index, print_loss=print_loss, loss_lists=loss_lists)
    loss = (nll + kl).mean()    # mean over batch
    return loss


def freebits_vae_loss(pred, target, mean, std, ignore_index=0, prior=Normal(0.0, 1.0), freebits=0.5, 
    print_loss=True, loss_lists=None):

    nll = cross_entropy(pred, target, ignore_index=ignore_index, reduction="none")
    nll = nll.sum(-1).mean() # First sum the nll over all dims, then average over batch

    q = Normal(mean, std)
    kl = kl_divergence(q, prior)
    kl = kl.mean(0) # Average over batch. Keep dimensions intact

    # If freebits is specified, the kl divergence along each dimension should be clamped to be higher than this value
    # The distributions used here are simple normal distributions, with no off-diagonal variance terms.
    # As such, the KL divergence is applied elementwise.
    kl = torch.clamp(kl, min = freebits)
    kl = kl.sum(-1) # Sum kl over all dimensions

    # elbo = log-likelihood - D_kl
    # max elbo <-> min -elbo
    # -elbo = -log-likelihood + D_kl
    loss = (nll + kl)

    if print_loss:
        print(
            "nll mean: {} \t kl mean: {} \t loss mean: {}".format(
                nll.item(), kl.item(), loss.item()
            )
        )
    if loss_lists is not None:
        loss_lists[0].append(nll.mean().item()) # store NLL in list
        loss_lists[1].append(kl.mean().item()) # store KL in list

    return loss


@torch.no_grad()
def perplexity(model, data_loader, device, num_samples):
    model.eval()
    prior = Normal(0.0, 1.0)
    padding_index = data_loader.dataset.tokenizer.pad_token_id
    sample_marginal = 0
    total_num_tokens = 0
    total_log_marginal = 0

    for bx, by, bl in tqdm(data_loader):
        samples = []
        num_tokens = sum(bl)
        input = bx.to(device)
        target = by.to(device)  # target shape: (batch_size, seq_length)
        batch_size = input.size(0)

        mean, std = model.encode(input, bl)
        q = Normal(mean, std)

        for k in trange(num_samples):
            z = q.sample()
            log_qz_x = q.log_prob(z).sum(-1)
            log_pz = prior.log_prob(z).sum(-1)

            logits = model.decode(input, z, bl)
            nll = cross_entropy(logits, target, ignore_index=padding_index, reduction="none")
            log_px_z = -nll.sum(-1)     # log p(x|z)
            log_pxz = log_px_z + log_pz     # log p(x, z) = log p(x|z) + log p(z)

            log_pxz_over_qz_x = (log_pxz - log_qz_x)     # log [p(x, z) / q(z|x)]

            samples.append(log_pxz_over_qz_x)

        samples = torch.stack(samples)
        K = torch.tensor(float(num_samples), device=device)
        log_marginal = torch.logsumexp(samples, dim=0) - torch.log(K)

        total_log_marginal += log_marginal.sum()
        total_num_tokens += num_tokens

    ppl = torch.exp(-total_log_marginal / total_num_tokens)

    return ppl



def marginal_ll(model, input, target, lengths, mean, std, sample_size,
                prior, batch_size, device, padding_index, debug = False):

    qz_x_dist = Normal(mean, std)
    argsums = torch.zeros((batch_size, 1), device=device)

    for k in range(sample_size):
        z_val = qz_x_dist.sample()
        qz_x_logprobs = qz_x_dist.log_prob(z_val)
        pz_logprobs = prior.log_prob(z_val)
        # decoder_input = model.word_dropout(input)
        decoder_input = input
        packed = model._embed_and_pack(decoder_input, lengths)
        decoded = model.decoder(z_val, packed)

        unpacked, lengths = pad_packed_sequence(decoded, batch_first = True)
        pred = model.decoded2vocab(unpacked)
        pred = pred.to(device)  # pred shape: (batch_size, vocab_size, seq_length)
        target = target.to(device)
        nll = cross_entropy(pred, target, ignore_index = padding_index, reduction = "none")
        logpx_z = -nll.sum(-1).reshape(batch_size, 1)
        logqz_x = torch.sum(qz_x_logprobs, axis = -1).reshape(batch_size, 1)
        logpz = torch.sum(pz_logprobs, axis = -1).reshape(batch_size, 1)
        argsum = logpx_z + logpz - logqz_x
        argsums += argsum

    logpx_batch = -np.log(sample_size) + argsums #logsumexp(argsums, dim = 1, keepdims = True) #TODO: Do we even need logsumexp here?

    return logpx_batch.to(device)

def perplexity_old(mll, batch_seq_lengths, batch_size):

    exponent = -torch.sum(mll) / np.sum(batch_seq_lengths)
    batch_ppl = torch.exp(exponent)

    return batch_ppl




def train_one_epoch(model, optimizer, data_loader, device, save_every, iter_start, padding_index, print_every=50, loss_lists = None):
    prior = Normal(0.0, 1.0)
    model.train()

    print("\nTraining for an epoch\n")

    for iteration, (bx, by, bl) in enumerate(tqdm(data_loader), start=iter_start):
        logp, mean, std = model(bx.to(device), bl)

        b, c, l = logp.shape
        pred = logp     # pred shape: (batch_size, vocab_size, seq_length)
        target = by.to(device)  # target shape: (batch_size, seq_length)

        print_loss = (iteration % print_every == 0)

        # TODO Is this fixed now? What kind of values are we supposed to get here?
        if model.freebits is None:
            loss = standard_vae_loss(pred, target, ignore_index=padding_index, mean=mean, std=std, print_loss=print_loss, loss_lists=loss_lists)
        elif model.freebits is not None: # Set up structure for when MDR is added
            loss = freebits_vae_loss(pred, target, ignore_index = padding_index, prior = prior, mean=mean, std=std, 
            freebits = model.freebits, print_loss=print_loss, loss_lists=loss_lists)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iteration % save_every) == 0:
            if model.freebits is not None:
                model.save_model(f"sentence_vae_FreeBits_{model.freebits}_{iteration}.pt")
            else:
                model.save_model(f"sentence_vae_{iteration}.pt")


    return iteration



def train_one_epoch_MDR(model, lagrangian, lagrangian_optimizer, general_optimizer, data_loader, 
                        device, save_every, iter_start, padding_index, minimum_rate=1.0, print_every=50, loss_lists=None):

    prior = Normal(0.0, 1.0)
    model.train()

    print("\nTraining one epoch:\n")

    for iteration, (bx, by, bl) in enumerate(data_loader, start=iter_start):
        logp, mean, std = model(bx.to(device), bl)

        b, c, l = logp.shape
        pred = logp  # pred shape: (batch_size, vocab_size, seq_length)
        target = by.to(device)  # target shape: (batch_size, seq_length)

        print_loss = (iteration % print_every == 0)

        nll, kl = standard_vae_loss_terms(pred, target, ignore_index=padding_index, mean=mean, std=std, print_loss=print_loss, loss_lists=loss_lists)
        elbo = (nll + kl).mean() # This is the negative elbo, which should be minimized
        lagrangian_loss = lagrangian(kl)

        loss = -(-elbo - lagrangian_loss) # MDR constrained optimization loss
        general_optimizer.zero_grad()
        lagrangian_optimizer.zero_grad()

        loss.backward()

        # Invert gradient for lagrangian parameter
        lagrangian.lagrangian_multiplier.grad *= -1

        # Update with the two optimizers
        lagrangian_optimizer.step()
        general_optimizer.step()
        lagrangian_optimizer.zero_grad()
        general_optimizer.zero_grad()

        if (iteration % save_every) == 0:
            model.save_model(f"sentence_vae_MDR_{minimum_rate}_{iteration}.pt")



    return iteration


def evaluate(model, data_loader, device, padding_index, print_every=50):
    model.eval()
    total_loss = 0
    total_num = 0
    with torch.no_grad():
        for iteration, (bx, by, bl) in enumerate(tqdm(data_loader)):
            logp, mean, std = model(bx.to(device), bl)

            b, c, l = logp.shape
            pred = logp  # pred shape: (batch_size, vocab_size, seq_length)
            target = by.to(device)  # target shape: (batch_size, seq_length)

            # TODO Is this fixed now? What kind of values are we supposed to get here?
            # TODO ignore index is hardcoded here
            print_loss = (iteration % print_every == 0)
            nll, kl = standard_vae_loss_terms(pred, target, mean, std, ignore_index=padding_index, print_loss=print_loss, loss_lists=None)
            loss = (nll + kl).sum()     # sum over batch
            total_loss += loss
            total_num += b

            mll = marginal_ll(model = model, input = bx.to(device), target = target.to(device),
                            lengths = bl, mean = mean, std = std, sample_size = 10,
                            prior = Normal(0.0, 1.0), batch_size = b, device = device,
                            padding_index = padding_index)

            ppl = perplexity_old(mll = mll, batch_seq_lengths = bl, batch_size = b)


    val_loss = total_loss / total_num

    return val_loss, ppl


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
    save_every,
    tensorboard_logging,
    model_save_path,
    early_stopping_patience,
    freebits,
    MDR,
    # losses_save_path,
    args=None,
):

    start_time = datetime.now()

    train_data, val_data, test_data = get_datasets(data_path)
    device = torch.device(device)
    vocab_size = train_data.tokenizer.vocab_size
    padding_index = train_data.tokenizer.pad_token_id

    model = SentenceVAE(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_layers=num_layers,
        word_dropout_probability=word_dropout,
        unk_token_idx=train_data.tokenizer.unk_token_id,
        freebits = freebits, # Freebits value is the lambda value as described in Kingma et al. 
        model_save_path=model_save_path
    )
    lagrangian = Lagrangian(MDR)

    model.to(device)
    lagrangian.to(device)

    if MDR is not None:
        ### Define lagrangian parameter and optimizers
        lagrangian_optimizer = RMSprop(lagrangian.parameters(), lr=learning_rate) # TODO: Move this to other scope and use args.lr
    optimizer = Adam(model.parameters(), lr=learning_rate)


    train_loader = DataLoader(
        train_data, batch_size=batch_size_train, shuffle=True, collate_fn=padded_collate
    )

    val_loader = DataLoader(
        val_data, batch_size=batch_size_valid, shuffle=False, collate_fn=padded_collate
    )

    iterations = 0
    patience = 0
    best_val_loss = torch.tensor(np.inf, device=device)
    best_model = None
    for epoch in range(num_epochs):

        epoch_start_time = datetime.now()
        try:
            nll_list = []
            kl_list = []
            lists = (nll_list, kl_list)

            if MDR is None:
                iterations = train_one_epoch(model, optimizer, train_loader, device, iter_start=iterations, 
                                            padding_index=padding_index, save_every=save_every, print_every=print_every, loss_lists=lists)
            else:
                iterations = train_one_epoch_MDR(model, lagrangian, lagrangian_optimizer, optimizer, train_loader, device, 
                    iter_start=iterations, padding_index=padding_index, save_every=save_every, minimum_rate=MDR, loss_lists=lists)
                
        except KeyboardInterrupt:
            print("Manually stopped current epoch")
            __import__('pdb').set_trace()

        print("Training this epoch took {}".format(datetime.now() - epoch_start_time))

        print("Validation phase:")
        val_loss, ppl = evaluate(model, val_loader, device, padding_index=padding_index, print_every=print_every)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.saved_model_files[-1]
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print("EARLY STOPPING")
                break


        print(f"###############################################################")
        print(f"Epoch {epoch} finished, validation loss: {val_loss}, ppl: {ppl}")
        print(f"###############################################################")
        print("Current epoch training took {}".format(datetime.now()-epoch_start_time))

        losses_file_name = f"MDR{MDR}-freebits{freebits}-word_dropout{word_dropout}-print_every{print_every}-iterations{iterations}"
        save_losses_path = Path(model_save_path) / losses_file_name
        with open(save_losses_path, 'wb') as file:
            print("Saving losses..")
            pickle.dump((lists, print_every, args), file)


    print("Training took {}".format(datetime.now() - start_time))
    print(f"Best validation loss: {best_val_loss}")
    print(f"Best model: {best_model}")
        
   


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='../Data/Dataset', help="Folder for the Penn Treebank dataset. This folder should contain the 3 files of the provided data: train, val and test. Default: ../Data/Dataset")
    parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cuda', 'cpu'], help="The device to use (either cpu or cuda). Default: gpu")

    parser.add_argument('-ne', '--num_epochs', type=int, default=10, help="Maximum number of epochs to train for. Default: 10")
    parser.add_argument('-sbt', '--batch_size_train', type=int, default=32, help="Batch size to use for training. Default: 32")
    parser.add_argument('-sbv', '--batch_size_valid', type=int, default=64, help="Batch size to use for validation. Default: 64")

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer. Default: 0.001")

    parser.add_argument('-nl', '--num_layers', type=int, default=1, help="Number of layers for the recurrent models. Default: 1")
    parser.add_argument('-se', '--embedding_size', type=int, default=300, help="Embedding size. Default: 300")
    parser.add_argument('-sh', '--hidden_size', type=int, default=256, help="Hidden size. Default: 256")
    parser.add_argument('-sl', '--latent_size', type=int, default=16, help="Latent size. Default: 16")

    parser.add_argument('-wd', '--word_dropout', type=float, default=1.0, help="Word dropout keep probability. Default: 1.0 (keep every word)")

    # TODO should we use dropout aftere the embedding?
    # parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-v','--print_every', type=int, default=50, help="Status printing interval. Default: 50")
    parser.add_argument('-tb','--tensorboard_logging', action='store_true')
    parser.add_argument('-m','--model_save_path', type=str, default='models', help="Folder to save the pretrained models to. If doesn't exist, it is created. Default: models")
    parser.add_argument('-si','--save_every', type=int, default=500, help="Checkpointing interval in iterations. Default: 500")
    parser.add_argument('-es', '--early_stopping_patience', type=int, default=2, help="Early stopping patience. Default: 2")
    parser.add_argument('-fb', '--freebits', type=float, default=None, help="Free Bits parameter (if not provided, free bits method is not used for training)")
    parser.add_argument('-mdr','--MDR', type=float, default=None, help='MDR minimum rate (if not provided, MDR won\'t be used for training)')
    args = parser.parse_args()
    return args



def approximate_nll(model, data_loader, device, num_samples, padding_index, print_every=1):
    model.eval()
    total_loss = 0
    total_kl_loss = 0
    total_num = 0
    with torch.no_grad():
        for iteration, (bx, by, bl) in enumerate(tqdm(data_loader)):
            input = bx.to(device)
            target = by.to(device)  # target shape: (batch_size, seq_length)

            mean, std = model.encode(input, bl)

            # This is not the most efficient way to do this :(

            # NLL and KL
            for sample in trange(num_samples):
                z = model.sampler(mean, std)
                logp = model.decode(input, z, bl)
                b, c, l = logp.shape
                pred = logp  # pred shape: (batch_size, vocab_size, seq_length)

                print_loss = (iteration % print_every == 0)
                nll, kl = standard_vae_loss_terms(pred, target, mean, std, ignore_index=padding_index, print_loss=print_loss, loss_lists=None)

                loss = nll.sum()     # sum over batch
                total_loss += loss

                kl_loss = kl.sum()
                total_kl_loss += kl_loss

                total_num += b


    approx_nll = total_loss / total_num
    approx_kl = total_kl_loss / total_num
    approx_ppl = None

    return approx_nll, approx_kl, approx_ppl


def test_nll_estimation(
        data_path,
        device,
        embedding_size,
        hidden_size,
        latent_size,
        num_layers,
        word_dropout,
        freebits,
        model_save_path,
        batch_size_valid,
        saved_model_file,
        num_samples,
        **kwargs):

    start_time = datetime.now()

    train_data, val_data, test_data = get_datasets(data_path)
    device = torch.device(device)
    vocab_size = train_data.tokenizer.vocab_size
    padding_index = train_data.tokenizer.pad_token_id

    model = SentenceVAE(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_layers=num_layers,
        word_dropout_probability=word_dropout,
        unk_token_idx=train_data.tokenizer.unk_token_id,
        freebits = freebits, # Freebits value is the lambda value as described in Kingma et al. 
        model_save_path=model_save_path
    )

    model.load_from(saved_model_file)
    model.to(device)

    test_loader = DataLoader(
        test_data, batch_size=batch_size_valid, shuffle=False, collate_fn=padded_collate
    )

    epoch_start_time = datetime.now()
    try:
        ppl = perplexity(model, data_loader=test_loader, device=device, num_samples=num_samples)
        loss, kl, _ = approximate_nll(model=model, data_loader=test_loader, device=device, padding_index=padding_index, num_samples=num_samples)


    except KeyboardInterrupt:
        print("Manually stopped current epoch")
        __import__('pdb').set_trace()


    print("Approximate NLL:")
    print(loss)

    print("Approximate KL:")
    print(kl)

    print("Testing took {}".format(datetime.now() - start_time))
    return loss, kl, ppl

def test():
    args = vars(parse_arguments())
    saved_model_file = "./results_final/results2/vanilla/models/sentence_vae_3500.pt"
    num_samples = 2
    test_nll_estimation(saved_model_file=saved_model_file, num_samples=num_samples, **args)

if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    args = vars(args)
    train(**args, args=args)
