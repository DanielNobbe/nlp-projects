import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set the random seed manually for reproducibility.
torch.manual_seed(2020)

# Check if GPU available
is_gpu_available = torch.cuda.is_available()

if is_gpu_available:
    device = torch.device("cuda")
    print('GPU is available.')
else:
    device = torch.device('cpu')
    print("GPU not available, CPU used instead.")

class RNNLM(nn.Module):
    """Container module for RNN Language Model. Consists of an encoder,
    a recurrent module, and a decoder."""

    def __init__(self, rnn_type = 'GRU', ntoken, ninp, nhid, nlayers, dropout = 0.5,
    tie_weights = False):

        super(RNNLM, self).__init__():
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout = dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH' : 'tanh', 'RNN_RELU' : 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied.
                                    Options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity = nonlinearity, dropout = dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    
