import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# The following code is based on Pytorch's example of an RNNModel for LM:
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py

class RNNLM(nn.Module):
    """Container module for RNN Language Model. Consists of an embedder,
    a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, rnn_type = 'GRU', dropout = 0.2):
        super(RNNLM, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(ntoken, ninp)

        self.rnn = nn.GRU(input_size = ninp, hidden_size = nhid,
                            num_layers = nlayers, bias = True, batch_first = True,
                            dropout = dropout, bidirectional = False)
        self.decoder = nn.Linear(nhid, ntoken)
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def _embed_and_pack(self, input, lengths):
        embedded = self.drop(self.embedding(input))
        packed = pack_padded_sequence(embedded, lengths, batch_first = True,
                                        enforce_sorted = False)
        return packed

    def forward(self, input, hidden, lengths):

        batch_size = input.size(0)
        embedded = self.drop(self.embedding(input))
        packed = pack_padded_sequence(embedded, lengths, batch_first = True,
                                        enforce_sorted = False)
        output, hidden = self.rnn(packed, hidden)
        unpacked, seq_lengths = pad_packed_sequence(output, batch_first = True)
        unpacked = self.drop(unpacked)
        decoded = self.decoder(unpacked)
        return decoded, hidden

    def init_hidden(self, bsz):
        if self.rnn_type == 'LSTM':
            return (torch.zeros(self.nlayers, bsz, self.nhid),
                    torch.zeros(self.nlayers, bsz, self.nhid))
        else:
            return torch.zeros(self.nlayers, bsz, self.nhid)
