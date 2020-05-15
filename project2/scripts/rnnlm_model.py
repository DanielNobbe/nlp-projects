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

    def __init__(self, ntoken, ninp, nhid, nlayers, rnn_type = 'GRU', dropout = 0.2,
    tie_weights = False):

        super(RNNLM, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(ntoken, ninp)

        self.rnn = nn.GRU(input_size = ninp, hidden_size = nhid,
                            num_layers = nlayers, bias = True, batch_first = True,
                            dropout = dropout, bidirectional = False)

        # if rnn_type in ['LSTM', 'GRU']:
        #     self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout = dropout)
        # else:
        #     try:
        #         nonlinearity = {'RNN_TANH' : 'tanh', 'RNN_RELU' : 'relu'}[rnn_type]
        #     except KeyError:
        #         raise ValueError("""An invalid option for `--model` was supplied.
        #                             Options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
        #     self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity = nonlinearity, dropout = dropout, batch_first = False)
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
            self.decoder.weight = self.embedding.weight

        # Call weight initialization function
        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _embed_and_pack(self, input, lengths):
        embedded = self.drop(self.embedding(input))
        packed = pack_padded_sequence(embedded, lengths, batch_first = True,
                                        enforce_sorted = False)
        return packed

    def forward(self, input, hidden, lengths):

        #print(input.
        batch_size = input.size(0)

        # print('Input shape')
        # print(input.shape)

        embedded = self.drop(self.embedding(input))

        # print('Embedded shape')
        # print(embedded.shape)

        packed = pack_padded_sequence(embedded, lengths, batch_first = True,
                                        enforce_sorted = False)



        #embedding = self.drop(self.embedding(input))
        #packed = self._embed_and_pack(input, lengths)

        # print('Packed type')
        # print(type(packed))
        #
        # print('Packed data shape')
        # print(packed.data.shape)
        #
        # print('NU')

        #output, hidden = self.rnn(embedded, hidden)
        output, hidden = self.rnn(packed, hidden)

        # print('Type output')
        # print(type(output))
        #
        # print('SHape output')
        # print(output.data.shape)
        #
        # print('Type hidden')
        # print(type(hidden))
        #
        # print('SHape hidden')
        # print(hidden.shape)

        #packed_decoder_input = _embed_and_pack(input, lengths)



        # output = self.drop(output.data)
        # decoded = self.decoder(output)
        #
        # unpacked, seq_lengths = pad_packed_sequence(decoded, batch_first = True)

        #unpacked, lengths = pad_packed_sequence(decoded, batch_first = True)
        #decoded = decoded.view(-1, self.ntoken) # TODO: either view(-1, seq_len, self.ntoken) or no resize

        # How about this?
        unpacked, seq_lengths = pad_packed_sequence(output, batch_first = True)

        unpacked = self.drop(unpacked)
        decoded = self.decoder(unpacked)

        #return F.log_softmax(decoded, dim = 1), hidden #TODO: use linear output and cross_entropy, or log_softmax and NLLLoss()
        return decoded, hidden #TODO: shouldn't this have a softmax layer like Equation (1e) in project description

    def init_hidden(self, bsz):
        weight = next(self.parameters())

        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
