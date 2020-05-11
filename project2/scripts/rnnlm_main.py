################################################################################
# This script contains the main training and evaluation functionality for the  #
# RNN Language Model.                                                          #
# A part of this code is an adaptation of Pytorch's example code for RNN LMs:  #
# https://github.com/pytorch/examples/blob/master/word_language_model/main.py  #
################################################################################

# Import statements
import argparse
import time
import math
import numpy as np
import os
import torch
import torch.cuda
import torch.nn as nn
import torch.utils.data as data
import torch.onnx
import pickle

from rnnlm_model import RNNLM
#import rnnlm_model
from Preprocessing_RNN import SentenceDataset


def main(args):

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    # Use GPU is possible
    # if args.cuda:
    # Check if GPU available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('GPU is available.')
    else:
        device = torch.device('cpu')
        print(type(device))
        print("GPU not available, CPU used instead.")

    # Load data

    # Get path of current directory
    file_path = os.path.dirname(os.path.abspath(__file__))

    # Find data path relative to current directory
    relative_pickle_path = '/../Data/Dataset/Dataloader.pkl'
    pickle_path = file_path + relative_pickle_path

    # Open and unpickle Dataloader.pkl
    with open(pickle_path, 'rb') as file:
        dataset = pickle.load(file)
    data_loader = data.DataLoader(dataset, args.batch_size, num_workers = 1)
    train_data = data.DataLoader(dataset._train_data,
                                args.batch_size, num_workers = 1)

    # print a couple of sequences to see if it works.
    for sequence in data_loader:
        print(sequence)
        print(type(sequence))
        print(sequence.size())
        break

    print('Hier')
    print(dataset._train_data)



    # Build RNN LM Model
    ntokens = dataset.vocab_size
    print(ntokens)
    model = RNNLM(ntoken = ntokens, ninp = args.emsize, nhid = args.nhid,
                    nlayers = args.nlayers, dropout = args.dropout).to(device)

    # use negative log-likelihood as loss / objective
    criterion = nn.NLLLoss()

    # Prerequisites for training

    def repackage_hidden(h):
        """Wraps hidden states in new tensors, to detatch them from history"""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)


    def train():

        # set model to training mode, enabling dropout
        model.train()

        total_loss = 0.
        start_time = time.time()

        ntokens = dataset.vocab_size

        # initialize hidden layers
        hidden = model.init_hidden(args.batch_size)

        # for batch, i in enumerate(range(0, data_loader.__len__() - 1, args.bptt)):








if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Main training script for RNN LM.')
    parser.add_argument('--seed', type = int, default = 2020,
                        help = 'seed for reproducibility of stochastic results')
    parser.add_argument('--model', type = str, default = 'GRU',
                        help = 'type of recurrent net (LSTM, GRU)')
    parser.add_argument('--emsize', type = int, default = 200,
                        help = 'size of word embeddings')
    parser.add_argument('--nhid', type = int, default = 200,
                        help = 'number of hidden units per layer')
    parser.add_argument('--nlayers', type = int, default = 2,
                        help = 'number of layers')
    parser.add_argument('--lr', type = float, default = 20,
                        help = 'initial learning rate')
    parser.add_argument('--clip', type = float, default = 0.25,
                        help = 'gradient clipping.') #TODO: Find out what gradient clipping is.
    parser.add_argument('--epochs', type = int, default = 5,
                        help = 'upper limit to number of epochs for training')
    parser.add_argument('--batch_size', type = int, default = 2,
                        help = 'batch size: number of sequences processed per step')
    parser.add_argument('--bptt', type = int, default = 142,
                        help = 'sequence length') #TODO: Why bptt? Is 142 always the sequence length?
    parser.add_argument('--dropout', type = float, default = 0.2,
                        help = 'probability of dropout applied to layers (set to 0 for no dropout)')
    parser.add_argument('--tied', action = 'store_false',
                        help = 'tie the word embedding and softmax weights') # TODO: figure out what this is and if should be store_true by default
    parser.add_argument('--cuda', action = 'store_true',
                        help = 'use CUDA') # TODO: figure out why CUDA isn't working
    parser.add_argument('--log-interval', type = int, default = 200, metavar = 'N',
                        help = 'report / log interval')
    parser.add_argument('--save', type = str, default = 'rnnlm_model.pt',
                        help = 'relative path to save the final model')
    parser.add_argument('--onnx-export', type = str, default = '',
                        help = 'path to export the final model in onnx format') #TODO: What?


    args = parser.parse_args()

    main(args)
