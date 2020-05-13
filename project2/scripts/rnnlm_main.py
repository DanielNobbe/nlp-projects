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
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
import torch.onnx
from torch.optim import Adam
import pickle

from rnnlm_model import RNNLM
#import rnnlm_model
#from Preprocessing_RNN import SentenceDataset

from data import padded_collate, get_datasets


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
        print("GPU not available, CPU used instead.")

    # Load data

    # # Get path of current directory
    # file_path = os.path.dirname(os.path.abspath(__file__))
    #
    # # Find data path relative to current directory
    # relative_pickle_path = '/../Data/Dataset/Dataloader.pkl'
    # pickle_path = file_path + relative_pickle_path
    #
    # # Open and unpickle Dataloader.pkl
    # with open(pickle_path, 'rb') as file:
    #     dataset = pickle.load(file)
    # data_loader = data.DataLoader(dataset, args.batch_size, num_workers = 1)
    # train_data = data.DataLoader(dataset._train_data,
    #                             args.batch_size, num_workers = 1)
    #
    # # print a couple of sequences to see if it works.
    # for sequence in data_loader:
    #     print(sequence)
    #     print(type(sequence))
    #     print(sequence.size())
    #     break



    # Build RNN LM Model
    #ntokens = dataset.vocab_size
    # print(ntokens)
    # model = RNNLM(ntoken = ntokens, ninp = args.emsize, nhid = args.nhid,
    #                 nlayers = args.nlayers, dropout = args.dropout).to(device)
    #
    # use negative log-likelihood as loss / objective
    # criterion = cross_entropy(ignore_index=0)

    # Prerequisites for training


    train_data, val_data, test_data = get_datasets()

    # Build model
    vocab_size = train_data.tokenizer.vocab_size #TODO: shouldn't this be taken over the entire dataset and not just the training set?
    model = RNNLM(ntoken = vocab_size, ninp = args.emsize, nhid = args.nhid,
                    nlayers = args.nlayers, dropout = args.dropout).to(device)

    train_loader = DataLoader(
        train_data, batch_size = args.batch_size, shuffle = True,
        collate_fn = padded_collate, num_workers = 1
    )

    val_loader = DataLoader(
        val_data, batch_size = args.eval_batch_size, shuffle = False, #TODO should this be true or false?
        collate_fn = padded_collate, num_workers = 1
    )

    test_loader = DataLoader(
        test_data, batch_size = args.eval_batch_size, shuffle = False, #TODO should this be true or false?
        collate_fn = padded_collate, num_workers = 1
    )

    # optimizer = Adam(model.parameters(), lr = args.lr)

    def repackage_hidden(h):
        """Wraps hidden states in new tensors, to detatch them from history"""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)


    def evaluate(data_loader):

        # Disable dropout by switching to evaluation mode
        model.eval()

        total_loss = 0.

        vocab_size = train_data.tokenizer.vocab_size #TODO: again, shouldn't this be the entire corpus vocab size? And how do we find this?

        hidden = model.init_hidden(args.eval_batch_size)

        with torch.no_grad():
            for batch, (source_batch, target_batch, lengths) in enumerate(data_loader, 0):

                output, hidden = model(source_batch, hidden, lengths)

                hidden = repackage_hidden(hidden)

                batches, seq_length, vocab_size = output.shape

                pred = output.view(batches * seq_length, vocab_size)
                target = target_batch.view(batches * seq_length)

                total_loss += cross_entropy(pred, target, ignore_index = 0, reduction = 'sum').item() / batches #TODO: we need to figure out the appropriate way of calculating this

            return total_loss /  (len(data_loader) - 1)



    def train():

    # Turn on training mode to enable dropout
        model.train()
        total_loss = 0.
        start_time = time.time()

        # initialize hidden variables of RNN
        hidden = model.init_hidden(args.batch_size)

        #for input_sentences_batch, target_sentences_batch, lengths in train_loader:
        for batch, (input_sentences_batch, target_sentences_batch, lengths) in enumerate(train_loader, 0):

            model.zero_grad()

            hidden = repackage_hidden(hidden)

            output, hidden = model(input_sentences_batch, hidden, lengths)

            batches, seq_length, vocab_size = output.shape

            pred = output.view(batches * seq_length, vocab_size)
            target = target_sentences_batch.view(batches * seq_length)

            # loss = criterion(output, target_sentences_batch)
            loss = cross_entropy(pred, target, ignore_index = 0, reduction = 'sum') / batches #TODO: should reduction be "sum" too?
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # `clip_grad_norm` is used to help prevent exploding gradient problem in RNNs (GRUs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)

            total_loss += loss.item()

            # Evaluate loss every args.log_interval steps
            if batch % args.log_interval == 0 and batch > 0:

                current_loss = total_loss / args.log_interval

                elapsed = time.time() - start_time

                print('| Current epoch: {:3d} | Learning rate: {:02.2f} | Current loss: {:5.2f} | Perplexity: {:8.2f} | Total loss {:5.2f}'.format(
                        epoch, lr, current_loss, math.exp(1), total_loss))

                total_loss = 0
                start_time = time.time()

            # To speed things up during debugging. TODO: remove
            if batch == 40:
                break







    # initial learning rate
    lr = args.lr
    optimizer = Adam(model.parameters(), lr = lr)

    # store best validation loss
    best_val_loss = None

    # Use Ctrl + C to break out of training at any time
    try:
        for epoch in range(1, args.epochs + 1):

            epoch_start_time = time.time()

            train()

            val_loss = evaluate(val_loader)

            print('-' * 89)

            print('| End of epoch {:3d} | Time: {:5.2f} | Validation loss: {:5.2f} |'.format(epoch,
             (time.time() - epoch_start_time), val_loss))

            print('-' * 89)

            #TODO: Save the model with the best validation loss until now in a folder containing all models in separate directories
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)

                best_val_loss = val_loss
            else:
                # Anneal the learning rate if we do not see improvement in validation loss
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Terminating training early.')



    # Load best model
    with open(args.save, 'rb') as f:
        model = torch.load(f)

        # Ensure rnn parameters are a continuous chunk of memory
        model.rnn.flatten_parameters()


    # Evaluate best model on test data
    test_loss = evaluate(test_loader)

    print('=' * 89)
    print('|End of training and testing. | Test loss {:5.2f}'.format(test_loss))
    print('=' * 89)











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
    parser.add_argument('--lr', type = float, default = 1,
                        help = 'initial learning rate')
    parser.add_argument('--clip', type = float, default = 0.25,
                        help = 'gradient clipping.') #TODO: Find out what gradient clipping is. --> Helps prevent exploding gradient problem for RNNs
    parser.add_argument('--epochs', type = int, default = 5,
                        help = 'upper limit to number of epochs for training')
    parser.add_argument('--batch_size', type = int, default = 2,
                        help = 'batch size: number of sequences processed per step')
    parser.add_argument('--eval_batch_size', type = int, default = 2,
                        help = 'Evaluation batch size: number of sequences processed per step during evaluation')
    parser.add_argument('--bptt', type = int, default = 142,
                        help = 'sequence length') #TODO: Why bptt? Is 142 always the sequence length?
    parser.add_argument('--dropout', type = float, default = 0.2,
                        help = 'probability of dropout applied to layers (set to 0 for no dropout)')
    parser.add_argument('--tied', action = 'store_false',
                        help = 'tie the word embedding and softmax weights') # TODO: figure out what this is and if should be store_true by default
    parser.add_argument('--cuda', action = 'store_true',
                        help = 'use CUDA') # TODO: figure out why CUDA isn't working
    parser.add_argument('--log_interval', type = int, default = 20, metavar = 'N',
                        help = 'report / log interval')
    parser.add_argument('--save', type = str, default = 'rnnlm_model.pt',
                        help = 'relative path to save the final model')
    parser.add_argument('--onnx-export', type = str, default = '',
                        help = 'path to export the final model in onnx format') #TODO: What?


    args = parser.parse_args()

    main(args)
