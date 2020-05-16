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
from data import padded_collate, get_datasets

def evaluate(data_loader, dataset, device, model):

    # Disable dropout by switching to evaluation mode
    model.eval()

    total_loss = 0.

    # TODO: find more elegant solution for this last batch problem
    # Adapt dimensions of hidden state to match last batch size if dataset size isnt multiple of batch size
    adapt_last_batch = False
    batch_mod_diff = len(dataset) % args.eval_batch_size
    if batch_mod_diff != 0:
        last_hidden = model.init_hidden((batch_mod_diff))
        adapt_last_batch = True

    with torch.no_grad():
        for batch, (source_batch, target_batch, lengths) in enumerate(data_loader, 0):

            hidden = model.init_hidden(args.eval_batch_size)

            if len(lengths) != args.eval_batch_size:
                hidden = last_hidden


            output, hidden = model(source_batch.to(device), hidden.to(device), lengths)

            # hidden = repackage_hidden(hidden)

            batches, seq_length, vocab_size = output.shape

            output = output.transpose(1, 2) # transpose output because cross_entropy expected the number of classes as second dim.

            target = target_batch.to(device)

            nll = cross_entropy(output, target, ignore_index = data_loader.dataset.tokenizer.pad_token_id)

            loss = nll

            total_loss += loss.item()

        return total_loss /  len(data_loader)


def train(model, train_data, train_loader, args, device, optimizer, epoch):

# Turn on training mode to enable dropout
    model.train()

    # Adapt dimensions of hidden state to match last batch size if dataset size isnt multiple of batch size
    adapt_last_batch = False
    batch_mod_diff = len(train_data) % args.batch_size
    if batch_mod_diff != 0:
        last_hidden = model.init_hidden((batch_mod_diff))
        adapt_last_batch = True

    for batch, (input_sentences_batch, target_sentences_batch, lengths) in enumerate(train_loader, 0):

        hidden = model.init_hidden(args.batch_size)

        if len(lengths) != args.batch_size:
            hidden = last_hidden

        output, hidden = model(input_sentences_batch.to(device),
                                hidden.to(device), lengths)

        batches, seq_length, vocab_size = output.shape

        ################
        # Daniel's idea:
        output = output.transpose(1, 2) # transpose output because cross_entropy expected the number of classes as second dim.

        target = target_sentences_batch.to(device)

        nll = cross_entropy(output, target, ignore_index = train_loader.dataset.tokenizer.pad_token_id)

        loss = nll

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % args.log_interval == 0 and batch > 0:

            print('| Current epoch: {} | Current loss: {} | Perplexity: {} |'.format(epoch, loss, torch.exp(loss)))


def main(args, layers, emsize, nhid):

    layers = layers
    emsize = emsize
    nhid = nhid

    # Check if GPU available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('GPU is available.')
    else:
        device = torch.device('cpu')
        print("GPU not available, CPU used instead.")

    # Prerequisites for training
    train_data, val_data, test_data, small_data = get_datasets()

    # Build model
    vocab_size = train_data.tokenizer.vocab_size #TODO: shouldn't this be taken over the entire dataset and not just the training set?
    model = RNNLM(ntoken = vocab_size, ninp = emsize, nhid = nhid,
                    nlayers = layers, dropout = args.dropout).to(device)

    train_loader = DataLoader(
        train_data, batch_size = args.batch_size, shuffle = False,
        collate_fn = padded_collate, num_workers = 1
    )

    val_loader = DataLoader(
        val_data, batch_size = args.eval_batch_size, shuffle = False,
        collate_fn = padded_collate, num_workers = 1
    )

    test_loader = DataLoader(
        test_data, batch_size = args.eval_batch_size, shuffle = False,
        collate_fn = padded_collate, num_workers = 1
    )

    # TODO: remove
    small_loader = DataLoader(
        small_data, batch_size = args.batch_size, shuffle = True,
        collate_fn = padded_collate, num_workers = 1
    )
    # Uncomment for quick testing/debugging
    # print('Small loader')
    # print(len(small_loader))
    #
    # print('Small data')
    # print(len(small_data))
    #
    # train_data = small_data
    # val_data = small_data
    # test_data = small_data
    # train_loader = small_loader
    # val_loader = small_loader
    # test_loader = small_loader

    # Till here

    print('Split sizes | Train: {} | Val: {} | Test: {} |'.format(len(train_loader), len(val_loader),
                                            len(test_loader)))

    optimizer = Adam(model.parameters())
    print(model)

    # store best validation loss
    best_val_loss = None

    # Use Ctrl + C to break out of training at any time
    try:
        for epoch in range(1, args.epochs + 1):

            epoch_start_time = time.time()

            train(model, train_data, train_loader, args, device, optimizer, epoch)

            val_loss = evaluate(val_loader, val_data, device, model)

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
                lr /= 1.25
    except KeyboardInterrupt:
        print('-' * 89)
        print('Terminating training early.')



    # Load best model
    with open(args.save, 'rb') as f:
        model = torch.load(f)

        # Ensure rnn parameters are a continuous chunk of memory
        model.rnn.flatten_parameters()

    test_loss = evaluate(test_loader, test_data, device, model)

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
    parser.add_argument('--lr', type = float, default = 0.001,
                        help = 'initial learning rate')
    parser.add_argument('--clip', type = float, default = 0.25,
                        help = 'gradient clipping.') #TODO: Find out what gradient clipping is. --> Helps prevent exploding gradient problem for RNNs
    parser.add_argument('--epochs', type = int, default = 5,
                        help = 'upper limit to number of epochs for training')
    parser.add_argument('--batch_size', type = int, default = 64,
                        help = 'batch size: number of sequences processed per step')
    parser.add_argument('--eval_batch_size', type = int, default = 64,
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
    parser.add_argument('--save_best', type = str, default = 'best_rnnlm_model.pt',
                        help = 'relative path to save best model after tuning.')


    args = parser.parse_args()

    main(args, args.nlayers, args.emsize, args.nhid)
