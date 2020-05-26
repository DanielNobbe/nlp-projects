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

        #`clip_grad_norm` is used to help prevent exploding gradient problem in RNNs (GRUs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-optimizer.param_groups[0]['lr'],
                        p.grad.data)


        if batch % args.log_interval == 0 and batch > 0:

            print('| Current epoch: {} | Current loss: {} | Perplexity: {} |'.format(epoch, loss, torch.exp(loss)))


def main(args):

    # If passed through command line, check if CUDA available and use GPU if possible
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device: {}'.format(device))
    else:
        device = torch.device('cpu')
        print('Device: {}'.format(device))

    # Prerequisites for training
    train_data, val_data, test_data = get_datasets()

    # Build model
    vocab_size = train_data.tokenizer.vocab_size
    model = RNNLM(ntoken = vocab_size, ninp = args.emsize, nhid = args.nhid,
                    nlayers = args.nlayers, dropout = args.dropout).to(device)

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

    # Till here

    print('Split sizes | Train: {} | Val: {} | Test: {} |'.format(len(train_loader), len(val_loader),
                                            len(test_loader)))

    optimizer = Adam(model.parameters(), lr = args.lr)
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

            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)

                best_val_loss = val_loss
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


def parse_arguments():
    parser = argparse.ArgumentParser(description = 'Main training script for RNN LM.')
    parser.add_argument('--model', type = str, default = 'GRU',
                        help = 'type of recurrent net (LSTM, GRU)')
    parser.add_argument('--emsize', type = int, default = 512,
                        help = 'size of word embeddings')
    parser.add_argument('--nhid', type = int, default = 512,
                        help = 'number of hidden units per layer')
    parser.add_argument('--nlayers', type = int, default = 1,
                        help = 'number of layers')
    parser.add_argument('--lr', type = float, default = 0.001,
                        help = 'initial learning rate')
    parser.add_argument('--clip', type = float, default = 0.25,
                        help = 'gradient clipping. Helps prevent exploding gradient problem for RNNs.')
    parser.add_argument('--epochs', type = int, default = 10,
                        help = 'upper limit to number of epochs for training')
    parser.add_argument('--batch_size', type = int, default = 32,
                        help = 'batch size: number of sequences processed per step')
    parser.add_argument('--eval_batch_size', type = int, default = 32,
                        help = 'Evaluation batch size: number of sequences processed per step during evaluation')
    parser.add_argument('--dropout', type = float, default = 0.34,
                        help = 'probability of dropout applied to layers (set to 0 for no dropout)')
    parser.add_argument('--cuda', action = 'store_true',
                        help = 'use CUDA')
    parser.add_argument('--log_interval', type = int, default = 200, metavar = 'N',
                        help = 'report / log interval')
    parser.add_argument('--save', type = str, default = 'rnnlm_model.pt',
                        help = 'relative path to save the final model')


    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()

    main(args)
