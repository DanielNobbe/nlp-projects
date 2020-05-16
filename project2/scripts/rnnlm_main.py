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

# def repackage_hidden(h):
#     """Wraps hidden states in new tensors, to detatch them from history"""

#     if isinstance(h, torch.Tensor):
#         return h.detach()
#     else:
#         return tuple(repackage_hidden(v) for v in h)


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

    #hidden = model.init_hidden(args.eval_batch_size)

    with torch.no_grad():
        for batch, (source_batch, target_batch, lengths) in enumerate(data_loader, 0):

            # TODO: See if this works by initialising for every batch
            hidden = model.init_hidden(args.batch_size)
            #print(len(lengths))

            if len(lengths) != args.eval_batch_size:
                hidden = last_hidden

            # hidden = repackage_hidden(hidden)

            output, hidden = model(source_batch.to(device), hidden.to(device), lengths)

            # hidden = repackage_hidden(hidden)

            batches, seq_length, vocab_size = output.shape

            pred = output.view(batches * seq_length, vocab_size)
            target = target_batch.view(batches * seq_length)

            nll = cross_entropy(pred, target, ignore_index = 0, reduction = "none") #TODO
            nll = nll.sum(-1)
            loss = nll.mean()
            total_loss += loss.item()

            #total_loss += cross_entropy(pred, target, ignore_index = 0, reduction = 'sum').item() / batches #TODO: we need to figure out the appropriate way of calculating this

        return total_loss /  (len(data_loader) - 1) # TODO: why minus 1?



def train(model, train_data, train_loader, args, device, optimizer, epoch):

# Turn on training mode to enable dropout
    
    # total_loss = 0.
    # start_time = time.time()

    # Adapt dimensions of hidden state to match last batch size if dataset size isnt multiple of batch size
    # adapt_last_batch = False
    # batch_mod_diff = len(train_data) % args.batch_size
    # if batch_mod_diff != 0:
        # last_hidden = model.init_hidden((batch_mod_diff))
        # adapt_last_batch = True

    # initialize hidden variables of RNN
    #hidden = model.init_hidden(args.batch_size)

    #for input_sentences_batch, target_sentences_batch, lengths in train_loader:
    for batch, (input_sentences_batch, target_sentences_batch, lengths) in enumerate(train_loader, 0):

        # TODO: See if this works by initialising for every batch
        hidden = model.init_hidden(args.batch_size)

        # model.zero_grad()

        if len(lengths) != args.batch_size:
            hidden = last_hidden

        # hidden = repackage_hidden(hidden)

        output, hidden = model(input_sentences_batch.to(device),
                                hidden.to(device), lengths) #TODO: should both the input and hidden be .to(device)?

        batches, seq_length, vocab_size = output.shape

        # pred = output.view(batches * seq_length, vocab_size)
        # target = target_sentences_batch.view(batches * seq_length).to(device)

        # loss = criterion(output, target_sentences_batch)
        #loss = cross_entropy(pred, target, ignore_index = 0, reduction = 'sum') / batches #TODO: should reduction be "sum" too?
        # nll = cross_entropy(pred, target, ignore_index = 0) 

        output = output.transpose(1,2) # transpose because cross_entropy expects number of classes as second dim

        target = target_sentences_batch.to(device)
        # print(output.shape, target.shape) # targets is not one-hot encoded, it is purely index-encoded.
        # print(train_loader.dataset.tokenizer.pad_token_id) # <-- this works
        nll = cross_entropy(output, target, ignore_index = train_loader.dataset.tokenizer.pad_token_id)




        # print("NLL: ", nll)
        loss = nll
        # nll = nll.sum(-1)
        # loss = nll.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # `clip_grad_norm` is used to help prevent exploding gradient problem in RNNs (GRUs)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)

        # total_loss += loss.item()
        print("Loss: ", loss.item())
        # Evaluate loss every args.log_interval steps
        # if batch % args.log_interval == 0 and batch > 0:

            # current_loss = total_loss / args.log_interval

            # elapsed = time.time() - start_time

            # print('| Current epoch: {:3d} | Learning rate: {:02.6f} | Current loss: {:5.2f} | Perplexity: {:8.2f} | Total loss {:5.2f}'.format(
                    # epoch, lr, current_loss, math.exp(1), total_loss))

            # total_loss = 0
            # start_time = time.time()

            # TODO: remove before tuning
            #break
        #To speed things up during debugging. TODO: remove
        # if batch == 1:
        break



def main(args, layers, emsize, nhid):

    # lr = lr
    layers = layers
    emsize = emsize
    nhid = nhid

    # Set the random seed manually for reproducibility.
    # torch.manual_seed(args.seed)

    # Use GPU is possible
    # if args.cuda:
    # Check if GPU available
    if torch.cuda.is_available() and args.cuda:
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
    vocab_size = train_data.tokenizer.vocab_size 
    model = RNNLM(ntoken = vocab_size, ninp = emsize, nhid = nhid,
                    nlayers = layers, dropout = args.dropout).to(device)

    train_loader = DataLoader(
        train_data, batch_size = args.batch_size, shuffle = False, # TODO: Revert to shuffling
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

    # TODO: remove
    # small_loader = DataLoader(
    #     small_data, batch_size = args.batch_size, shuffle = True,
    #     collate_fn = padded_collate, num_workers = 1
    # )
    # print('Small loader')
    # print(len(small_loader))

    # print('Small data')
    # print(len(small_data))

    # train_data = small_data
    # val_data = small_data
    # train_loader = small_loader
    # val_loader = small_loader

    # Till here

    # optimizer = Adam(model.parameters(), lr = args.lr)








    # initial learning rate
    #lr = args.lr
    optimizer = Adam(model.parameters())
    print(model)
    # store best validation loss
    best_val_loss = None
    model.train() # TODO: move this back to previous place
    # Use Ctrl + C to break out of training at any time
    # try:
    for epoch in range(1, args.epochs + 1):

        epoch_start_time = time.time()

        train(model, train_data, train_loader, args, device, optimizer, epoch)

            # val_loss = evaluate(val_loader, val_data, device, model)

            # print('-' * 89)

            # print('| End of epoch {:3d} | Time: {:5.2f} | Validation loss: {:5.2f} |'.format(epoch,
            #  (time.time() - epoch_start_time), val_loss))

            # print('-' * 89)

            #TODO: Save the model with the best validation loss until now in a folder containing all models in separate directories
            # if not best_val_loss or val_loss < best_val_loss:
            #     with open(args.save, 'wb') as f:
            #         torch.save(model, f)

            #     best_val_loss = val_loss
            # else:
            #     # Anneal the learning rate if we do not see improvement in validation loss
            #     lr /= 1.0
    # except KeyboardInterrupt:
    #     print('-' * 89)
    #     print('Terminating training early.')



    # Load best model
    with open(args.save, 'rb') as f:
        model = torch.load(f)

        # Ensure rnn parameters are a continuous chunk of memory
        model.rnn.flatten_parameters()


    # Evaluate best model on test data
    #test_loss = evaluate(test_loader, test_data)
    test_loss = evaluate(val_loader, val_data, device, model)

    print('=' * 89)
    print('|End of training and testing. | Test loss {:5.2f}'.format(test_loss))
    print('=' * 89)





    return test_loss, model





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
    parser.add_argument('--clip', type = float, default = 10,
                        help = 'gradient clipping.') #TODO: Find out what gradient clipping is. --> Helps prevent exploding gradient problem for RNNs
    parser.add_argument('--epochs', type = int, default = 100,
                        help = 'upper limit to number of epochs for training')
    parser.add_argument('--batch_size', type = int, default = 64,
                        help = 'batch size: number of sequences processed per step')
    parser.add_argument('--eval_batch_size', type = int, default = 64,
                        help = 'Evaluation batch size: number of sequences processed per step during evaluation')
    parser.add_argument('--bptt', type = int, default = 142,
                        help = 'sequence length') #TODO: Why bptt? Is 142 always the sequence length?
    parser.add_argument('--dropout', type = float, default = 0.0,
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

    #main(args)

    lr_vec = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    # lr_vec = [0.05, 0.1]

    layers_vec = [1, 2, 8, 16, 32, 64]
    # layers_vec = [1, 2]

    emsize_vec = [128, 256, 512, 1024]
    # emsize_vec = [128, 256]

    nhid_vec = [128, 256, 512, 1024, 2048]
    # nhid_vec = [128, 256]

    best_test_loss = None
    best_model = None
    best_hp = {}

    # for lr in lr_vec:
    #     for layers in layers_vec:
    #         for emsize in emsize_vec:
    #             for nhid in nhid_vec:

    current_test_loss, current_model = main(args, 1,
                                            256, 256)
    # if not best_test_loss or current_test_loss < best_test_loss:
    #     best_test_loss = current_test_loss
    #     best_model = current_model
    #     best_hp['Learning rate'] = lr
    #     best_hp['Layers'] = layers
    #     best_hp['Embedding size'] = emsize
    #     best_hp['Hidden units'] = nhid



    # print('Best hyperparameters')
    # print(best_hp)
    # with open(args.save_best, 'wb') as f:
    #     torch.save(best_model, f)