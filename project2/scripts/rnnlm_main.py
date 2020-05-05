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
from Preprocessing_RNN import SentenceDataset

# Set the random seed manually for reproducibility.
torch.manual_seed(2020)

# Check if GPU available
is_gpu_available = torch.cuda.is_available()
print(is_gpu_available)

if is_gpu_available:
    device = torch.device("cuda")
    print('GPU is available.')
else:
    device = torch.device('cpu')
    print("GPU not available, CPU used instead.")

parser = argparse.ArgumentParser(description = 'Main training script for RNN LM.')
parser.add_argument('--model', type = str, default = 'GRU',
                    help = 'type of recurrent net (LSTM, GRU)')
parser.add_argument('--emsize', type = int, default = 200,
                    help = 'size of word embeddings')
parser.add_argument('--nhid', type = int, default = 200,
                    help = 'number of hidden units per layer')
parser.add_argument('--nlayers', type = int, default = 2,
                    help = 'number of layers')
