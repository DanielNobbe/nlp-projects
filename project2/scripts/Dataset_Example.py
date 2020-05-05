
import os
import numpy as np

import torch
import torch.utils.data as data
import pickle

from Preprocessing_RNN import SentenceDataset

def main():
    file_path = os.path.dirname(os.path.abspath(__file__))

    relative_pickle_path = '/../Data/Dataset/Dataloader.pkl'
    pickle_path = file_path + relative_pickle_path

    with open(pickle_path, 'rb') as file:
        dataset = pickle.load(file)

    batch_size = 2
    data_loader = data.DataLoader(dataset, batch_size, num_workers=1)

    for sequence in data_loader:
        print(sequence)
        break


if __name__ == '__main__':
    main()