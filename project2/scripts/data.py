from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizers import WordTokenizer
from preprocessing import preprocess_lines


class SentenceDataset(Dataset):
    def __init__(self, sentences, tokenizer):
        self.sentences = sentences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(self.sentences[idx], add_special_tokens=True)
        input_encoded = encoded[:-1]
        target_encoded = encoded[1:]
        return input_encoded, target_encoded


def padded_collate(batch, pad_idx=0):
    """Pad sentences, return sentences and labels as LongTensors."""
    sentences, targets = zip(*batch)
    lengths = [len(s) for s in sentences]
    max_length = max(lengths)
    # Pad each sentence with zeros to max_length
    padded_sentences = [s + [pad_idx] * (max_length - len(s)) for s in sentences]
    padded_targets = [s + [pad_idx] * (max_length - len(s)) for s in targets]

    return torch.LongTensor(padded_sentences), torch.LongTensor(padded_targets), lengths


def get_datasets(data_path="../Data/Dataset"):
    file_directory = Path(__file__).parent.absolute()
    data_path = file_directory / Path(data_path)

    training_set_path = data_path / "train"
    test_set_path = data_path / "test"
    val_set_path = data_path / "val"

    with open(training_set_path, "r") as train_file:
        train_text = train_file.readlines()

    with open(val_set_path, "r") as val_file:
        val_text = val_file.readlines()

    with open(test_set_path, "r") as test_file:
        test_text = test_file.readlines()

    train_sentences = preprocess_lines(train_text)
    val_sentences = preprocess_lines(val_text)
    test_sentences = preprocess_lines(test_text)

    tokenizer = WordTokenizer(train_sentences)

    train_data = SentenceDataset(train_sentences, tokenizer)
    val_data = SentenceDataset(val_sentences, tokenizer)
    test_data = SentenceDataset(test_sentences, tokenizer)

    return train_data, val_data, test_data


if __name__ == "__main__":
    train_data, val_data, test_data = get_datasets("../Data/Dataset")
    train_loader = DataLoader(
        train_data, batch_size=64, shuffle=True, collate_fn=padded_collate
    )
    val_loader = DataLoader(
        train_data, batch_size=256, shuffle=False, collate_fn=padded_collate
    )

    for a in islice(train_loader, 10):
        print(a)
