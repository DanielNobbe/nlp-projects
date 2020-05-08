import os
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.utils.data as data

from tokenizers import WordTokenizer


class SentenceDataset(data.Dataset):
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


def remove_closing_bracket(word):
    i = word.index(")")
    return word[:i]


def preprocess_line(line):
    words = line.strip().split(" ")
    new_words = [
        remove_closing_bracket(word) for word in words if not word.startswith("(")
    ]
    # Remove the . at the end of the sentence
    new_words = new_words[:-1]
    preprocessed = " ".join(new_words)
    return preprocessed


def preprocess_lines(lines):
    preprocessed = [preprocess_line(line) for line in lines]
    return preprocessed


def get_datasets(data_path="../Data/Dataset"):
    data_path = Path(data_path)

    training_set_path = data_path / "train"
    test_set_path = data_path / "test"
    val_set_path = data_path / "val"

    with open(training_set_path, "r") as train_file:
        train_text = train_file.readlines()

    with open(val_set_path, "r") as val_file:
        val_text = val_file.readlines()

    with open(test_set_path, "r") as test_file:
        test_text = test_file.readlines()

    tokenizer = WordTokenizer(train_text)

    train_sentences = preprocess_lines(train_text)
    val_sentences = preprocess_lines(val_text)
    test_sentences = preprocess_lines(test_text)

    train_data = SentenceDataset(train_sentences, tokenizer)
    val_data = SentenceDataset(val_sentences, tokenizer)
    test_data = SentenceDataset(test_sentences, tokenizer)

    return train_data, val_data, test_data


if __name__ == "__main__":
    train_data, val_data, test_data = get_datasets("../Data/Dataset")
