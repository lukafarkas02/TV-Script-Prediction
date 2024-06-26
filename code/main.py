import os
import glob
import torch
import numpy as np
from torch import device
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn


def load_scripts(path):
    scripts = []
    for filename in glob.glob(os.path.join(path, '*.txt')):
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            scripts.append(content)
    return scripts


def tokenize(text):
    return text.split()


def build_vocab(tokenized_texts):
    counter = Counter()
    for text in tokenized_texts:
        counter.update(text)
    return {word: idx for idx, (word, _) in enumerate(counter.items(), 1)}


def text_to_sequences(tokenized_texts, vocab):
    return [[vocab[word] for word in text if word in vocab] for text in tokenized_texts]


def create_sequences(tokenized_texts, sequence_length=50):
    sequences = []
    for text in tokenized_texts:
        for i in range(0, len(text) - sequence_length):
            sequences.append(text[i:i + sequence_length + 1])
    return sequences


class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx][:-1]), torch.tensor(self.sequences[idx][1:])

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets


if __name__ == '__main__':
    episodes = load_scripts("scripts")
    print(f"Loaded {len(episodes)} episodes")

    tokenized_texts = [tokenize(script) for script in episodes]
    vocab = build_vocab(tokenized_texts)
    sequences = text_to_sequences(tokenized_texts, vocab)
    sequences = create_sequences(sequences)

    sequences = sequences
    print(f"Number of sequences: {len(sequences)}")

    train_data, test_data = train_test_split(sequences, test_size=0.2)
    train_data, val_data = train_test_split(train_data, test_size=0.2)

    train_dataset = TextDataset(train_data)
    val_dataset = TextDataset(val_data)
    test_dataset = TextDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")






