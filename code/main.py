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




if __name__ == '__main__':
    episodes = load_scripts("scripts")
    print(f"Loaded {len(episodes)} episodes")

    tokenized_texts = [tokenize(script) for script in episodes]
    vocab = build_vocab(tokenized_texts)
    sequences = text_to_sequences(tokenized_texts, vocab)
    sequences = create_sequences(sequences)

    sequences = sequences
    print(f"Number of sequences: {len(sequences)}")





