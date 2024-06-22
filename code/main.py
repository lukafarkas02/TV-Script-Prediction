import os
import numpy as np
# import random
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.nn.utils.rnn import pad_sequence

tokenizer = get_tokenizer('basic_english')
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

class FriendsDataset(Dataset):
    def __init__(self, data_folder):
        self.data = []
        files = os.listdir(data_folder)
        for file in files:
            with open(os.path.join(data_folder, file), 'r', encoding='utf-8') as f:
                script = f.read()
                self.data.extend([ord(c) for c in script])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        for script in self.data:
            yield script

def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset.indices[i:i + lookback]
        target = dataset.indices[i + 1:i + lookback + 1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)


def yield_tokens(data_iter):
    for text in data_iter:
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        normalized_tokens = [word.lower() for word in filtered_tokens if word.isalnum()]
        yield normalized_tokens

# def text_to_vector(text):
#     tokens = word_tokenize(text)
#     filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
#     normalized_tokens = [word.lower() for word in filtered_tokens if word.isalnum()]
#     return [vocab[token] for token in normalized_tokens]

def load_scripts(folder_path):
    return FriendsDataset(folder_path)


def tokenize_text(text):
    return tokenizer(text)


if __name__ == '__main__':
    dataset = load_scripts("scripts")

    train_size = int(0.70 * len(dataset.data))
    val_size = int(0.15 * len(dataset.data))
    test_size = len(dataset.data) - train_size - val_size

    # print(train_size)
    # print(val_size)
    # print(test_size)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    # print(len(tokenized_text))

    X_train, y_train = create_dataset(train_dataset, 2)
    # print(X_train)
    # input()
    # print(y_train)

    # vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
    # vocab.set_default_index(vocab["<unk>"])
    #
    # counter = 0
    # for word, index in vocab.get_stoi().items():
    #     counter+=1
    #     # print(f'Word: {word}, Index: {index}')
    #
    # print(counter)
    # vectors = [torch.tensor(text_to_vector(text), dtype=torch.long) for text in dataset]
    #
    # padded_vectors = pad_sequence(vectors, batch_first=True, padding_value=0)
    # print(padded_vectors)


