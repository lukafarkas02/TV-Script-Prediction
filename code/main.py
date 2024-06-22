import os

# import numpy as np
import torch
# import random
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer('basic_english')

class FriendsDataset(Dataset):
    def __init__(self, data_folder):
        self.data = []
        files = os.listdir(data_folder)
        for file in files:
            with open(os.path.join(data_folder, file), 'r', encoding='utf-8') as f:
                script = f.read()
                self.data.append(script)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_scripts(folder_path):
    return  FriendsDataset(folder_path)




def tokenize_text(text):
    return tokenizer(text)


if __name__ == '__main__':
    dataset = load_scripts("scripts")
    # data = " ".join(dataset.data)
    train_size = int(0.70 * len(dataset.data))
    val_size = int(0.15 * len(dataset.data))
    test_size = len(dataset.data) - train_size - val_size
    print(train_size)
    print(val_size)
    print(test_size)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Primer kako izgleda podatak
    print(train_dataset[0])

    # Primjer tokenizacije teksta


    tokenized_text = tokenize_text(train_dataset[0])
    print(tokenized_text)


