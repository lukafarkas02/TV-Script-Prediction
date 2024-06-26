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


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_(),
                weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_())


def train(model, dataloader, criterion, optimizer, epoch, vocab_size):
    model.train()
    total_loss = 0
    hidden = model.init_hidden(dataloader.batch_size)
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        hidden = tuple([each.data for each in hidden])
        model.zero_grad()
        output, hidden = model(inputs, hidden)
        loss = criterion(output.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}")


def evaluate(model, dataloader, criterion, vocab_size):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_count = 0
    hidden = model.init_hidden(dataloader.batch_size)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = tuple([each.data for each in hidden])
            output, hidden = model(inputs, hidden)

            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()

            _, predicted = torch.max(output, dim=2)
            total_correct += (predicted == targets).sum().item()
            total_count += targets.numel()

    accuracy = total_correct / total_count
    return total_loss / len(dataloader), accuracy



def generate_text(model, start_text, vocab, idx_to_word, max_length=1000):
    model.eval()
    tokens = tokenize(start_text)
    sequences = [vocab.get(token, 0) for token in tokens]
    inputs = torch.tensor(sequences).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)

    generated_text = start_text

    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = model(inputs, hidden)
            output = output[:, -1, :]
            _, top_idx = torch.topk(output, 1)
            top_idx = top_idx.item()
            word = idx_to_word[top_idx]
            generated_text += ' ' + word
            inputs = torch.cat((inputs, torch.tensor([[top_idx]]).to(device)), dim=1)

    return generated_text


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(vocab) + 1
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2

    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch, vocab_size)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, vocab_size)
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    # TEST AND ACCURACY
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, vocab_size)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    idx_to_word = {idx: word for word, idx in vocab.items()}
    # print(idx_to_word)

    # New episode
    start_text = "New episode: " + '\n\n\n'
    generated_text = generate_text(model, start_text, vocab, idx_to_word)
    print(generated_text)





