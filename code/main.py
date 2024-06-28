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

# Randomness of the script
randomness = {
    "1": 0.5,
    "2": 1,
    "3": 1.5
}


def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')
    # Alternative if above doesn't work
    # print('\033c', end='')

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
    i = 1
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = tuple([each.data for each in hidden])
        model.zero_grad()
        output, hidden = model(inputs, hidden)
        loss = criterion(output.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 100 == 0:
            clear_terminal()
            print("Training...")
            print(f"{i}/{len(dataloader)} done for this epoch.")
        i += 1

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


def generate_text(model, start_text, vocab, idx_to_word):
    clear_terminal()
    max_length = int(input("Length of the script:\n> "))
    temperature_type = input("Randomness: \n1. Low\n2. Normal \n3. High\n> ")
    temperature = randomness[temperature_type]

    model.eval()
    tokens = tokenize(start_text)
    sequences = [vocab.get(token, 0) for token in tokens]
    inputs = torch.tensor(sequences).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)
    generated_text = start_text

    clear_terminal()
    print("Generating text...")

    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = model(inputs, hidden)
            output = output[:, -1, :].squeeze()

            #Applying temperature
            output = output / temperature
            probabilities = torch.softmax(output, dim=-1)
            top_idx = torch.multinomial(probabilities, 1).item()

            # word = idx_to_word[top_idx]
            word = idx_to_word.get(top_idx, '<UNK>')
            generated_text += ' ' + word
            if word[-1] in ['.', '?', '!', ')', ']']:  # TODO: Find a better way to format the generated script.
                generated_text += '\n\n'

            inputs = torch.cat((inputs, torch.tensor([[top_idx]]).to(device)), dim=1)

    return generated_text


def save_model(model):
    file_name = input("Enter the desired file name to save the model:\n> ")
    torch.save(model.state_dict(), "models/"+file_name+".pth")
    print(f"Model saved successfully as {file_name}!")


def load_model(vocab_size, embedding_dim, hidden_dim, num_layers):
    clear_terminal()
    model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
    if not model_files:
        print("No saved model files found.")
        return None

    print("Available model files:")
    for idx, file in enumerate(model_files):
        print(f"{idx + 1}. {file}")

    choice = int(input("Enter the number of the model file to load:\n> ")) - 1
    if choice < 0 or choice >= len(model_files):
        print("Invalid choice.")
        return None

    file_name = model_files[choice]
    file_path = os.path.join('models', file_name)
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    model.load_state_dict(torch.load(file_path))
    model.to(device)
    print(f"Model loaded successfully from {file_name}!")
    return model


# TODO: Separate by modules


if __name__ == '__main__':
    episodes = load_scripts("scripts")
    print(f"Loaded {len(episodes)} episodes")

    tokenized_texts = [tokenize(script) for script in episodes]
    vocab = build_vocab(tokenized_texts)
    vocab_size = len(vocab) + 1

    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU for training.\n")


    while True:
        option = input("1. New Training\n2. Existing Training\n3. Generate Episode\n4. Exit\n> ")

        match option:
            case "1":
                sequences = text_to_sequences(tokenized_texts, vocab)
                sequences = create_sequences(sequences)

                sequences = sequences[:120000]  # TODO: Make it so users can select the amount of training data.
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
                vocab_size = len(vocab) + 1

                model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                num_epochs = 10
                for epoch in range(1, num_epochs + 1):
                    train(model, train_loader, criterion, optimizer, epoch, vocab_size)
                    val_loss, val_accuracy = evaluate(model, val_loader, criterion, vocab_size)
                    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

                # SAVING THE MODEL
                save_model(model)

                # TEST AND ACCURACY
                test_loss, test_accuracy = evaluate(model, test_loader, criterion, vocab_size)
                print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

            case "2":
                model = load_model(vocab_size, embedding_dim, hidden_dim, num_layers)

            case "3":
                try:
                    idx_to_word = {idx: word for word, idx in vocab.items()}
                    start_text = "New episode: " + '\n\n\n'
                    generated_text = generate_text(model, start_text, vocab, idx_to_word) + '\n'
                    print(generated_text)
                    with open("generated_scripts/script.txt", "a") as text_file:
                        text_file.write(generated_text)
                except Exception as e:  # TODO: What exception is this???
                    print("Unable to generate text, no model present!")
            case _:
                break
