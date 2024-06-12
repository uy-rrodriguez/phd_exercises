"""
Tools to train and execute a Deep Neural Network with PyTorch, using
EmbeddingBag.

https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
"""

import collections
import csv
import time
import typing

from nltk import word_tokenize
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, Dataset


batch_size = 16
embeddings_size = 64
log_on = False
log_interval = 200
label_map = {
    "neutral": 0,
    "positive": 1,
    "negative": 2,
}
unknown_word = "<unk>"
lexicon: collections.OrderedDict[str, typing.Any] = collections.OrderedDict()
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class CustomTweetsDataset(Dataset):
    """
    Dataset representing Tweets with an emotion label.
    """
    def __init__(self, tweets_file, transform=None, target_transform=None):
        with open(tweets_file, "r") as f:
            reader = csv.reader(f, delimiter='\t', quotechar='"')
            self.lines = [line for line in reader]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        sample = self.lines[idx]
        label = sample[1]
        message = sample[2]
        if self.transform:
            message = self.transform(message)
        if self.target_transform:
            label = self.target_transform(label)
        return label, message


def yield_tokens(train_file_path: str) -> list[str]:
    """
    Generator that returns each tweet of the given file, line by line, as a list
    of words.
    """
    with open(train_file_path, "r") as f:
        reader = csv.reader(f, delimiter='\t', quotechar='"')
        for line in reader:
            _id, emotion, text = line
            yield word_tokenize(text)


def text_pipeline(text: str) -> list[int]:
    """
    Transforms the text in a list of word identifiers as defined in the
    `lexicon`.
    """
    return [lexicon.get(w, lexicon[unknown_word]) for w in word_tokenize(text)]


def label_pipeline(label: str) -> int:
    """
    Transforms the label in a numeric identifier as defined in `label_map`.
    """
    return label_map[label]


def collate_batch(batch):
    """
    Function to use in DataLoader to collate samples in a single batch.
    """
    label_list, text_list, offset_list = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offset_list.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offset_list = torch.tensor(offset_list[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offset_list


class TextClassificationModel(nn.Module):
    """
    Neuronal Network to classify text, defined by the layers:
        - EmbeddingBag:
            Transforms a tensor of concatenated samples into embeddings of equal
            length.
        - Linear:
            y = W.x + b
            Predicts an output class based on the embedding input.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_class: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        """
        Define initial weights and bias for each layer.
        """
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()

    def forward(self, text: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        `offsets` will be used to determine the position of each sample
        concatenated in `text`.
        """
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


def train(model: nn.Module, dataloader: DataLoader, criterion: _Loss,
          optimiser: torch.optim.Optimizer):
    """
    Trains the model with the given DataLoader, using `criterion` as loss
    function and optimising parameters with the strategy defined by `optimiser`.
    """
    model.train()
    size = len(dataloader.dataset)
    for idx, (label, text, offsets) in enumerate(dataloader):
        label, text, offsets = label.to(device), text.to(device), offsets.to(device)

        # Reset gradients
        optimiser.zero_grad()
        # Predict label
        predicted_label: torch.Tensor = model(text, offsets)
        # Calculate loss
        loss = criterion(predicted_label, label)
        # Back-propagation (calculate gradients)
        loss.backward()
        # Normalise gradients
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        # Optimise (adjust) parameters
        optimiser.step()

        if log_on and idx % log_interval == 0 and idx > 0:
            loss, current = loss.item(), (idx + 1) * batch_size
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: _Loss):
    """
    Evaluates the model with the given DataLoader, using `criterion` as loss
    function.
    """
    model.eval()
    num_batches = len(dataloader)
    total_loss, total_acc, total_count = 0, 0, 0
    with torch.no_grad():
        label: torch.Tensor
        text: torch.Tensor
        offsets: torch.Tensor
        for idx, (label, text, offsets) in enumerate(dataloader):
            label, text, offsets = label.to(device), text.to(device), offsets.to(device)
            predicted_label = model(text, offsets)
            total_loss += criterion(predicted_label, label).item()
            total_acc += (predicted_label.argmax(1) == label).type(torch.float).sum().item()
            total_count += label.size(0)
    total_loss /= num_batches
    total_acc /= total_count
    print(f"Test Error: \n Accuracy: {(100 * total_acc):>0.1f}%, Avg loss: {total_loss:>8f}")
    return total_acc


def main():
    """
    Main program.
    """
    # Load datasets
    train_ds = CustomTweetsDataset("res/twitter-2013train-A.txt")
    dev_ds = CustomTweetsDataset("res/twitter-2013dev-A.txt")
    test_ds = CustomTweetsDataset("res/twitter-2013test-A.txt")

    # Initiate lexicon from training data
    lexicon[unknown_word] = 0
    for i in range(len(train_ds)):
        _label, message = train_ds[i]
        for w in word_tokenize(message):
            if w not in lexicon:
                lexicon[w] = len(lexicon)

    # Create data loaders.
    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    dev_dataloader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    test_dataloader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    for y, X, offsets in train_dataloader:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        print(f"Shape of offsets: {offsets.shape}")
        break

    # Create model
    model = TextClassificationModel(len(lexicon), embeddings_size, len(label_map)).to(device)
    print(model)

    # Define training parameters
    epochs = 5
    learning_rate = 10  # Initial rate, will change with scheduler
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    # Stochastic gradient descent
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, 1, gamma=0.1)
    max_acc = None

    # Train model with training dataset
    for e in range(epochs):
        print("-" * 59 + f"\nEpoch {e + 1}\n" + "-" * 59)
        epoch_start_time = time.time()
        train(model, train_dataloader, criterion, optimiser)
        dev_acc = evaluate(model, dev_dataloader, criterion)
        if max_acc is not None and max_acc > dev_acc:
            # Decrease learning rate if accuracy dropped
            lr_scheduler.step()
        else:
            max_acc = dev_acc
        print(f"Time: {(time.time() - epoch_start_time):>5.2f}s")

    # Evaluate model with test dataset
    print("-" * 59)
    print("Checking the results of test dataset.")
    evaluate(model, test_dataloader, criterion)
    print("\nDone!")


if __name__ == "__main__":
    main()
