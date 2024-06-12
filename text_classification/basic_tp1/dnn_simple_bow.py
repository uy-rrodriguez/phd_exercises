"""
Tools to train and execute a Deep Neural Network with PyTorch, using the technique Bag of Words.

https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
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
log_on = False
log_interval = 200
label_map = {
    "neutral": 0,
    "positive": 1,
    "negative": 2,
}
unknown_word = "<unk>"
lexicon: collections.OrderedDict[str, typing.Any] = collections.OrderedDict()


class CustomTweetsDataset(Dataset):
    """
    Dataset representing Tweets with an emotion label.
    """
    def __init__(self, tweets_file):
        with open(tweets_file, "r") as f:
            reader = csv.reader(f, delimiter='\t', quotechar='"')
            self.lines = [line for line in reader]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        _id, label, message = self.lines[idx]
        words = word_tokenize(message)
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0.0) + 1
        return (
            torch.tensor(label_map[label]),
            torch.tensor([
                word_counts.get(w, 0.0)
                for w in lexicon.keys()
            ]),
        )


class TextClassificationModel(nn.Module):
    """
    Neuronal Network to classify text, defined by the layers:
        - Linear:
            y = W.x + b
            Predicts an output class based on the text input represented as a
            vector in the lexicon.
    """
    def __init__(self, vocab_size: int, num_class: int):
        super().__init__()
        self.fc = nn.Linear(vocab_size, num_class)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        return self.fc(text)


def train(model: nn.Module, dataloader: DataLoader, loss_fn: _Loss,
          optimiser: torch.optim.Optimizer):
    """
    Trains the model with the given DataLoader, using the loss function
    `loss_fn` to measure prediction errors, and optimising parameters with the
    strategy defined by `optimiser`.
    """
    model.train()
    size = len(dataloader.dataset)
    y: torch.Tensor
    X: torch.Tensor
    for idx, (y, X) in enumerate(dataloader):
        # Compute prediction error
        predicted: torch.Tensor = model(X)
        loss = loss_fn(predicted, y)

        # Back-propagation (calculate gradients)
        loss.backward()
        # Optimise (adjust) parameters
        optimiser.step()
        # Reset gradients
        optimiser.zero_grad()

        if log_on and idx % log_interval == 0 and idx > 0:
            loss, current = loss.item(), (idx + 1) * batch_size
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def evaluate(model: nn.Module, dataloader: DataLoader, loss_fn: _Loss) -> float:
    """
    Evaluates the model with the given DataLoader, using the loss function
    `loss_fn` to measure prediction errors.
    """
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        y: torch.Tensor
        X: torch.Tensor
        for y, X in dataloader:
            predicted: torch.Tensor = model(X)
            total_loss += loss_fn(predicted, y).item()
            total_acc += (predicted.argmax(1) == y).sum().item()
    total_loss /= num_batches
    total_acc /= size
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
    with open("res/twitter-2013train-A.txt", "r") as f:
        reader = csv.reader(f, delimiter='\t', quotechar='"')
        for line in reader:
            _id, _label, message = line
            for w in word_tokenize(message):
                if w not in lexicon:
                    lexicon[w] = len(lexicon)

    # Create data loaders
    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    dev_dataloader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    for y, X in train_dataloader:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Create model
    model = TextClassificationModel(len(lexicon), len(label_map))
    print(model)

    # Define training parameters
    epochs = 5
    learning_rate = 10  # Initial rate, will change with scheduler
    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # Stochastic gradient descent
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, 1, gamma=0.1)
    max_acc = None

    # Train model with training dataset
    for e in range(epochs):
        print("-" * 59 + f"\nEpoch {e + 1}\n" + "-" * 59)
        epoch_start_time = time.time()
        train(model, train_dataloader, loss_fn, optimiser)
        dev_acc = evaluate(model, dev_dataloader, loss_fn)
        if max_acc is not None and max_acc > dev_acc:
            # Decrease learning rate if accuracy dropped
            lr_scheduler.step()
        else:
            max_acc = dev_acc
        print(f"Time: {(time.time() - epoch_start_time):>5.2f}s")

    # Evaluate model with test dataset
    print("-" * 59)
    print("Checking the results of test dataset.")
    evaluate(model, test_dataloader, loss_fn)
    print("\nDone!")


if __name__ == "__main__":
    main()
