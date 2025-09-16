"""
Usage:
    python scripts/week3_torch.py --batch-size 32 --lr 3.0 --epochs 8 --shuffle

Example taken from:
    https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
with some modifications.


"""

import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor


class NeuralNetwork(nn.Module):
    def __init__(self, seed: int = 2025):
        super().__init__()
        self.flatten = nn.Flatten()
        self.f1, self.f2 = self._init_as_mn(seed)
        self.sigmoid = nn.Sigmoid()

    def _init_as_mn(self, seed: int):
        """Init. weights exactly as MN.

        This is only to compare with: scripts/week2_network.py

        The weights are drawn from N(0, 0.01^2) and the biases are initialized
        as vectors of zeros.

        See https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
        for how `nn.Linear` initializes the weights.
        """
        rng = np.random.default_rng(seed=seed)

        f1 = nn.Linear(784, 30)  # input -> hidden
        f2 = nn.Linear(30, 10)  # hidden -> output

        with torch.no_grad():
            f1.weight.copy_(
                torch.tensor(rng.normal(0, 0.1, size=(30, 784)), dtype=torch.float32)
            )
            f1.bias.zero_()
            f2.weight.copy_(
                torch.tensor(rng.normal(0, 0.1, size=(10, 30)), dtype=torch.float32)
            )
            f2.bias.zero_()
        return f1, f2

    def forward(self, x):
        """
        NOTE: no softmax because `nn.CrossEntropyLoss` applies it internally
        """
        logits = self.f2(self.sigmoid(self.f1(self.flatten(x))))
        return logits


def train(
    dataloader,
    model,
    loss_fn,
    optimizer,
    device,
):
    model.train()
    tot_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()  # computes gradients
        optimizer.step()  # updates parameters using gradients
        optimizer.zero_grad()  # reset gradients

        tot_loss += loss * X.shape[0]
    n = len(dataloader.dataset)
    return tot_loss / n


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return test_loss, correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3.0, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle data each epoch"
    )
    args = parser.parse_args()

    # Download training and test data from open datasets.
    _training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    training_data = Subset(_training_data, range(0, 50_000))  # Match Nielsen
    val_data = Subset(_training_data, range(50_000, 60_000))  # Match Nielsen

    # Create data loaders.
    train_dataloader = DataLoader(
        training_data, batch_size=args.batch_size, shuffle=args.shuffle
    )
    val_dataloader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=args.shuffle
    )
    test_dataloader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=args.shuffle
    )

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )

    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=3.0)

    print(f"Using {device} device")
    print("Training net...")
    try:
        for i in range(args.epochs):
            avg_loss = train(train_dataloader, model, loss_fn, optimizer, device)
            val_loss, val_acc = test(val_dataloader, model, loss_fn, device)
            print(
                f"Epoch {i:<2}: "
                f"Train loss: {avg_loss:.4f}, "
                f"Val loss: {val_loss:.4f}, "
                f"Val acc: {val_acc:.4f}"
            )
    except KeyboardInterrupt:
        test_loss, test_acc = test(test_dataloader, model, loss_fn, device)
        print(f"Final model: Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
    else:
        test_loss, test_acc = test(test_dataloader, model, loss_fn, device)
        print(f"Final model: Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
