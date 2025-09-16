"""
Estimate simple neural network using a simple training loop.

Usage:
    python scripts/week2_network.py --batch-size 32 --lr 3.0 --epochs 8

"""

import argparse
import random

import numpy as np
from numpy.typing import NDArray

from ai4h.models.network import BasicNetwork, cross_entropy, load_mnist, prepare_nielsen


def avg_acc_loss(data: list[tuple[NDArray, NDArray]], model: BasicNetwork):
    """helper to compute avg loss and acc. on val or test data"""
    loss = 0
    acc = 0

    for x, y in data:
        yhat = model.predict(x)
        loss += cross_entropy(y, yhat)
        yhat_label = yhat.ravel().argmax().item()
        y_label = y.ravel().argmax().item()
        acc += yhat_label == y_label

    n_val = len(data)
    loss /= n_val
    acc /= n_val
    return loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3.0, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle data each epoch"
    )
    args = parser.parse_args()

    X, y = load_mnist()
    X_train, y_train = X[:50_000], y[:50_000]
    X_val, y_val = X[50_000:60_000], y[50_000:60_000]
    X_test, y_test = X[60_000:], y[60_000:]

    training_data = prepare_nielsen(X_train, y_train)
    test_data = prepare_nielsen(X_test, y_test)
    val_data = prepare_nielsen(X_val, y_val)

    n = len(training_data)

    # Set seed for shuffle of mini batches and weight initialization
    seed = 2025
    random.seed(seed)
    rng = np.random.default_rng(seed=seed)
    net = BasicNetwork(rng=rng)

    print("Training net...")
    try:
        for i in range(args.epochs):
            # shuffle training data as MN does
            if args.shuffle:
                random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + args.batch_size]
                for k in range(0, n, args.batch_size)
            ]

            # mini-batch SGD
            avg_loss = 0
            for mini_batch in mini_batches:
                avg_loss += net.train_update(mini_batch, lr=args.lr) * len(mini_batch)
            avg_loss /= n

            val_loss, val_acc = avg_acc_loss(val_data, net)
            print(
                f"Epoch {i:<2}: "
                f"Train loss: {avg_loss:.4f}, "
                f"Val loss: {val_loss:.4f}, "
                f"Val acc: {val_acc:.4f}"
            )
    except KeyboardInterrupt:
        test_loss, test_acc = avg_acc_loss(test_data, net)
        print(f"Final model: Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
    else:
        test_loss, test_acc = avg_acc_loss(test_data, net)
        print(f"Final model: Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
