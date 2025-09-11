from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import fetch_openml

fp_proj = Path(__file__).parents[3]
fp_data = fp_proj.joinpath("data")
fp_data.mkdir(exist_ok=True)
mnist_file = fp_data / "mnist.npz"


def onehot(y: NDArray[np.uint8], n_classes: int = 10) -> NDArray[np.uint8]:
    return np.eye(n_classes, dtype=np.uint8)[y]


def prepare_nielsen(
    X: NDArray[np.int64],
    y: NDArray[np.uint8],
    y_transform=lambda x: x.reshape(10, 1),
    one_hot: bool = True,
) -> list[tuple[NDArray, NDArray]]:
    """Massages data into the shape MN's code expects."""
    return [
        (x.reshape(784, 1) / 255, y_transform(v))
        for x, v in zip(X, onehot(y) if one_hot else y)
    ]


def download_mnist():
    if mnist_file.exists():
        print("Mnist data already exist in data folder")
        return

    mnist = fetch_openml("mnist_784", as_frame=False, parser="liac-arff")
    X = mnist["data"].astype(np.uint8)  # type: ignore
    y = mnist["target"].astype(np.uint8)  # type: ignore

    np.savez_compressed(mnist_file, X=X, y=y)
    print(f"Mnist data saved to {fp_data}")


def load_mnist() -> tuple[NDArray[np.int64], NDArray[np.uint8]]:
    if not mnist_file.exists():
        print("Mnist data hasn't been downloaded")
        download_mnist()
    data = np.load(fp_data / "mnist.npz")
    X, y = data["X"], data["y"]
    return X, y


def cross_entropy(y: NDArray, yhat: NDArray):
    eps = 1e-9
    return -np.sum(y * np.log(yhat + eps)).item()


def sigmoid(z: NDArray):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z: NDArray):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z: NDArray, axis: int = 1) -> NDArray[np.float64]:
    z_shift = z - z.max(axis=axis, keepdims=True)  # numerical stability
    expz = np.exp(z_shift)
    return expz / expz.sum(axis=axis, keepdims=True)


class BasicNetwork:
    """Basic network with 1 hidden layer (30 units)."""

    def __init__(self, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        self.W_1 = rng.normal(0, 0.1, size=(30, 784))
        self.b_1 = np.zeros((30, 1))
        self.W_2 = rng.normal(0, 0.1, size=(10, 30))
        self.b_2 = np.zeros((10, 1))

    def predict(self, x: NDArray) -> NDArray:
        """Forward pass for a single column vector"""
        z1 = self.W_1 @ x + self.b_1
        a1 = sigmoid(z1)
        z2 = self.W_2 @ a1 + self.b_2
        return softmax(z2, axis=0)  # (10,1)

    def train_step(
        self,
        x: NDArray,
        y: NDArray,
    ) -> tuple[float, NDArray, NDArray, NDArray, NDArray]:
        """Forward + backward for one sample. Returns loss and grads."""

        # forward
        a0 = x  # (784,1)
        z1 = self.W_1 @ a0 + self.b_1
        a1 = sigmoid(z1)
        z2 = self.W_2 @ a1 + self.b_2
        yhat = softmax(z2, axis=0)

        # cross-entropy loss
        loss = cross_entropy(y, yhat)

        # backward
        delta2 = yhat - y  # (10,1)
        dW2 = delta2 @ a1.T  # (10,30)
        db2 = delta2  # (10,1)

        delta1 = (self.W_2.T @ delta2) * sigmoid_prime(z1)  # (30,1)
        dW1 = delta1 @ a0.T  # (30,784)
        db1 = delta1  # (30,1)

        return loss, dW1, db1, dW2, db2

    def train_update(
        self, mini_batch: list[tuple[NDArray, NDArray]], lr: float
    ) -> float:
        """Aggregate gradients over mini_batch and apply update."""
        B = len(mini_batch)

        # accumulators
        sum_loss = 0.0
        dW1 = np.zeros_like(self.W_1)
        db1 = np.zeros_like(self.b_1)
        dW2 = np.zeros_like(self.W_2)
        db2 = np.zeros_like(self.b_2)

        for x, y in mini_batch:
            loss, gW1, gb1, gW2, gb2 = self.train_step(x, y)
            sum_loss += loss
            dW1 += gW1
            db1 += gb1
            dW2 += gW2
            db2 += gb2

        # average gradients
        dW1 /= B
        db1 /= B
        dW2 /= B
        db2 /= B
        avg_loss = sum_loss / B

        # SGD update
        self.W_1 -= lr * dW1
        self.b_1 -= lr * db1
        self.W_2 -= lr * dW2
        self.b_2 -= lr * db2

        return avg_loss
