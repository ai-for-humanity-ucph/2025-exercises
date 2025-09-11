import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def get_iris_data_split():
    """helper to load and split data for tests"""
    from typing import cast

    # https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = cast(
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.int64],
            NDArray[np.int64],
        ],
        train_test_split(X, y, random_state=2025, train_size=0.8),
    )

    return X_train, X_test, y_train, y_test
