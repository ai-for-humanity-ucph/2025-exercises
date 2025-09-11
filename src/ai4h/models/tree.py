from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.random.mtrand import RandomState
from numpy.typing import NDArray

DEFAULT_SEED = 0


def cprobs(y: NDArray[np.integer]):
    return np.bincount(y) / y.shape[0]


def misclassification(y: NDArray[np.integer]):
    """misclassification measure"""
    p_j_t = cprobs(y)
    k_m = p_j_t.argmax().item()  # majority class
    return (1 - p_j_t[k_m]).item()


def gini(y: NDArray[np.integer]):
    """gini impurity function"""
    p_j_t = cprobs(y)
    return np.sum(p_j_t * (1 - p_j_t)).item()


def mse(y: NDArray[np.float64]):
    """mse impurity function"""
    return float(np.mean((y - y.mean()) ** 2))


def goodness_split(
    y_left: NDArray,
    y_right: NDArray,
    imp_parent: float,
    impurity_fn: Callable[[NDArray], float] = gini,
) -> float:
    n_left, n_right = y_left.shape[0], y_right.shape[0]
    n = n_left + n_right
    imp_left, imp_right = impurity_fn(y_left), impurity_fn(y_right)
    p_L = n_left / n
    p_R = n_right / n
    return imp_parent - p_L * imp_left - p_R * imp_right


def best_split(
    X: NDArray,
    y: NDArray,
    *,
    impurity_fn: Callable[[NDArray], float] = gini,
    min_samples_leaf: int = 1,
    max_features: int | None = None,
    random_state: RandomState = np.random.RandomState(DEFAULT_SEED),
) -> tuple[int, float, float]:
    """finds the best split of the data in a greedy fashion"""
    n, d = X.shape

    if n < 2 * min_samples_leaf:
        return -1, np.nan, 0.0

    if max_features is None:
        features = list(range(d))
    else:
        # NOTE: sklearn does it in a bit more convoluted way; ask if interested
        features = draw_features(X.shape[1], max_features, random_state=random_state)

    imp_parent = impurity_fn(y)
    best_feat, best_thr, best_gain = -1, np.nan, 0.0

    for j in features:
        xj = X[:, j].astype(np.float32, copy=False)

        order = np.argsort(xj, kind="stable")
        x_sorted, y_sorted = xj[order], y[order]

        for i in range(min_samples_leaf - 1, n - min_samples_leaf):
            if x_sorted[i] == x_sorted[i + 1]:
                continue  # no threshold between identical values

            gain = goodness_split(
                y_sorted[: i + 1],
                y_sorted[i + 1 :],
                imp_parent,
                impurity_fn=impurity_fn,
            )
            if gain > best_gain:
                best_thr = 0.5 * (x_sorted[i] + x_sorted[i + 1])
                best_gain, best_feat = gain, j

    return best_feat, best_thr, best_gain


def compute_pred(y: NDArray) -> float:
    """computes predicted values.

    Guesses whether classification or regression tree by inspecting dtype of
    outcome variable.
    """
    if np.issubdtype(y.dtype, np.integer):
        return int(np.bincount(y).argmax())
    return float(y.mean())


@dataclass
class TerminalNode:
    prediction: float
    impurity: float
    depth: int
    n_samples: int

    def predict_one(self, _):
        return self.prediction

    def predict(self, X: NDArray):
        return np.full(len(X), self.prediction)

    def pretty(self, indent=""):
        return f"{indent}Terminal(pred={self.prediction:.2f}, n={self.n_samples})\n"


@dataclass
class NonterminalNode:
    feature: int
    threshold: float
    left: Node
    right: Node
    impurity: float
    depth: int
    n_samples: int

    def predict_one(self, x: NDArray):
        return (
            self.left.predict_one(x)
            if x[self.feature] <= self.threshold
            else self.right.predict_one(x)
        )

    def predict(self, X: NDArray):
        return np.array([self.predict_one(x) for x in X])

    def pretty(self, indent=""):
        out = f"{indent}NonTerminalNode(f{self.feature}<={self.threshold:.2f}, n={self.n_samples})\n"
        return out + self.left.pretty(indent + "  ") + self.right.pretty(indent + "  ")


Node = TerminalNode | NonterminalNode


def build_tree(
    X: NDArray,
    y: NDArray,
    *,
    impurity_fn: Callable[[NDArray], float] = gini,
    max_depth: int = 32,
    min_samples_leaf: int = 1,
    depth: int = 0,
    max_features: int | None = None,
    random_state: RandomState = np.random.RandomState(DEFAULT_SEED),
) -> Node:
    """builds a tree using recursive binary splitting"""
    n = y.shape[0]

    # too deep or not enough samples
    if depth >= max_depth or n <= 2 * min_samples_leaf:
        return TerminalNode(
            prediction=compute_pred(y),
            depth=depth,
            n_samples=n,
            impurity=impurity_fn(y),
        )

    # return early if all in node are of same class
    if np.unique(y).size == 1:
        return TerminalNode(
            prediction=compute_pred(y),
            depth=depth,
            n_samples=n,
            impurity=impurity_fn(y),
        )

    j, thr, gain = best_split(
        X,
        y,
        impurity_fn=impurity_fn,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
    )

    # `best_split` returned no valid split; return TerminalNode
    if j == -1 or not np.isfinite(thr) or gain <= 0:
        return TerminalNode(
            prediction=compute_pred(y),
            depth=depth,
            n_samples=n,
            impurity=impurity_fn(y),
        )

    mask = X[:, j] <= thr
    return NonterminalNode(
        feature=j,
        threshold=thr,
        left=build_tree(
            X[mask],
            y[mask],
            impurity_fn=impurity_fn,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            depth=depth + 1,
            max_features=max_features,
            random_state=random_state,
        ),
        right=build_tree(
            X[~mask],
            y[~mask],
            impurity_fn=impurity_fn,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            depth=depth + 1,
            max_features=max_features,
            random_state=random_state,
        ),
        depth=depth,
        n_samples=n,
        impurity=impurity_fn(y),
    )


# NOTE: Extra/Optional stuff

RandomStateType = int | None | RandomState


def check_random_state(random_state: RandomStateType):
    """helper function that follows sklearn"""
    match random_state:
        case int():
            return np.random.RandomState(random_state)
        case RandomState():
            return random_state
        case None:
            return np.random.RandomState(0)
    raise ValueError


def draw_features(
    n_features: int, max_features: int, random_state: RandomStateType = None
):
    """draw features."""
    rng = check_random_state(random_state)
    return rng.choice(n_features, size=max_features, replace=False)


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth: int = 32,
        min_samples_leaf: int = 1,
        criterion: Callable[[NDArray], float] = gini,
        max_features: int | None = None,
        random_state: RandomStateType = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree_: Node
        self.random_state = check_random_state(random_state)
        self.max_features = max_features

    def fit(self, X: NDArray, y: NDArray):
        if self.max_features is None:
            self.max_features = X.shape[1]
        self.tree_ = build_tree(
            X=X,
            y=y,
            impurity_fn=self.criterion,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"max_depth={self.max_depth}, "
            f"min_samples_leaf={self.min_samples_leaf}, "
            f"criterion={self.criterion.__name__ if callable(self.criterion) else self.criterion}, "
            f"random_state={self.random_state}"
            f")"
        )

    def predict(self, X: NDArray) -> NDArray:
        if self.tree_ is None:
            raise ValueError("Not fitted yet")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.tree_.predict(X)
