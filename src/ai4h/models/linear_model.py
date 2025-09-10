from numpy.typing import NDArray
import numpy as np
from pathlib import Path


def test():
    project_root = Path(__file__).resolve().parents[3]
    relpath = Path(__file__).resolve().relative_to(project_root)
    print(f"Hello from `{relpath}`!")


def expit(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the expit (logistic) function."""
    return 1 / (1 + np.exp(-x))


def generate_logreg_data(
    N: int,
    beta: NDArray[np.float64],
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(seed=seed)
    p = beta.ravel().shape[0] - 1
    X = np.column_stack((np.ones((N, 1)), rng.uniform(low=-1, high=1, size=(N, p))))
    y = rng.binomial(n=1, p=expit(X @ beta))
    return X, y


def irls(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    tol: float = 1e-8,
    max_iter: int = 100,
):
    """Iteratively reweighted least squares solver."""
    beta_t = np.zeros(shape=(X.shape[1], 1))
    not_converged = True
    iters = 0
    if y.ndim != 2:
        y = y.reshape(-1, 1)
    Xt = X.T
    while not_converged and iters < max_iter:
        eta = X @ beta_t
        p = expit(eta)
        W_t = p * (1 - p)
        Z_t = eta + (y - p) / W_t
        beta_t_next = np.linalg.solve(Xt @ (X * W_t), Xt @ (W_t * Z_t))
        not_converged = np.any(np.abs(beta_t_next - beta_t) > tol)
        beta_t = beta_t_next
        iters += 1
    return {
        "estimates": beta_t.ravel(),
        "iterations": iters,
        "converged": not not_converged,
    }


class LogisticRegression:
    def __init__(self):
        self.coef_ = None
        self.n_iter_ = None
        self.converged_ = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]):
        result = irls(X, y)
        self.coef_ = result["estimates"]
        self.n_iter_ = result["iterations"]
        self.converged_ = result["converged"]

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute predicted probabilities."""
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet.")
        return expit(X @ self.coef_)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict binary outcomes based on a 0.5 threshold."""
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet.")
        probs = expit(X @ self.coef_)
        return (probs >= 0.5).astype(np.int64)


def neg_loglik(
    X: NDArray[np.float64], y: NDArray[np.int64], beta: NDArray[np.float64]
) -> float:
    """negative log likehood for logistic regression"""
    linear = X @ beta
    return -np.sum(y * linear - np.log1p(np.exp(linear)))
