import numpy as np
from ai4h.models.linear_model import (
    LogisticRegression,
    generate_logreg_data,
    neg_loglik,
)


def solution():
    n = 10_000
    beta = np.array([1, 1.0, -1.0])
    X, y = generate_logreg_data(n, beta, seed=42)

    logreg = LogisticRegression()
    logreg.fit(X, y)
    print("Converged:", logreg.converged_)
    print("Iterations:", logreg.n_iter_)
    print("Estimates:", logreg.coef_)


def bonus():
    """bonus exercise"""

    from scipy.optimize import minimize

    n = 10_000
    beta = np.array([1, 1.0, -1.0])
    X, y = generate_logreg_data(n, beta, seed=42)
    res = minimize(
        lambda b: neg_loglik(X, y, b),
        x0=np.zeros(X.shape[1]),
        method="BFGS",
    )

    if res.success:
        print("Estimates:", res.x)
    else:
        print("Optimization failed:", res.message)


def main():
    solution()


if __name__ == "__main__":
    main()
