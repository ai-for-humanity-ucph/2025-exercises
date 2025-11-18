"""
Solutions week 11.

Usage:
    # No sim study
    python scripts/week11_dml_sim.py estimates
    # With sim study
    python scripts/week11_dml_sim.py estimates --do-sim-study
    # Test against DoubleML implementation
    python scripts/week11_dml_sim.py validate-dml-pkg
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import doubleml as dml
import numpy as np
import polars as pl
import polars.selectors as cs
import statsmodels.formula.api as smf
import typer
from doubleml.datasets import make_irm_data
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


app = typer.Typer(no_args_is_help=True)


def res_to_dict(
    values, cols: list[str] = ["coef", "std", "t", "pval", "lower", "upper"]
):
    return dict(zip(cols, values))


def m_res(m, coef: str = "d"):
    return res_to_dict(m.summary2().tables[1].loc[coef].values.tolist()) | {
        "method": "ols"
    }


def dml_res(dml_obj, coef: str = "d"):
    return res_to_dict(dml_obj.summary.loc[coef].values.tolist()) | {"method": "dml"}


def xvars_str(num: int, prefix: str = "X", suffix: str = ""):
    return " + ".join([f"{prefix}{i}{suffix}" for i in range(1, num + 1)])


def ate_ols_plugin(df: pl.DataFrame, xvars: str):
    md0 = smf.ols(f"y ~ {xvars}", data=df.filter(pl.col("d").eq(0))).fit()
    md1 = smf.ols(f"y ~ {xvars}", data=df.filter(pl.col("d").eq(1))).fit()
    p0 = md0.predict(df.with_columns(D=pl.lit(0)))
    p1 = md1.predict(df.with_columns(D=pl.lit(1)))
    ate_plugin = np.mean(p1 - p0)
    return ate_plugin


def ate_ols(df: pl.DataFrame, num_vars: int = 10):
    """Computes the same as plugin but with standard errors also."""
    xvars = xvars_str(num_vars, "X", "_C")
    df = df.with_columns(
        ((sel := cs.matches(r"^X\d+$")) - sel.mean()).name.suffix("_C")
    )
    m = smf.ols(f"y ~ d + {xvars} + d : ({xvars})", data=df).fit(cov_type="HC0")
    return m


def sim_study(B: int, sim_round_fn: Callable[[], list[dict]]):
    np.random.seed(1111)
    res = []
    try:
        for _ in tqdm(range(B)):
            res.extend(sim_round_fn())
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user (Ctrl+C). Returning partial results.")
        return res
    return res


def res_to_df(res):
    return (
        pl.DataFrame(res)
        .group_by("method")
        .agg(
            pl.col("coef").mean(),
            pl.col("coef").std().alias("std"),
        )
    )


@dataclass
class DmlAteRes:
    res: dict
    score: np.ndarray


def dml_ate(
    X,
    y,
    d,
    ml_g,
    ml_m,
    splits,
    verbose: bool = False,
    thresh: float = 0.01,
):
    l_kc_0 = ml_g
    l_kc_1 = clone(ml_g)
    r_kc = clone(ml_m)
    n = y.shape[0]
    preds = np.zeros_like(y)
    for k, (I_k_c, I_k) in enumerate(splits):
        if verbose:
            print(f"Estimating fold {k=}")

        X_kc = X[I_k_c]
        y_kc = y[I_k_c]
        d_kc = d[I_k_c]

        # Estimate on I_k^c
        l_kc_0.fit(X_kc[d_kc == 0], y_kc[d_kc == 0])  # E[Y | X, D = 0]
        l_kc_1.fit(X_kc[d_kc == 1], y_kc[d_kc == 1])  # E[Y | X, D = 1]
        r_kc.fit(X_kc, d_kc)  # P(D = 1 | X)

        # Predict on I_k
        l_0_hat = l_kc_0.predict(X[I_k])
        l_1_hat = l_kc_1.predict(X[I_k])
        r_hat = r_kc.predict_proba(X[I_k])[:, 1]
        r_hat = np.clip(a=r_hat, a_min=thresh, a_max=1 - thresh)

        # Compute estimator values for I_k
        d_k = d[I_k]
        y_k = y[I_k]
        alpha_hat = d_k / r_hat - (1 - d_k) / (1 - r_hat)
        l_hat = d_k * l_1_hat + (1 - d_k) * l_0_hat
        score_k = alpha_hat * (y_k - l_hat) + l_1_hat - l_0_hat
        preds[I_k] = score_k

    est = preds.mean()
    se = np.sqrt(((preds - est) ** 2).mean() / n)
    lower, upper = est - 1.96 * se, est + 1.96 * se
    res = dict(
        zip(
            ["est", "se", "lower", "upper"],
            np.array([est, se, lower, upper]).tolist(),
        )
    )
    return DmlAteRes(res, score=preds - est)


def propensity(X, clf, thresh: float = 0.01) -> np.ndarray:
    r_hat = clf.predict_proba(X)[:, 1]
    r_hat = np.clip(a=r_hat, a_min=thresh, a_max=1 - thresh)
    return r_hat


def alpha_n(D, r):
    return D / r - (1 - D) / (1 - r)


@app.command()
def estimates(do_sim_study: bool = False):
    """Compute estimates using the three score functions"""

    ml_g = RandomForestRegressor(
        n_estimators=100, max_features=10, max_depth=5, min_samples_leaf=2
    )
    ml_m = RandomForestClassifier(
        n_estimators=100, max_features=10, max_depth=5, min_samples_leaf=2
    )

    def sim_round(X, y, d):
        # Fit models
        l_0 = ml_g
        l_1 = clone(ml_g)
        l_0.fit(X[d == 0], y[d == 0])  # E[Y | X, D = 0]
        l_1.fit(X[d == 1], y[d == 1])  # E[Y | X, D = 1]
        ml_m.fit(X, d)  # P(D = 1 | X)

        # Use estimated nuisance functions on whole dataset
        r_hat = propensity(X, clf=ml_m)
        alpha_p = alpha_n(d, r_hat)
        l_0_p = l_0.predict(X)
        l_1_p = l_1.predict(X)
        l_p = d * l_1_p + (1 - d) * l_0_p

        # RA
        est_ra = np.mean(l_1_p - l_0_p)
        # IPW
        est_ipw = (alpha_p * y).mean()
        # DR
        est_dr = np.mean(alpha_p * (y - l_p) + l_1_p - l_0_p)

        testdata = pl.DataFrame(
            np.column_stack((X, y, d)),
            schema=[f"X{i + 1}" for i in range(10)] + ["y", "d"],
        )
        ols_est = ate_ols_plugin(testdata, xvars_str(10))
        return {"ols": ols_est, "ipw": est_ipw, "ra": est_ra, "dr": est_dr}

    # 1.4
    X, y, d = make_irm_data(theta=0.5, n_obs=1_000, dim_x=10, return_type="np.ndarray")
    ests = sim_round(X, y, d)
    print(pl.DataFrame(ests))

    # 1.5

    if do_sim_study:
        B = 200

        all_ests = []
        for b in tqdm(range(B)):
            X, y, d = make_irm_data(
                theta=0.5, n_obs=1_000, dim_x=10, return_type="np.ndarray"
            )
            all_ests.append(sim_round(X, y, d))
        print(res := pl.DataFrame(all_ests))
        print(res.describe())


@app.command()
def validate_dml_pkg():
    # Exercise 2.1-2
    ml_g = LinearRegression()
    ml_m = LogisticRegression()

    data = make_irm_data(theta=0.5, n_obs=10_000, dim_x=10, return_type="DataFrame")

    obj_dml_data = dml.DoubleMLData(data, "y", "d")
    df = pl.from_pandas(data)  # type: ignore

    X = df.select(cs.matches(r"^X\d+$")).to_numpy()
    y = df.select("y").to_numpy().ravel()
    d = df.select("d").to_numpy().ravel()

    kf = KFold(n_splits=10)
    splits = list(kf.split(data))

    # Apply algorithm
    dmlr = dml_ate(X, y, d, ml_g, ml_m, splits)

    dml_irm_obj_external = dml.DoubleMLIRM(
        obj_dml_data, ml_g, ml_m, draw_sample_splitting=False
    )
    dml_irm_obj_external.set_sample_splitting(splits)
    dml_irm_obj_external.fit().summary

    dml_res_pkg = dml_res(dml_irm_obj_external)
    print(dml_res(dml_irm_obj_external))
    print(dmlr.res)

    est_pkg = dml_res_pkg["coef"]
    est = dmlr.res["est"]
    se_pkg = dml_res_pkg["std"]
    se = dmlr.res["se"]

    # Assert our estimates matches the package
    assert np.allclose(est, est_pkg)
    assert np.allclose(se, se_pkg)
    assert np.allclose(dmlr.score, dml_irm_obj_external.psi.ravel())


if __name__ == "__main__":
    app()
