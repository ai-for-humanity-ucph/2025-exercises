"""
Usage:
    python scripts/week10_causal.py
"""

import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from causalinference import CausalModel


def ex1():
    df = pl.DataFrame(
        [
            [1, 3, 2, 0],
            [2, 4, 2, 0],
            [3, 5, 2, 0],
            [4, 6, 2, 1],
            [5, 13, 3, 1],
            [5, 10, 6, 1],
        ],
        schema=["id", "Y(1)", "Y(0)", "D"],
        orient="row",
    )
    df = df.with_columns(dy=pl.col("Y(1)") - pl.col("Y(0)"))
    gpt = df.filter(pl.col("D").eq(1))
    print("ATT:", gpt.select(pl.col("dy").mean()).item())
    print("ATE:", df.select(pl.col("dy").mean()).item())


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def est(m, coef: str = "treat"):
    return dict(
        zip(
            ["coef", "std", "t", "pval", "lower", "upper"],
            m.summary2().tables[1].loc[coef].values,
        )
    )


def ex2():
    df_sim = pl.read_csv("data/ci_sim.csv")
    Y = df_sim.select("Y").to_numpy().ravel()
    D = df_sim.select("D").to_numpy().ravel()
    X = df_sim.select("X").to_numpy()

    # Naive difference in means (biased due to X-driven selection)
    naive_ate = Y[D == 1].mean() - Y[D == 0].mean()
    m = smf.ols("Y ~ D + X", data=df_sim).fit()
    adj_ate = est(m, coef="D")["coef"]
    md0 = smf.ols("Y ~ X", data=df_sim.filter(pl.col("D").eq(0))).fit()
    md1 = smf.ols("Y ~ X", data=df_sim.filter(pl.col("D").eq(1))).fit()
    p0 = md0.predict(df_sim.with_columns(D=pl.lit(0)))
    p1 = md1.predict(df_sim.with_columns(D=pl.lit(1)))
    ate_plugin = np.mean(p1 - p0)
    me = smf.ols(
        "Y ~ D + Xc + D : Xc",
        data=df_sim.with_columns(Xc=pl.col("X") - pl.col("X").mean()),
    ).fit()
    adj_ci_ate = est(me, coef="D")["coef"]
    print(f"Naive ATE:  {naive_ate: .3f}   (biased)")
    print(f"Adj. ATE:   {adj_ate: .3f}     (≈ unbiased given correct spec)")
    print(f"Plugin ATE: {ate_plugin: .3f}     ")

    assert np.allclose(adj_ci_ate, ate_plugin)

    causal = CausalModel(Y, D, X)
    causal.est_via_ols()
    ci_ate_2 = causal.estimates["ols"]["ate"]
    assert np.allclose(ci_ate_2, ate_plugin)

    causal.est_via_ols(adj=1)
    ci_ate_1 = causal.estimates["ols"]["ate"]

    assert np.allclose(ci_ate_1, adj_ate)

    causal.est_via_matching(matches=4, bias_adj=True)
    print(causal.estimates["matching"]["ate"])


def ex3():
    data = pl.read_csv("data/lalonde.csv")
    Y = data.select("re78").to_numpy().ravel()
    D = data.select("treat").to_numpy().ravel()
    X = data.select(
        ["age", "educ", "black", "hispanic", "married", "nodegree", "re75"]
    ).to_numpy()

    causal = CausalModel(Y, D, X)

    causal.est_via_ols()
    print(causal.estimates)
    causal.est_via_matching(matches=7)
    print(causal.estimates)
    causal.est_via_matching(matches=7, bias_adj=True)
    print(causal.estimates)


def main():
    ex1()
    ex2()
    ex3()


if __name__ == "__main__":
    main()
