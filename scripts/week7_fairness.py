"""
Solution for the exercises of week 7.

Usage:
    python scripts/week7_fairness.py
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

figsize = (6, 5)
colors = {
    "Asian": "blue",
    "White": "green",
    "Hispanic": "red",
    "Black": "lightblue",
}
linestyles = {
    "Asian": "-",
    "White": "--",
    "Hispanic": "-.",
    "Black": ":",
}
np.random.seed(2025)


def plot_cdf(df: pl.DataFrame):
    figsize = (6, 5)
    fig, ax = plt.subplots(figsize=figsize)
    for (group,), sub in df.group_by("group", maintain_order=True):
        x_sorted = np.sort(sub["FICO"])
        F = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        ax.plot(
            x_sorted,
            F,
            label=group,
            color=colors.get(group),
            linestyle=linestyles.get(group),
            linewidth=2,
        )
    ax.set_xlabel("FICO score")
    ax.set_ylabel("Fraction of group below")
    ax.set_title("CDF of FICO score by group")
    ax.set(xlim=(300, 900), ylim=(0, 1))
    ax.legend(title="Group")
    ax.legend()
    plt.tight_layout()
    fig.savefig("figs/fairness/cdf.png")


def plot_ndrate(df_p: pl.DataFrame):
    gp = (
        df_p.with_columns(
            pl.col("FICO")
            .cut(
                nps := np.linspace(350, 850, 10).tolist(),
                left_closed=False,
                include_breaks=True,
                labels=[f"b{i}" for i in range(len(nps) + 1)],
            )
            .alias("bin")
        )
        .group_by("bin", "group")
        .agg(pl.col("ND").mean())
        .unnest("bin")
        .sort("group", "breakpoint")
        .select("group", "ND", "breakpoint")
        .with_columns(pl.col("breakpoint").sub(25))
    )
    fig, ax = plt.subplots(figsize=figsize)
    for (group,), sub in gp.group_by("group", maintain_order=True):
        ax.plot(
            sub["breakpoint"],
            sub["ND"] * 100,
            label=group,
            color=colors.get(group, "black"),
            linestyle=linestyles.get(group, "-"),
            linewidth=2,
        )
    ax.set(
        xlabel="FICO score",
        ylabel="Non-default rate",
        title="Non-default rate by FICO score",
    )
    ax.set(xlim=(300, 900), ylim=(0, 100))
    ax.set_yticks(pcts := [0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{p}%" for p in pcts])
    ax.legend(title="")
    plt.tight_layout()
    fig.savefig("figs/fairness/ndrate.png")


def plot_ndrate_within(df_p: pl.DataFrame):
    # percentiles within-group
    gp = (
        df_p.with_columns(
            pl.col("FICO")
            .qcut(
                nps := np.linspace(0.1, 0.9, 9).tolist(),
                left_closed=True,
                include_breaks=True,
                labels=(qs := [f"Q{i + 1}" for i in range(len(nps) + 1)]),
            )
            .over("group")
            .alias("bin")
        )
        .group_by("bin", "group")
        .agg(pl.col("ND").mean())
        .unnest("bin")
        .select("group", "ND", "category")
        .with_columns(
            pct=pl.col("category").replace_strict(
                dict(zip(qs, range(5, 105, 10))), return_dtype=pl.Int8
            )
        )
        .sort("group", "pct")
    )

    fig, ax = plt.subplots(figsize=figsize)
    for (group,), sub in gp.group_by("group", maintain_order=True):
        ax.plot(
            sub["pct"],
            sub["ND"] * 100,
            label=group,
            color=colors.get(group, "black"),
            linestyle=linestyles.get(group, "-"),
            linewidth=2,
        )
    ax.set(
        xlabel="Within-group FICO score percentile",
        ylabel="Non-default rate",
        title="Non-default rate by FICO score",
    )
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.set_yticks(pcts := [0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{p}%" for p in pcts])
    ax.legend(title="")
    plt.tight_layout()
    fig.savefig("figs/fairness/ndrate-within.png")


def compute_metrics(Y_hat: np.ndarray, Y: np.ndarray):
    Y_p = Y == 1
    Y_n = Y == 0
    # Among (Y = 1)
    TPR = np.mean(Y_hat[Y_p] == 1).mean()
    FNR = np.mean(Y_hat[Y_p] == 0).mean()
    # Among (Y = 0)
    FPR = np.mean(Y_hat[Y_n] == 1).mean()
    TNR = np.mean(Y_hat[Y_n] == 0).mean()
    metrics = dict(zip(["TPR", "FNR", "FPR", "TNR"], [TPR, FNR, FPR, TNR]))
    return metrics


def roc_data(data: pl.DataFrame):
    Y = data["ND"].to_numpy()  # Non-default
    R = data["FICO"].to_numpy()  # FICO Score
    rs = np.quantile(R, q=np.arange(0, 1 + 0.01, step=0.01))
    metrics = []
    for r in rs:
        Y_hat = (R > r).astype(int)
        metrics.append(compute_metrics(Y_hat, Y))
    df_m = pl.DataFrame(metrics).sort("FPR")
    return df_m


def main():
    # 1.1
    # Load data
    df_p = pl.read_csv("data/fico.csv")

    # 1.2
    # Empirical default and non-default rates; overall and groups
    table = (
        pl.concat(
            # cmt
            (
                df_p.group_by("group").agg(
                    (1 - pl.col("ND").mean()).alias("Default rate"),
                    pl.col("ND").mean().alias("Non-default rate"),
                ),
                df_p.select(
                    pl.lit("All").alias("group"),
                    (1 - pl.col("ND").mean()).alias("Default rate"),
                    pl.col("ND").mean().alias("Non-default rate"),
                ),
            )
        )
        .with_columns(pl.selectors.numeric().round(2))
        .sort("group")
    )
    print(table)

    # 1.3
    gp1 = (
        df_p.group_by(pl.col("FICO").ge(620).cast(pl.Int8).alias("Above r"))
        .agg(pl.col("ND").mean())
        .with_columns(pl.lit("All").alias("group"))
        .select("Above r", "group", "ND")
    )
    gp2 = (
        df_p.group_by(pl.col("FICO").ge(620).cast(pl.Int8).alias("Above r"), "group")
        .agg(pl.col("ND").mean())
        .sort("group", "Above r")
    )
    # Default-rates for r = 620
    print(
        pl.concat((gp2, gp1))
        .sort("Above r", "group")
        .with_columns(pl.selectors.numeric().round(2)),
        sep="\n",
    )

    # 1.4: CDF of FICO score by group and Non-default rate by group
    plot_ndrate(df_p)
    plot_cdf(df_p)

    # 1.5: Within
    plot_ndrate_within(df_p)

    # 2.2
    Y = df_p["ND"].to_numpy()  # Non-default
    R = df_p["FICO"].to_numpy()  # FICO Score
    r = 620
    Y_hat = (R > r).astype(int)
    print(
        pl.DataFrame(compute_metrics(Y_hat, Y)).with_columns(
            pl.selectors.numeric().round(2)
        )
    )

    # 2.3
    df_m = roc_data(df_p)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df_m["FPR"], df_m["TPR"], linewidth=2, color="black", linestyle="--")
    ax.set(
        xlabel="Fraction defaulters getting loan",
        ylabel="Fraction non-defaulters getting loan",
        title="ROC curve\nclassifying non-defaulters using FICO score ",
    )
    ax.set(xlim=(0, 1), ylim=(0, 1))
    plt.tight_layout()
    fig.savefig("figs/fairness/roc1.png")

    # 2.4
    gp = pl.concat(
        [
            df.pipe(roc_data).with_columns(group=pl.lit(g))
            for (g,), df in df_p.group_by("group")
        ]
    )
    fig, ax = plt.subplots(figsize=figsize)
    for (group,), sub in gp.group_by("group", maintain_order=True):
        ax.plot(
            sub["FPR"],
            sub["TPR"],
            label=group,
            color=colors.get(group, "black"),
            linestyle=linestyles.get(group, "-"),
            linewidth=2,
        )
    ax.set(
        xlabel="Fraction defaulters getting loan",
        ylabel="Fraction non-defaulters getting loan",
        title="Per-group ROC curve\nclassifying non-defaulters using FICO score ",
    )
    ax.legend(title="")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    plt.tight_layout()
    fig.savefig("figs/fairness/roc2.png")

    # 2.5
    fig2, ax2 = plt.subplots(figsize=figsize)
    for (group,), sub in gp.group_by("group", maintain_order=True):
        ax2.plot(
            sub["FPR"],
            sub["TPR"],
            label=group,
            color=colors.get(group, "black"),
            linestyle=linestyles.get(group, "-"),
            linewidth=2,
        )
    ax2.set(
        xlabel="Fraction defaulters getting loan",
        ylabel="Fraction non-defaulters getting loan",
        title="Zoomed ROC (detail view)",
        xlim=(0, 0.35),  # zoom in on FPR
        ylim=(0.3, 1.0),  # zoom in on TPR
    )
    ax2.legend(title="")
    plt.tight_layout()
    fig2.savefig("figs/fairness/roc2_zoom.png")


if __name__ == "__main__":
    main()
