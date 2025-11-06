"""
Solutions for week 9.

Usage:
    python scripts/week9_pp.py
"""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pl.read_csv("data/crime.csv")


# Exercise 1: train test split with 0.2 and random_state 42

# 1.2
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = cast(pl.DataFrame, train_df)
test_df = cast(pl.DataFrame, test_df)
features = ["PriorFTA", "FelonyArrests", "Age"]
outcome = ["FTA"]
X_train = train_df.select(features).to_numpy()
y_train = train_df.select(outcome).to_numpy().ravel()
X_test = test_df.select(features).to_numpy()
y_test = test_df.select(outcome).to_numpy().ravel()


# 1.3
mask = train_df["FTA"].is_not_null().to_numpy().ravel()
model = LogisticRegression()
model.fit(X_train[mask], y_train[mask])

maskt = test_df["FTA"].is_not_null().to_numpy().ravel()
preds = model.predict(X_test[maskt])
print("Accuracy:", np.mean(preds == y_test[maskt]))

# 1.4-6 + 1.8
gp = (
    test_df.with_columns(risk=pl.lit(model.predict_proba(X_test)[:, 1]))
    .filter(
        # Selective labels
        pl.col("FTA").is_not_null()
    )
    .with_columns(
        # Individuals partitioned into quantile bins
        pl.col("risk")
        # NOTE: qcut happens here
        .qcut(
            np.linspace(0.10, 1, 10).tolist(),
            labels=(labs := [f"Q{i}" for i in range(1, 10 + 2)]),
        )
        .alias("bin")
        .cast(pl.Enum(labs))
    )
    .sort("risk")
    # NOTE: 1.8 happens here
    .pipe(
        lambda df: df.with_columns(
            sbin=pl.lit(npr.randint(1, 10 + 1, size=df.shape[0])),
        )
    )
    .with_columns(
        pl.col("sbin")
        .replace_strict(dict(zip(range(1, 10 + 1), labs)))
        .cast(pl.Enum(labs))
    )
)

print(
    gp.group_by("bin")
    .agg(
        pl.col("risk").mean(),
        pl.col("FTA").mean(),
        pl.col("Release").mean(),
    )
    .sort("bin"),
    gp.group_by("sbin")
    .agg(
        pl.col("risk").mean(),
        pl.col("FTA").mean(),
        pl.col("Release").mean(),
    )
    .sort("sbin"),
    sep="\n",
)

print(train_df.shape, test_df.shape)


# 1.7

res = []
for lab in ["Q10", "Q9", "Q8", "Q7", "Q6", "Q5", "Q4", "Q3", "Q2", "Q1"]:

    def rates(col: str, lab: str):
        subset = gp.with_columns(
            pl.when(
                # bin and sbin enums, hence we can compare to str like this
                # ge for e.g. col >= Q10
                pl.col(col).ge(lab)
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("Jailed"),
        ).with_columns(
            # Set FTA to 0 for those jailed
            pl.when(pl.col("Jailed").eq(0))
            .then(pl.col("FTA"))
            .otherwise(pl.lit(0))
            .alias("FTA")
        )
        return subset.select(
            pl.col("Jailed").mean(),
            pl.col("FTA").mean(),
        ).with_columns(pl.lit(col).alias("model"), pl.lit(lab).alias("lab"))

    res.extend(rates("bin", lab).to_dicts() + rates("sbin", lab).to_dicts())


# 1.9
# Compute baseline FTA rate and compute FTA rate relative to baseline
baseline_fta = gp["FTA"].mean()
dfres = (
    pl.DataFrame(res)
    .with_columns(pl.col("lab").str.strip_chars_start("Q").cast(pl.Int8))
    .with_columns(
        FTApct=(pl.col("FTA").sub(baseline_fta).truediv(baseline_fta).mul(100))
    )
)


# 1.10

styles = {"bin": "-", "sbin": "--"}
name_map = {"bin": "Algorithm", "sbin": "Random"}
colors = {"bin": "black", "sbin": "grey"}
fig, ax = plt.subplots(figsize=(6, 4))
for (mlab,), dfm in dfres.group_by("model"):
    ax.plot(
        dfm["Jailed"],
        dfm["FTApct"],
        label=name_map[mlab],
        linestyle=styles[mlab],
        color=colors[mlab],
    )
ax.set_xlabel("Jail rate")
ax.set_ylabel("Percent Decline in Crime Rate")
ax.legend(title="Decision Maker")
ax.invert_xaxis()
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
fig.savefig("figs/contraction.png")
