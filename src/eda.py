"""
eda.py  —  Phase 2: Exploratory Data Analysis helpers
=====================================================

WHAT THIS FILE DOES
-------------------
Provides a small library of EDA functions that you call from a notebook.
They handle the boilerplate (`plt.figure(...)`, axis labels, sorting,
correlation math) so the notebook stays focused on *findings*, not plumbing.

Functions are grouped into:

  * Summary tables    -- numeric_summary, categorical_summary, missingness
  * Distribution plots -- plot_distribution
  * Relationship plots -- plot_numeric_vs_target, plot_categorical_vs_target
  * Correlation tools  -- top_correlations, plot_correlation_heatmap

WHY A LIBRARY INSTEAD OF INLINE CELLS?
--------------------------------------
The first time you do EDA you'll write `df["col"].hist()` ten times
in a notebook. The second time, you'll copy-paste the styling code.
The third time, you'll wish you'd named the function.

Building tiny named helpers from day one means: (a) your notebook reads
as a *story* of findings, not a wall of matplotlib config, and
(b) you can call the same helper in later phases without re-inventing it.

USAGE
-----
    from src.data_loader import load_ames
    from src.preprocessing import clean_ames
    from src import eda

    df = clean_ames(load_ames())
    eda.numeric_summary(df).head(10)
    eda.plot_distribution(df["saleprice"], log=True)
    eda.top_correlations(df, target="saleprice", n=15)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Aesthetics: set once, apply everywhere
# ---------------------------------------------------------------------------
# A consistent visual style makes a notebook feel polished without effort.
# We pick seaborn's "whitegrid" theme + a colorblind-safe palette.
# Change these defaults here, not in every plot.
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams["figure.dpi"] = 90  # crisper inline rendering in Jupyter


# ============================================================================
# 1. SUMMARIES
# ============================================================================

def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-numeric-column stats table.

    Returns a DataFrame indexed by column name with:
        count, mean, std, min, 25%, median, 75%, max, skew, n_missing
    Sorted by absolute skew (most lopsided distributions on top), which
    is where surprises tend to hide.
    """
    numeric = df.select_dtypes(include=np.number)
    if numeric.empty:
        return pd.DataFrame()

    # `.describe().T` flips the standard describe table so each ROW is a column.
    desc = numeric.describe().T
    desc["skew"] = numeric.skew()
    desc["n_missing"] = numeric.isna().sum()
    # Reorder columns into a friendlier reading order.
    desc = desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "skew", "n_missing"]]
    desc = desc.rename(columns={"50%": "median"})
    return desc.sort_values("skew", key=lambda s: s.abs(), ascending=False)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-categorical-column stats: unique count, top value, top-value share."""
    cat = df.select_dtypes(include="object")
    if cat.empty:
        return pd.DataFrame()

    rows = []
    for col in cat.columns:
        s = cat[col]
        top_val = s.mode().iloc[0] if not s.mode().empty else None
        rows.append({
            "column": col,
            "n_unique": s.nunique(dropna=True),
            "top_value": top_val,
            "top_share_pct": round(100 * (s == top_val).mean(), 1) if top_val is not None else np.nan,
            "n_missing": s.isna().sum(),
        })
    return pd.DataFrame(rows).set_index("column").sort_values("n_unique", ascending=False)


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missingness, sorted from worst to best.

    Only returns columns with at least one missing value.
    """
    n = len(df)
    counts = df.isna().sum()
    pct = (counts / n * 100).round(2)
    out = pd.DataFrame({"n_missing": counts, "pct_missing": pct})
    return out[out["n_missing"] > 0].sort_values("n_missing", ascending=False)


# ============================================================================
# 2. DISTRIBUTION PLOTS
# ============================================================================

def plot_distribution(
    series: pd.Series,
    *,
    log: bool = False,
    bins: int = 40,
    title: Optional[str] = None,
) -> None:
    """Histogram + KDE for a single numeric series.

    Parameters
    ----------
    series : pd.Series
        Numeric series to plot.
    log : bool, default False
        If True, also plots `log1p(series)` side-by-side. Useful for
        right-skewed targets like SalePrice.
    bins : int
        Number of histogram bins.
    title : str or None
        Custom suptitle. If None, uses the series name.

    Why histogram + KDE?
    --------------------
    A histogram tells you *where* the data is. A KDE (Kernel Density
    Estimate) smooths it into a continuous curve so the *shape* is
    obvious. Together they're complementary; alone, each can mislead.
    """
    name = title or (series.name or "value")

    if log:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.histplot(series.dropna(), bins=bins, kde=True, ax=axes[0])
        axes[0].set_title(f"{name} -- raw")
        sns.histplot(np.log1p(series.dropna()), bins=bins, kde=True, ax=axes[1])
        axes[1].set_title(f"{name} -- log1p")
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(series.dropna(), bins=bins, kde=True, ax=ax)
        ax.set_title(name)

    plt.tight_layout()
    plt.show()


def plot_numeric_vs_target(
    df: pd.DataFrame,
    feature: str,
    target: str = "saleprice",
    *,
    log_y: bool = False,
) -> None:
    """Scatter plot of `feature` vs `target` with a regression line overlay.

    Use this to *visually* judge whether a feature has a linear,
    nonlinear, or no relationship with the target.

    log_y=True is handy when the target is right-skewed -- patterns
    that are messy on raw price often look clean on log price.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.log1p(df[target]) if log_y else df[target]
    sns.regplot(
        x=df[feature], y=y, ax=ax,
        scatter_kws={"alpha": 0.4, "s": 18},
        line_kws={"color": "red"},
    )
    ax.set_xlabel(feature)
    ax.set_ylabel(f"log1p({target})" if log_y else target)
    ax.set_title(f"{feature}  vs  {target}{' (log1p)' if log_y else ''}")
    plt.tight_layout()
    plt.show()


def plot_categorical_vs_target(
    df: pd.DataFrame,
    feature: str,
    target: str = "saleprice",
    *,
    order_by_median: bool = True,
    max_categories: int = 30,
) -> None:
    """Boxplot of target distribution per category of `feature`.

    Box plots show the central tendency AND the spread within each
    category. If category A's box is way above category B's, that
    feature carries real information about the target.

    `order_by_median=True` sorts categories from lowest-median to
    highest-median, which makes the plot read as a ranking. Without
    this, alphabetical ordering buries the signal.
    """
    if feature not in df.columns:
        raise KeyError(f"column {feature!r} not in df")

    s = df[feature]
    # If the column has too many distinct values, the plot becomes unreadable.
    # Bail with a hint rather than silently truncating.
    if s.nunique() > max_categories:
        print(f"[plot_categorical_vs_target] {feature!r} has {s.nunique()} categories "
              f"(>{max_categories}). Skipping. Consider grouping or filtering first.")
        return

    order = None
    if order_by_median:
        order = df.groupby(feature)[target].median().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * s.nunique() + 4), 4))
    sns.boxplot(x=feature, y=target, data=df, order=order, ax=ax)
    ax.set_title(f"{target} by {feature}{' (sorted by median)' if order_by_median else ''}")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 3. CORRELATION
# ============================================================================

def top_correlations(
    df: pd.DataFrame,
    target: str = "saleprice",
    n: int = 15,
    method: str = "pearson",
) -> pd.Series:
    """Top `n` numeric features by absolute correlation with `target`.

    Returns a Series indexed by column name, values = correlation,
    sorted by absolute strength (positive and negative mixed).

    `method='pearson'` measures LINEAR correlation. Use `'spearman'` if
    you suspect a monotonic-but-nonlinear relationship (e.g., quality
    rating vs. price -- doubling the rating doesn't double the price).
    """
    numeric = df.select_dtypes(include=np.number)
    if target not in numeric.columns:
        raise KeyError(f"target {target!r} must be numeric and present in df.")

    corr = numeric.corr(method=method)[target].drop(target)
    return corr.reindex(corr.abs().sort_values(ascending=False).index).head(n)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    target: str = "saleprice",
    n: int = 15,
    method: str = "pearson",
) -> None:
    """Heatmap of correlations among the top-N features (by |corr with target|).

    This is more revealing than the obvious "all-pairs heatmap" because
    on a 200-column DataFrame the all-pairs version is unreadable.
    We restrict to the features that actually matter for the target.
    """
    top = top_correlations(df, target=target, n=n, method=method)
    cols = [target, *top.index.tolist()]
    corr = df[cols].corr(method=method)

    fig, ax = plt.subplots(figsize=(0.6 * len(cols) + 2, 0.5 * len(cols) + 2))
    sns.heatmap(
        corr,
        annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        square=True, cbar_kws={"shrink": 0.6},
        ax=ax,
    )
    ax.set_title(f"Correlation heatmap -- top {n} features vs {target}")
    plt.tight_layout()
    plt.show()


# ============================================================================
# 4. Convenience runner -- one-call EDA snapshot
# ============================================================================

def quick_overview(df: pd.DataFrame, target: str = "saleprice", n_top: int = 12) -> None:
    """Print + plot a one-shot EDA snapshot. Useful for kicking the tires."""
    print(f"=== Shape: {df.shape[0]:,} rows x {df.shape[1]:,} cols ===\n")

    miss = missingness_table(df)
    if miss.empty:
        print("Missing values: none. Clean data!\n")
    else:
        print(f"Columns with missing values: {len(miss)} (worst 10)")
        print(miss.head(10), "\n")

    if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        print(f"=== Target ({target}) ===")
        print(df[target].describe().round(2).to_string(), "\n")
        plot_distribution(df[target], log=True)

        print(f"\n=== Top {n_top} features by |correlation| with {target} ===")
        top = top_correlations(df, target=target, n=n_top)
        print(top.round(3).to_string(), "\n")

        plot_correlation_heatmap(df, target=target, n=n_top)
    else:
        print(f"[quick_overview] target {target!r} missing or non-numeric -- skipping target plots.")
