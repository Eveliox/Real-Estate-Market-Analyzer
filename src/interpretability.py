"""
interpretability.py  —  Phase 6: Interpretability & ROI
========================================================

WHAT THIS FILE DOES
-------------------
Answers two questions that follow naturally after training a model:

  1. **WHY does the model predict what it predicts?**
        -> SHAP (SHapley Additive exPlanations) values assign each feature
           a dollar-denominated contribution to a specific prediction.
        -> Global summary: which features matter most, on average?
        -> Local explanation: why did the model value *this* house at $X?

  2. **WHAT is a renovation worth?**
        -> The ROI simulator changes one or more features of a home and
           re-runs the model to estimate the price lift.
        -> Compare lift vs. estimated renovation cost → ROI %.

WHY SHAP?
---------
Tree-based models (Random Forest, XGBoost) are often called "black boxes"
because their predictions come from hundreds of trees, not from a single
readable equation. SHAP solves this by computing, for each prediction, the
marginal contribution of every feature — grounded in game-theory (Shapley
values). The key properties:

  * **Consistency**: if a feature is used more, its SHAP value increases.
  * **Local accuracy**: SHAP values sum to (prediction − baseline).
  * **Model-agnostic**: works for trees, neural nets, anything.

For tree models, ``shap.TreeExplainer`` uses a fast exact algorithm
(not sampling), so it is both exact and fast.

USAGE
-----
    from src import interpretability as I
    from src import models as M
    from src.features import prepare_modeling_data
    from src.data_loader import load_ames, load_zillow_zhvi

    model = M.load_model("xgb_ames_tuned")
    X, y  = prepare_modeling_data(load_ames(), zhvi=load_zillow_zhvi())
    _, X_test, _, y_test = M.split(X, y)

    explainer = I.get_explainer(model, X_test)
    sv        = I.shap_values(explainer, X_test)

    I.plot_summary(sv, X_test)
    I.plot_importance(sv, X_test)
    I.plot_waterfall(sv, X_test, idx=0)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


# ============================================================================
# 1. SHAP EXPLAINER
# ============================================================================

def get_explainer(model, X_background: pd.DataFrame = None) -> shap.TreeExplainer:
    """Build a SHAP TreeExplainer from the fitted XGBoost pipeline.

    Parameters
    ----------
    model : fitted sklearn Pipeline
        The pipeline saved in Phase 4 (``M.load_model("xgb_ames_tuned")``).
        Must have a step named ``"model"`` that is an XGBRegressor.
    X_background : pd.DataFrame, optional
        Unused — kept for API compatibility. TreeExplainer uses the
        tree-path-dependent algorithm which does not require a background
        dataset and is both faster and exact for tree models.

    Returns
    -------
    shap.TreeExplainer
    """
    xgb_model = model.named_steps["model"]
    return shap.TreeExplainer(xgb_model)


def shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for every row in X.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
        Each entry [i, j] is the SHAP value for sample i, feature j.
        Units are the same as the model target (log-price here).
    """
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[0]
    return sv


def shap_values_dollars(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    baseline_log: float,
) -> np.ndarray:
    """SHAP values converted to approximate dollar contributions.

    Because the target is log1p(SalePrice), a SHAP value of +0.1 means
    the feature pushed the *log* prediction up by 0.1. Converting back:
        dollar_contribution ≈ exp(baseline + shap) - exp(baseline)

    This is an approximation — it treats each feature independently —
    but gives a human-readable "this feature added ~$8,000" framing.

    Parameters
    ----------
    explainer : shap.TreeExplainer
    X : pd.DataFrame
    baseline_log : float
        The model's expected log-price (``explainer.expected_value``).

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
        Approximate dollar contribution per feature per sample.
    """
    sv_log = shap_values(explainer, X)
    baseline = float(baseline_log)
    dollar_sv = np.zeros_like(sv_log)
    for j in range(sv_log.shape[1]):
        dollar_sv[:, j] = np.expm1(baseline + sv_log[:, j]) - np.expm1(baseline)
    return dollar_sv


# ============================================================================
# 2. GLOBAL PLOTS
# ============================================================================

def plot_summary(
    sv: np.ndarray,
    X: pd.DataFrame,
    max_display: int = 20,
    title: str = "SHAP Summary — Feature Impact on log(SalePrice)",
) -> None:
    """Beeswarm plot: every dot is one home, colored by feature value.

    The horizontal position shows whether the feature pushed the prediction
    up (right) or down (left). Color shows whether the raw feature value
    was high (red) or low (blue).

    High feature value + rightward dot → high value of this feature
    increases the predicted price. That's intuitive for things like
    ``total_sf`` (bigger = more expensive).

    Parameters
    ----------
    sv : np.ndarray
        Output of ``shap_values``.
    X : pd.DataFrame
        The same rows that were passed to ``shap_values``.
    max_display : int
        How many features to show (top N by mean |SHAP|).
    title : str
        Plot title.
    """
    plt.figure(figsize=(10, max_display * 0.45 + 2))
    shap.summary_plot(sv, X, max_display=max_display, show=False)
    plt.title(title, fontsize=12, pad=12)
    plt.tight_layout()
    plt.show()


def plot_importance(
    sv: np.ndarray,
    X: pd.DataFrame,
    top_n: int = 20,
    title: str = "Mean |SHAP| — Global Feature Importance",
) -> plt.Axes:
    """Horizontal bar chart of mean absolute SHAP value per feature.

    This is the most compact "which features matter" summary. Each bar
    shows the average size of that feature's impact on the model output,
    regardless of direction.

    Parameters
    ----------
    sv : np.ndarray
        Output of ``shap_values``.
    X : pd.DataFrame
        Same rows passed to ``shap_values``.
    top_n : int
        Number of features to display.
    title : str
        Plot title.

    Returns
    -------
    plt.Axes
    """
    mean_abs = np.abs(sv).mean(axis=0)
    importance = (
        pd.Series(mean_abs, index=X.columns)
        .sort_values(ascending=True)
        .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, top_n * 0.40 + 1.5))
    importance.plot.barh(ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value| (log-price units)")
    ax.set_title(title, fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return ax


def plot_dependence(
    sv: np.ndarray,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: str = "auto",
    title: Optional[str] = None,
) -> None:
    """Scatter of one feature's values vs. its SHAP values.

    Shows how the model's sensitivity to ``feature`` changes across its
    range. Points are colored by an interaction feature (by default, SHAP
    picks the one most correlated with the residual).

    A flat horizontal band → the feature has a linear, constant effect.
    A curve → the effect is non-linear (the model learned diminishing returns,
    a threshold, etc.).

    Parameters
    ----------
    sv : np.ndarray
        Output of ``shap_values``.
    X : pd.DataFrame
        Same rows used to compute ``sv``.
    feature : str
        Column name to put on the x-axis.
    interaction_feature : str or int
        Column name (or index) to color by. ``"auto"`` lets SHAP choose.
    title : str, optional
        Override the default title.
    """
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not in X. Did you mean one of: {[c for c in X.columns if feature.lower().replace(' ','_') in c]}")

    col_idx = list(X.columns).index(feature)
    feat_vals = X[feature].values
    shap_vals = sv[:, col_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(feat_vals, shap_vals, c=feat_vals, cmap="coolwarm", alpha=0.6, s=15)
    plt.colorbar(sc, ax=ax, label=feature)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value for {feature}")
    ax.set_title(title or f"SHAP Dependence — {feature}", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 3. LOCAL (SINGLE-PREDICTION) PLOTS
# ============================================================================

def plot_waterfall(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    idx: int,
    max_display: int = 15,
    title: Optional[str] = None,
) -> None:
    """Waterfall plot: why did the model predict THIS price for HOME idx?

    Starting from the average prediction (the baseline), the waterfall
    shows each feature pushing the prediction up or down until we arrive
    at the final output. The longest bars dominate this particular home's
    price estimate.

    Parameters
    ----------
    explainer : shap.TreeExplainer
    X : pd.DataFrame
        The dataset (use the test set so the home was not seen at training).
    idx : int
        Row index of the home to explain.
    max_display : int
        How many features to show before collapsing the rest into "other".
    title : str, optional
        Override the auto-generated title.
    """
    row = X.iloc[[idx]]
    sv_row = explainer(row)

    plt.figure(figsize=(10, max_display * 0.5 + 2))
    shap.plots.waterfall(sv_row[0], max_display=max_display, show=False)
    if title:
        plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def explain_row(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    idx: int,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the top-N SHAP contributors for home `idx` as a DataFrame.

    Useful for tabular display in a notebook cell when you want numbers,
    not a plot.

    Returns
    -------
    pd.DataFrame with columns: feature, feature_value, shap_value
        Sorted by |shap_value| descending.
    """
    row = X.iloc[[idx]]
    sv_row = shap_values(explainer, row)[0]

    df = pd.DataFrame({
        "feature": X.columns,
        "feature_value": row.values[0],
        "shap_value": sv_row,
    })
    df["abs_shap"] = df["shap_value"].abs()
    return (
        df.sort_values("abs_shap", ascending=False)
        .head(top_n)
        .drop(columns="abs_shap")
        .reset_index(drop=True)
    )


# ============================================================================
# 4. ROI SIMULATOR
# ============================================================================

def predict_dollars(model, X: pd.DataFrame) -> np.ndarray:
    """Run the model and convert log-price predictions back to dollars."""
    return np.expm1(model.predict(X))


def simulate_upgrade(
    model,
    X: pd.DataFrame,
    idx: int,
    upgrades: Dict[str, float],
) -> Tuple[float, float, float]:
    """Estimate the price lift from a set of feature changes on home `idx`.

    Parameters
    ----------
    model : fitted sklearn Pipeline
    X : pd.DataFrame
        Feature matrix (typically the test set).
    idx : int
        Row index of the home to upgrade.
    upgrades : dict
        Mapping of ``{feature_name: new_value}``. Only these features change;
        everything else stays the same.

    Returns
    -------
    (before_dollars, after_dollars, lift_dollars)
        Predicted price before upgrades, after upgrades, and the difference.

    Example
    -------
    Adding a full bathroom (``total_bath += 1``):

        before, after, lift = simulate_upgrade(
            model, X_test, idx=5,
            upgrades={"total_bath": X_test.iloc[5]["total_bath"] + 1},
        )
        print(f"Lift: ${lift:,.0f}")
    """
    before = predict_dollars(model, X.iloc[[idx]])[0]

    row_upgraded = X.iloc[[idx]].copy()
    for feature, new_val in upgrades.items():
        if feature not in row_upgraded.columns:
            raise ValueError(f"Feature '{feature}' not in X. Available: {list(X.columns)[:10]}...")
        row_upgraded[feature] = new_val

    after = predict_dollars(model, row_upgraded)[0]
    return before, after, after - before


def roi_table(
    model,
    X: pd.DataFrame,
    idx: int,
    scenarios: List[Dict],
) -> pd.DataFrame:
    """Tabulate upgrade ROI for a set of renovation scenarios.

    Parameters
    ----------
    model : fitted sklearn Pipeline
    X : pd.DataFrame
        Feature matrix.
    idx : int
        Home row index.
    scenarios : list of dict, each with keys:
        ``label``    str   Human-readable name (e.g. "Add full bathroom")
        ``upgrades`` dict  Passed to ``simulate_upgrade``
        ``cost``     float Estimated renovation cost in dollars

    Returns
    -------
    pd.DataFrame with columns:
        scenario, before ($), after ($), price_lift ($), cost ($), roi_%
    """
    rows = []
    for s in scenarios:
        before, after, lift = simulate_upgrade(model, X, idx, s["upgrades"])
        cost = s.get("cost", 0)
        roi = (lift / cost * 100) if cost > 0 else float("nan")
        rows.append({
            "scenario": s["label"],
            "before ($)": before,
            "after ($)": after,
            "price_lift ($)": lift,
            "cost ($)": cost,
            "roi_%": roi,
        })

    return pd.DataFrame(rows).set_index("scenario")
