"""
models.py  —  Phase 4: ML Model Training
=========================================

WHAT THIS FILE DOES
-------------------
The modeling toolbox for the project. Provides:

  * Model builders (one per algorithm family), each returning a sklearn
    Pipeline so preprocessing and the estimator are inseparable.
        - build_ridge_pipeline       (linear, with L2 regularization)
        - build_random_forest_pipeline (bagged trees)
        - build_xgboost_pipeline     (gradient-boosted trees)

  * Train/test split helper -- a one-liner with a fixed seed.

  * Evaluation utilities    -- RMSE, MAE, R^2, both in log-space and dollar-space.

  * Cross-validation        -- k-fold CV that returns per-fold metrics as a DataFrame.

  * Hyperparameter tuning   -- RandomizedSearchCV wrapper with a sensible grid for XGBoost.

  * Persistence             -- joblib save/load.

  * Diagnostic plotting     -- predicted-vs-actual scatter and residuals histogram.

WHY PIPELINES?
--------------
A scikit-learn `Pipeline` glues preprocessing steps (like StandardScaler)
to the estimator. The crucial property: when you call `.fit(X_train)`,
the pipeline learns the scaler's parameters from X_train ONLY, then uses
them on X_test. This automatically prevents the "preprocessing-leakage"
trap from Phase 2's doc 03.

In short: a Pipeline is a contract that says "this preprocessing and
this model are ONE thing; you cannot use the model without applying the
preprocessing exactly the same way I learned at fit time."

WHY THREE MODELS?
-----------------
The three families cover the spectrum of common tabular regression:

    Linear (Ridge)    -- simple, interpretable, baseline.
    Random Forest     -- non-parametric, robust, captures interactions.
    XGBoost           -- gradient boosting, usually the strongest tabular model.

Comparing them on the same data is how you find out which family the
data is best suited to. See docs/07_model_selection.md for the full tour.

USAGE
-----
    from src.features import prepare_modeling_data
    from src.data_loader import load_ames, load_zillow_zhvi
    from src import models

    X, y = prepare_modeling_data(load_ames(), zhvi=load_zillow_zhvi())
    X_train, X_test, y_train, y_test = models.split(X, y)

    ridge   = models.build_ridge_pipeline()
    rforest = models.build_random_forest_pipeline()
    xgb     = models.build_xgboost_pipeline()

    comparison = models.compare_cv({"ridge": ridge, "rf": rforest, "xgb": xgb},
                                   X_train, y_train, cv=5)
    print(comparison)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# Where to save fitted models. The .gitignore already excludes *.joblib.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "data" / "outputs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 1. TRAIN / TEST SPLIT
# ----------------------------------------------------------------------------
# We hold out 20% of the data for a FINAL, honest estimate of how the model
# performs on data it has never seen. The remaining 80% is what we train
# and cross-validate on.
#
# random_state=42 fixes the split so re-runs are reproducible. Without a
# seed, your "test score" would change every time you ran the notebook --
# which is harmless until you start tuning hyperparameters and accidentally
# convince yourself that fold variance is real improvement.
# ============================================================================

DEFAULT_RANDOM_STATE: int = 42


def split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.20,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """80/20 train/test split with a fixed seed."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ============================================================================
# 2. MODEL BUILDERS
# ----------------------------------------------------------------------------
# Each builder returns a Pipeline. Default hyperparameters are sensible
# starting points -- callers can override via kwargs.
#
# Why Ridge instead of plain LinearRegression?
#   With 230+ features and visible multicollinearity (Phase 2 EDA), plain
#   OLS coefficients are unstable. Ridge adds an L2 penalty that shrinks
#   correlated coefficients toward each other -- gives near-OLS predictions
#   with much more stability. See docs/07_model_selection.md for the math.
# ============================================================================

def build_ridge_pipeline(*, alpha: float = 1.0) -> Pipeline:
    """Ridge regression with feature standardization.

    Standardization (mean=0, std=1) matters for Ridge because the L2
    penalty is applied uniformly to all coefficients. If one feature is
    in dollars (10^5) and another is 0/1, the dollar feature dominates
    the penalty without scaling.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha, random_state=DEFAULT_RANDOM_STATE)),
    ])


def build_random_forest_pipeline(
    *,
    n_estimators: int = 400,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = -1,
) -> Pipeline:
    """Random Forest regressor. No scaling needed -- trees are scale-invariant."""
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return Pipeline([("model", rf)])


def build_xgboost_pipeline(
    *,
    n_estimators: int = 600,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 1.0,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = -1,
) -> Pipeline:
    """XGBoost regressor with sensible defaults for ~3k-row tabular data.

    learning_rate * n_estimators is the headline tradeoff -- lower
    learning rate + more trees = slower but usually more accurate.
    """
    xgb = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",   # fast histogram-based algorithm
        verbosity=0,
    )
    return Pipeline([("model", xgb)])


# ============================================================================
# 3. EVALUATION METRICS
# ----------------------------------------------------------------------------
# We compute three metrics. See docs/08_evaluation_metrics.md for the
# full intuition on each.
#
#   RMSE -- root mean squared error. Penalizes big mistakes more.
#   MAE  -- mean absolute error. Robust to outliers.
#   R^2  -- "fraction of variance explained". 1.0 = perfect, 0.0 = no better
#           than predicting the mean, < 0 = worse than the mean.
# ============================================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, R^2 for one set of predictions."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def evaluate(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    target_is_log: bool = True,
) -> dict[str, float]:
    """Predict on X and report metrics in BOTH log space and dollar space.

    If `target_is_log` is True (the default for this project), we assume
    `y` is `log1p(saleprice)`. We exponentiate predictions and ground
    truth back to dollars and report a second set of metrics there.
    """
    y_pred = model.predict(X)
    out = {f"log_{k}": v for k, v in regression_metrics(np.asarray(y), y_pred).items()}

    if target_is_log:
        y_dollars = np.expm1(np.asarray(y))
        y_pred_dollars = np.expm1(y_pred)
        for k, v in regression_metrics(y_dollars, y_pred_dollars).items():
            out[f"dollar_{k}"] = v

    return out


# ============================================================================
# 4. CROSS-VALIDATION
# ----------------------------------------------------------------------------
# k-fold CV repeatedly splits the training data into K folds. We train on
# K-1 folds and score on the held-out fold. Average across all K folds.
# This gives a robust estimate of "out-of-sample" performance using only
# the training data -- the test set is held back for the *final* check.
#
# We use 5-fold CV with a fixed shuffle. negative-RMSE is sklearn convention.
# ============================================================================

CV_SCORING: dict[str, str] = {
    "rmse": "neg_root_mean_squared_error",
    "mae":  "neg_mean_absolute_error",
    "r2":   "r2",
}


def cv_score(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Return a DataFrame with one row per fold and one column per metric."""
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = cross_validate(
        model, X, y,
        cv=splitter,
        scoring=CV_SCORING,
        n_jobs=-1,
        return_train_score=False,
    )
    # sklearn returns negative RMSE/MAE -- flip them back so positive = bigger error.
    df = pd.DataFrame({
        "fold": np.arange(1, cv + 1),
        "rmse": -results["test_rmse"],
        "mae":  -results["test_mae"],
        "r2":    results["test_r2"],
    })
    return df


def compare_cv(
    models: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv: int = 5,
) -> pd.DataFrame:
    """Run cv_score for each model and return a side-by-side mean+std table."""
    rows = []
    for name, mdl in models.items():
        folds = cv_score(mdl, X, y, cv=cv)
        rows.append({
            "model": name,
            "rmse_mean": folds["rmse"].mean(),
            "rmse_std":  folds["rmse"].std(),
            "mae_mean":  folds["mae"].mean(),
            "r2_mean":   folds["r2"].mean(),
        })
    return pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)


# ============================================================================
# 5. HYPERPARAMETER TUNING
# ----------------------------------------------------------------------------
# We use RandomizedSearchCV rather than GridSearchCV: randomized search
# tends to find good configurations faster because most hyperparameter
# combinations are equivalent to each other. See doc 09 for the proof.
# ============================================================================

# A reasonable XGBoost search space for ~3k-row tabular regression.
XGBOOST_PARAM_DIST: dict = {
    "model__n_estimators":     [300, 500, 800, 1200],
    "model__learning_rate":    [0.03, 0.05, 0.08, 0.10],
    "model__max_depth":        [3, 4, 5, 6],
    "model__subsample":        [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "model__reg_lambda":       [0.5, 1.0, 2.0, 5.0],
}


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_iter: int = 30,
    cv: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[Pipeline, dict, pd.DataFrame]:
    """Randomized search over XGBOOST_PARAM_DIST. Returns (best_model, best_params, cv_results)."""
    base = build_xgboost_pipeline()
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=XGBOOST_PARAM_DIST,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=splitter,
        n_jobs=-1,
        random_state=random_state,
        refit=True,           # refit on full training data with best params
        return_train_score=False,
        verbose=0,
    )
    search.fit(X, y)

    # Cleaner CV results table for reporting.
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results = cv_results[["mean_test_score", "std_test_score", "params"]].copy()
    cv_results["mean_rmse"] = -cv_results["mean_test_score"]
    cv_results = cv_results.sort_values("mean_rmse").reset_index(drop=True)
    cv_results = cv_results[["mean_rmse", "std_test_score", "params"]]

    return search.best_estimator_, search.best_params_, cv_results


# ============================================================================
# 6. PERSISTENCE
# ----------------------------------------------------------------------------
# joblib serializes the full Pipeline (preprocessing + model) into one file.
# At prediction time, load the file and call `.predict(X)` -- the
# preprocessing replays automatically.
# ============================================================================

def save_model(model: Pipeline, name: str) -> Path:
    """Save a fitted pipeline to data/outputs/<name>.joblib."""
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str) -> Pipeline:
    """Inverse of save_model."""
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)


# ============================================================================
# 7. DIAGNOSTIC PLOTS
# ============================================================================

def plot_predictions_vs_actuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Predictions vs Actuals",
    label_xy: tuple[str, str] = ("actual", "predicted"),
) -> None:
    """Scatter of predicted vs actual, with y=x reference line.

    A perfect model puts all points on the diagonal. Points above the
    diagonal are over-predictions; points below are under-predictions.
    A widening cone toward the right is the classic "heteroskedasticity"
    visual -- your model gets less precise on expensive homes.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20)

    # Diagonal y = x reference line over the actual data range.
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1)

    ax.set_xlabel(label_xy[0])
    ax.set_ylabel(label_xy[1])
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Residuals",
) -> None:
    """Histogram of residuals (y_true - y_pred).

    For a well-fit model the residuals should look roughly normal, centered
    on zero, with no obvious skew. A skewed residual histogram means the
    model is systematically biased.
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(residuals, kde=True, bins=40, ax=ax)
    ax.axvline(0, color="red", linewidth=1)
    ax.set_xlabel("y_true - y_pred")
    ax.set_title(f"{title}  (mean={residuals.mean():.4f}, std={residuals.std():.4f})")
    plt.tight_layout()
    plt.show()


# ============================================================================
# 8. Smoke test entrypoint
# ============================================================================

if __name__ == "__main__":
    from src.data_loader import load_ames, load_zillow_zhvi
    from src.features import prepare_modeling_data

    X, y = prepare_modeling_data(load_ames(), zhvi=load_zillow_zhvi())
    X_train, X_test, y_train, y_test = split(X, y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    models_to_test = {
        "ridge":         build_ridge_pipeline(),
        "random_forest": build_random_forest_pipeline(),
        "xgboost":       build_xgboost_pipeline(),
    }

    comparison = compare_cv(models_to_test, X_train, y_train, cv=5)
    print("\n5-fold CV on training set:")
    print(comparison.round(4).to_string(index=False))

    # Final test-set evaluation with XGBoost (typically the best).
    print("\nFinal XGBoost test-set evaluation:")
    xgb = build_xgboost_pipeline().fit(X_train, y_train)
    print({k: round(v, 4) for k, v in evaluate(xgb, X_test, y_test).items()})
