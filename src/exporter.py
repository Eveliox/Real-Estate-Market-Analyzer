"""
exporter.py  —  Phase 7: Power BI Export & Investor ROI
=========================================================

WHAT THIS FILE DOES
-------------------
Two responsibilities:

  1. **Data Export** — assembles everything produced in Phases 2-6 into a
     set of clean CSVs *and* a single SQLite database that Power BI (or any
     BI tool) can connect to directly.

     Tables exported:
       ames_clean        Cleaned Ames housing data (2,930 homes × ~80 cols)
       ames_predictions  Model predictions + actuals + error for every home
       shap_importance   Mean |SHAP| per feature (global importance ranking)
       zhvi_forecasts    Iowa ZIP-level 24-month price forecasts (Phase 5)
       investment_roi    Cap rate / cash-on-cash / IRR per Ames home

  2. **Investor ROI Metrics** — given rental income assumptions, computes
     three standard real estate investment metrics for each home:

       Cap Rate        Net Operating Income ÷ Property Value.
                       "If I bought it cash, what yield do I get?"

       Cash-on-Cash    Annual cash flow after debt service ÷ cash invested.
                       "What does my actual cash return look like with a
                        mortgage?"

       IRR             Internal Rate of Return over a 10-year hold with
                       assumed 3% annual appreciation. Accounts for the
                       time value of money; harder to game than cap rate.

WHY SQLITE?
-----------
Power BI can connect directly to an SQLite file with the SQLite ODBC driver
(free). All five tables live in one file — no CSV folder to manage, no
import wizard for each table. A single .pbix file can query all tables with
relationships set up between them.

USAGE
-----
    from src import exporter as E
    from src import models as M, interpretability as I
    from src.features import prepare_modeling_data
    from src.data_loader import load_ames, load_zillow_zhvi

    model = M.load_model("xgb_ames_tuned")
    X, y  = prepare_modeling_data(load_ames(), zhvi=load_zillow_zhvi())
    _, X_test, _, y_test = M.split(X, y)

    explainer = I.get_explainer(model)
    sv        = I.shap_values(explainer, X_test)

    db_path = E.build_sqlite(model, X, y, sv, X_test, y_test)
    print(f"Power BI database: {db_path}")
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
OUTPUTS_DIR: Path = PROJECT_ROOT / "data" / "outputs"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 1. INDIVIDUAL CSV EXPORTERS
# ============================================================================

def export_ames_clean(df: pd.DataFrame, name: str = "ames_clean") -> Path:
    """Write cleaned Ames data to data/processed/ as CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``preprocessing.clean_ames(load_ames())``.
    name : str
        File stem (no extension).

    Returns
    -------
    Path
        Full path to the written CSV.
    """
    path = PROCESSED_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"[export] ames_clean       → {path}  ({len(df):,} rows)")
    return path


def export_predictions(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    name: str = "ames_predictions",
) -> pd.DataFrame:
    """Build a predictions table and write it to data/outputs/.

    Each row is one home from the full Ames feature matrix with columns:
        home_id       row index from X
        actual_log    log1p(SalePrice) — the model target
        pred_log      model's prediction in log space
        actual_usd    actual SalePrice in dollars
        pred_usd      predicted SalePrice in dollars
        error_usd     pred_usd − actual_usd (positive = over-predicted)
        abs_error_usd |error_usd|
        pct_error     abs_error_usd / actual_usd × 100

    Returns
    -------
    pd.DataFrame
        The predictions table (also saved to CSV).
    """
    pred_log = model.predict(X)
    actual_usd = np.expm1(y.to_numpy())
    pred_usd = np.expm1(pred_log)

    df = pd.DataFrame({
        "home_id": np.arange(len(X)),
        "actual_log": y.to_numpy(),
        "pred_log": pred_log,
        "actual_usd": actual_usd,
        "pred_usd": pred_usd,
        "error_usd": pred_usd - actual_usd,
        "abs_error_usd": np.abs(pred_usd - actual_usd),
        "pct_error": np.abs(pred_usd - actual_usd) / actual_usd * 100,
    })

    path = OUTPUTS_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"[export] ames_predictions → {path}  ({len(df):,} rows)")
    return df


def export_shap_importance(
    sv: np.ndarray,
    X: pd.DataFrame,
    name: str = "shap_importance",
) -> pd.DataFrame:
    """Write the global SHAP importance table to data/outputs/.

    Columns: feature, mean_abs_shap, rank
    """
    mean_abs = np.abs(sv).mean(axis=0)
    df = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    df["rank"] = df.index + 1

    path = OUTPUTS_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"[export] shap_importance  → {path}  ({len(df):,} features)")
    return df


# ============================================================================
# 2. INVESTOR ROI METRICS
# ============================================================================

def _npv(rate: float, cash_flows: List[float]) -> float:
    """Net present value of a series of cash flows at a given discount rate."""
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))


def _irr(cash_flows: List[float], guess: float = 0.10) -> float:
    """Internal Rate of Return via bisection search.

    Finds the discount rate r such that NPV(r, cash_flows) == 0.
    Returns NaN if no solution is found (e.g. all cash flows same sign).
    """
    try:
        from scipy.optimize import brentq
        lo, hi = -0.999, 10.0
        if _npv(lo, cash_flows) * _npv(hi, cash_flows) > 0:
            return float("nan")
        return brentq(_npv, lo, hi, args=(cash_flows,), xtol=1e-8)
    except Exception:
        return float("nan")


def compute_investor_roi(
    pred_usd: np.ndarray,
    *,
    gross_rent_multiplier: float = 150,
    vacancy_rate: float = 0.05,
    expense_ratio: float = 0.40,
    down_payment_pct: float = 0.20,
    mortgage_rate: float = 0.07,
    loan_term_years: int = 30,
    appreciation_rate: float = 0.03,
    hold_years: int = 10,
) -> pd.DataFrame:
    """Compute cap rate, cash-on-cash return, and IRR for each home.

    Parameters
    ----------
    pred_usd : np.ndarray
        Predicted home values in dollars (one per home).
    gross_rent_multiplier : float
        Monthly rent = property_value / gross_rent_multiplier.
        150 is a common rule-of-thumb for mid-tier US markets.
    vacancy_rate : float
        Fraction of time the property is vacant (default 5%).
    expense_ratio : float
        Operating expenses as a fraction of gross rent (default 40%).
        Covers maintenance, insurance, property tax, management.
    down_payment_pct : float
        Down payment as a fraction of purchase price (default 20%).
    mortgage_rate : float
        Annual mortgage interest rate (default 7%).
    loan_term_years : int
        Mortgage amortization period (default 30 years).
    appreciation_rate : float
        Annual home value appreciation assumed during hold (default 3%).
    hold_years : int
        Investment horizon in years (default 10).

    Returns
    -------
    pd.DataFrame with columns:
        home_id, pred_value_usd, monthly_rent, annual_noi,
        cap_rate_pct, annual_cash_flow, cash_invested,
        cash_on_cash_pct, irr_pct
    """
    n_months = loan_term_years * 12
    monthly_rate = mortgage_rate / 12

    rows = []
    for i, value in enumerate(pred_usd):
        monthly_rent = value / gross_rent_multiplier
        annual_gross = monthly_rent * 12 * (1 - vacancy_rate)
        annual_noi = annual_gross * (1 - expense_ratio)

        cap_rate = annual_noi / value if value > 0 else 0.0

        loan = value * (1 - down_payment_pct)
        if monthly_rate > 0:
            monthly_payment = loan * (monthly_rate * (1 + monthly_rate) ** n_months) / \
                              ((1 + monthly_rate) ** n_months - 1)
        else:
            monthly_payment = loan / n_months
        annual_debt_service = monthly_payment * 12
        annual_cash_flow = annual_noi - annual_debt_service

        cash_invested = value * down_payment_pct
        coc = annual_cash_flow / cash_invested if cash_invested > 0 else 0.0

        # IRR: year-0 outflow is cash_invested; years 1-hold are annual_cash_flow;
        # at year hold_years, add sale proceeds minus remaining loan balance.
        sale_price = value * (1 + appreciation_rate) ** hold_years
        remaining_balance = loan * (1 + monthly_rate) ** (hold_years * 12)
        for _ in range(hold_years * 12):
            remaining_balance = remaining_balance * (1 + monthly_rate) - monthly_payment
        remaining_balance = max(remaining_balance, 0)
        terminal_cf = annual_cash_flow + sale_price - remaining_balance

        cf_series = [-cash_invested] + [annual_cash_flow] * (hold_years - 1) + [terminal_cf]
        irr = _irr(cf_series)

        rows.append({
            "home_id": i,
            "pred_value_usd": round(value, 2),
            "monthly_rent": round(monthly_rent, 2),
            "annual_noi": round(annual_noi, 2),
            "cap_rate_pct": round(cap_rate * 100, 3),
            "annual_cash_flow": round(annual_cash_flow, 2),
            "cash_invested": round(cash_invested, 2),
            "cash_on_cash_pct": round(coc * 100, 3),
            "irr_pct": round(irr * 100, 3) if not np.isnan(irr) else None,
        })

    df = pd.DataFrame(rows)
    path = OUTPUTS_DIR / "investment_roi.csv"
    df.to_csv(path, index=False)
    print(f"[export] investment_roi   → {path}  ({len(df):,} rows)")
    return df


# ============================================================================
# 3. SQLITE DATABASE BUILDER
# ============================================================================

def build_sqlite(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    sv: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    forecasts_path: Optional[Path] = None,
    db_name: str = "real_estate.db",
) -> Path:
    """Assemble all outputs into a single SQLite file for Power BI.

    Parameters
    ----------
    model : fitted sklearn Pipeline
        The saved XGBoost model from Phase 4.
    X, y : pd.DataFrame, pd.Series
        Full feature matrix and log-price target (all homes).
    sv : np.ndarray
        SHAP values for X_test (from Phase 6).
    X_test, y_test : pd.DataFrame, pd.Series
        The held-out test set.
    forecasts_path : Path, optional
        Path to the ZHVI forecasts CSV from Phase 5. Defaults to
        ``data/outputs/iowa_zhvi_forecasts.csv``.
    db_name : str
        Filename of the SQLite database in data/outputs/.

    Returns
    -------
    Path
        Full path to the .db file.
    """
    db_path = OUTPUTS_DIR / db_name
    if db_path.exists():
        db_path.unlink()

    print(f"\nBuilding SQLite database: {db_path}\n")

    # --- predictions (all homes) ---
    pred_df = export_predictions(model, X, y)

    # --- SHAP importance (from test set) ---
    shap_df = export_shap_importance(sv, X_test)

    # --- investor ROI (all homes) ---
    roi_df = compute_investor_roi(pred_df["pred_usd"].values)

    # --- ZHVI forecasts ---
    fp = forecasts_path or (OUTPUTS_DIR / "iowa_zhvi_forecasts.csv")
    if fp.exists():
        zhvi_df = pd.read_csv(fp)
        print(f"[export] zhvi_forecasts   → loaded from {fp}  ({len(zhvi_df):,} rows)")
    else:
        zhvi_df = pd.DataFrame(columns=["zip", "date", "forecast", "lower_95", "upper_95"])
        print(f"[warn]  zhvi_forecasts not found at {fp} — table will be empty")

    # --- write to SQLite ---
    con = sqlite3.connect(db_path)
    tables = {
        "ames_predictions": pred_df,
        "shap_importance": shap_df,
        "investment_roi": roi_df,
        "zhvi_forecasts": zhvi_df,
    }
    for table_name, df in tables.items():
        df.to_sql(table_name, con, if_exists="replace", index=False)
        print(f"[sqlite] wrote {table_name:<22} ({len(df):,} rows)")

    con.close()

    size_mb = db_path.stat().st_size / 1e6
    print(f"\nDone. Database size: {size_mb:.2f} MB  →  {db_path}")
    return db_path


# ============================================================================
# 4. SUMMARY REPORT
# ============================================================================

def print_export_summary(db_path: Path) -> None:
    """Print a table of all SQLite tables with row counts and columns."""
    con = sqlite3.connect(db_path)
    cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    print(f"\n{'─' * 55}")
    print(f"  SQLite database: {db_path.name}")
    print(f"{'─' * 55}")
    print(f"  {'Table':<25} {'Rows':>8}  {'Columns':>8}")
    print(f"{'─' * 55}")
    for t in tables:
        df = pd.read_sql(f"SELECT * FROM {t} LIMIT 1", con)
        count = pd.read_sql(f"SELECT COUNT(*) as n FROM {t}", con)["n"][0]
        print(f"  {t:<25} {count:>8,}  {len(df.columns):>8}")
    print(f"{'─' * 55}\n")
    con.close()
