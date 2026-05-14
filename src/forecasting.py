"""
forecasting.py  —  Phase 5: Time-Series Forecasting
====================================================

WHAT THIS FILE DOES
-------------------
Turns the Zillow ZHVI ZIP-level monthly panel into forward-looking price
forecasts using classical time-series models (ARIMA / SARIMA).

Key functions:

  melt_zhvi(df)                Wide → long (tidy) reshape of the Zillow CSV.
  filter_zips(long_df, zips)   Keep only the requested ZIP codes.
  get_top_zips(...)            Pick the N most data-complete ZIPs in a state.
  fit_arima(series, ...)       Fit a SARIMAX model on a single time series.
  forecast_zip(...)            End-to-end: filter → fit → predict one ZIP.
  plot_forecast(...)           History + forecast + confidence interval ribbon.
  plot_multi_forecast(...)     Side-by-side panel for several ZIPs.
  decompose(series)            Trend / seasonal / residual decomposition plot.
  save_forecasts(...)          Export combined forecast CSV for Phase 7 / Power BI.

WHY TIME-SERIES?
----------------
The ZHVI data is a **panel**: multiple time series (one per ZIP), each
recorded at regular monthly intervals. This is qualitatively different from
the Ames cross-section used in Phases 2-4:

  Ames (Phases 2-4)  → snapshot: every row is ONE house sale.
  Zillow (Phase 5)   → panel:    every row is ONE ZIP, one month.

The extra structure — time order — lets us exploit autocorrelation:
"this month's value is correlated with last month's." ARIMA models make
that explicit and use it to forecast.

ARIMA REFRESHER
---------------
ARIMA(p, d, q):
  p = autoregressive order   how many lag values to include
  d = differencing order     how many times to difference for stationarity
  q = moving-average order   how many lag error terms to include

SARIMA adds a seasonal layer: (P, D, Q)_s where s=12 for monthly data.
We fit with statsmodels SARIMAX, which handles both in one class.

USAGE
-----
    from src import forecasting as F
    from src.data_loader import load_zillow_zhvi

    long  = F.melt_zhvi(load_zillow_zhvi())
    fc    = F.forecast_zip(long, zip_code="50010", periods=24)
    F.plot_forecast("50010", long, fc)
    plt.show()
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
OUTPUTS_DIR: Path = PROJECT_ROOT / "data" / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Columns in the wide Zillow CSV that are metadata, not dates.
_ZHVI_META_COLS = [
    "RegionID", "SizeRank", "RegionName", "RegionType",
    "StateName", "State", "City", "Metro", "CountyName",
]


# ============================================================================
# 1. DATA RESHAPING
# ============================================================================

def melt_zhvi(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape the wide Zillow ZHVI CSV into tidy (long) format.

    The raw file has one row per ZIP and one column per month-end date
    (e.g. "2024-01-31"). Tidy format has one row per (ZIP, month) pair —
    much easier to filter, group, and pass to a time-series model.

    Parameters
    ----------
    df : pd.DataFrame
        Raw output of ``load_zillow_zhvi()``.

    Returns
    -------
    pd.DataFrame with columns:
        zip     str      5-digit ZIP code (zero-padded)
        state   str      2-letter state abbreviation (when present in source)
        date    Period   monthly period, freq='M'
        zhvi    float    Zillow Home Value Index ($)
    """
    meta_present = [c for c in _ZHVI_META_COLS if c in df.columns]
    date_cols = [c for c in df.columns if c not in meta_present]

    long = df.melt(
        id_vars=meta_present,
        value_vars=date_cols,
        var_name="date",
        value_name="zhvi",
    )

    long["date"] = pd.to_datetime(long["date"]).dt.to_period("M")
    long = long.rename(columns={"RegionName": "zip", "StateName": "state"})
    long["zip"] = long["zip"].astype(str).str.zfill(5)
    long = long.dropna(subset=["zhvi"])
    long = long.sort_values(["zip", "date"]).reset_index(drop=True)

    keep = [c for c in ["zip", "state", "date", "zhvi"] if c in long.columns]
    return long[keep]


def filter_zips(long_df: pd.DataFrame, zips: List[str]) -> pd.DataFrame:
    """Return rows for the given ZIP codes only."""
    zips_padded = [str(z).zfill(5) for z in zips]
    return long_df[long_df["zip"].isin(zips_padded)].copy()


def get_top_zips(
    long_df: pd.DataFrame,
    n: int = 5,
    state: Optional[str] = None,
) -> List[str]:
    """Return the N ZIPs with the most non-null monthly observations.

    Parameters
    ----------
    long_df : pd.DataFrame
        Tidy Zillow data from ``melt_zhvi``.
    n : int
        Number of ZIPs to return.
    state : str, optional
        Filter to one state first (2-letter code, e.g. ``"IA"``).

    Returns
    -------
    list[str]
        ZIP codes sorted by descending observation count.
    """
    df = long_df.copy()
    if state:
        df = df[df["state"] == state.upper()]
    counts = df.groupby("zip")["zhvi"].count().sort_values(ascending=False)
    return counts.head(n).index.tolist()


# ============================================================================
# 2. MODEL FITTING
# ============================================================================

def fit_arima(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 12),
):
    """Fit a SARIMAX model on a monthly time series.

    Parameters
    ----------
    series : pd.Series
        ZHVI values in chronological order (Period index or plain array).
    order : (p, d, q)
        Non-seasonal ARIMA order. (1,1,1) is a solid default for a smoothed
        economic series.
    seasonal_order : (P, D, Q, s)
        Seasonal order; s=12 for monthly data.

    Returns
    -------
    SARIMAXResults
        Fitted model. Call ``.get_forecast(steps=n)`` to generate predictions.

    Notes
    -----
    ``enforce_stationarity=False`` prevents spurious errors when the
    optimizer lands near the boundary of the parameter space. Fine for
    forecasting; tighten for inference.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            series.values.astype(float),
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False)


def _select_order(series: pd.Series) -> Tuple[Tuple, Tuple]:
    """Light AIC grid search over a narrow (p,d,q)×(P,D,Q,12) space.

    Tries p,q ∈ {0,1,2} and P,Q ∈ {0,1} with d=D=1 fixed (one round of
    regular + seasonal differencing almost always works for price series).
    Returns the pair with the lowest AIC.

    Intentionally narrow so it finishes in a few seconds per ZIP. For a
    production search use ``pmdarima.auto_arima`` which covers more ground.
    """
    best_aic = np.inf
    best_order = (1, 1, 1)
    best_seasonal = (1, 1, 0, 12)

    for p in [0, 1, 2]:
        for q in [0, 1, 2]:
            for P in [0, 1]:
                for Q in [0, 1]:
                    try:
                        result = fit_arima(
                            series,
                            order=(p, 1, q),
                            seasonal_order=(P, 1, Q, 12),
                        )
                        if result.aic < best_aic:
                            best_aic = result.aic
                            best_order = (p, 1, q)
                            best_seasonal = (P, 1, Q, 12)
                    except Exception:
                        pass

    return best_order, best_seasonal


# ============================================================================
# 3. FORECASTING
# ============================================================================

def forecast_zip(
    long_df: pd.DataFrame,
    zip_code: str,
    periods: int = 24,
    order: Optional[Tuple] = None,
    seasonal_order: Optional[Tuple] = None,
    auto: bool = True,
) -> pd.DataFrame:
    """Fit SARIMA on one ZIP and return a forecast DataFrame.

    Parameters
    ----------
    long_df : pd.DataFrame
        Tidy Zillow data from ``melt_zhvi``.
    zip_code : str
        5-digit ZIP code to forecast.
    periods : int
        Months ahead to forecast (default 24 = 2 years).
    order, seasonal_order : tuple, optional
        Override the ARIMA orders. When None and ``auto=True`` a small AIC
        grid search selects them automatically.
    auto : bool
        Run AIC order selection when orders are not provided.

    Returns
    -------
    pd.DataFrame with columns:
        date       Period   future month
        forecast   float    point estimate ($)
        lower_95   float    lower bound of 95% confidence interval
        upper_95   float    upper bound of 95% confidence interval
    """
    zip_code = str(zip_code).zfill(5)
    series = (
        long_df[long_df["zip"] == zip_code]
        .set_index("date")["zhvi"]
        .sort_index()
    )
    if series.empty:
        raise ValueError(f"ZIP {zip_code} not found in dataset.")

    # Rare internal gaps → fill with linear interpolation before modeling.
    series = series.interpolate(method="linear")

    if order is None or seasonal_order is None:
        if auto:
            order, seasonal_order = _select_order(series)
        else:
            order, seasonal_order = (1, 1, 1), (1, 1, 0, 12)

    fitted = fit_arima(series, order=order, seasonal_order=seasonal_order)
    pred = fitted.get_forecast(steps=periods)
    ci = pred.conf_int(alpha=0.05)

    last_period = series.index[-1]
    future_periods = pd.period_range(start=last_period + 1, periods=periods, freq="M")

    return pd.DataFrame({
        "date": future_periods,
        "forecast": pred.predicted_mean,
        "lower_95": ci[:, 0],
        "upper_95": ci[:, 1],
    }).reset_index(drop=True)


def forecast_multiple_zips(
    long_df: pd.DataFrame,
    zip_codes: List[str],
    periods: int = 24,
    auto: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Forecast several ZIPs and return a dict of zip → forecast DataFrame.

    Parameters
    ----------
    long_df, zip_codes, periods, auto
        Passed through to ``forecast_zip`` for each ZIP.
    verbose : bool
        Print progress as each ZIP is fitted.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are 5-digit ZIP strings; values are forecast DataFrames.
    """
    results = {}
    for z in zip_codes:
        if verbose:
            print(f"  Fitting ZIP {z}...", end=" ", flush=True)
        try:
            results[z] = forecast_zip(long_df, z, periods=periods, auto=auto)
            if verbose:
                print("done")
        except Exception as exc:
            if verbose:
                print(f"FAILED ({exc})")
    return results


# ============================================================================
# 4. PLOTTING
# ============================================================================

def plot_forecast(
    zip_code: str,
    long_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    history_months: int = 60,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot historical ZHVI and the SARIMA forecast with a 95% CI ribbon.

    Parameters
    ----------
    zip_code : str
        ZIP code (used for label and default title).
    long_df : pd.DataFrame
        Full tidy Zillow history.
    forecast_df : pd.DataFrame
        Output of ``forecast_zip``.
    history_months : int
        How many months of history to show to the left of the forecast line.
    ax : plt.Axes, optional
        Axes to draw into. A new figure is created when None.
    title : str, optional
        Override the default title.

    Returns
    -------
    plt.Axes
    """
    zip_code = str(zip_code).zfill(5)
    history = (
        long_df[long_df["zip"] == zip_code]
        .set_index("date")["zhvi"]
        .sort_index()
        .tail(history_months)
    )

    hist_dates = history.index.to_timestamp()
    fc_dates = forecast_df["date"].dt.to_timestamp()

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    ax.plot(hist_dates, history.values, color="steelblue", linewidth=2,
            label="Historical ZHVI")
    ax.plot(fc_dates, forecast_df["forecast"], color="tomato", linewidth=2,
            linestyle="--", label="Forecast")
    ax.fill_between(
        fc_dates,
        forecast_df["lower_95"],
        forecast_df["upper_95"],
        color="tomato", alpha=0.15, label="95% CI",
    )
    ax.axvline(hist_dates[-1], color="gray", linestyle=":", linewidth=1)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_title(title or f"ZIP {zip_code} — SARIMA Forecast", fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("ZHVI ($)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return ax


def plot_multi_forecast(
    zip_codes: List[str],
    long_df: pd.DataFrame,
    forecasts: Dict[str, pd.DataFrame],
    history_months: int = 60,
) -> plt.Figure:
    """Grid of forecast panels, one per ZIP.

    Parameters
    ----------
    zip_codes : list[str]
        ZIPs to include (must be keys in ``forecasts``).
    long_df : pd.DataFrame
        Full tidy Zillow history.
    forecasts : dict[str, pd.DataFrame]
        Output of ``forecast_multiple_zips``.
    history_months : int
        Months of history shown in each panel.

    Returns
    -------
    plt.Figure
    """
    n = len(zip_codes)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(12 * ncols, 5 * nrows),
        squeeze=False,
    )

    for i, z in enumerate(zip_codes):
        ax = axes[i // ncols][i % ncols]
        plot_forecast(z, long_df, forecasts[z], history_months=history_months, ax=ax)

    for j in range(len(zip_codes), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    return fig


def decompose(
    long_df: pd.DataFrame,
    zip_code: str,
    model: str = "additive",
) -> plt.Figure:
    """Seasonal decomposition — trend + seasonal + residual.

    Parameters
    ----------
    long_df : pd.DataFrame
        Tidy Zillow data.
    zip_code : str
        ZIP to decompose.
    model : {"additive", "multiplicative"}
        Additive: value = trend + seasonal + residual.
        Multiplicative: value = trend × seasonal × residual.
        For a price series that grows over time, additive works well once
        the long-run growth is modest; use multiplicative if the seasonal
        swings grow proportionally with price level.

    Returns
    -------
    plt.Figure
    """
    zip_code = str(zip_code).zfill(5)
    series = (
        long_df[long_df["zip"] == zip_code]
        .set_index("date")["zhvi"]
        .sort_index()
    )
    if series.empty:
        raise ValueError(f"ZIP {zip_code} not found in dataset.")

    series.index = series.index.to_timestamp()
    result = seasonal_decompose(series, model=model, period=12)
    fig = result.plot()
    fig.suptitle(f"ZIP {zip_code} — {model.capitalize()} Decomposition", y=1.02)
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    return fig


# ============================================================================
# 5. EXPORT
# ============================================================================

def save_forecasts(
    forecasts: Dict[str, pd.DataFrame],
    name: str = "zhvi_forecasts",
) -> Path:
    """Concatenate all ZIP forecasts and save to data/outputs/ as a CSV.

    Each row is one (ZIP, future month) pair. The flat CSV is easy to import
    into Power BI, Tableau, or Excel for Phase 7 visualisations.

    Returns
    -------
    Path
        Full path to the saved file.
    """
    combined = pd.concat(
        [df.assign(zip=z) for z, df in forecasts.items()],
        ignore_index=True,
    )[["zip", "date", "forecast", "lower_95", "upper_95"]]
    combined["date"] = combined["date"].dt.to_timestamp()

    out_path = OUTPUTS_DIR / f"{name}.csv"
    combined.to_csv(out_path, index=False)
    print(f"Saved {len(combined):,} rows → {out_path}")
    return out_path
