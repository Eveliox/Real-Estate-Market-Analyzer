"""
features.py  —  Phase 3: Feature Engineering
=============================================

WHAT THIS FILE DOES
-------------------
Takes the cleaned Ames DataFrame (output of `preprocessing.clean_ames`)
and adds new columns that are *more useful to a model than the raw inputs*.

The five families of engineered features we build here:

  1. AGGREGATION    -- collapse correlated columns into one informative total
                       (total_sf, total_bath, total_porch_sf)

  2. TEMPORAL       -- turn raw years into "age at sale" style features
                       (home_age, years_since_remodel, is_remodeled, is_new)

  3. CYCLIC         -- encode month-of-year so December and January are neighbors
                       (mo_sold_sin, mo_sold_cos)

  4. BINARY FLAGS   -- "does this home have an X at all?"
                       (has_pool, has_basement, has_2nd_floor, ...)

  5. CROSS-DATASET  -- join Zillow's monthly market index onto each sale
                       (ames_market_index)

  6. TARGET TRANSFORM -- add log1p(saleprice) so we can train on a less-skewed target

WHY ENGINEER FEATURES AT ALL?
-----------------------------
Models can only "see" what you show them. Some examples:

  - `1st_flr_sf`, `2nd_flr_sf`, `total_bsmt_sf` are three correlated columns.
    A linear model treats each independently. The TOTAL square footage is
    a single, more interpretable signal. We engineer it explicitly.

  - `year_built = 1995` is a different "kind of fact" from `home_age = 13
    years at time of sale`. Trees can derive one from the other, but giving
    the model `home_age` directly means it uses one split where it would
    otherwise need two or three.

  - `mo_sold = 12` (December) is closer to `mo_sold = 1` (January) than to
    `mo_sold = 6` (June). The integer encoding hides this. The sin/cos
    encoding restores the cyclic topology.

In real ML projects, a thoughtful set of engineered features usually
beats a fancier algorithm on raw features. This is where domain
knowledge becomes performance.

USAGE
-----
    from src.data_loader import load_ames, load_zillow_zhvi
    from src.preprocessing import clean_ames
    from src.features import engineer_features

    raw = load_ames()
    clean = clean_ames(raw)
    zhvi = load_zillow_zhvi()
    engineered = engineer_features(clean, zhvi=zhvi)
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================================
# 1. AGGREGATION FEATURES
# ----------------------------------------------------------------------------
# Collapse groups of correlated columns into a single, more informative one.
# We do NOT drop the originals -- trees may still find a use for them, and
# linear models in Phase 4 will have feature selection options.
# ============================================================================

def add_total_sf(df: pd.DataFrame) -> pd.DataFrame:
    """Add `total_sf` = sum of first-floor, second-floor, and basement square feet.

    This single column usually correlates with SalePrice *more strongly than
    any of its three parts*. The reason: the total captures "how big is this
    home?" without the parts cancelling each other (a home with a big
    basement and small upstairs is the same size as the reverse).
    """
    out = df.copy()
    parts = ["1st_flr_sf", "2nd_flr_sf", "total_bsmt_sf"]
    available = [c for c in parts if c in out.columns]
    if available:
        out["total_sf"] = out[available].sum(axis=1)
    return out


def add_total_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    """Add `total_bath` = full_bath + 0.5*half_bath, summed across main and basement.

    Half-baths count as 0.5 by real-estate convention. This collapses four
    correlated columns into one.
    """
    out = df.copy()

    def col(name: str) -> pd.Series:
        return out[name] if name in out.columns else pd.Series(0, index=out.index)

    out["total_bath"] = (
        col("full_bath") + 0.5 * col("half_bath")
        + col("bsmt_full_bath") + 0.5 * col("bsmt_half_bath")
    )
    return out


def add_total_porch_sf(df: pd.DataFrame) -> pd.DataFrame:
    """Add `total_porch_sf` = sum of all outdoor square footage columns."""
    out = df.copy()
    parts = ["open_porch_sf", "enclosed_porch", "3ssn_porch", "screen_porch", "wood_deck_sf"]
    available = [c for c in parts if c in out.columns]
    if available:
        out["total_porch_sf"] = out[available].sum(axis=1)
    return out


# ============================================================================
# 2. TEMPORAL FEATURES
# ----------------------------------------------------------------------------
# `year_built = 1995` is fine, but `home_age = 13 years at sale` is more
# directly useful: "old homes" vs "new homes" is a more model-friendly
# framing than "homes built in 1965 vs 1995".
# ============================================================================

def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add age-related features derived from year columns.

    New columns:
      home_age            -- yr_sold - year_built
      years_since_remodel -- yr_sold - year_remod_add
      is_remodeled        -- 1 if the home was ever remodeled, else 0
      is_new              -- 1 if built in the same year it sold, else 0

    Note: clamping at zero handles the rare data-entry quirk where a sale
    year is *before* the build year (which would yield a negative age).
    """
    out = df.copy()
    if "yr_sold" in out.columns and "year_built" in out.columns:
        out["home_age"] = (out["yr_sold"] - out["year_built"]).clip(lower=0)
        out["is_new"] = (out["yr_sold"] == out["year_built"]).astype("int8")
    if "yr_sold" in out.columns and "year_remod_add" in out.columns:
        out["years_since_remodel"] = (out["yr_sold"] - out["year_remod_add"]).clip(lower=0)
    if "year_built" in out.columns and "year_remod_add" in out.columns:
        # `year_remod_add` equals `year_built` when the home was never
        # remodeled (per the Ames data dictionary). So:
        out["is_remodeled"] = (out["year_remod_add"] != out["year_built"]).astype("int8")
    return out


# ============================================================================
# 3. CYCLIC ENCODING
# ----------------------------------------------------------------------------
# Month is a cyclic variable: December (12) and January (1) are adjacent.
# Encoding it as an integer 1-12 tells the model December is 11 units
# "further" from January than from November -- which is wrong.
#
# The fix is the sin/cos trick. We map each month to a (sin, cos) pair on
# the unit circle:
#         month_rad = 2 * pi * month / 12
#         sin_feature = sin(month_rad)
#         cos_feature = cos(month_rad)
#
# Now December (cos=1, sin~0) and January (cos~1, sin~0) are close. June
# (cos=-1, sin~0) is far from both. The model "sees" the cyclic topology.
# ============================================================================

def add_cyclic_month(df: pd.DataFrame, col: str = "mo_sold") -> pd.DataFrame:
    """Add `<col>_sin` and `<col>_cos` so the model sees month cyclically."""
    out = df.copy()
    if col not in out.columns:
        return out
    month_rad = 2 * np.pi * out[col] / 12.0
    out[f"{col}_sin"] = np.sin(month_rad)
    out[f"{col}_cos"] = np.cos(month_rad)
    return out


# ============================================================================
# 4. BINARY "HAS-IT" FLAGS
# ----------------------------------------------------------------------------
# Sometimes "the home has a pool at all" is more predictive than "the pool
# quality is 4 (Good)". A model that has both can decide which to use.
# ============================================================================

# (column, condition that means "has it"). We use a callable for flexibility.
BINARY_FLAG_SOURCES: list[tuple[str, str]] = [
    ("has_pool",         "pool_area"),       # >0
    ("has_2nd_floor",    "2nd_flr_sf"),      # >0
    ("has_basement",     "total_bsmt_sf"),   # >0
    ("has_garage",       "garage_area"),     # >0
    ("has_fireplace",    "fireplaces"),      # >0
    ("has_porch",        "total_porch_sf"),  # >0 (depends on add_total_porch_sf having run)
]


def add_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add 0/1 indicator columns for whether the home has each feature at all."""
    out = df.copy()
    for new_col, source_col in BINARY_FLAG_SOURCES:
        if source_col in out.columns:
            out[new_col] = (out[source_col] > 0).astype("int8")
    return out


# ============================================================================
# 5. TARGET TRANSFORM
# ----------------------------------------------------------------------------
# `saleprice` is right-skewed: a few mansions stretch the upper tail.
# Linear regression assumes the residuals are roughly normal; trees don't
# care about distributional shape but DO benefit from a target compressed
# into a smaller numeric range. So in either case, training on
# `log1p(saleprice)` is the standard move.
#
# At prediction time you `np.expm1(...)` (= exp(x) - 1) to undo the log.
# ============================================================================

def add_log_target(df: pd.DataFrame, target: str = "saleprice") -> pd.DataFrame:
    """Add `log_<target>` = log1p(<target>). Keep the original column for reference."""
    out = df.copy()
    if target in out.columns:
        out[f"log_{target}"] = np.log1p(out[target])
    return out


# ============================================================================
# 6. CROSS-DATASET FEATURE -- the Zillow market index
# ----------------------------------------------------------------------------
# Ames is per-home but lacks "market conditions". Zillow ZHVI has those
# conditions at ZIP * month granularity. We aggregate ZHVI for the few
# Ames ZIP codes into a single monthly "Ames market index", then attach
# the right month's value to each home sale.
#
# This is a *huge* signal: the 2007-2010 Ames sales straddle the great
# recession. A home sold for $200k in 2007 was the same home sold for
# $180k in 2009; without a market feature the model has to learn the
# whole macroeconomic crash from the date column alone.
# ============================================================================

# ZIP codes that cover the city of Ames, IA. We average ZHVI across them.
AMES_ZIPS: tuple[int, ...] = (50010, 50011, 50012, 50013, 50014)

# Match a Zillow value column whose name looks like "YYYY-MM-DD".
_DATE_COL_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def compute_ames_market_index(zhvi: pd.DataFrame) -> pd.Series:
    """Return a Series indexed by (year, month) of the average Ames ZHVI.

    Implementation details worth knowing:
      * We filter to the handful of Ames ZIPs.
      * We average across them to get one number per month-end.
      * We pivot from wide (one col per date) to long (one row per month).
      * We index by (year, month) so the per-row Ames join is a fast lookup.
    """
    # 1. Filter to Ames-area ZIPs. RegionName in Zillow is an integer ZIP code.
    ames = zhvi[zhvi["RegionName"].isin(AMES_ZIPS)].copy()
    if ames.empty:
        raise ValueError(
            "No Ames ZIPs found in Zillow ZHVI. "
            f"Looked for {AMES_ZIPS}. Re-download with `python -m src.data_loader`."
        )

    # 2. Pick out the date-named columns and average across the filtered rows.
    date_cols = [c for c in ames.columns if _DATE_COL_PATTERN.match(c)]
    monthly_avg = ames[date_cols].mean(axis=0)
    monthly_avg.index = pd.to_datetime(monthly_avg.index)

    # 3. Reindex by (year, month) for downstream join speed.
    monthly_avg.index = pd.MultiIndex.from_arrays(
        [monthly_avg.index.year, monthly_avg.index.month],
        names=["year", "month"],
    )
    return monthly_avg.rename("ames_market_index")


def add_market_index(df: pd.DataFrame, zhvi: pd.DataFrame) -> pd.DataFrame:
    """Attach the Ames market index for each home's sale year+month.

    Requires columns `yr_sold` and `mo_sold` in the input DataFrame.
    Rows whose sale month is outside Zillow's range get NaN, which we
    then fill with the median of the index (an honest "average market"
    fallback).
    """
    if "yr_sold" not in df.columns or "mo_sold" not in df.columns:
        # Nothing to join on -- just return df unchanged with a note.
        print("[add_market_index] yr_sold or mo_sold missing -- skipping join.")
        return df

    index_series = compute_ames_market_index(zhvi)

    out = df.copy()
    keys = list(zip(out["yr_sold"].astype(int), out["mo_sold"].astype(int)))
    # `.reindex()` with the list of (year, month) tuples returns NaN for misses.
    market = index_series.reindex(keys).reset_index(drop=True)
    market.index = out.index
    median_value = float(index_series.median())
    out["ames_market_index"] = market.fillna(median_value)
    return out


# ============================================================================
# 7. ORCHESTRATOR
# ============================================================================

def engineer_features(
    df: pd.DataFrame,
    *,
    zhvi: Optional[pd.DataFrame] = None,
    add_log_y: bool = True,
) -> pd.DataFrame:
    """Run the full Phase 3 feature engineering pipeline on a cleaned df.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned Ames DataFrame (output of `preprocessing.clean_ames`).
    zhvi : pd.DataFrame, optional
        Loaded Zillow ZHVI DataFrame. If None, the market index feature is skipped.
    add_log_y : bool, default True
        If True, adds `log_saleprice` for log-target training.
    """
    df = add_total_sf(df)
    df = add_total_bathrooms(df)
    df = add_total_porch_sf(df)
    df = add_age_features(df)
    df = add_cyclic_month(df)
    df = add_binary_flags(df)        # must come AFTER add_total_porch_sf for has_porch
    if zhvi is not None:
        df = add_market_index(df, zhvi)
    if add_log_y:
        df = add_log_target(df)
    return df


# ============================================================================
# 8. Convenience: a single function for "give me model-ready data"
# ============================================================================

def prepare_modeling_data(
    raw_ames: pd.DataFrame,
    zhvi: Optional[pd.DataFrame] = None,
    *,
    target: str = "saleprice",
    use_log_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Clean + engineer + split into (X, y) in one call.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (does NOT include the target or any log-target column).
    y : pd.Series
        Target vector (log-transformed if use_log_target=True).
    """
    from src.preprocessing import clean_ames  # local import to keep cycle clean

    df = clean_ames(raw_ames)
    df = engineer_features(df, zhvi=zhvi, add_log_y=use_log_target)

    drop_cols = [target]
    log_col = f"log_{target}"
    if log_col in df.columns:
        drop_cols.append(log_col)

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[log_col] if use_log_target and log_col in df.columns else df[target]
    return X, y


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    from src.data_loader import load_ames, load_zillow_zhvi
    from src.preprocessing import clean_ames

    raw = load_ames()
    zhvi = load_zillow_zhvi()
    clean = clean_ames(raw)

    engineered = engineer_features(clean, zhvi=zhvi)
    new_cols = sorted(set(engineered.columns) - set(clean.columns))
    print(f"Clean:      {clean.shape[0]:,} x {clean.shape[1]:,}")
    print(f"Engineered: {engineered.shape[0]:,} x {engineered.shape[1]:,}")
    print(f"New columns ({len(new_cols)}): {new_cols}")

    # Sanity check the market index
    print()
    print("Ames market index range:")
    print(engineered["ames_market_index"].describe().round(0).to_string())
