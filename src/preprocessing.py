"""
preprocessing.py  —  Phase 2: Data Cleaning
===========================================

WHAT THIS FILE DOES
-------------------
Takes the raw Ames Housing DataFrame (as returned by `data_loader.load_ames`)
and turns it into a "model-ready" DataFrame:

  * column names standardized (lowercase, underscores, no spaces)
  * identifier columns dropped (so the model can't cheat on PID)
  * structural NAs (e.g. PoolQC=NaN means "no pool") filled with "None"
  * true missing numerics imputed with the median
  * true missing categoricals imputed with the mode
  * ordinal categoricals (quality scales) mapped to integers preserving rank
  * nominal categoricals one-hot encoded
  * known outliers removed (the Ames paper warns about a few homes)

The cleaning is exposed both as small focused functions (good for *learning*
and *inspecting* each step) and a single `clean_ames(df)` orchestrator
(good for *using* in notebooks and downstream code).

WHY FUNCTIONS INSTEAD OF AN sklearn TRANSFORMER?
------------------------------------------------
sklearn's `Pipeline` + `ColumnTransformer` is the production-correct way
to package preprocessing. It auto-prevents data leakage, supports
serialization, and chains cleanly with a model. **We will use it in
Phase 4** when we start modeling.

For Phase 2 we keep things as plain functions because they are easier to
read line-by-line, easier to debug, and force you to *see* every
transformation explicitly. Once you understand what each step does, the
sklearn version in Phase 4 will read as "the same thing, wrapped".

HOW TO USE
----------
    from src.data_loader import load_ames
    from src.preprocessing import clean_ames

    raw = load_ames()
    clean = clean_ames(raw)
    print(clean.shape, clean.columns[:5].tolist())
"""

from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd


# ============================================================================
# 1. Column-name standardization
# ----------------------------------------------------------------------------
# The raw Ames file uses names like "MS SubClass" and "Bsmt Qual". Spaces in
# column names are technically legal but they:
#   * break dot-access syntax (df.MS SubClass is a syntax error)
#   * make code visually noisy ("MS SubClass" vs ms_subclass)
#   * silently break joins if one source has spaces and another doesn't
# Standardizing to snake_case once, up front, is a one-line tax that saves
# you from a thousand papercuts later.
# ============================================================================

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with snake_case column names.

    "MS SubClass" -> "ms_subclass"
    "Lot Frontage" -> "lot_frontage"
    "SalePrice" -> "saleprice"

    We `.copy()` so the caller's original frame isn't mutated. This is a
    pandas habit worth keeping: pure functions are easier to reason about.
    """
    out = df.copy()
    out.columns = (
        out.columns
        .str.strip()                 # remove leading/trailing whitespace
        .str.replace(" ", "_")       # spaces -> underscores
        .str.replace("/", "_")       # slashes (e.g. "Year Remod/Add") -> underscores
        .str.lower()                 # everything lowercase
    )
    return out


# ============================================================================
# 2. Identifier columns -- drop before modeling
# ----------------------------------------------------------------------------
# `pid` is the parcel ID (a tax number). `order` is just the row position in
# the original file. Neither has predictive meaning -- and worse, both are
# *unique* to each home, so any model that sees them will memorize and
# *appear* perfect while having learned nothing useful.
# This is the simplest form of "data leakage". We kill it early.
# ============================================================================

IDENTIFIER_COLS: list[str] = ["pid", "order"]


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that uniquely identify rows. They are not features."""
    return df.drop(columns=[c for c in IDENTIFIER_COLS if c in df.columns])


# ============================================================================
# 3. Structural NAs -- "this feature does not exist for this home"
# ----------------------------------------------------------------------------
# In Ames, many `NA` values are not "we don't know" but "doesn't have one".
# Example: pool_qc = NaN means "the home has no pool" (and therefore has no
# pool quality). The data dictionary lists each such column.
#
# If we imputed these with the mode, we'd be telling the model:
# "this home has a typical-quality pool" -- which is a lie, and erases the
# very real signal "has a pool" vs "no pool".
#
# So we replace these NaN values with the literal string "None", which
# becomes its own category during one-hot encoding.
# ============================================================================

# Columns where NaN means "structurally absent", per the Ames data dictionary.
STRUCTURAL_NA_CATEGORICAL: list[str] = [
    "alley",            # NaN = no alley access
    "bsmt_qual",        # NaN = no basement
    "bsmt_cond",
    "bsmt_exposure",
    "bsmtfin_type_1",
    "bsmtfin_type_2",
    "fireplace_qu",     # NaN = no fireplace
    "garage_type",      # NaN = no garage
    "garage_finish",
    "garage_qual",
    "garage_cond",
    "pool_qc",          # NaN = no pool
    "fence",            # NaN = no fence
    "misc_feature",     # NaN = no misc feature
    "mas_vnr_type",     # NaN = no masonry veneer
]

# Numeric columns where NaN means "doesn't apply" -- fill with 0, NOT median.
# Example: garage_yr_blt is NaN when there is no garage. Imputing the median
# year would suggest a garage was built in ~1980, which is nonsense.
STRUCTURAL_NA_NUMERIC: list[str] = [
    "garage_yr_blt",    # NaN = no garage -> 0 (we'll engineer "has_garage" later)
    "mas_vnr_area",     # NaN = no masonry -> 0
    "bsmtfin_sf_1",     # NaN = no basement -> 0
    "bsmtfin_sf_2",
    "bsmt_unf_sf",
    "total_bsmt_sf",
    "bsmt_full_bath",
    "bsmt_half_bath",
    "garage_cars",
    "garage_area",
]


def fill_structural_nas(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN where it *means* 'doesn't apply', not 'unknown'."""
    out = df.copy()
    for col in STRUCTURAL_NA_CATEGORICAL:
        if col in out.columns:
            out[col] = out[col].fillna("None")
    for col in STRUCTURAL_NA_NUMERIC:
        if col in out.columns:
            out[col] = out[col].fillna(0)
    return out


# ============================================================================
# 4. True missing values -- impute with the column's central tendency
# ----------------------------------------------------------------------------
# After step 3, any remaining NaN really is "we don't know". The two
# go-to imputation strategies:
#
#   * Numeric columns      -> fill with the MEDIAN
#       (robust to outliers; the mean would be pulled by a few mansions)
#
#   * Categorical columns  -> fill with the MODE (most common value)
#
# Sophisticated alternatives exist (k-NN imputation, model-based imputation,
# multiple imputation). We use simple imputation here because:
#   1. It is the right baseline. Always start simple.
#   2. The fraction of truly-missing values in Ames is small after step 3.
#   3. Imputation method has a small effect compared to feature engineering.
# ============================================================================

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining NaN: median for numeric, mode for categorical."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include=np.number).columns
    categorical_cols = out.select_dtypes(include="object").columns

    for col in numeric_cols:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())

    for col in categorical_cols:
        if out[col].isna().any():
            # .mode() can return >1 value if there's a tie. Take the first.
            out[col] = out[col].fillna(out[col].mode().iloc[0])

    return out


# ============================================================================
# 5. Outlier removal -- only when domain knowledge justifies it
# ----------------------------------------------------------------------------
# In his 2011 paper, Dean De Cock specifically calls out a handful of homes
# with `gr_liv_area` > 4000 sq ft. Two are partial sales (parents-to-child
# transfers, not market deals) and two are weirdly cheap mansions. The paper
# *recommends* removing them when fitting a regression model on this data.
#
# We do NOT generically remove "statistical" outliers (e.g. anything beyond
# 3 standard deviations). Doing that throws away real, informative tail
# observations -- a $750k home is not noise, it's a $750k home.
# We only remove outliers when the domain expert (De Cock here) has a
# specific story for why they are not representative.
# ============================================================================

def remove_known_outliers(df: pd.DataFrame, target: str = "saleprice") -> pd.DataFrame:
    """Drop the specific Ames outliers flagged by De Cock (2011).

    The rule: keep all homes EXCEPT those with `gr_liv_area > 4000`.
    De Cock explicitly recommends this filter in the dataset paper.
    """
    if "gr_liv_area" not in df.columns:
        return df  # column missing -- nothing to filter on
    mask = df["gr_liv_area"] <= 4000
    n_removed = (~mask).sum()
    if n_removed > 0:
        # We print a small notice so the cleaning is auditable.
        # In production code, prefer logging.info(...) over print(...).
        print(f"[outliers] removed {n_removed} row(s) with gr_liv_area > 4000 (De Cock 2011).")
    return df.loc[mask].copy()


# ============================================================================
# 6. Ordinal encoding -- preserve the ranking
# ----------------------------------------------------------------------------
# An "ordinal" categorical has a meaningful *order*. Kitchen quality goes
# Excellent > Good > Typical > Fair > Poor. The right encoding maps these
# to integers that preserve the order.
#
# We do this manually (rather than using sklearn's OrdinalEncoder) because:
#   1. The mapping is short, and seeing it explicitly is the lesson.
#   2. sklearn's encoder requires you to specify the order anyway -- there
#      is no machine-readable "Ex > Gd > TA" baked into the strings.
#
# After this step, what was a column of strings becomes a column of ints
# from 0 (worst / absent) to 5 (best).
# ============================================================================

# The standard Ames quality scale, shared across many columns.
QUALITY_SCALE: dict[str, int] = {
    "None": 0,   # the sentinel we inserted in step 3
    "Po": 1,     # Poor
    "Fa": 2,     # Fair
    "TA": 3,     # Typical / Average
    "Gd": 4,     # Good
    "Ex": 5,     # Excellent
}

# Columns that follow the QUALITY_SCALE.
QUALITY_COLS: list[str] = [
    "exter_qual", "exter_cond",
    "bsmt_qual", "bsmt_cond",
    "heating_qc",
    "kitchen_qual",
    "fireplace_qu",
    "garage_qual", "garage_cond",
    "pool_qc",
]

# Columns with their own, dataset-specific ordinal scales.
OTHER_ORDINAL_MAPS: dict[str, dict[str, int]] = {
    "bsmt_exposure":  {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "bsmtfin_type_1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "bsmtfin_type_2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "garage_finish":  {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "functional":     {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8},
    # land_slope and lot_shape are technically ordinal too:
    "land_slope":     {"Sev": 1, "Mod": 2, "Gtl": 3},
    "lot_shape":      {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    # paved_drive: N=no, P=partial, Y=yes
    "paved_drive":    {"N": 0, "P": 1, "Y": 2},
    # central_air: N/Y -> 0/1 (binary but logically ordinal)
    "central_air":    {"N": 0, "Y": 1},
    # street and alley: paving level
    "street":         {"Grvl": 0, "Pave": 1},
}


def encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """Map ordinal-categorical columns to ranked integers."""
    out = df.copy()
    for col in QUALITY_COLS:
        if col in out.columns:
            out[col] = out[col].map(QUALITY_SCALE).astype("Int64")
    for col, mapping in OTHER_ORDINAL_MAPS.items():
        if col in out.columns:
            out[col] = out[col].map(mapping).astype("Int64")
    return out


# ============================================================================
# 7. Nominal encoding -- one-hot
# ----------------------------------------------------------------------------
# A "nominal" categorical has NO inherent order. `neighborhood` is the
# classic example: "Crawfor" is not greater or less than "NAmes", they
# are just different neighborhoods.
#
# The standard encoding is *one-hot*: turn one column into many columns,
# each a 0/1 indicator for one category. So a 28-neighborhood column
# becomes 28 binary columns. The model can then learn a coefficient per
# neighborhood independently.
#
# `pd.get_dummies(drop_first=True)` drops one category per column to
# avoid perfect multicollinearity (every column being a linear combo of
# the others). Tree-based models don't care; linear models do.
# ============================================================================

def encode_nominals(df: pd.DataFrame, *, drop_first: bool = True) -> pd.DataFrame:
    """One-hot encode remaining object columns."""
    out = df.copy()
    object_cols = out.select_dtypes(include="object").columns.tolist()
    if not object_cols:
        return out
    return pd.get_dummies(out, columns=object_cols, drop_first=drop_first, dtype="int8")


# ============================================================================
# 8. Orchestrator
# ----------------------------------------------------------------------------
# Glues all the above together in the right order. The order matters:
#   * standardize names FIRST so all later step lookups work
#   * fill structural NAs BEFORE impute_missing (so the imputer doesn't see
#     a "this home has no pool" NaN and impute it as median pool quality)
#   * encode ordinals BEFORE one-hot, so quality columns don't get one-hot'd
#   * outlier removal can happen anywhere AFTER column standardization;
#     putting it early is slightly more efficient (less data to transform).
# ============================================================================

def clean_ames(
    df: pd.DataFrame,
    *,
    drop_outliers: bool = True,
    one_hot: bool = True,
) -> pd.DataFrame:
    """Run the full Ames cleaning pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Ames DataFrame from `data_loader.load_ames()`.
    drop_outliers : bool, default True
        If True, removes the De Cock-flagged outliers.
    one_hot : bool, default True
        If True, one-hot encodes nominal categoricals. Set False if you
        plan to use a model (like CatBoost) that handles raw categoricals.

    Returns
    -------
    pd.DataFrame
        Cleaned and encoded DataFrame, ready for modeling.
    """
    df = standardize_column_names(df)
    df = drop_identifier_columns(df)
    if drop_outliers:
        df = remove_known_outliers(df)
    df = fill_structural_nas(df)
    df = impute_missing(df)
    df = encode_ordinals(df)
    if one_hot:
        df = encode_nominals(df)
    return df


# ============================================================================
# 9. Convenience: split features and target
# ============================================================================

def split_features_target(
    df: pd.DataFrame,
    target: str = "saleprice",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) where X has every column except the target."""
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in DataFrame. Have: {df.columns.tolist()[:10]}...")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


# ============================================================================
# Quick self-test if executed directly
# ============================================================================

if __name__ == "__main__":
    # Smoke test: load -> clean -> print shapes.
    from src.data_loader import load_ames

    raw = load_ames()
    print(f"Raw:    {raw.shape[0]:,} rows x {raw.shape[1]:,} cols, "
          f"{raw.isna().sum().sum():,} NaN cells")

    cleaned = clean_ames(raw)
    print(f"Clean:  {cleaned.shape[0]:,} rows x {cleaned.shape[1]:,} cols, "
          f"{cleaned.isna().sum().sum():,} NaN cells")

    X, y = split_features_target(cleaned)
    print(f"Split:  X={X.shape}, y={y.shape}")
