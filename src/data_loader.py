"""
data_loader.py  —  Phase 1: Data Acquisition
============================================

WHAT THIS FILE DOES
-------------------
Downloads the two public datasets this project depends on, saves them to
`data/raw/`, and exposes small helper functions to load them as pandas
DataFrames later in the pipeline.

The two datasets are intentionally complementary:

  1. **Ames Housing** (Dean De Cock, 2011)
        -> Used for the REGRESSION pipeline (Phases 2-4, 6).
        -> 2,930 individual home sales in Ames, Iowa (2006-2010)
        -> ~80 descriptive features per home

  2. **Zillow ZHVI** (Zillow Home Value Index, ZIP-level, monthly)
        -> Used for the TIME-SERIES + GEO pipeline (Phases 5, 7).
        -> Monthly typical-home-value per US ZIP from 2000 to today.

We download them with the `requests` library, stream to disk (so very large
files don't blow up your RAM), and show a progress bar via `tqdm`.

WHY HAVE A DATA LOADER MODULE AT ALL?
-------------------------------------
"Just call pd.read_csv(...)!"  ← That works once. The reason we wrap it in a
module:
  * **Reproducibility** - anyone can rebuild data/raw/ with one command.
  * **Caching**         - we don't re-download a 30MB file every notebook run.
  * **Single source of truth** - if a URL changes, we fix it here, ONCE.
  * **Clear errors**    - if a URL is dead we print recovery instructions
                          instead of silently failing.

HOW TO RUN
----------
From the project root, with your venv activated:

    python -m src.data_loader

That populates `data/raw/` and prints a short summary of each file.
"""

from __future__ import annotations

# --- Standard library imports ----------------------------------------------
# `pathlib.Path` is the modern, OS-aware way to handle file paths in Python.
# It works on Windows ("\") and POSIX ("/") without you needing to think.
from pathlib import Path
from typing import Optional
import sys

# --- Third-party imports ---------------------------------------------------
# These come from requirements.txt. If any import fails, you forgot to
# activate the venv or run `pip install -r requirements.txt`.
import requests          # HTTP client: downloads bytes from a URL
import pandas as pd      # DataFrames: tabular data with named columns
from tqdm import tqdm    # Progress bar for the download loop


# ============================================================================
# Configuration
# ----------------------------------------------------------------------------
# We keep configuration at the top of the file so a teammate (or future-you)
# can scan one block to understand "where does this data come from?".
# ============================================================================

# Path to data/raw/ relative to this file. The "..parents[1]" trick walks up:
#   __file__              = .../src/data_loader.py
#   Path(__file__).parent = .../src/
#   .parents[1]           = .../              (the project root)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"

# ---- Ames Housing -----------------------------------------------------------
# Dean De Cock published this dataset alongside his 2011 paper in the
# Journal of Statistics Education. The URL has been stable since.
# The file is tab-separated, ~1 MB, 2,930 rows x ~82 columns.
AMES_URL: str = "http://jse.amstat.org/v19n3/decock/AmesHousing.txt"
AMES_DOC_URL: str = "http://jse.amstat.org/v19n3/decock/DataDocumentation.txt"
AMES_FILENAME: str = "ames_housing.tsv"
AMES_DOC_FILENAME: str = "ames_data_dictionary.txt"

# ---- Zillow ZHVI ------------------------------------------------------------
# ZHVI = Zillow Home Value Index. The specific file below is:
#   "ZIP-level, smoothed, seasonally-adjusted, all-homes (SFR+condo),
#    middle tier (33rd-67th percentile of home values)".
#
# It is published openly at zillow.com/research/data/. The CSV is "wide":
# one row per ZIP, one column per month, with columns like "2024-01-31".
# Roughly 30 MB and ~30,000 rows.
ZILLOW_ZHVI_URL: str = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)
ZILLOW_FILENAME: str = "zillow_zhvi_zip_monthly.csv"


# ============================================================================
# Core download helper
# ============================================================================

def _download_file(
    url: str,
    dest: Path,
    *,
    force: bool = False,
    timeout_seconds: int = 60,
) -> Path:
    """Stream-download `url` to `dest`. Skip the download if `dest` exists.

    Parameters
    ----------
    url : str
        Full HTTP(S) URL of the file to fetch.
    dest : Path
        Where to save it locally.
    force : bool, default False
        If True, re-download even if the file already exists. Useful when you
        suspect a partial download or want a fresh copy.
    timeout_seconds : int, default 60
        Seconds to wait for the initial HTTP response before giving up.

    Returns
    -------
    Path
        Same as `dest`, returned so callers can chain operations.

    Notes for learners
    ------------------
    `requests.get(..., stream=True)` does NOT load the whole response into
    RAM at once. Instead it gives us a generator we can iterate over in
    chunks. That matters for files larger than your free memory.

    `tqdm` wraps the chunk loop to draw a progress bar without us having to
    track percentages manually.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  [cache] {dest.name} already present ({size_mb:.2f} MB) — skipping download.")
        return dest

    print(f"  [GET ] {url}")
    try:
        # `stream=True` -> headers only first, body in chunks below
        response = requests.get(url, stream=True, timeout=timeout_seconds)
        response.raise_for_status()  # raises if HTTP status is 4xx or 5xx
    except requests.RequestException as e:
        # We catch *any* requests error here (DNS, timeout, 404, etc.) and
        # convert it into a friendly RuntimeError with recovery hints.
        # The original exception is preserved as `__cause__` via `from e`.
        raise RuntimeError(
            f"Failed to download {url}\n"
            f"  Reason: {e}\n"
            f"  Fix: check your internet connection, or download manually\n"
            f"       and place the file at: {dest}"
        ) from e

    # Content-Length is the file size in bytes from the response headers.
    # It can be missing (some servers don't send it), in which case tqdm
    # just shows a spinner with no ETA.
    total_bytes: Optional[int] = (
        int(response.headers["Content-Length"])
        if "Content-Length" in response.headers
        else None
    )

    chunk_size = 1024 * 64  # 64 KB chunks — a good speed/overhead tradeoff
    with open(dest, "wb") as f, tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"  [save] {dest.name}",
        leave=False,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive new-chunks (rare)
                f.write(chunk)
                pbar.update(len(chunk))

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  [done] {dest.name} saved ({size_mb:.2f} MB).")
    return dest


# ============================================================================
# Public functions: download + load each dataset
# ============================================================================

def download_ames(*, force: bool = False) -> Path:
    """Download the Ames Housing dataset and its data dictionary.

    Returns the path to the housing data file (not the dictionary).
    The dictionary is saved alongside it so future-you can grep it.
    """
    print("[Ames] Downloading housing data...")
    data_path = _download_file(AMES_URL, RAW_DIR / AMES_FILENAME, force=force)

    # The data dictionary is small and human-readable - keep it next to the
    # data so anyone exploring data/raw/ can immediately understand columns.
    try:
        _download_file(AMES_DOC_URL, RAW_DIR / AMES_DOC_FILENAME, force=force)
    except RuntimeError as e:
        # Failing to grab the dictionary is annoying but not fatal — we can
        # still train models without it. Warn and continue.
        print(f"  [warn] could not fetch data dictionary: {e}")

    return data_path


def download_zillow_zhvi(*, force: bool = False) -> Path:
    """Download Zillow's ZIP-level ZHVI monthly time series."""
    print("[Zillow] Downloading ZHVI ZIP-level monthly series...")
    return _download_file(ZILLOW_ZHVI_URL, RAW_DIR / ZILLOW_FILENAME, force=force)


def load_ames(path: Optional[Path] = None) -> pd.DataFrame:
    """Load Ames Housing into a DataFrame.

    The file is **tab**-separated, not comma-separated, hence `sep="\\t"`.
    A common beginner pitfall: defaulting to read_csv with no separator on
    a TSV gives you one column containing the whole row.
    """
    path = path or (RAW_DIR / AMES_FILENAME)
    if not path.exists():
        raise FileNotFoundError(
            f"Ames file not found at {path}. Run `python -m src.data_loader` first."
        )
    # `low_memory=False` tells pandas: read the whole file before guessing
    # column dtypes. This avoids spurious mixed-type warnings on the Ames
    # file, which has many columns where the first chunk looks numeric but
    # later rows contain strings (e.g., "NA" markers).
    return pd.read_csv(path, sep="\t", low_memory=False)


def load_zillow_zhvi(path: Optional[Path] = None) -> pd.DataFrame:
    """Load Zillow ZHVI ZIP-level monthly series.

    The file is "wide": one row per ZIP, one column per month-end date.
    We keep it wide on load - Phase 5 reshapes it to long format ("tidy")
    when we get to forecasting.
    """
    path = path or (RAW_DIR / ZILLOW_FILENAME)
    if not path.exists():
        raise FileNotFoundError(
            f"Zillow file not found at {path}. Run `python -m src.data_loader` first."
        )
    return pd.read_csv(path)


# ============================================================================
# Quick sanity printout
# ============================================================================

def _summarize(name: str, df: pd.DataFrame) -> None:
    """Print a compact one-screen summary so we can eyeball each dataset.

    Why bother? After downloading, you want a 5-second confidence check
    that the file you got is the file you expected. A glance at shape,
    head, and dtypes catches 90% of "wrong file" or "encoding broke"
    problems before they pollute the rest of the pipeline.
    """
    print(f"\n=== {name} ===")
    print(f"shape: {df.shape[0]:,} rows  x  {df.shape[1]:,} columns")
    print(f"memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print("first 5 columns:")
    print(df.iloc[:5, :5].to_string(index=False))
    print(f"dtype counts: {df.dtypes.value_counts().to_dict()}")


# ============================================================================
# Script entrypoint
# ----------------------------------------------------------------------------
# Running this file as a script (e.g., `python -m src.data_loader`) triggers
# downloads for both datasets and prints a summary of each.
# ============================================================================

def main(force: bool = False) -> int:
    """Download both datasets and print a summary. Returns POSIX exit code."""
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data dir: {RAW_DIR}\n")

    try:
        ames_path = download_ames(force=force)
        zhvi_path = download_zillow_zhvi(force=force)
    except RuntimeError as e:
        # Already a human-friendly message from _download_file.
        print(f"\nERROR: {e}", file=sys.stderr)
        print(
            "\nManual download fallback:\n"
            f"  Ames:   {AMES_URL}\n"
            f"          -> save as {RAW_DIR / AMES_FILENAME}\n"
            f"  Zillow: visit https://www.zillow.com/research/data/ ,\n"
            f"          choose 'ZHVI All Homes (SFR, Condo/Co-op) Time Series,\n"
            f"          Smoothed, Seasonally Adjusted' at the ZIP geography,\n"
            f"          -> save as {RAW_DIR / ZILLOW_FILENAME}",
            file=sys.stderr,
        )
        return 1

    # Re-open the freshly-downloaded files to verify they parse cleanly.
    ames_df = load_ames(ames_path)
    zhvi_df = load_zillow_zhvi(zhvi_path)

    _summarize("Ames Housing", ames_df)
    _summarize("Zillow ZHVI (ZIP, monthly)", zhvi_df)

    print(
        "\nDone. Next step: open `notebooks/01_data_exploration.ipynb`"
        "\n   and walk through it cell-by-cell."
    )
    return 0


if __name__ == "__main__":
    # `--force` flag re-downloads even if cached. Useful when you suspect a
    # truncated file. We parse it manually to avoid a heavyweight argparse
    # dependency for one boolean flag.
    force_flag = "--force" in sys.argv
    sys.exit(main(force=force_flag))
