"""Phase 2 smoke test — verifies preprocessing + EDA work end-to-end on real data.

Not part of the educational pipeline. Safe to delete after Phase 2 ships.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_ames
from src.preprocessing import clean_ames
from src import eda


def main() -> int:
    raw = load_ames()
    df = clean_ames(raw)

    print(f"Cleaned: {df.shape[0]:,} x {df.shape[1]:,}")
    print(f"NaN cells: {df.isna().sum().sum()}")
    print(f"Non-numeric cols left: {df.select_dtypes(exclude='number').shape[1]}")

    out_path = PROJECT_ROOT / "data" / "processed" / "ames_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    print()
    print("=== Top 8 features by |corr| with saleprice ===")
    print(eda.top_correlations(df, target="saleprice", n=8).round(3).to_string())

    print()
    print("=== Numeric summary (top 5 by |skew|) ===")
    cols = ["mean", "median", "std", "skew"]
    print(eda.numeric_summary(df).head(5)[cols].round(2).to_string())

    print("\nPhase 2 smoke test: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
