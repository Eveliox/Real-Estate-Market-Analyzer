"""Phase 3 smoke test -- runs the full pipeline through feature engineering."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_ames, load_zillow_zhvi
from src.preprocessing import clean_ames
from src.features import engineer_features
from src import eda


def main() -> int:
    raw = load_ames()
    zhvi = load_zillow_zhvi()

    clean = clean_ames(raw)
    df = engineer_features(clean, zhvi=zhvi)

    print(f"Engineered: {df.shape[0]:,} x {df.shape[1]:,}")
    print(f"NaN cells:  {df.isna().sum().sum()}")

    new_cols = sorted(set(df.columns) - set(clean.columns))
    print(f"Added {len(new_cols)} columns:")
    for c in new_cols:
        print(f"  - {c}")

    out_path = PROJECT_ROOT / "data" / "processed" / "ames_engineered.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Top correlations including engineered features
    print("\n=== Top 12 by |corr| with saleprice (engineered df) ===")
    print(eda.top_correlations(
        df.drop(columns=["log_saleprice"], errors="ignore"),
        target="saleprice", n=12,
    ).round(3).to_string())

    # Top correlations against log_saleprice
    print("\n=== Top 12 by |corr| with log_saleprice ===")
    print(eda.top_correlations(
        df.drop(columns=["saleprice"], errors="ignore"),
        target="log_saleprice", n=12,
    ).round(3).to_string())

    print("\nPhase 3 smoke test: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
