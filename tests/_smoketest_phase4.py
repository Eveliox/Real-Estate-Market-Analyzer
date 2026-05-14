"""Phase 4 smoke test -- full pipeline through training, tuning, save/load."""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_ames, load_zillow_zhvi
from src.features import prepare_modeling_data
from src import models as M


def main() -> int:
    X, y = prepare_modeling_data(load_ames(), zhvi=load_zillow_zhvi())
    X_train, X_test, y_train, y_test = M.split(X, y, test_size=0.20)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # 1. Compare three families with CV.
    print("\n=== 5-fold CV comparison (training set) ===")
    cv_comp = M.compare_cv(
        {
            "ridge":         M.build_ridge_pipeline(),
            "random_forest": M.build_random_forest_pipeline(),
            "xgboost":       M.build_xgboost_pipeline(),
        },
        X_train, y_train, cv=5,
    )
    print(cv_comp.round(4).to_string(index=False))

    # 2. Tune XGBoost.
    print("\n=== Tuning XGBoost (RandomizedSearch, n_iter=20, cv=5) ===")
    best, best_params, cv_results = M.tune_xgboost(X_train, y_train, n_iter=20, cv=5)
    print("best_params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"best_cv_rmse: {cv_results.iloc[0]['mean_rmse']:.4f}")

    # 3. Test-set evaluation.
    print("\n=== Final test-set evaluation (tuned XGBoost) ===")
    final = M.evaluate(best, X_test, y_test, target_is_log=True)
    for k, v in final.items():
        # Only RMSE and MAE are in dollars; R^2 is always unitless.
        if k.startswith("dollar_") and not k.endswith("_r2"):
            print(f"  {k:14s}: ${v:,.0f}")
        else:
            print(f"  {k:14s}: {v:.4f}")

    # 4. Save and round-trip.
    path = M.save_model(best, name="xgb_ames_tuned")
    print(f"\nSaved to: {path}")
    loaded = M.load_model("xgb_ames_tuned")
    assert np.allclose(loaded.predict(X_test), best.predict(X_test)), "round-trip mismatch!"
    print("Save/load round-trip: OK")

    print("\nPhase 4 smoke test: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
