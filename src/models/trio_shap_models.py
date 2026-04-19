# run_all_models.py - prosty skrypt do uruchamiania wszystkich modeli
"""
Run all SHAP-10 models sequentially and collect results.
"""

import logging
import time
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from lgbm_shap_10 import LGBM_SHAP_10
from xgb_model import XGBoostModel
from catboost_model import CatBoostModel


def run_all_models():
    """Run LGBM, XGBoost, and CatBoost sequentially."""
    logging.basicConfig(level=logging.INFO)

    total_start = time.time()

    # 1. LGBM
    print("\n" + "=" * 80)
    print("RUNNING LGBM SHAP-10")
    print("=" * 80)
    lgbm_start = time.time()
    lgbm = LGBM_SHAP_10()
    lgbm_results = lgbm.train_all_horizons()
    lgbm.generate_final_submission(lgbm_results)
    print(f"LGBM completed in {time.time() - lgbm_start:.2f}s")

    # 2. XGBoost
    print("\n" + "=" * 80)
    print("RUNNING XGBoost SHAP-10")
    print("=" * 80)
    xgb_start = time.time()
    xgb = XGBoostModel()
    xgb_results = xgb.train_all_horizons()
    xgb.generate_final_submission(xgb_results)
    print(f"XGBoost completed in {time.time() - xgb_start:.2f}s")

    # 3. CatBoost
    print("\n" + "=" * 80)
    print("RUNNING CatBoost SHAP-10")
    print("=" * 80)
    cat_start = time.time()
    cat = CatBoostModel()
    cat_results = cat.train_all_horizons()
    cat.generate_final_submission(cat_results)
    print(f"CatBoost completed in {time.time() - cat_start:.2f}s")

    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"ALL MODELS COMPLETED in {total_time:.2f}s ({total_time / 60:.2f}m)")
    print("=" * 80)


if __name__ == "__main__":
    run_all_models()