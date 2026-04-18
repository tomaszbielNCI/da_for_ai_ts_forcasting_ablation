"""
Machine Learning Models for Time Series Forecasting

This module contains implementations of various models used in the ablation study:
- LightGBM variants (baseline, SHAP-engineered, target encoding)
- XGBoost and CatBoost baselines
- Bayesian Neural Networks (pre/post SHAP)
- Mamba model as negative control
"""

from .baseline_lgbm import BaselineLGBM
from .shap_analyzer import SHAPAnalyzer
from .lgbm_shap_10 import LGBM_SHAP_10
from .lgbm_shap_20 import LGBM_SHAP_20
from .lgbm_all_plus_shap import LGBM_All_Plus_SHAP
from .xgb_model import XGBoostModel
from .catboost_model import CatBoostModel
from .trio_shap_models import TrioSHAPModels
from .ensemble_shap import EnsembleSHAP
# from .lgbm_base import LightGBMBase
# from .lgbm_shap import LightGBMSHAP
# from .xgb import XGBoostModel
# from .catboost import CatBoostModel
# from .bnn_pre import BNNPreSHAP
# from .bnn_post import BNNPostSHAP
# from .mamba import MambaModel

__all__ = [
    "BaselineLGBM",
    "SHAPAnalyzer",
    "LGBM_SHAP_10",
    "LGBM_SHAP_20", 
    "LGBM_All_Plus_SHAP",
    "XGBoostModel",
    "CatBoostModel",
    "TrioSHAPModels",
    "EnsembleSHAP",
    # "LightGBMBase",
    # "LightGBMSHAP", 
    # "XGBoostModel",
    # "CatBoostModel",
    # "BNNPreSHAP",
    # "BNNPostSHAP",
    # "MambaModel"
]
