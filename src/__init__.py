"""
Time Series Forecasting Ablation Study

A comprehensive framework for multi-horizon time series forecasting
with SHAP-based feature selection and ensemble methods.

This package implements the methodology for the NCI PGDAI_SEP25
academic project on hedge fund data analytics.
"""

__version__ = "1.0.0"
__author__ = "Tomasz Biel"
__email__ = "x25113186@student.ncirl.ie"

from .data_loader import DataLoader
from .preprocessor import DataPreprocessor
# from .feature_selector import FeatureSelector
# from .validator import DataValidator
# from .ensemble import EnsembleManager
# from .evaluator import ModelEvaluator
# from .utils import save_results, load_config

__all__ = [
    "DataLoader",
    "DataPreprocessor", 
    # "FeatureSelector",
    # "DataValidator",
    # "EnsembleManager",
    # "ModelEvaluator",
    # "save_results",
    # "load_config"
]
