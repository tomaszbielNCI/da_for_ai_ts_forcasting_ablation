#!/usr/bin/env python3
"""
Trio SHAP Models Pipeline

Pipeline combining LGBM, XGBoost, and CatBoost models with top 10 SHAP features.
Runs all three models sequentially and compares results per horizon.
Based on trio-shap-features.ipynb with modular design.
"""

import polars as pl
import numpy as np
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from .lgbm_shap_10 import LGBM_SHAP_10
from .xgb_model import XGBoostModel
from .catboost_model import CatBoostModel
from ..metrics.evaluation import TimeSeriesMetrics, MetricResults

@dataclass
class TrioModelResults:
    """Container for trio model comparison results"""
    horizon: int
    lgbm_time_validation: Optional[MetricResults]
    lgbm_full_validation: Optional[MetricResults]
    xgb_time_validation: Optional[MetricResults]
    xgb_full_validation: Optional[MetricResults]
    catboost_time_validation: Optional[MetricResults]
    catboost_full_validation: Optional[MetricResults]
    lgbm_time: float
    xgb_time: float
    catboost_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'horizon': self.horizon,
            'lgbm_time_validation': self.lgbm_time_validation.to_dict() if self.lgbm_time_validation else None,
            'lgbm_full_validation': self.lgbm_full_validation.to_dict() if self.lgbm_full_validation else None,
            'xgb_time_validation': self.xgb_time_validation.to_dict() if self.xgb_time_validation else None,
            'xgb_full_validation': self.xgb_full_validation.to_dict() if self.xgb_full_validation else None,
            'catboost_time_validation': self.catboost_time_validation.to_dict() if self.catboost_time_validation else None,
            'catboost_full_validation': self.catboost_full_validation.to_dict() if self.catboost_full_validation else None,
            'training_times': {
                'lgbm': self.lgbm_time,
                'xgb': self.xgb_time,
                'catboost': self.catboost_time
            }
        }

class TrioSHAPModels:
    """
    Pipeline for training and comparing LGBM, XGBoost, and CatBoost models.
    
    Features:
    - Sequential training of all three models
    - Per-horizon comparison
    - Comprehensive metrics comparison
    - Timing analysis
    - Results aggregation and visualization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize trio models pipeline with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Paths
        self.results_dir = Path('results/trio_shap')
        self.comparisons_dir = self.results_dir / 'comparisons'
        self.models_dir = self.results_dir / 'models'
        
        # Create directories
        for dir_path in [self.results_dir, self.comparisons_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Horizons
        self.horizons = self.config.get('horizons', [1, 3, 10, 25])
        
        # Initialize models
        self.lgbm_model = LGBM_SHAP_10(self.config)
        self.xgb_model = XGBoostModel(self.config)
        self.catboost_model = CatBoostModel(self.config)
        
        # Results storage
        self.results = {}
    
    def run_single_model(self, model_name: str, model) -> Dict[int, Dict[str, Any]]:
        """Run a single model and return results."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"RUNNING {model_name.upper()} MODEL")
        self.logger.info(f"{'='*80}")
        
        start_time = time.time()
        model_results = model.train_all_horizons()
        total_time = time.time() - start_time
        
        self.logger.info(f"{model_name.upper()} completed in {total_time:.2f} seconds")
        
        return model_results
    
    def compare_models_per_horizon(self, lgbm_results: Dict, xgb_results: Dict, 
                                  catboost_results: Dict) -> Dict[int, TrioModelResults]:
        """Compare results from all three models per horizon."""
        comparison = {}
        
        for horizon in self.horizons:
            self.logger.info(f"Comparing models for H={horizon}...")
            
            # Extract results
            lgbm_time_res = lgbm_results[horizon]['time_validation']
            lgbm_full_res = lgbm_results[horizon]['full_validation']
            xgb_time_res = xgb_results[horizon]['time_validation']
            xgb_full_res = xgb_results[horizon]['full_validation']
            catboost_time_res = catboost_results[horizon]['time_validation']
            catboost_full_res = catboost_results[horizon]['full_validation']
            
            # Create comparison object
            trio_result = TrioModelResults(
                horizon=horizon,
                lgbm_time_validation=lgbm_time_res.valid_metrics,
                lgbm_full_validation=lgbm_full_res.train_metrics,
                xgb_time_validation=xgb_time_res.valid_metrics,
                xgb_full_validation=xgb_full_res.train_metrics,
                catboost_time_validation=catboost_time_res.valid_metrics,
                catboost_full_validation=catboost_full_res.train_metrics,
                lgbm_time=lgbm_results[horizon]['training_time'],
                xgb_time=xgb_results[horizon]['training_time'],
                catboost_time=catboost_results[horizon]['training_time']
            )
            
            comparison[horizon] = trio_result
        
        return comparison
    
    def print_comparison_summary(self, comparison: Dict[int, TrioModelResults]) -> None:
        """Print detailed comparison summary."""
        print(f"\n{'='*100}")
        print("TRIO MODELS COMPARISON SUMMARY")
        print(f"{'='*100}")
        
        for horizon, result in comparison.items():
            print(f"\nHORIZON {horizon}:")
            print(f"{'-'*80}")
            
            # Time validation comparison
            print(f"Time Validation (Weighted RMSE):")
            print(f"  LGBM:     {result.lgbm_time_validation.weighted_rmse:.6f}")
            print(f"  XGBoost:  {result.xgb_time_validation.weighted_rmse:.6f}")
            print(f"  CatBoost: {result.catboost_time_validation.weighted_rmse:.6f}")
            
            # Full validation comparison
            print(f"Full Validation (Weighted RMSE):")
            print(f"  LGBM:     {result.lgbm_full_validation.weighted_rmse:.6f}")
            print(f"  XGBoost:  {result.xgb_full_validation.weighted_rmse:.6f}")
            print(f"  CatBoost: {result.catboost_full_validation.weighted_rmse:.6f}")
            
            # Training time comparison
            print(f"Training Time (seconds):")
            print(f"  LGBM:     {result.lgbm_time:.2f}")
            print(f"  XGBoost:  {result.xgb_time:.2f}")
            print(f"  CatBoost: {result.catboost_time:.2f}")
            
            # Find best model
            time_best = min([
                ('LGBM', result.lgbm_time_validation.weighted_rmse),
                ('XGBoost', result.xgb_time_validation.weighted_rmse),
                ('CatBoost', result.catboost_time_validation.weighted_rmse)
            ], key=lambda x: x[1])
            
            full_best = min([
                ('LGBM', result.lgbm_full_validation.weighted_rmse),
                ('XGBoost', result.xgb_full_validation.weighted_rmse),
                ('CatBoost', result.catboost_full_validation.weighted_rmse)
            ], key=lambda x: x[1])
            
            print(f"Best Model - Time Validation: {time_best[0]} ({time_best[1]:.6f})")
            print(f"Best Model - Full Validation: {full_best[0]} ({full_best[1]:.6f})")
        
        print(f"\n{'='*100}")
    
    def save_comparison_results(self, comparison: Dict[int, TrioModelResults]) -> None:
        """Save comparison results to files."""
        # Save detailed results as JSON
        comparison_dict = {h: result.to_dict() for h, result in comparison.items()}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.comparisons_dir / f'trio_comparison_{timestamp}.json'
        
        with open(results_path, 'w') as f:
            json.dump(comparison_dict, f, indent=2)
        
        # Save summary CSV
        summary_data = []
        for horizon, result in comparison.items():
            summary_data.append({
                'horizon': horizon,
                'lgbm_time_rmse': result.lgbm_time_validation.weighted_rmse,
                'xgb_time_rmse': result.xgb_time_validation.weighted_rmse,
                'catboost_time_rmse': result.catboost_time_validation.weighted_rmse,
                'lgbm_full_rmse': result.lgbm_full_validation.weighted_rmse,
                'xgb_full_rmse': result.xgb_full_validation.weighted_rmse,
                'catboost_full_rmse': result.catboost_full_validation.weighted_rmse,
                'lgbm_time': result.lgbm_time,
                'xgb_time': result.xgb_time,
                'catboost_time': result.catboost_time
            })
        
        summary_df = pl.DataFrame(summary_data)
        summary_path = self.comparisons_dir / f'trio_summary_{timestamp}.csv'
        summary_df.write_csv(summary_path)
        
        self.logger.info(f"Comparison results saved to: {results_path}")
        self.logger.info(f"Summary CSV saved to: {summary_path}")
    
    def run_trio_pipeline(self) -> Dict[int, TrioModelResults]:
        """Run complete trio models pipeline."""
        total_start_time = time.time()
        self.logger.info("Starting Trio SHAP Models Pipeline...")
        
        # Run all three models
        lgbm_results = self.run_single_model('lgbm', self.lgbm_model)
        xgb_results = self.run_single_model('xgb', self.xgb_model)
        catboost_results = self.run_single_model('catboost', self.catboost_model)
        
        # Compare results
        comparison = self.compare_models_per_horizon(lgbm_results, xgb_results, catboost_results)
        
        # Print summary
        self.print_comparison_summary(comparison)
        
        # Save results
        self.save_comparison_results(comparison)
        
        total_time = time.time() - total_start_time
        
        print(f"\n{'='*80}")
        print("PIPELINE COMPLETION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Pipeline Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Results saved to: {self.results_dir}")
        print(f"{'='*80}")
        
        return comparison
    
    def get_best_model_per_horizon(self, comparison: Dict[int, TrioModelResults], 
                                  validation_type: str = 'time') -> Dict[int, str]:
        """Get best model name per horizon."""
        best_models = {}
        
        for horizon, result in comparison.items():
            if validation_type == 'time':
                best_models[horizon] = min([
                    ('LGBM', result.lgbm_time_validation.weighted_rmse),
                    ('XGBoost', result.xgb_time_validation.weighted_rmse),
                    ('CatBoost', result.catboost_time_validation.weighted_rmse)
                ], key=lambda x: x[1])[0]
            else:  # full validation
                best_models[horizon] = min([
                    ('LGBM', result.lgbm_full_validation.weighted_rmse),
                    ('XGBoost', result.xgb_full_validation.weighted_rmse),
                    ('CatBoost', result.catboost_full_validation.weighted_rmse)
                ], key=lambda x: x[1])[0]
        
        return best_models

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    trio = TrioSHAPModels()
    comparison_results = trio.run_trio_pipeline()
    
    # Get best models
    best_time = trio.get_best_model_per_horizon(comparison_results, 'time')
    best_full = trio.get_best_model_per_horizon(comparison_results, 'full')
    
    print(f"\nBest models (time validation): {best_time}")
    print(f"Best models (full validation): {best_full}")
