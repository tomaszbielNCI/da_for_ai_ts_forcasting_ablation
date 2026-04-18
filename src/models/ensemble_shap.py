#!/usr/bin/env python3
"""
Ensemble SHAP Models

Ensemble model combining LGBM SHAP-10 and LGBM SHAP-20 predictions.
Creates weighted ensemble with validation metrics and submission generation.
Based on trio-shap-features.ipynb ensemble approach.
"""

import polars as pl
import numpy as np
import logging
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from .lgbm_shap_10 import LGBM_SHAP_10
from .lgbm_shap_20 import LGBM_SHAP_20
from ..metrics.evaluation import TimeSeriesMetrics, MetricResults

@dataclass
class EnsembleResults:
    """Container for ensemble results"""
    horizon: int
    shap10_time_validation: Optional[MetricResults]
    shap20_time_validation: Optional[MetricResults]
    ensemble_time_validation: Optional[MetricResults]
    shap10_full_validation: Optional[MetricResults]
    shap20_full_validation: Optional[MetricResults]
    ensemble_full_validation: Optional[MetricResults]
    ensemble_weights: Dict[str, float]
    training_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'horizon': self.horizon,
            'shap10_time_validation': self.shap10_time_validation.to_dict() if self.shap10_time_validation else None,
            'shap20_time_validation': self.shap20_time_validation.to_dict() if self.shap20_time_validation else None,
            'ensemble_time_validation': self.ensemble_time_validation.to_dict() if self.ensemble_time_validation else None,
            'shap10_full_validation': self.shap10_full_validation.to_dict() if self.shap10_full_validation else None,
            'shap20_full_validation': self.shap20_full_validation.to_dict() if self.shap20_full_validation else None,
            'ensemble_full_validation': self.ensemble_full_validation.to_dict() if self.ensemble_full_validation else None,
            'ensemble_weights': self.ensemble_weights,
            'training_time': self.training_time
        }

class EnsembleSHAP:
    """
    Ensemble model combining SHAP-10 and SHAP-20 LGBM models.
    
    Features:
    - Weighted ensemble of SHAP-10 and SHAP-20 predictions
    - Dynamic weight optimization based on validation performance
    - Comprehensive ensemble validation metrics
    - Submission generation
    - Per-horizon optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ensemble SHAP model with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Paths
        self.results_dir = Path('results/ensemble_shap')
        self.predictions_dir = self.results_dir / 'predictions'
        self.models_dir = self.results_dir / 'models'
        
        # Create directories
        for dir_path in [self.results_dir, self.predictions_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Horizons
        self.horizons = self.config.get('horizons', [1, 3, 10, 25])
        
        # Initialize component models
        self.shap10_model = LGBM_SHAP_10(self.config)
        self.shap20_model = LGBM_SHAP_20(self.config)
        
        # Results storage
        self.results = {}
    
    def load_model_predictions(self, model_name: str, horizon: int) -> Dict[str, np.ndarray]:
        """Load predictions from trained models."""
        if model_name == 'shap10':
            predictions_path = self.shap10_model.predictions_dir / f'lgbm_shap_10_h{horizon}_predictions.npz'
        elif model_name == 'shap20':
            predictions_path = self.shap20_model.predictions_dir / f'lgbm_shap_20_h{horizon}_predictions.npz'
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions not found for {model_name} H={horizon}")
        
        predictions = np.load(predictions_path)
        return {
            'train_pred': predictions['train_pred'],
            'valid_pred': predictions['valid_pred'],
            'train_true': predictions['train_true'],
            'valid_true': predictions['valid_true']
        }
    
    def load_full_predictions(self, model_name: str, horizon: int) -> Dict[str, np.ndarray]:
        """Load full validation predictions from trained models."""
        if model_name == 'shap10':
            predictions_path = self.shap10_model.predictions_dir / f'lgbm_shap_10_h{horizon}_full_predictions.npz'
        elif model_name == 'shap20':
            predictions_path = self.shap20_model.predictions_dir / f'lgbm_shap_20_h{horizon}_full_predictions.npz'
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        if not predictions_path.exists():
            raise FileNotFoundError(f"Full predictions not found for {model_name} H={horizon}")
        
        predictions = np.load(predictions_path)
        return {
            'train_pred': predictions['train_pred'],
            'test_pred': predictions['test_pred'],
            'train_true': predictions['train_true']
        }
    
    def optimize_ensemble_weights(self, shap10_pred: np.ndarray, shap20_pred: np.ndarray, 
                                 y_true: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """Optimize ensemble weights using validation performance."""
        # Simple weight optimization based on validation RMSE
        shap10_rmse = np.sqrt(np.mean((y_true - shap10_pred) ** 2))
        shap20_rmse = np.sqrt(np.mean((y_true - shap20_pred) ** 2))
        
        # Inverse RMSE weighting (better performance gets higher weight)
        inv_shap10 = 1.0 / shap10_rmse
        inv_shap20 = 1.0 / shap20_rmse
        
        total_inv = inv_shap10 + inv_shap20
        weight_shap10 = inv_shap10 / total_inv
        weight_shap20 = inv_shap20 / total_inv
        
        return {
            'shap10': weight_shap10,
            'shap20': weight_shap20
        }
    
    def create_ensemble_predictions(self, shap10_pred: np.ndarray, shap20_pred: np.ndarray, 
                                  weights: Dict[str, float]) -> np.ndarray:
        """Create ensemble predictions using optimized weights."""
        return weights['shap10'] * shap10_pred + weights['shap20'] * shap20_pred
    
    def evaluate_ensemble(self, shap10_pred: np.ndarray, shap20_pred: np.ndarray, 
                         y_true: np.ndarray, weights: np.ndarray) -> Tuple[MetricResults, Dict[str, float]]:
        """Evaluate ensemble performance and optimize weights."""
        # Optimize weights
        ensemble_weights = self.optimize_ensemble_weights(shap10_pred, shap20_pred, y_true, weights)
        
        # Create ensemble predictions
        ensemble_pred = self.create_ensemble_predictions(shap10_pred, shap20_pred, ensemble_weights)
        
        # Evaluate ensemble
        ensemble_metrics = TimeSeriesMetrics.evaluate_all(
            y_true, ensemble_pred, weights, 
            include_weighted_mape=True,
            y_train=y_true
        )
        
        return ensemble_metrics, ensemble_weights
    
    def train_ensemble_for_horizon(self, horizon: int) -> EnsembleResults:
        """Train ensemble for a specific horizon."""
        self.logger.info(f"Training ensemble for H={horizon}...")
        horizon_start = time.time()
        
        # Load time validation predictions
        shap10_time_preds = self.load_model_predictions('shap10', horizon)
        shap20_time_preds = self.load_model_predictions('shap20', horizon)
        
        # Load validation weights (use SHAP-10 weights as reference)
        shap10_model_path = self.shap10_model.models_dir / f'lgbm_shap_10_h{horizon}.pkl'
        with open(shap10_model_path, 'rb') as f:
            shap10_model = pickle.load(f)
        
        # Get validation weights from SHAP-10 model
        engineered_data = self.shap10_model.load_engineered_data()
        train_df, _ = engineered_data[horizon]
        valid_split = train_df.filter(
            (pl.col('ts_index') >= self.shap10_model.time_split_valid_start) & 
            (pl.col('ts_index') <= self.shap10_model.time_split_valid_end)
        )
        _, _, valid_weights = self.shap10_model.prepare_data(valid_split, [], is_test=False)
        
        # Evaluate time validation ensemble
        ensemble_time_metrics, time_weights = self.evaluate_ensemble(
            shap10_time_preds['valid_pred'], 
            shap20_time_preds['valid_pred'],
            shap10_time_preds['valid_true'],
            valid_weights
        )
        
        # Load full validation predictions
        shap10_full_preds = self.load_full_predictions('shap10', horizon)
        shap20_full_preds = self.load_full_predictions('shap20', horizon)
        
        # Get full training weights
        train_full = train_df.filter(pl.col('ts_index') <= self.shap10_model.full_train_end)
        _, _, train_weights = self.shap10_model.prepare_data(train_full, [], is_test=False)
        
        # Evaluate full validation ensemble
        ensemble_full_metrics, full_weights = self.evaluate_ensemble(
            shap10_full_preds['train_pred'],
            shap20_full_preds['train_pred'],
            shap10_full_preds['train_true'],
            train_weights
        )
        
        # Load individual model metrics for comparison
        shap10_time_metrics = TimeSeriesMetrics.evaluate_all(
            shap10_time_preds['valid_true'], shap10_time_preds['valid_pred'], valid_weights,
            include_weighted_mape=True, y_train=shap10_time_preds['train_true']
        )
        
        shap20_time_metrics = TimeSeriesMetrics.evaluate_all(
            shap20_time_preds['valid_true'], shap20_time_preds['valid_pred'], valid_weights,
            include_weighted_mape=True, y_train=shap20_time_preds['train_true']
        )
        
        shap10_full_metrics = TimeSeriesMetrics.evaluate_all(
            shap10_full_preds['train_true'], shap10_full_preds['train_pred'], train_weights,
            include_weighted_mape=True, y_train=shap10_full_preds['train_true']
        )
        
        shap20_full_metrics = TimeSeriesMetrics.evaluate_all(
            shap20_full_preds['train_true'], shap20_full_preds['train_pred'], train_weights,
            include_weighted_mape=True, y_train=shap20_full_preds['train_true']
        )
        
        # Create ensemble test predictions
        ensemble_test_pred = self.create_ensemble_predictions(
            shap10_full_preds['test_pred'], shap20_full_preds['test_pred'], full_weights
        )
        
        # Save ensemble test predictions
        test_predictions_path = self.predictions_dir / f'ensemble_h{horizon}_test_predictions.npz'
        np.savez(test_predictions_path, 
                test_pred=ensemble_test_pred,
                ensemble_weights=full_weights)
        
        # Create submission file
        self.create_ensemble_submission(horizon, ensemble_test_pred, full_weights)
        
        horizon_time = time.time() - horizon_start
        
        return EnsembleResults(
            horizon=horizon,
            shap10_time_validation=shap10_time_metrics,
            shap20_time_validation=shap20_time_metrics,
            ensemble_time_validation=ensemble_time_metrics,
            shap10_full_validation=shap10_full_metrics,
            shap20_full_validation=shap20_full_metrics,
            ensemble_full_validation=ensemble_full_metrics,
            ensemble_weights=full_weights,
            training_time=horizon_time
        )
    
    def create_ensemble_submission(self, horizon: int, predictions: np.ndarray, 
                                  weights: Dict[str, float]) -> None:
        """Create ensemble submission file."""
        # Load test data for IDs
        engineered_data = self.shap10_model.load_engineered_data()
        _, test_df = engineered_data[horizon]
        
        # Create submission DataFrame
        submission_df = pl.DataFrame({
            'id': test_df['id'],
            'prediction': predictions
        })
        
        # Save submission with weights in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weight_str = f"w{weights['shap10']:.3f}_{weights['shap20']:.3f}"
        submission_path = self.predictions_dir / f'ensemble_h{horizon}_{weight_str}_submission_{timestamp}.csv'
        submission_df.write_csv(submission_path)
        
        self.logger.info(f"Ensemble submission saved: {submission_path}")
    
    def train_all_ensembles(self) -> Dict[int, EnsembleResults]:
        """Train ensembles for all horizons."""
        start_time = time.time()
        self.logger.info("Starting ensemble training for all horizons...")
        
        results = {}
        
        for horizon in self.horizons:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"ENSEMBLE HORIZON: {horizon}")
            self.logger.info(f"{'='*60}")
            
            try:
                result = self.train_ensemble_for_horizon(horizon)
                results[horizon] = result
            except FileNotFoundError as e:
                self.logger.error(f"Skipping H={horizon}: {e}")
                continue
        
        total_time = time.time() - start_time
        
        # Print summary
        self.print_summary(results, total_time)
        
        # Save results
        self.save_results(results)
        
        return results
    
    def print_summary(self, results: Dict[int, EnsembleResults], total_time: float) -> None:
        """Print ensemble training summary."""
        print(f"\n{'='*100}")
        print("ENSEMBLE SHAP TRAINING SUMMARY")
        print(f"{'='*100}")
        
        for horizon, result in results.items():
            print(f"\nHORIZON {horizon}:")
            print(f"{'-'*80}")
            
            print(f"Time Validation (Weighted RMSE):")
            print(f"  SHAP-10:  {result.shap10_time_validation.weighted_rmse:.6f}")
            print(f"  SHAP-20:  {result.shap20_time_validation.weighted_rmse:.6f}")
            print(f"  ENSEMBLE: {result.ensemble_time_validation.weighted_rmse:.6f}")
            
            print(f"Full Validation (Weighted RMSE):")
            print(f"  SHAP-10:  {result.shap10_full_validation.weighted_rmse:.6f}")
            print(f"  SHAP-20:  {result.shap20_full_validation.weighted_rmse:.6f}")
            print(f"  ENSEMBLE: {result.ensemble_full_validation.weighted_rmse:.6f}")
            
            print(f"Ensemble Weights: SHAP-10={result.ensemble_weights['shap10']:.3f}, "
                  f"SHAP-20={result.ensemble_weights['shap20']:.3f}")
            print(f"Training Time: {result.training_time:.2f} seconds")
        
        print(f"\n{'='*80}")
        print("TIMING SUMMARY")
        print(f"{'='*80}")
        print(f"Total Ensemble Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Results saved to: {self.results_dir}")
        print(f"{'='*100}")
    
    def save_results(self, results: Dict[int, EnsembleResults]) -> None:
        """Save ensemble results to files."""
        # Save detailed results as JSON
        results_dict = {h: result.to_dict() for h, result in results.items()}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.results_dir / f'ensemble_results_{timestamp}.json'
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save summary CSV
        summary_data = []
        for horizon, result in results.items():
            summary_data.append({
                'horizon': horizon,
                'shap10_time_rmse': result.shap10_time_validation.weighted_rmse,
                'shap20_time_rmse': result.shap20_time_validation.weighted_rmse,
                'ensemble_time_rmse': result.ensemble_time_validation.weighted_rmse,
                'shap10_full_rmse': result.shap10_full_validation.weighted_rmse,
                'shap20_full_rmse': result.shap20_full_validation.weighted_rmse,
                'ensemble_full_rmse': result.ensemble_full_validation.weighted_rmse,
                'shap10_weight': result.ensemble_weights['shap10'],
                'shap20_weight': result.ensemble_weights['shap20'],
                'training_time': result.training_time
            })
        
        summary_df = pl.DataFrame(summary_data)
        summary_path = self.results_dir / f'ensemble_summary_{timestamp}.csv'
        summary_df.write_csv(summary_path)
        
        self.logger.info(f"Ensemble results saved to: {results_path}")
        self.logger.info(f"Summary CSV saved to: {summary_path}")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    ensemble = EnsembleSHAP()
    results = ensemble.train_all_ensembles()
