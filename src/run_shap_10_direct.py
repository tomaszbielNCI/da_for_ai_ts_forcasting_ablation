#!/usr/bin/env python3
"""
Direct run script for LGBM SHAP-10 model without import issues
"""

import polars as pl
import numpy as np
import lightgbm as lgb
import logging
import time
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Import metrics directly
import sys
sys.path.append(str(Path(__file__).parent))
from metrics.evaluation import TimeSeriesMetrics, MetricResults

# Import feature engineering
from features.shap_features import SHAPFeatureEngineer

@dataclass
class ValidationResults:
    """Container for validation results"""
    horizon: int
    train_metrics: Optional[MetricResults]
    valid_metrics: Optional[MetricResults]
    test_metrics: Optional[MetricResults]
    model_path: str
    predictions_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            'horizon': self.horizon,
            'train_metrics': self.train_metrics.to_dict() if self.train_metrics else None,
            'valid_metrics': self.valid_metrics.to_dict() if self.valid_metrics else None,
            'test_metrics': self.test_metrics.to_dict() if self.test_metrics else None,
            'model_path': self.model_path,
            'predictions_path': self.predictions_path
        }
        return result

class LGBM_SHAP_10_Direct:
    """Direct implementation of LGBM SHAP-10 without import issues."""
    
    def __init__(self):
        """Initialize model."""
        self.logger = logging.getLogger(__name__)
        
        # Paths
        project_root = Path(__file__).parent.parent
        self.processed_dir = project_root / 'data/processed/top_10'
        self.models_dir = project_root / 'results/models/lgbm_shap_10'
        self.predictions_dir = project_root / 'results/predictions/lgbm_shap_10'
        
        # Create directories
        for dir_path in [self.models_dir, self.predictions_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Validation split parameters
        self.time_split_train = 3000
        self.time_split_valid_start = 3001
        self.time_split_valid_end = 3600
        self.full_train_end = 3601
        
        # Horizons
        self.horizons = [1, 3, 10, 25]
        
        # Horizon-specific parameters
        self.horizon_params = {
            1: {'num_leaves': 240, 'lr': 0.028, 'min_child': 77, 'l2': 3.89, 'depth': 10},
            3: {'num_leaves': 50, 'lr': 0.05, 'min_child': 20, 'l2': 0.1, 'depth': 10},
            10: {'num_leaves': 50, 'lr': 0.05, 'min_child': 20, 'l2': 0.1, 'depth': 10},
            25: {'num_leaves': 350, 'lr': 0.021, 'min_child': 250, 'l2': 20.0, 'depth': 16}
        }
    
    def check_shap_results(self) -> bool:
        """Check if SHAP results exist."""
        project_root = Path(__file__).parent.parent
        shap_results_path = project_root / 'results/shap/data/horizon_specific_top_features.json'
        
        if shap_results_path.exists():
            self.logger.info(f"Found existing SHAP results: {shap_results_path}")
            return True
        else:
            self.logger.warning(f"SHAP results not found: {shap_results_path}")
            return False
    
    def get_lgbm_params(self, horizon: int, seed: int = 42) -> Dict[str, Any]:
        """Get LightGBM parameters for specific horizon."""
        h_params = self.horizon_params[horizon]
        
        return {
            'objective': 'regression', 
            'metric': 'rmse',
            'num_leaves': h_params['num_leaves'], 
            'learning_rate': h_params['lr'],
            'n_estimators': 1000, 
            'max_depth': h_params['depth'],
            'min_child_samples': h_params['min_child'], 
            'subsample': 0.85,
            'colsample_bytree': 0.75, 
            'reg_lambda': h_params['l2'],
            'random_state': seed, 
            'verbose': -1
        }
    
    def get_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """Get feature columns (all except target, weight, and metadata)."""
        exclude_cols = ['y_target', 'weight', 'horizon', 'ts_index', 'code', 'sub_code', 'sub_category', 'id']
        
        # Get only numeric columns
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols:
                # Check if column is numeric
                dtype = df[col].dtype
                if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]:
                    feature_cols.append(col)
        
        self.logger.info(f"Found {len(feature_cols)} numeric feature columns")
        return feature_cols
    
    def prepare_data(self, df: pl.DataFrame, feature_cols: List[str], is_test: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training/prediction."""
        if is_test:
            # Test data doesn't have y_target and weight
            X = df.select(feature_cols).to_numpy()
            y = np.zeros(len(X))  # Dummy
            w = np.ones(len(X))    # Dummy
        else:
            X = df.select(feature_cols).to_numpy()
            y = df['y_target'].to_numpy().ravel()
            w = df['weight'].to_numpy().ravel()
        
        return X, y, w
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray, 
                    horizon: int, seed: int = 42) -> lgb.LGBMRegressor:
        """Train LightGBM model."""
        params = self.get_lgbm_params(horizon, seed)
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, sample_weight=w_train)
        
        return model
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           weights: np.ndarray, y_train: Optional[np.ndarray] = None) -> MetricResults:
        """Evaluate predictions with comprehensive metrics."""
        return TimeSeriesMetrics.evaluate_all(
            y_true, y_pred, weights, 
            include_weighted_mape=True,
            y_train=y_train
        )
    
    def train_horizon_time_validation(self, horizon: int, train_df: pl.DataFrame) -> ValidationResults:
        """Train model with time validation for a specific horizon."""
        self.logger.info(f"Training LGBM SHAP-10 for H={horizon} with time validation...")
        
        # Split data
        train_split = train_df.filter(pl.col('ts_index') <= self.time_split_train)
        valid_split = train_df.filter(
            (pl.col('ts_index') >= self.time_split_valid_start) & 
            (pl.col('ts_index') <= self.time_split_valid_end)
        )
        
        # Get features
        feature_cols = self.get_feature_columns(train_split)
        
        # Prepare data
        X_train, y_train, w_train = self.prepare_data(train_split, feature_cols, is_test=False)
        X_valid, y_valid, w_valid = self.prepare_data(valid_split, feature_cols, is_test=False)
        
        # Train model
        model = self.train_model(X_train, y_train, w_train, horizon)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)
        
        # Evaluation
        train_metrics = self.evaluate_predictions(y_train, y_train_pred, w_train, y_train)
        valid_metrics = self.evaluate_predictions(y_valid, y_valid_pred, w_valid, y_train)
        
        # Save model
        model_path = self.models_dir / f'lgbm_shap_10_h{horizon}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save predictions
        predictions_path = self.predictions_dir / f'lgbm_shap_10_h{horizon}_predictions.npz'
        np.savez(predictions_path, 
                train_pred=y_train_pred, 
                valid_pred=y_valid_pred,
                train_true=y_train,
                valid_true=y_valid)
        
        return ValidationResults(
            horizon=horizon,
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            test_metrics=None,
            model_path=str(model_path),
            predictions_path=str(predictions_path)
        )
    
    def train_horizon_full_validation(self, horizon: int, train_df: pl.DataFrame, 
                                    test_df: pl.DataFrame) -> ValidationResults:
        """Train model with full validation for a specific horizon."""
        self.logger.info(f"Training LGBM SHAP-10 for H={horizon} with full validation...")
        
        # Use full training data
        train_full = train_df.filter(pl.col('ts_index') <= self.full_train_end)
        
        # Get features
        feature_cols = self.get_feature_columns(train_full)
        
        # Prepare data
        X_train, y_train, w_train = self.prepare_data(train_full, feature_cols, is_test=False)
        X_test, _, _ = self.prepare_data(test_df, feature_cols, is_test=True)
        
        # Train model
        model = self.train_model(X_train, y_train, w_train, horizon)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Evaluation (train only, no test targets)
        train_metrics = self.evaluate_predictions(y_train, y_train_pred, w_train, y_train)
        
        # Save model
        model_path = self.models_dir / f'lgbm_shap_10_h{horizon}_full.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save predictions and create submission
        predictions_path = self.predictions_dir / f'lgbm_shap_10_h{horizon}_full_predictions.npz'
        np.savez(predictions_path, 
                train_pred=y_train_pred, 
                test_pred=y_test_pred,
                train_true=y_train)
        
        # Create submission file
        self.create_submission_file(horizon, test_df, y_test_pred)
        
        return ValidationResults(
            horizon=horizon,
            train_metrics=train_metrics,
            valid_metrics=None,
            test_metrics=None,  # No test targets available
            model_path=str(model_path),
            predictions_path=str(predictions_path)
        )
    
    def create_submission_file(self, horizon: int, test_df: pl.DataFrame, predictions: np.ndarray) -> None:
        """Create submission file for horizon."""
        # Create submission DataFrame
        submission_df = pl.DataFrame({
            'id': test_df['id'],
            'prediction': predictions
        })
        
        # Save submission
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = self.predictions_dir / f'lgbm_shap_10_h{horizon}_submission_{timestamp}.csv'
        submission_df.write_csv(submission_path)
        
        self.logger.info(f"Submission saved: {submission_path}")
    
    def train_all_horizons(self) -> Dict[int, Dict[str, Any]]:
        """Train models for all horizons with both validation types."""
        start_time = time.time()
        self.logger.info("Starting LGBM SHAP-10 training for all horizons...")
        
        # Check SHAP results exist
        if not self.check_shap_results():
            raise FileNotFoundError("SHAP results not found. Please run SHAP analysis first.")
        
        # Load engineered data
        self.logger.info("Loading engineered features data...")
        engineered_data = {}
        
        for horizon in self.horizons:
            train_path = self.processed_dir / f'train_h{horizon}_engineered.parquet'
            test_path = self.processed_dir / f'test_h{horizon}_engineered.parquet'
            
            if not train_path.exists() or not test_path.exists():
                raise FileNotFoundError(f"Engineered data not found for H={horizon}. Run feature engineering first.")
            
            train_df = pl.read_parquet(train_path)
            test_df = pl.read_parquet(test_path)
            
            engineered_data[horizon] = (train_df, test_df)
            self.logger.info(f"Loaded H={horizon}: train {train_df.shape}, test {test_df.shape}")
        
        results = {}
        horizon_times = {}
        
        for horizon in self.horizons:
            horizon_start = time.time()
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"HORIZON: {horizon}")
            self.logger.info(f"{'='*60}")
            
            train_df, test_df = engineered_data[horizon]
            
            # Log feature count for verification
            feature_cols = self.get_feature_columns(train_df)
            self.logger.info(f"Training with {len(feature_cols)} features for H={horizon}")
            
            # Time validation
            time_res = self.train_horizon_time_validation(horizon, train_df)
            
            # Full validation
            full_res = self.train_horizon_full_validation(horizon, train_df, test_df)
            
            horizon_time = time.time() - horizon_start
            horizon_times[horizon] = horizon_time
            
            results[horizon] = {
                'time_validation': time_res,
                'full_validation': full_res,
                'training_time': horizon_time,
                'feature_count': len(feature_cols)
            }
        
        total_time = time.time() - start_time
        
        # Print summary
        self.print_summary(results, horizon_times, total_time)
        
        return results
    
    def print_summary(self, results: Dict[int, Dict[str, Any]], 
                      horizon_times: Dict[int, float], total_time: float) -> None:
        """Print training summary."""
        print(f"\n{'='*80}")
        print("LGBM SHAP-10 TRAINING SUMMARY")
        print(f"{'='*80}")
        
        for horizon, horizon_results in results.items():
            time_res = horizon_results['time_validation']
            full_res = horizon_results['full_validation']
            horizon_time = horizon_results['training_time']
            feature_count = horizon_results['feature_count']
            
            print(f"\nHORIZON {horizon}:")
            print(f"  Features Used: {feature_count} (should be ~90: 10 SHAP + engineered)")
            print(f"  Time Validation - Weighted RMSE: {time_res.valid_metrics.weighted_rmse:.6f}")
            print(f"  Full Validation - Weighted RMSE: {time_res.train_metrics.weighted_rmse:.6f}")
            print(f"  Training Time: {horizon_time:.2f} seconds")
        
        print(f"\n{'='*60}")
        print("TIMING SUMMARY")
        print(f"{'='*60}")
        print(f"Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        for horizon, h_time in horizon_times.items():
            print(f"  H={horizon}: {h_time:.2f}s ({h_time/60:.2f}min)")
        
        print(f"\nModels saved to: {self.models_dir}")
        print(f"Predictions saved to: {self.predictions_dir}")
        print(f"SHAP results used from: results/shap/data/")
        print(f"Engineered features from: data/processed/top_10/")
        print(f"{'='*80}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    model = LGBM_SHAP_10_Direct()
    results = model.train_all_horizons()
