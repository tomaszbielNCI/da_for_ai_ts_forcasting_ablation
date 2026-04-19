#!/usr/bin/env python3
"""
Baseline LightGBM Model

Implements baseline LGBM with dual validation:
1. Time split validation (train ts <= 3000, valid 3001-3600)
2. Full train/test validation (train ts <= 3601, test ts > 3601)

Uses cleaned data from preprocessor and comprehensive metrics.
"""

import polars as pl
import numpy as np
import lightgbm as lgb
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass

from ..metrics.evaluation import TimeSeriesMetrics, MetricResults


@dataclass
class ValidationResults:
    """Container for validation results"""
    train_metrics: MetricResults
    valid_metrics: MetricResults
    test_metrics: Optional[MetricResults] = None
    model: Optional[lgb.LGBMRegressor] = None
    feature_names: Optional[List[str]] = None
    training_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            'training_time_seconds': self.training_time_seconds,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names) if self.feature_names else 0
        }

        if self.train_metrics:
            result['train_metrics'] = self.train_metrics.to_dict()
        if self.valid_metrics:
            result['valid_metrics'] = self.valid_metrics.to_dict()
        if self.test_metrics:
            result['test_metrics'] = self.test_metrics.to_dict()

        return result


class BaselineLGBM:
    """
    Baseline LightGBM model with comprehensive evaluation.

    Features:
    - Dual validation (time split + full train/test)
    - Comprehensive metrics including Kaggle Weighted RMSE
    - Per-horizon training and evaluation
    - Model saving and loading
    - Submission generation
    - Saves metrics to JSON and CSV
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize baseline LGBM with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Data paths
        self.data_dir = Path('data')
        self.cleaned_dir = self.data_dir / 'cleaned'
        self.results_dir = Path('results') / 'models'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Metrics directory
        self.metrics_dir = Path('results') / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Validation splits
        self.time_split_train = 3000
        self.time_split_valid_start = 3001
        self.time_split_valid_end = 3600
        self.full_train_end = 3601

        # Horizons
        self.horizons = self.config.get('horizons', [1, 3, 10, 25])

        # Model parameters
        self.model_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 50,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'max_depth': 10,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1
        }

        # Update params with config
        if 'model_params' in self.config:
            self.model_params.update(self.config['model_params'])

    def load_cleaned_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load cleaned data from preprocessor."""
        self.logger.info("Loading cleaned data...")

        train_path = self.cleaned_dir / 'train_clean.parquet'
        test_path = self.cleaned_dir / 'test_clean.parquet'

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError("Cleaned data not found. Run preprocessor first.")

        train_df = pl.read_parquet(train_path)
        test_df = pl.read_parquet(test_path)

        self.logger.info(f"Loaded cleaned data: train {train_df.shape}, test {test_df.shape}")
        return train_df, test_df

    def get_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """Get feature columns (feature_*) from dataframe."""
        feature_cols = [c for c in df.columns if c.startswith('feature_')]
        self.logger.info(f"Found {len(feature_cols)} feature columns")
        return feature_cols

    def split_data_time_validation(self, df: pl.DataFrame, horizon: int) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split data for time validation.
        Train: ts <= 3000, Valid: 3001-3600
        """
        horizon_df = df.filter(pl.col('horizon') == horizon)

        train_split = horizon_df.filter(pl.col('ts_index') <= self.time_split_train)
        valid_split = horizon_df.filter(
            (pl.col('ts_index') >= self.time_split_valid_start) &
            (pl.col('ts_index') <= self.time_split_valid_end)
        )

        self.logger.info(f"Time validation split - Train: {len(train_split):,}, Valid: {len(valid_split):,}")
        return train_split, valid_split

    def split_data_full_validation(self, train_df: pl.DataFrame, test_df: pl.DataFrame, horizon: int) -> Tuple[
        pl.DataFrame, pl.DataFrame]:
        """
        Split data for full validation.
        Train: ts <= 3601, Test: ts > 3601
        """
        train_horizon = train_df.filter(pl.col('horizon') == horizon)
        test_horizon = test_df.filter(pl.col('horizon') == horizon)

        train_full = train_horizon.filter(pl.col('ts_index') <= self.full_train_end)

        self.logger.info(f"Full validation split - Train: {len(train_full):,}, Test: {len(test_horizon):,}")
        return train_full, test_horizon

    def prepare_data(self, df: pl.DataFrame, feature_cols: List[str], is_test: bool = False) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training (features, target, weights)."""
        X = df.select(feature_cols).to_numpy()

        if is_test:
            # Test data doesn't have target or weight
            y = np.zeros(len(df))  # Dummy target
            w = np.ones(len(df))  # Dummy weights
        else:
            y = df['y_target'].to_numpy().ravel()
            w = df['weight'].to_numpy().ravel()

        return X, y, w

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray) -> lgb.LGBMRegressor:
        """Train LightGBM model."""
        start_time = time.time()

        model = lgb.LGBMRegressor(**self.model_params)
        model.fit(X_train, y_train, sample_weight=w_train)

        training_time = time.time() - start_time
        self.logger.info(f"Model trained in {training_time:.2f} seconds")

        return model

    def evaluate_model(self, model: lgb.LGBMRegressor, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> MetricResults:
        """Evaluate model and return metrics."""
        y_pred = model.predict(X)
        metrics = TimeSeriesMetrics.evaluate_all(y, y_pred, w)
        return metrics

    def _save_metrics(self, horizon: int, validation_type: str,
                      train_metrics: MetricResults, valid_metrics: MetricResults,
                      training_time: float, feature_count: int) -> None:
        """Save metrics to JSON and CSV files."""
        metrics_dict = {
            'horizon': horizon,
            'model': 'baseline_lgbm',
            'validation_type': validation_type,
            'timestamp': datetime.now().isoformat(),
            'train': train_metrics.to_dict(),
            'valid': valid_metrics.to_dict(),
            'training_time_seconds': training_time,
            'features_used': feature_count
        }

        # Save JSON
        json_path = self.metrics_dir / f'metrics_h{horizon}_{validation_type}.json'
        TimeSeriesMetrics.save_metrics_to_json(metrics_dict, json_path)

        # Save CSV (appends to single file)
        csv_path = self.metrics_dir / 'all_metrics.csv'
        TimeSeriesMetrics.save_metrics_to_csv(metrics_dict, csv_path, 'train')
        TimeSeriesMetrics.save_metrics_to_csv(metrics_dict, csv_path, 'valid')

    def train_horizon_time_validation(self, train_df: pl.DataFrame, horizon: int,
                                      feature_cols: List[str]) -> ValidationResults:
        """Train model for single horizon with time validation."""
        self.logger.info(f"Training horizon {horizon} with time validation...")

        # Split data
        train_split, valid_split = self.split_data_time_validation(train_df, horizon)

        # Prepare data
        X_train, y_train, w_train = self.prepare_data(train_split, feature_cols, is_test=False)
        X_valid, y_valid, w_valid = self.prepare_data(valid_split, feature_cols, is_test=False)

        # Train model
        start_time = time.time()
        model = self.train_model(X_train, y_train, w_train)
        training_time = time.time() - start_time

        # Evaluate
        train_metrics = self.evaluate_model(model, X_train, y_train, w_train)
        valid_metrics = self.evaluate_model(model, X_valid, y_valid, w_valid)

        # Save metrics
        self._save_metrics(horizon, 'time_validation', train_metrics, valid_metrics,
                           training_time, len(feature_cols))

        return ValidationResults(
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            model=model,
            feature_names=feature_cols,
            training_time_seconds=training_time
        )

    def train_horizon_full_validation(self, train_df: pl.DataFrame, test_df: pl.DataFrame, horizon: int,
                                      feature_cols: List[str]) -> ValidationResults:
        """Train model for single horizon with full validation."""
        self.logger.info(f"Training horizon {horizon} with full validation...")

        # Split data
        train_full, test_horizon = self.split_data_full_validation(train_df, test_df, horizon)

        # Prepare data
        X_train, y_train, w_train = self.prepare_data(train_full, feature_cols, is_test=False)
        X_test, _, _ = self.prepare_data(test_horizon, feature_cols, is_test=True)

        # Train model on full training data
        start_time = time.time()
        model = self.train_model(X_train, y_train, w_train)
        training_time = time.time() - start_time

        # Evaluate on training data only (test has no targets)
        train_metrics = self.evaluate_model(model, X_train, y_train, w_train)

        # Save metrics (no validation metrics for full validation)
        metrics_dict = {
            'horizon': horizon,
            'model': 'baseline_lgbm',
            'validation_type': 'full_validation',
            'timestamp': datetime.now().isoformat(),
            'train': train_metrics.to_dict(),
            'training_time_seconds': training_time,
            'features_used': len(feature_cols),
            'test_samples': len(test_horizon)
        }

        json_path = self.metrics_dir / f'metrics_h{horizon}_full_validation.json'
        TimeSeriesMetrics.save_metrics_to_json(metrics_dict, json_path)

        csv_path = self.metrics_dir / 'all_metrics.csv'
        TimeSeriesMetrics.save_metrics_to_csv(metrics_dict, csv_path, 'train')

        return ValidationResults(
            train_metrics=train_metrics,
            valid_metrics=None,
            test_metrics=None,
            model=model,
            feature_names=feature_cols,
            training_time_seconds=training_time
        )

    def train_all_horizons(self, train_df: pl.DataFrame, test_df: Optional[pl.DataFrame] = None) -> Dict[
        int, Dict[str, ValidationResults]]:
        """
        Train models for all horizons with both validation types.

        Returns:
            Dict with results per horizon and validation type
        """
        feature_cols = self.get_feature_columns(train_df)
        results = {}

        for horizon in self.horizons:
            self.logger.info(f"Processing horizon {horizon}...")
            results[horizon] = {}

            # Time validation
            time_results = self.train_horizon_time_validation(train_df, horizon, feature_cols)
            results[horizon]['time_validation'] = time_results

            # Print time validation results
            self.logger.info(f"Time Validation Results - Horizon {horizon}:")
            TimeSeriesMetrics.print_metrics(time_results.valid_metrics, f"Time Validation - Horizon {horizon}")

            # Full validation (if test data provided)
            if test_df is not None:
                full_results = self.train_horizon_full_validation(train_df, test_df, horizon, feature_cols)
                results[horizon]['full_validation'] = full_results

                # Print full validation info
                self.logger.info(f"Full Validation - Horizon {horizon}: Model trained on full data, "
                                 f"{len(test_df.filter(pl.col('horizon') == horizon)):,} test samples ready")

        return results

    def save_results(self, results: Dict[int, Dict[str, ValidationResults]], filename: Optional[str] = None) -> Path:
        """Save training results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"baseline_lgbm_results_{timestamp}.json"

        results_path = self.results_dir / filename

        # Convert results to serializable format
        serializable_results = {}
        for horizon, horizon_results in results.items():
            serializable_results[str(horizon)] = {}
            for validation_type, validation_results in horizon_results.items():
                serializable_results[str(horizon)][validation_type] = validation_results.to_dict()

        # Save to JSON
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Results saved to {results_path}")
        return results_path

    def generate_submission(self, train_df: pl.DataFrame, test_df: pl.DataFrame,
                            results: Dict[int, Dict[str, ValidationResults]], filename: Optional[str] = None) -> Path:
        """Generate submission file using full validation models."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"baseline_lgbm_submission_{timestamp}.csv"

        submission_path = Path('results') / filename
        submission_path.parent.mkdir(exist_ok=True)

        all_ids = []
        all_preds = []

        for horizon in self.horizons:
            if horizon not in results or 'full_validation' not in results[horizon]:
                continue

            model = results[horizon]['full_validation'].model
            feature_cols = results[horizon]['full_validation'].feature_names

            # Get test data for this horizon
            test_horizon = test_df.filter(pl.col('horizon') == horizon)
            X_test = test_horizon.select(feature_cols).to_numpy()

            # Predict
            preds = model.predict(X_test)

            # Collect IDs and predictions
            ids = test_horizon['id'].to_numpy().ravel()
            all_ids.extend(ids)
            all_preds.extend(preds)

            self.logger.info(f"Horizon {horizon}: {len(preds):,} predictions")

        # Create submission dataframe
        submission_df = pl.DataFrame({
            'id': all_ids,
            'prediction': all_preds
        })

        # Save submission
        submission_df.write_csv(submission_path)

        self.logger.info(f"Submission saved: {submission_path}")
        self.logger.info(f"Shape: {submission_df.shape}")
        self.logger.info(f"Prediction range: [{np.min(all_preds):.4f}, {np.max(all_preds):.4f}]")

        return submission_path

    def run_full_pipeline(self) -> Dict[int, Dict[str, ValidationResults]]:
        """Run complete baseline LGBM pipeline."""
        self.logger.info("Starting baseline LGBM pipeline...")

        # Load data
        train_df, test_df = self.load_cleaned_data()

        # Train all horizons
        results = self.train_all_horizons(train_df, test_df)

        # Save results
        self.save_results(results)

        # Generate submission
        self.generate_submission(train_df, test_df, results)

        # Print summary
        self.print_summary(results)

        # Print metrics files location
        print(f"\n✅ Metrics saved to: {self.metrics_dir}")
        print(f"   - Individual JSON files: metrics_h*_*.json")
        print(f"   - Combined CSV: all_metrics.csv")

        return results

    def print_summary(self, results: Dict[int, Dict[str, ValidationResults]]) -> None:
        """Print comprehensive summary of results."""
        print("\n" + "=" * 80)
        print("BASELINE LGBM - COMPREHENSIVE SUMMARY")
        print("=" * 80)

        print(
            f"\n{'Horizon':<8} {'Validation':<12} {'Weighted RMSE':<15} {'Pearson':<12} {'RMSE':<12} {'Training Time':<14}")
        print("-" * 80)

        for horizon in self.horizons:
            if horizon not in results:
                continue

            horizon_results = results[horizon]

            # Time validation
            if 'time_validation' in horizon_results:
                time_res = horizon_results['time_validation']
                print(f"{horizon:<8} {'Time Valid':<12} {time_res.valid_metrics.weighted_rmse:<15.6f} "
                      f"{time_res.valid_metrics.pearson:<12.6f} {time_res.valid_metrics.rmse:<12.6f} "
                      f"{time_res.training_time_seconds:<14.2f}")

            # Full validation
            if 'full_validation' in horizon_results:
                full_res = horizon_results['full_validation']
                if full_res.train_metrics:
                    print(f"{horizon:<8} {'Full Valid':<12} {full_res.train_metrics.weighted_rmse:<15.6f} "
                          f"{full_res.train_metrics.pearson:<12.6f} {full_res.train_metrics.rmse:<12.6f} "
                          f"{full_res.training_time_seconds:<14.2f}")
                else:
                    print(f"{horizon:<8} {'Full Valid':<12} {'N/A':<15} {'N/A':<12} {'N/A':<12} "
                          f"{full_res.training_time_seconds:<14.2f}")

        print("\n" + "=" * 80)
        print("Key Insights:")
        print("=" * 80)

        # Calculate average metrics
        time_wrmses = []

        for horizon in self.horizons:
            if horizon in results and 'time_validation' in results[horizon]:
                time_wrmses.append(results[horizon]['time_validation'].valid_metrics.weighted_rmse)

        if time_wrmses:
            print(f"Average Time Validation Weighted RMSE: {np.mean(time_wrmses):.6f}")

        print("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    baseline = BaselineLGBM()
    results = baseline.run_full_pipeline()