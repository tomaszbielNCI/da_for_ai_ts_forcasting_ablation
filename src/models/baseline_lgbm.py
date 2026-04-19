#!/usr/bin/env python3
"""
Baseline LightGBM Model

Implements baseline LGBM with dual validation:
1. Time split validation (train ts <= 3000, valid 3001-3600)
2. Full train/test validation (train ts <= 3601, test ts > 3601)

Uses RAW CLEANED data (all original feature_* columns, no engineered features).
Saves metrics to JSON and CSV files.
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

# Add src to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.metrics.evaluation import TimeSeriesMetrics, MetricResults


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
    Uses RAW CLEANED data (all feature_* columns, no engineering).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize baseline LGBM with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Paths - use raw cleaned data, NOT engineered!
        project_root = Path(__file__).parent.parent.parent
        self.cleaned_dir = project_root / 'data/cleaned'
        self.results_dir = project_root / 'results' / 'models'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Metrics directory
        self.metrics_dir = project_root / 'results' / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Validation splits
        self.train_split = 3000
        self.valid_start = 3001
        self.valid_end = 3600
        self.max_ts_train = 3601

        # Horizons
        self.horizons = self.config.get('horizons', [1, 3, 10, 25])

        # Model parameters (simple baseline)
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
        """Load raw cleaned data (no feature engineering)."""
        train_path = self.cleaned_dir / 'train_clean.parquet'
        test_path = self.cleaned_dir / 'test_clean.parquet'

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Cleaned data not found. Run preprocessor first.")

        train_df = pl.read_parquet(train_path)
        test_df = pl.read_parquet(test_path)

        self.logger.info(f"Loaded cleaned data: train {train_df.shape}, test {test_df.shape}")
        return train_df, test_df

    def get_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """Get feature columns (all feature_* columns)."""
        feature_cols = [c for c in df.columns if c.startswith('feature_')]
        self.logger.info(f"Found {len(feature_cols)} raw feature columns")
        return feature_cols

    def prepare_data(self, df: pl.DataFrame, feature_cols: List[str], is_test: bool = False) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training (features, target, weights)."""
        X = df.select(feature_cols).to_numpy()

        if is_test:
            # Test data doesn't have target or weight
            y = np.zeros(len(df))
            w = np.ones(len(df))
        else:
            y = df['y_target'].to_numpy().ravel()
            w = df['weight'].to_numpy().ravel()

        return X, y, w

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
                    X_valid: np.ndarray, y_valid: np.ndarray, w_valid: np.ndarray) -> lgb.LGBMRegressor:
        """Train LightGBM model with early stopping."""
        start_time = time.time()

        model = lgb.LGBMRegressor(**self.model_params)
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_valid, y_valid)],
            eval_sample_weight=[w_valid],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        training_time = time.time() - start_time
        self.logger.info(f"Model trained in {training_time:.2f} seconds, best iteration: {model.best_iteration_}")

        return model

    def evaluate_model(self, model: lgb.LGBMRegressor, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> MetricResults:
        """Evaluate model and return metrics."""
        y_pred = model.predict(X)
        metrics = TimeSeriesMetrics.evaluate_all(y, y_pred, w, y_train=y)
        return metrics

    def _save_metrics(self, horizon: int, validation_type: str,
                      train_metrics: MetricResults, valid_metrics: MetricResults,
                      training_time: float, feature_count: int, best_iter: int) -> None:
        """Save metrics to JSON and CSV files."""
        metrics_dict = {
            'horizon': horizon,
            'model': 'baseline_lgbm_raw',
            'validation_type': validation_type,
            'timestamp': datetime.now().isoformat(),
            'train': train_metrics.to_dict(),
            'valid': valid_metrics.to_dict(),
            'training_time_seconds': training_time,
            'features_used': feature_count,
            'best_iteration': best_iter
        }

        # Save JSON
        json_path = self.metrics_dir / f'metrics_h{horizon}_baseline_raw_{validation_type}.json'
        TimeSeriesMetrics.save_metrics_to_json(metrics_dict, json_path)

        # Save CSV (appends to single file)
        csv_path = self.metrics_dir / 'all_metrics.csv'
        TimeSeriesMetrics.save_metrics_to_csv(metrics_dict, csv_path, 'train')
        TimeSeriesMetrics.save_metrics_to_csv(metrics_dict, csv_path, 'valid')

    def train_horizon(self, horizon: int, train_df: pl.DataFrame, test_df: pl.DataFrame) -> ValidationResults:
        """Train model for single horizon with time validation."""
        self.logger.info(f"Training H={horizon} with time validation...")

        # Filter by horizon
        train_h = train_df.filter(pl.col('horizon') == horizon)
        test_h = test_df.filter(pl.col('horizon') == horizon)

        # Get features (all feature_* columns)
        feature_cols = self.get_feature_columns(train_h)

        # Time-based split (from notebook)
        train_split = train_h.filter(pl.col('ts_index') <= self.train_split)
        valid_split = train_h.filter(
            (pl.col('ts_index') >= self.valid_start) &
            (pl.col('ts_index') <= self.valid_end)
        )
        train_full = train_h.filter(pl.col('ts_index') <= self.max_ts_train)

        # Prepare data
        X_train, y_train, w_train = self.prepare_data(train_split, feature_cols, is_test=False)
        X_valid, y_valid, w_valid = self.prepare_data(valid_split, feature_cols, is_test=False)
        X_full, y_full, w_full = self.prepare_data(train_full, feature_cols, is_test=False)
        X_test, _, _ = self.prepare_data(test_h, feature_cols, is_test=True)

        # Train with early stopping
        model = self.train_model(X_train, y_train, w_train, X_valid, y_valid, w_valid)
        best_iter = model.best_iteration_

        # Retrain on full data with best iterations
        final_model = lgb.LGBMRegressor(**self.model_params)
        final_model.set_params(n_estimators=best_iter)
        final_model.fit(X_full, y_full, sample_weight=w_full)

        # Evaluate
        y_train_pred = final_model.predict(X_full)
        train_metrics = self.evaluate_model(final_model, X_full, y_full, w_full)

        y_valid_pred = model.predict(X_valid)
        valid_metrics = self.evaluate_model(model, X_valid, y_valid, w_valid)

        # Test predictions
        y_test_pred = final_model.predict(X_test)

        # Save metrics
        self._save_metrics(horizon, 'time_validation', train_metrics, valid_metrics,
                           0, len(feature_cols), best_iter)

        # Print results
        print(f"\n  {'=' * 50}")
        print(f"  HORIZON {horizon} RESULTS (BASELINE RAW)")
        print(f"  {'=' * 50}")
        print(f"  Train Weighted RMSE: {train_metrics.weighted_rmse:.6f}")
        print(f"  Train Pearson:       {train_metrics.pearson:.6f}")
        print(f"  Train RMSE:          {train_metrics.rmse:.6f}")
        print(f"  Valid Weighted RMSE: {valid_metrics.weighted_rmse:.6f}")
        print(f"  Valid Pearson:       {valid_metrics.pearson:.6f}")
        print(f"  Valid RMSE:          {valid_metrics.rmse:.6f}")
        print(f"  Best iteration:      {best_iter}")
        print(f"  Features used:       {len(feature_cols)}")
        print(f"  Test predictions:    {len(y_test_pred):,}")
        print(f"  Test mean:           {np.mean(y_test_pred):.6f}")
        print(f"  Test std:            {np.std(y_test_pred):.6f}")
        print(f"  {'=' * 50}")

        return ValidationResults(
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            model=final_model,
            feature_names=feature_cols,
            training_time_seconds=0
        )

    def train_all_horizons(self, train_df: pl.DataFrame, test_df: pl.DataFrame) -> Dict[int, ValidationResults]:
        """Train models for all horizons."""
        self.logger.info("Starting baseline LGBM training on raw data...")

        results = {}
        for horizon in self.horizons:
            print(f"\n{'=' * 60}")
            print(f"HORIZON: {horizon}")
            print(f"{'=' * 60}")

            result = self.train_horizon(horizon, train_df, test_df)
            results[horizon] = result

        self._print_summary(results)
        return results

    def generate_submission(self, results: Dict[int, ValidationResults], test_df: pl.DataFrame) -> Path:
        """Generate final submission combining all horizons."""
        all_ids = []
        all_preds = []

        for horizon in self.horizons:
            # Get test data for this horizon
            test_h = test_df.filter(pl.col('horizon') == horizon)
            ids = test_h['id'].to_numpy().ravel()
            all_ids.extend(ids)

            # Get predictions from model
            model = results[horizon].model
            feature_cols = results[horizon].feature_names
            X_test = test_h.select(feature_cols).to_numpy()
            preds = model.predict(X_test)
            all_preds.extend(preds)

        submission_df = pl.DataFrame({'id': all_ids, 'prediction': all_preds})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = self.results_dir / f'baseline_lgbm_raw_submission_{timestamp}.csv'
        submission_df.write_csv(submission_path)

        self.logger.info(f"Submission saved: {submission_path}")
        self.logger.info(f"Shape: {submission_df.shape}")
        self.logger.info(f"Prediction range: [{np.min(all_preds):.4f}, {np.max(all_preds):.4f}]")

        return submission_path

    def _print_summary(self, results: Dict[int, ValidationResults]) -> None:
        """Print comprehensive summary."""
        print(f"\n{'=' * 80}")
        print("BASELINE LGBM (RAW DATA) - FINAL SUMMARY")
        print(f"{'=' * 80}")

        print(
            f"\n{'Horizon':<8} {'Train WRMSE':<14} {'Valid WRMSE':<14} {'Train Pearson':<14} {'Valid Pearson':<14} {'Best Iter':<10} {'Features':<10}")
        print("-" * 90)

        for h in self.horizons:
            r = results[h]
            print(f"{h:<8} {r.train_metrics.weighted_rmse:<14.6f} {r.valid_metrics.weighted_rmse:<14.6f} "
                  f"{r.train_metrics.pearson:<14.6f} {r.valid_metrics.pearson:<14.6f} "
                  f"{r.model.best_iteration_ if hasattr(r.model, 'best_iteration_') else 0:<10} {len(r.feature_names):<10}")

        print(f"\n{'=' * 60}")
        print(f"Metrics saved to: {self.metrics_dir}")
        print(f"Models saved in memory")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    baseline = BaselineLGBM()

    # Load raw cleaned data
    train_df, test_df = baseline.load_cleaned_data()

    # Train on all horizons
    results = baseline.train_all_horizons(train_df, test_df)

    # Generate submission
    baseline.generate_submission(results, test_df)

    print(f"\n✅ Baseline LGBM (raw data) complete")
    print(f"✅ Metrics saved to: results/metrics/")