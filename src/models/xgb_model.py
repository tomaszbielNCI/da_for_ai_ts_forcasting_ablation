#!/usr/bin/env python3
"""
XGBoost Model with Top 10 SHAP Features

XGBoost model using top 10 SHAP features per horizon with horizon-specific parameters.
Based on final notebook version (without early stopping for XGBoost).
Saves metrics to JSON and CSV files.
"""

import polars as pl
import numpy as np
import xgboost as xgb
import logging
import time
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Add src to path and import metrics
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.metrics.evaluation import TimeSeriesMetrics, MetricResults


@dataclass
class ValidationResults:
    """Container for validation results"""
    horizon: int
    train_metrics: MetricResults
    valid_metrics: MetricResults
    feature_count: int
    training_time: float
    model_path: str


class XGBoostModel:
    """
    XGBoost model using top 10 SHAP features per horizon.

    Features:
    - Horizon-specific parameters from notebook
    - Top 10 SHAP features from existing SHAP analysis
    - Time split validation (train ts <= 3000, valid 3001-3600)
    - NO early stopping (simplified, works reliably)
    - Comprehensive metrics (Weighted RMSE, Pearson, RMSE)
    - Saves metrics to JSON and CSV files
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize XGBoost SHAP 10 model."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Paths
        project_root = Path(__file__).parent.parent.parent
        self.processed_dir = project_root / 'data/processed/top_10'
        self.models_dir = project_root / 'results/models/xgb_shap_10'
        self.predictions_dir = project_root / 'results/predictions/xgb_shap_10'
        self.metrics_dir = project_root / 'results/metrics'
        self.shap_results_path = project_root / 'results/shap/data/horizon_specific_top_features.json'

        # Create directories
        for dir_path in [self.models_dir, self.predictions_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Validation split parameters (from notebook)
        self.train_split = 3000
        self.valid_start = 3001
        self.valid_end = 3600
        self.max_ts_train = 3601

        # Horizons
        self.horizons = self.config.get('horizons', [1, 3, 10, 25])
        self.seed = self.config.get('seed', 42)

        # Horizon-specific parameters for XGBoost (simplified, no early stopping)
        self.horizon_params = {
            1: {'max_depth': 10, 'learning_rate': 0.028, 'min_child_weight': 77,
                'reg_lambda': 3.89, 'n_estimators': 500},
            3: {'max_depth': 10, 'learning_rate': 0.05, 'min_child_weight': 20,
                'reg_lambda': 0.1, 'n_estimators': 500},
            10: {'max_depth': 10, 'learning_rate': 0.05, 'min_child_weight': 20,
                 'reg_lambda': 0.1, 'n_estimators': 500},
            25: {'max_depth': 16, 'learning_rate': 0.021, 'min_child_weight': 250,
                 'reg_lambda': 20.0, 'n_estimators': 500}
        }

    def check_shap_results(self) -> bool:
        """Check if SHAP results exist."""
        if self.shap_results_path.exists():
            self.logger.info(f"Found SHAP results: {self.shap_results_path}")
            return True
        else:
            self.logger.error(f"SHAP results not found: {self.shap_results_path}")
            return False

    def check_engineered_data(self) -> List[int]:
        """Check which horizons have engineered data available."""
        missing = []
        for horizon in self.horizons:
            train_path = self.processed_dir / f'train_h{horizon}_engineered.parquet'
            if not train_path.exists():
                missing.append(horizon)
        return missing

    def prepare_engineered_data(self) -> Dict[int, Tuple[pl.DataFrame, pl.DataFrame]]:
        """Load engineered features data."""
        self.logger.info("Loading engineered features data...")

        # Check SHAP results
        if not self.check_shap_results():
            raise FileNotFoundError(
                f"\n{'=' * 60}\n"
                f"SHAP RESULTS NOT FOUND\n"
                f"{'=' * 60}\n"
                f"Expected location: {self.shap_results_path}\n\n"
                f"SOLUTION: Run SHAP analysis first.\n"
                f"{'=' * 60}"
            )

        # Check engineered data
        missing = self.check_engineered_data()
        if missing:
            raise FileNotFoundError(
                f"\n{'=' * 60}\n"
                f"ENGINEERED DATA NOT FOUND\n"
                f"{'=' * 60}\n"
                f"Missing horizons: {missing}\n"
                f"Expected directory: {self.processed_dir}\n\n"
                f"SOLUTION: Run feature engineering first with n_features=10\n"
                f"{'=' * 60}"
            )

        # Load data
        engineered_data = {}
        for horizon in self.horizons:
            train_path = self.processed_dir / f'train_h{horizon}_engineered.parquet'
            test_path = self.processed_dir / f'test_h{horizon}_engineered.parquet'

            train_df = pl.read_parquet(train_path)
            test_df = pl.read_parquet(test_path)

            engineered_data[horizon] = (train_df, test_df)
            self.logger.info(f"Loaded H={horizon}: train {train_df.shape}, test {test_df.shape}")

        return engineered_data

    def get_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """Get feature columns (exclude metadata and target)."""
        exclude = ['id', 'code', 'sub_code', 'sub_category', 'horizon',
                   'ts_index', 'y_target', 'weight']

        features = [c for c in df.columns if c not in exclude]
        self.logger.info(f"Using {len(features)} features")
        return features

    def get_xgb_params(self, horizon: int) -> Dict[str, Any]:
        """Get XGBoost parameters for specific horizon (simplified, no early stopping)."""
        hp = self.horizon_params[horizon]

        return {
            'objective': 'reg:squarederror',
            'n_estimators': hp['n_estimators'],
            'max_depth': hp['max_depth'],
            'learning_rate': hp['learning_rate'],
            'min_child_weight': hp['min_child_weight'],
            'subsample': 0.85,
            'colsample_bytree': 0.75,
            'reg_lambda': hp['reg_lambda'],
            'random_state': self.seed,
            'verbosity': 0
        }

    def _save_metrics(self, horizon: int, train_metrics: MetricResults,
                      valid_metrics: MetricResults, feature_count: int) -> None:
        """Save metrics to JSON and CSV files."""
        metrics_dict = {
            'horizon': horizon,
            'model': 'xgb_shap_10',
            'timestamp': datetime.now().isoformat(),
            'train': train_metrics.to_dict(),
            'valid': valid_metrics.to_dict(),
            'features_used': feature_count
        }

        # Save JSON
        json_path = self.metrics_dir / f'metrics_h{horizon}_xgb_shap_10.json'
        TimeSeriesMetrics.save_metrics_to_json(metrics_dict, json_path)

        # Save CSV (appends to single file)
        csv_path = self.metrics_dir / 'all_metrics.csv'
        TimeSeriesMetrics.save_metrics_to_csv(metrics_dict, csv_path, 'train')
        TimeSeriesMetrics.save_metrics_to_csv(metrics_dict, csv_path, 'valid')

    def train_horizon(self, horizon: int, train_df: pl.DataFrame, test_df: pl.DataFrame) -> ValidationResults:
        """Train model for single horizon with time validation."""
        self.logger.info(f"Training XGB H={horizon}...")

        # Get features
        feature_cols = self.get_feature_columns(train_df)

        # Time-based split (from notebook)
        train_split = train_df.filter(pl.col('ts_index') <= self.train_split)
        valid_split = train_df.filter(
            (pl.col('ts_index') >= self.valid_start) &
            (pl.col('ts_index') <= self.valid_end)
        )

        # Full training data for final model
        train_full = train_df.filter(pl.col('ts_index') <= self.max_ts_train)

        # Prepare data
        X_train = train_split.select(feature_cols).to_numpy()
        y_train = train_split['y_target'].to_numpy().ravel()
        w_train = train_split['weight'].to_numpy().ravel()

        X_valid = valid_split.select(feature_cols).to_numpy()
        y_valid = valid_split['y_target'].to_numpy().ravel()
        w_valid = valid_split['weight'].to_numpy().ravel()

        X_full = train_full.select(feature_cols).to_numpy()
        y_full = train_full['y_target'].to_numpy().ravel()
        w_full = train_full['weight'].to_numpy().ravel()

        X_test = test_df.select(feature_cols).to_numpy()

        # Train on split data (NO early stopping - simplified)
        params = self.get_xgb_params(horizon)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=w_train, verbose=False)

        # Predict on validation
        y_valid_pred = model.predict(X_valid)
        valid_metrics = TimeSeriesMetrics.evaluate_all(y_valid, y_valid_pred, w_valid, y_train=y_full)

        # Retrain on full data
        final_model = xgb.XGBRegressor(**params)
        final_model.fit(X_full, y_full, sample_weight=w_full, verbose=False)

        # Evaluate on training data
        y_train_pred = final_model.predict(X_full)
        train_metrics = TimeSeriesMetrics.evaluate_all(y_full, y_train_pred, w_full, y_train=y_full)

        # Predict on test
        y_test_pred = final_model.predict(X_test)

        # Save metrics
        self._save_metrics(horizon, train_metrics, valid_metrics, len(feature_cols))

        # Save model
        model_path = self.models_dir / f'xgb_shap_10_h{horizon}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(final_model, f)

        # Save predictions
        pred_path = self.predictions_dir / f'xgb_shap_10_h{horizon}_predictions.npz'
        np.savez(pred_path, train_pred=y_train_pred, test_pred=y_test_pred)

        # Create submission
        self._save_submission(horizon, test_df, y_test_pred)

        # Print results
        print(f"\n  {'=' * 50}")
        print(f"  HORIZON {horizon} RESULTS (XGBoost SHAP-10)")
        print(f"  {'=' * 50}")
        print(f"  Train Weighted RMSE: {train_metrics.weighted_rmse:.6f}")
        print(f"  Train Pearson:       {train_metrics.pearson:.6f}")
        print(f"  Train RMSE:          {train_metrics.rmse:.6f}")
        print(f"  Valid Weighted RMSE: {valid_metrics.weighted_rmse:.6f}")
        print(f"  Valid Pearson:       {valid_metrics.pearson:.6f}")
        print(f"  Valid RMSE:          {valid_metrics.rmse:.6f}")
        print(f"  Features used:       {len(feature_cols)}")
        print(f"  Test predictions:    {len(y_test_pred):,}")
        print(f"  Test mean:           {np.mean(y_test_pred):.6f}")
        print(f"  Test std:            {np.std(y_test_pred):.6f}")
        print(f"  {'=' * 50}")

        return ValidationResults(
            horizon=horizon,
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            feature_count=len(feature_cols),
            training_time=0,
            model_path=str(model_path)
        )

    def train_all_horizons(self) -> Dict[int, ValidationResults]:
        """Train models for all horizons."""
        start_time = time.time()
        self.logger.info("Starting XGBoost SHAP-10 training...")

        # Load engineered data
        engineered_data = self.prepare_engineered_data()

        results = {}
        for horizon in self.horizons:
            train_df, test_df = engineered_data[horizon]
            results[horizon] = self.train_horizon(horizon, train_df, test_df)

        total_time = time.time() - start_time
        self._print_summary(results, total_time)

        return results

    def _save_submission(self, horizon: int, test_df: pl.DataFrame, predictions: np.ndarray) -> None:
        """Save submission file for horizon."""
        submission_df = pl.DataFrame({
            'id': test_df['id'],
            'prediction': predictions
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = self.predictions_dir / f'xgb_shap_10_h{horizon}_submission_{timestamp}.csv'
        submission_df.write_csv(submission_path)
        self.logger.info(f"Submission saved: {submission_path}")

    def generate_final_submission(self, results: Dict[int, ValidationResults]) -> Path:
        """Generate final submission combining all horizons."""
        all_ids = []
        all_preds = []

        for horizon in self.horizons:
            # Load predictions from saved file
            pred_path = self.predictions_dir / f'xgb_shap_10_h{horizon}_predictions.npz'
            data = np.load(pred_path)
            predictions = data['test_pred']

            # Load test data to get IDs
            test_path = self.processed_dir / f'test_h{horizon}_engineered.parquet'
            test_df = pl.read_parquet(test_path)
            ids = test_df['id'].to_numpy().ravel()

            all_ids.extend(ids)
            all_preds.extend(predictions)

        submission_df = pl.DataFrame({'id': all_ids, 'prediction': all_preds})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = self.predictions_dir / f'xgb_shap_10_submission_{timestamp}.csv'
        submission_df.write_csv(submission_path)

        self.logger.info(f"Final submission saved: {submission_path}")
        self.logger.info(f"Shape: {submission_df.shape}")
        self.logger.info(f"Prediction range: [{np.min(all_preds):.4f}, {np.max(all_preds):.4f}]")

        return submission_path

    def _print_summary(self, results: Dict[int, ValidationResults], total_time: float) -> None:
        """Print comprehensive summary."""
        print(f"\n{'=' * 80}")
        print("XGBoost SHAP-10 - FINAL SUMMARY")
        print(f"{'=' * 80}")

        print(
            f"\n{'Horizon':<8} {'Train WRMSE':<14} {'Valid WRMSE':<14} {'Train Pearson':<14} {'Valid Pearson':<14} {'Features':<10}")
        print("-" * 80)

        for h in self.horizons:
            r = results[h]
            print(f"{h:<8} {r.train_metrics.weighted_rmse:<14.6f} {r.valid_metrics.weighted_rmse:<14.6f} "
                  f"{r.train_metrics.pearson:<14.6f} {r.valid_metrics.pearson:<14.6f} "
                  f"{r.feature_count:<10}")

        print(f"\n{'=' * 60}")
        print(f"Total training time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
        print(f"Models saved to: {self.models_dir}")
        print(f"Predictions saved to: {self.predictions_dir}")
        print(f"Metrics saved to: {self.metrics_dir}")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = XGBoostModel()
    results = model.train_all_horizons()
    model.generate_final_submission(results)