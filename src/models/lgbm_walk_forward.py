#!/usr/bin/env python3
"""
Walk-Forward Validation - LGBM Only

Implements walk-forward validation with 4 large windows using LGBM SHAP-10.
Based on notebook walk-forward logic with 4 windows and exponential weights.

Windows:
- Window 1: train [1-1000] → valid [1001-1500]
- Window 2: train [1001-2000] → valid [2001-2500]
- Window 3: train [2001-3000] → valid [3001-3500]
- Window 4: train [2501-3300] → valid [3301-3601]
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

# Add src to path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.metrics.evaluation import TimeSeriesMetrics, MetricResults


@dataclass
class WindowResult:
    """Container for single window results"""
    window_idx: int
    window_name: str
    weight: float
    train_metrics: MetricResults
    valid_metrics: MetricResults
    best_iteration: int
    predictions: np.ndarray


class WalkForwardLGB:
    """
    Walk-forward validation with 4 large windows using LGBM SHAP-10.

    Features:
    - 4 time-based windows (non-overlapping)
    - Exponential weights (more recent windows have higher weight)
    - Weighted ensemble of window predictions
    - Same features and parameters as LGBM SHAP-10
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize walk-forward LGBM model."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Paths
        project_root = Path(__file__).parent.parent.parent
        self.processed_dir = project_root / 'data/processed/top_10'
        self.models_dir = project_root / 'results/models/walkforward_lgb'
        self.predictions_dir = project_root / 'results/predictions/walkforward_lgb'

        # Create directories
        for dir_path in [self.models_dir, self.predictions_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 4 large windows (from notebook)
        self.windows = [
            {'name': 'Window 1', 'train_start': 1, 'train_end': 1000,
             'valid_start': 1001, 'valid_end': 1500},
            {'name': 'Window 2', 'train_start': 1001, 'train_end': 2000,
             'valid_start': 2001, 'valid_end': 2500},
            {'name': 'Window 3', 'train_start': 2001, 'train_end': 3000,
             'valid_start': 3001, 'valid_end': 3500},
            {'name': 'Window 4', 'train_start': 2501, 'train_end': 3300,
             'valid_start': 3301, 'valid_end': 3601}
        ]

        # Horizons
        self.horizons = self.config.get('horizons', [1, 3, 10, 25])
        self.seed = self.config.get('seed', 42)

        # LGBM parameters (same as SHAP-10)
        self.lgbm_params = {
            1: {'num_leaves': 240, 'learning_rate': 0.028, 'min_child_samples': 77,
                'reg_lambda': 3.89, 'max_depth': 10},
            3: {'num_leaves': 50, 'learning_rate': 0.05, 'min_child_samples': 20,
                'reg_lambda': 0.1, 'max_depth': 10},
            10: {'num_leaves': 50, 'learning_rate': 0.05, 'min_child_samples': 20,
                 'reg_lambda': 0.1, 'max_depth': 10},
            25: {'num_leaves': 350, 'learning_rate': 0.021, 'min_child_samples': 250,
                 'reg_lambda': 20.0, 'max_depth': 16}
        }

        # Store results
        self.window_results = {h: [] for h in self.horizons}

    def get_lgbm_params(self, horizon: int) -> Dict[str, Any]:
        """Get LightGBM parameters for specific horizon."""
        hp = self.lgbm_params[horizon]

        return {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': hp['num_leaves'],
            'learning_rate': hp['learning_rate'],
            'n_estimators': 500,
            'max_depth': hp['max_depth'],
            'min_child_samples': hp['min_child_samples'],
            'subsample': 0.85,
            'colsample_bytree': 0.75,
            'reg_lambda': hp['reg_lambda'],
            'random_state': self.seed,
            'verbose': -1
        }

    def load_engineered_data(self, horizon: int) -> pl.DataFrame:
        """Load engineered data for a specific horizon."""
        train_path = self.processed_dir / f'train_h{horizon}_engineered.parquet'

        if not train_path.exists():
            raise FileNotFoundError(f"Engineered data not found: {train_path}")

        return pl.read_parquet(train_path)

    def get_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """Get feature columns (exclude metadata and target)."""
        exclude = ['id', 'code', 'sub_code', 'sub_category', 'horizon',
                   'ts_index', 'y_target', 'weight']

        features = [c for c in df.columns if c not in exclude]
        return features

    def prepare_data(self, df: pl.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training/prediction."""
        X = df.select(feature_cols).to_numpy()
        y = df['y_target'].to_numpy().ravel()
        w = df['weight'].to_numpy().ravel()
        return X, y, w

    def train_window(self, horizon: int, window: Dict, w_idx: int) -> WindowResult:
        """Train model for a single window."""
        self.logger.info(f"Training H={horizon}, {window['name']}...")

        # Load data
        train_df = self.load_engineered_data(horizon)

        # Split by time
        train_split = train_df.filter(
            (pl.col('ts_index') >= window['train_start']) &
            (pl.col('ts_index') <= window['train_end'])
        )
        valid_split = train_df.filter(
            (pl.col('ts_index') >= window['valid_start']) &
            (pl.col('ts_index') <= window['valid_end'])
        )

        if len(train_split) < 100 or len(valid_split) < 10:
            self.logger.warning(f"Not enough data for H={horizon}, {window['name']}")
            return None

        # Get features
        feature_cols = self.get_feature_columns(train_split)

        # Prepare data
        X_train, y_train, w_train = self.prepare_data(train_split, feature_cols)
        X_valid, y_valid, w_valid = self.prepare_data(valid_split, feature_cols)

        # Train model
        params = self.get_lgbm_params(horizon)
        model = lgb.LGBMRegressor(**params)

        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_valid, y_valid)],
            eval_sample_weight=[w_valid],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )

        best_iter = model.best_iteration_

        # Predictions
        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)

        # Metrics
        train_metrics = TimeSeriesMetrics.evaluate_all(y_train, y_train_pred, w_train, y_train=y_train)
        valid_metrics = TimeSeriesMetrics.evaluate_all(y_valid, y_valid_pred, w_valid, y_train=y_train)

        # Exponential weight (more recent windows have higher weight)
        n_windows = len(self.windows)
        weight = np.exp(-(n_windows - 1 - w_idx) * 0.3)

        # Predict on test data
        test_df = pl.read_parquet(self.processed_dir / f'test_h{horizon}_engineered.parquet')
        X_test, _, _ = self.prepare_data(test_df, feature_cols)
        test_pred = model.predict(X_test)

        return WindowResult(
            window_idx=w_idx,
            window_name=window['name'],
            weight=weight,
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            best_iteration=best_iter,
            predictions=test_pred
        )

    def run_horizon(self, horizon: int) -> Tuple[List[WindowResult], np.ndarray]:
        """Run walk-forward for a single horizon."""
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"HORIZON: {horizon}")
        self.logger.info(f"{'=' * 60}")

        results = []
        all_test_preds = []
        all_weights = []

        for w_idx, window in enumerate(self.windows):
            result = self.train_window(horizon, window, w_idx)
            if result is not None:
                results.append(result)
                all_test_preds.append(result.predictions)
                all_weights.append(result.weight)

                print(f"  {window['name']}: weight={result.weight:.4f}")
                print(f"    Train WRMSE: {result.train_metrics.weighted_rmse:.6f}")
                print(f"    Valid WRMSE: {result.valid_metrics.weighted_rmse:.6f}")
                print(f"    Best iter: {result.best_iteration}")

        # Weighted ensemble of test predictions
        if all_test_preds:
            all_preds_array = np.array(all_test_preds)
            all_weights_array = np.array(all_weights)
            ensemble_pred = np.average(all_preds_array, axis=0, weights=all_weights_array)
        else:
            ensemble_pred = np.zeros(1)

        return results, ensemble_pred

    def run_all_horizons(self) -> Dict[int, Tuple[List[WindowResult], np.ndarray]]:
        """Run walk-forward for all horizons."""
        start_time = time.time()
        self.logger.info("Starting walk-forward LGBM validation...")

        results = {}

        for horizon in self.horizons:
            window_results, ensemble_pred = self.run_horizon(horizon)
            results[horizon] = (window_results, ensemble_pred)

            # Save submission for this horizon
            self._save_horizon_submission(horizon, ensemble_pred)

        total_time = time.time() - start_time

        # Print summary
        self._print_summary(results, total_time)

        # Generate final submission
        self._generate_final_submission(results)

        return results

    def _save_horizon_submission(self, horizon: int, predictions: np.ndarray) -> None:
        """Save submission file for a single horizon."""
        test_path = self.processed_dir / f'test_h{horizon}_engineered.parquet'
        test_df = pl.read_parquet(test_path)

        submission_df = pl.DataFrame({
            'id': test_df['id'],
            'prediction': predictions
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = self.predictions_dir / f'walkforward_lgb_h{horizon}_submission_{timestamp}.csv'
        submission_df.write_csv(submission_path)
        self.logger.info(f"Submission saved: {submission_path}")

    def _generate_final_submission(self, results: Dict[int, Tuple[List[WindowResult], np.ndarray]]) -> None:
        """Generate final submission combining all horizons."""
        all_ids = []
        all_preds = []

        for horizon in self.horizons:
            _, ensemble_pred = results[horizon]

            test_path = self.processed_dir / f'test_h{horizon}_engineered.parquet'
            test_df = pl.read_parquet(test_path)
            ids = test_df['id'].to_numpy().ravel()

            all_ids.extend(ids)
            all_preds.extend(ensemble_pred)

        submission_df = pl.DataFrame({'id': all_ids, 'prediction': all_preds})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = self.predictions_dir / f'walkforward_lgb_final_{timestamp}.csv'
        submission_df.write_csv(submission_path)

        self.logger.info(f"Final submission saved: {submission_path}")
        self.logger.info(f"Shape: {submission_df.shape}")
        self.logger.info(f"Prediction range: [{np.min(all_preds):.4f}, {np.max(all_preds):.4f}]")

    def _print_summary(self, results: Dict[int, Tuple[List[WindowResult], np.ndarray]], total_time: float) -> None:
        """Print walk-forward summary."""
        print(f"\n{'=' * 80}")
        print("WALK-FORWARD LGBM - SUMMARY")
        print(f"{'=' * 80}")

        print(f"\n{'Horizon':<8} {'Valid WRMSE (mean)':<20} {'Valid WRMSE (std)':<20} {'Windows':<10}")
        print("-" * 60)

        for horizon in self.horizons:
            window_results, _ = results[horizon]
            if window_results:
                wrmse_vals = [r.valid_metrics.weighted_rmse for r in window_results]
                print(f"{horizon:<8} {np.mean(wrmse_vals):<20.6f} {np.std(wrmse_vals):<20.6f} {len(window_results)}")
            else:
                print(f"{horizon:<8} {'N/A':<20} {'N/A':<20} {0}")

        print(f"\n{'=' * 60}")
        print(f"Total time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
        print(f"Models saved to: {self.models_dir}")
        print(f"Predictions saved to: {self.predictions_dir}")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    wf = WalkForwardLGB()
    results = wf.run_all_horizons()