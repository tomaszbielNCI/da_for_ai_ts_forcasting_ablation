#!/usr/bin/env python3
"""
Walk-Forward Validation - Trio (LGBM + XGBoost + CatBoost)

Implements walk-forward validation with 4 large windows using all three models.
Based on notebook walk-forward logic with 4 windows and exponential weights.
Saves metrics to JSON and CSV files.

Windows:
- Window 1: train [1-1000] → valid [1001-1500]
- Window 2: train [1001-2000] → valid [2001-2500]
- Window 3: train [2001-3000] → valid [3001-3500]
- Window 4: train [2501-3300] → valid [3301-3601]
"""

import polars as pl
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import logging
import time
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

# Add src to path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.metrics.evaluation import TimeSeriesMetrics, MetricResults


@dataclass
class WindowResult:
    """Container for single window results for one model"""
    window_idx: int
    window_name: str
    weight: float
    train_metrics: MetricResults
    valid_metrics: MetricResults
    best_iteration: int
    predictions: np.ndarray


@dataclass
class TrioWindowResult:
    """Container for single window results for all three models"""
    window_idx: int
    window_name: str
    weight: float
    lgbm: Optional[WindowResult] = None
    xgb: Optional[WindowResult] = None
    cat: Optional[WindowResult] = None


class WalkForwardTrio:
    """
    Walk-forward validation with 4 large windows using LGBM, XGBoost, and CatBoost.

    Features:
    - 4 time-based windows (non-overlapping)
    - Exponential weights (more recent windows have higher weight)
    - Weighted ensemble of window predictions per model
    - Final ensemble: average of three models
    - Saves metrics to JSON and CSV files
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize walk-forward trio model."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Paths
        project_root = Path(__file__).parent.parent.parent
        self.processed_dir = project_root / 'data/processed/top_10'
        self.models_dir = project_root / 'results/models/walkforward_trio'
        self.predictions_dir = project_root / 'results/predictions/walkforward_trio'
        self.metrics_dir = project_root / 'results/metrics'

        # Create directories
        for dir_path in [self.models_dir, self.predictions_dir, self.metrics_dir]:
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
                'reg_lambda': 3.89, 'max_depth': 10, 'n_estimators': 500},
            3: {'num_leaves': 50, 'learning_rate': 0.05, 'min_child_samples': 20,
                'reg_lambda': 0.1, 'max_depth': 10, 'n_estimators': 500},
            10: {'num_leaves': 50, 'learning_rate': 0.05, 'min_child_samples': 20,
                 'reg_lambda': 0.1, 'max_depth': 10, 'n_estimators': 500},
            25: {'num_leaves': 350, 'learning_rate': 0.021, 'min_child_samples': 250,
                 'reg_lambda': 20.0, 'max_depth': 16, 'n_estimators': 500}
        }

        # XGBoost parameters (simplified, no early stopping)
        self.xgb_params = {
            1: {'max_depth': 10, 'learning_rate': 0.028, 'min_child_weight': 77,
                'reg_lambda': 3.89, 'n_estimators': 500},
            3: {'max_depth': 10, 'learning_rate': 0.05, 'min_child_weight': 20,
                'reg_lambda': 0.1, 'n_estimators': 500},
            10: {'max_depth': 10, 'learning_rate': 0.05, 'min_child_weight': 20,
                 'reg_lambda': 0.1, 'n_estimators': 500},
            25: {'max_depth': 16, 'learning_rate': 0.021, 'min_child_weight': 250,
                 'reg_lambda': 20.0, 'n_estimators': 500}
        }

        # CatBoost parameters (with early stopping)
        self.cat_params = {
            1: {'depth': 10, 'learning_rate': 0.028, 'min_data_in_leaf': 77,
                'l2_leaf_reg': 3.89, 'iterations': 500},
            3: {'depth': 10, 'learning_rate': 0.05, 'min_data_in_leaf': 20,
                'l2_leaf_reg': 0.1, 'iterations': 500},
            10: {'depth': 10, 'learning_rate': 0.05, 'min_data_in_leaf': 20,
                 'l2_leaf_reg': 0.1, 'iterations': 500},
            25: {'depth': 16, 'learning_rate': 0.021, 'min_data_in_leaf': 250,
                 'l2_leaf_reg': 20.0, 'iterations': 500}
        }

        # Store results
        self.window_results = {h: [] for h in self.horizons}

    def get_lgbm_params(self, horizon: int) -> Dict[str, Any]:
        """Get LightGBM parameters."""
        hp = self.lgbm_params[horizon]
        return {
            'objective': 'regression', 'metric': 'rmse',
            'num_leaves': hp['num_leaves'], 'learning_rate': hp['learning_rate'],
            'n_estimators': hp['n_estimators'], 'max_depth': hp['max_depth'],
            'min_child_samples': hp['min_child_samples'], 'subsample': 0.85,
            'colsample_bytree': 0.75, 'reg_lambda': hp['reg_lambda'],
            'random_state': self.seed, 'verbose': -1
        }

    def get_xgb_params(self, horizon: int) -> Dict[str, Any]:
        """Get XGBoost parameters (simplified, no early stopping)."""
        hp = self.xgb_params[horizon]
        return {
            'objective': 'reg:squarederror',
            'n_estimators': hp['n_estimators'], 'max_depth': hp['max_depth'],
            'learning_rate': hp['learning_rate'], 'min_child_weight': hp['min_child_weight'],
            'subsample': 0.85, 'colsample_bytree': 0.75,
            'reg_lambda': hp['reg_lambda'], 'tree_method': 'hist',
            'random_state': self.seed, 'verbosity': 0
        }

    def get_cat_params(self, horizon: int) -> Dict[str, Any]:
        """Get CatBoost parameters (with early stopping)."""
        hp = self.cat_params[horizon]
        return {
            'objective': 'RMSE', 'eval_metric': 'RMSE',
            'depth': hp['depth'], 'learning_rate': hp['learning_rate'],
            'iterations': hp['iterations'], 'min_data_in_leaf': hp['min_data_in_leaf'],
            'subsample': 0.85, 'colsample_bylevel': 0.75,
            'l2_leaf_reg': hp['l2_leaf_reg'], 'random_seed': self.seed,
            'verbose': False
        }

    def _save_metrics(self, horizon: int, window_name: str, model_name: str,
                      train_metrics: MetricResults, valid_metrics: MetricResults,
                      best_iter: int) -> None:
        """Save metrics to JSON and CSV files."""
        metrics_dict = {
            'horizon': horizon,
            'window': window_name,
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'train': train_metrics.to_dict(),
            'valid': valid_metrics.to_dict(),
            'best_iteration': best_iter
        }

        # Save JSON
        json_path = self.metrics_dir / f'metrics_wf_h{horizon}_{window_name}_{model_name}.json'
        TimeSeriesMetrics.save_metrics_to_json(metrics_dict, json_path)

        # Save CSV (appends to single file)
        csv_path = self.metrics_dir / 'all_metrics_walkforward.csv'
        TimeSeriesMetrics.save_metrics_to_csv(metrics_dict, csv_path, 'train')
        TimeSeriesMetrics.save_metrics_to_csv(metrics_dict, csv_path, 'valid')

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
        return [c for c in df.columns if c not in exclude]

    def prepare_data(self, df: pl.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training/prediction."""
        X = df.select(feature_cols).to_numpy()
        y = df['y_target'].to_numpy().ravel()
        w = df['weight'].to_numpy().ravel()
        return X, y, w

    def train_lgbm_window(self, horizon: int, window: Dict, w_idx: int,
                          X_train, y_train, w_train, X_valid, y_valid, w_valid,
                          X_test) -> Optional[WindowResult]:
        """Train LGBM for a single window."""
        params = self.get_lgbm_params(horizon)
        model = lgb.LGBMRegressor(**params)

        model.fit(
            X_train, y_train, sample_weight=w_train,
            eval_set=[(X_valid, y_valid)], eval_sample_weight=[w_valid],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )

        best_iter = model.best_iteration_
        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)
        test_pred = model.predict(X_test)

        train_metrics = TimeSeriesMetrics.evaluate_all(y_train, y_train_pred, w_train, y_train=y_train)
        valid_metrics = TimeSeriesMetrics.evaluate_all(y_valid, y_valid_pred, w_valid, y_train=y_train)

        # Save metrics
        self._save_metrics(horizon, window['name'], 'lgbm', train_metrics, valid_metrics, best_iter)

        n_windows = len(self.windows)
        weight = np.exp(-(n_windows - 1 - w_idx) * 0.3)

        return WindowResult(
            window_idx=w_idx, window_name=window['name'], weight=weight,
            train_metrics=train_metrics, valid_metrics=valid_metrics,
            best_iteration=best_iter, predictions=test_pred
        )

    def train_xgb_window(self, horizon: int, window: Dict, w_idx: int,
                         X_train, y_train, w_train, X_valid, y_valid, w_valid,
                         X_test) -> Optional[WindowResult]:
        """Train XGBoost for a single window (no early stopping)."""
        params = self.get_xgb_params(horizon)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=w_train, verbose=False)

        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)
        test_pred = model.predict(X_test)

        train_metrics = TimeSeriesMetrics.evaluate_all(y_train, y_train_pred, w_train, y_train=y_train)
        valid_metrics = TimeSeriesMetrics.evaluate_all(y_valid, y_valid_pred, w_valid, y_train=y_train)

        # Save metrics
        self._save_metrics(horizon, window['name'], 'xgb', train_metrics, valid_metrics, params['n_estimators'])

        n_windows = len(self.windows)
        weight = np.exp(-(n_windows - 1 - w_idx) * 0.3)

        return WindowResult(
            window_idx=w_idx, window_name=window['name'], weight=weight,
            train_metrics=train_metrics, valid_metrics=valid_metrics,
            best_iteration=params['n_estimators'], predictions=test_pred
        )

    def train_cat_window(self, horizon: int, window: Dict, w_idx: int,
                         X_train, y_train, w_train, X_valid, y_valid, w_valid,
                         X_test) -> Optional[WindowResult]:
        """Train CatBoost for a single window (with early stopping)."""
        params = self.get_cat_params(horizon)
        model = CatBoostRegressor(**params)

        model.fit(
            X_train, y_train, sample_weight=w_train,
            eval_set=(X_valid, y_valid), early_stopping_rounds=20, verbose=False
        )

        best_iter = model.get_best_iteration()
        if best_iter is None:
            best_iter = params['iterations']

        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)
        test_pred = model.predict(X_test)

        train_metrics = TimeSeriesMetrics.evaluate_all(y_train, y_train_pred, w_train, y_train=y_train)
        valid_metrics = TimeSeriesMetrics.evaluate_all(y_valid, y_valid_pred, w_valid, y_train=y_train)

        # Save metrics
        self._save_metrics(horizon, window['name'], 'cat', train_metrics, valid_metrics, best_iter)

        n_windows = len(self.windows)
        weight = np.exp(-(n_windows - 1 - w_idx) * 0.3)

        return WindowResult(
            window_idx=w_idx, window_name=window['name'], weight=weight,
            train_metrics=train_metrics, valid_metrics=valid_metrics,
            best_iteration=best_iter, predictions=test_pred
        )

    def run_window(self, horizon: int, window: Dict, w_idx: int) -> Optional[TrioWindowResult]:
        """Run all three models for a single window."""
        # Load and prepare data
        train_df = self.load_engineered_data(horizon)

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

        feature_cols = self.get_feature_columns(train_split)

        X_train, y_train, w_train = self.prepare_data(train_split, feature_cols)
        X_valid, y_valid, w_valid = self.prepare_data(valid_split, feature_cols)

        test_df = pl.read_parquet(self.processed_dir / f'test_h{horizon}_engineered.parquet')
        X_test, _, _ = self.prepare_data(test_df, feature_cols)

        # Train all three models
        lgbm_res = self.train_lgbm_window(horizon, window, w_idx, X_train, y_train, w_train, X_valid, y_valid, w_valid,
                                          X_test)
        xgb_res = self.train_xgb_window(horizon, window, w_idx, X_train, y_train, w_train, X_valid, y_valid, w_valid,
                                        X_test)
        cat_res = self.train_cat_window(horizon, window, w_idx, X_train, y_train, w_train, X_valid, y_valid, w_valid,
                                        X_test)

        return TrioWindowResult(
            window_idx=w_idx,
            window_name=window['name'],
            weight=lgbm_res.weight if lgbm_res else 0,
            lgbm=lgbm_res,
            xgb=xgb_res,
            cat=cat_res
        )

    def run_horizon(self, horizon: int) -> Tuple[List[TrioWindowResult], np.ndarray]:
        """Run walk-forward for a single horizon."""
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"HORIZON: {horizon}")
        self.logger.info(f"{'=' * 60}")

        results = []
        all_lgb_preds, all_xgb_preds, all_cat_preds = [], [], []
        all_weights = []

        for w_idx, window in enumerate(self.windows):
            result = self.run_window(horizon, window, w_idx)
            if result is not None:
                results.append(result)
                all_weights.append(result.weight)

                if result.lgbm:
                    all_lgb_preds.append(result.lgbm.predictions)
                if result.xgb:
                    all_xgb_preds.append(result.xgb.predictions)
                if result.cat:
                    all_cat_preds.append(result.cat.predictions)

                print(f"\n  {window['name']}: weight={result.weight:.4f}")
                if result.lgbm:
                    print(
                        f"    LGBM - Valid WRMSE: {result.lgbm.valid_metrics.weighted_rmse:.6f}, iter={result.lgbm.best_iteration}")
                if result.xgb:
                    print(
                        f"    XGB  - Valid WRMSE: {result.xgb.valid_metrics.weighted_rmse:.6f}, iter={result.xgb.best_iteration}")
                if result.cat:
                    print(
                        f"    CAT  - Valid WRMSE: {result.cat.valid_metrics.weighted_rmse:.6f}, iter={result.cat.best_iteration}")

        # Weighted ensemble per model
        lgb_ensemble = np.average(np.array(all_lgb_preds), axis=0, weights=all_weights) if all_lgb_preds else None
        xgb_ensemble = np.average(np.array(all_xgb_preds), axis=0, weights=all_weights) if all_xgb_preds else None
        cat_ensemble = np.average(np.array(all_cat_preds), axis=0, weights=all_weights) if all_cat_preds else None

        # Final trio ensemble (average of three models)
        if lgb_ensemble is not None and xgb_ensemble is not None and cat_ensemble is not None:
            final_ensemble = (lgb_ensemble + xgb_ensemble + cat_ensemble) / 3.0
        elif lgb_ensemble is not None and cat_ensemble is not None:
            final_ensemble = (lgb_ensemble + cat_ensemble) / 2.0
        elif lgb_ensemble is not None:
            final_ensemble = lgb_ensemble
        else:
            final_ensemble = np.zeros(1)

        return results, final_ensemble

    def run_all_horizons(self) -> Dict[int, Tuple[List[TrioWindowResult], np.ndarray]]:
        """Run walk-forward for all horizons."""
        start_time = time.time()
        self.logger.info("Starting walk-forward trio validation...")

        results = {}

        for horizon in self.horizons:
            window_results, ensemble_pred = self.run_horizon(horizon)
            results[horizon] = (window_results, ensemble_pred)
            self._save_horizon_submission(horizon, ensemble_pred)

        total_time = time.time() - start_time
        self._print_summary(results, total_time)
        self._generate_final_submission(results)

        # Print metrics location
        print(f"\n✅ Metrics saved to: {self.metrics_dir}")
        print(f"   - Individual JSON files: metrics_wf_h*_*.json")
        print(f"   - Combined CSV: all_metrics_walkforward.csv")

        return results

    def _save_horizon_submission(self, horizon: int, predictions: np.ndarray) -> None:
        """Save submission file for a single horizon."""
        test_path = self.processed_dir / f'test_h{horizon}_engineered.parquet'
        test_df = pl.read_parquet(test_path)

        submission_df = pl.DataFrame({'id': test_df['id'], 'prediction': predictions})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = self.predictions_dir / f'walkforward_trio_h{horizon}_submission_{timestamp}.csv'
        submission_df.write_csv(submission_path)
        self.logger.info(f"Submission saved: {submission_path}")

    def _generate_final_submission(self, results: Dict[int, Tuple[List[TrioWindowResult], np.ndarray]]) -> None:
        """Generate final submission combining all horizons."""
        all_ids = []
        all_preds = []

        for horizon in self.horizons:
            _, ensemble_pred = results[horizon]
            test_path = self.processed_dir / f'test_h{horizon}_engineered.parquet'
            test_df = pl.read_parquet(test_path)
            all_ids.extend(test_df['id'].to_numpy().ravel())
            all_preds.extend(ensemble_pred)

        submission_df = pl.DataFrame({'id': all_ids, 'prediction': all_preds})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = self.predictions_dir / f'walkforward_trio_final_{timestamp}.csv'
        submission_df.write_csv(submission_path)

        self.logger.info(f"Final submission saved: {submission_path}")
        self.logger.info(f"Shape: {submission_df.shape}")
        self.logger.info(f"Prediction range: [{np.min(all_preds):.4f}, {np.max(all_preds):.4f}]")

    def _print_summary(self, results: Dict[int, Tuple[List[TrioWindowResult], np.ndarray]], total_time: float) -> None:
        """Print walk-forward summary."""
        print(f"\n{'=' * 80}")
        print("WALK-FORWARD TRIO - SUMMARY")
        print(f"{'=' * 80}")

        print(f"\n{'Horizon':<8} {'Model':<10} {'Valid WRMSE (mean)':<20} {'Valid WRMSE (std)':<20} {'Best Iter':<12}")
        print("-" * 75)

        for horizon in self.horizons:
            window_results, _ = results[horizon]
            if not window_results:
                print(f"{horizon:<8} {'N/A':<10} {'N/A':<20} {'N/A':<20} {'N/A':<12}")
                continue

            # LGBM stats
            lgb_wrmse = [r.lgbm.valid_metrics.weighted_rmse for r in window_results if r.lgbm]
            if lgb_wrmse:
                lgb_iter = [r.lgbm.best_iteration for r in window_results if r.lgbm]
                print(
                    f"{horizon:<8} {'LGBM':<10} {np.mean(lgb_wrmse):<20.6f} {np.std(lgb_wrmse):<20.6f} {np.mean(lgb_iter):<12.1f}")

            # XGB stats
            xgb_wrmse = [r.xgb.valid_metrics.weighted_rmse for r in window_results if r.xgb]
            if xgb_wrmse:
                xgb_iter = [r.xgb.best_iteration for r in window_results if r.xgb]
                print(
                    f"{horizon:<8} {'XGB':<10} {np.mean(xgb_wrmse):<20.6f} {np.std(xgb_wrmse):<20.6f} {np.mean(xgb_iter):<12.1f}")

            # CAT stats
            cat_wrmse = [r.cat.valid_metrics.weighted_rmse for r in window_results if r.cat]
            if cat_wrmse:
                cat_iter = [r.cat.best_iteration for r in window_results if r.cat]
                print(
                    f"{horizon:<8} {'CAT':<10} {np.mean(cat_wrmse):<20.6f} {np.std(cat_wrmse):<20.6f} {np.mean(cat_iter):<12.1f}")

        print(f"\n{'=' * 60}")
        print(f"Total time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
        print(f"Models saved to: {self.models_dir}")
        print(f"Predictions saved to: {self.predictions_dir}")
        print(f"Metrics saved to: {self.metrics_dir}")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    wf = WalkForwardTrio()
    results = wf.run_all_horizons()