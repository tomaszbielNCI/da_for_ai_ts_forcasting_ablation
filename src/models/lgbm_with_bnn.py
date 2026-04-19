#!/usr/bin/env python3
"""
LGBM Model with BNN Features (SHAP-10 or Aggregated)

LightGBM model using top 10 SHAP features + BNN predictions from either:
- bnn_shap10 (top 10 SHAP + target encoding)
- bnn_aggregated (aggregated groups)

Compares performance with and without BNN features.
Saves metrics to JSON and CSV files.
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
from src.features.shap_features import SHAPFeatureEngineer


@dataclass
class ValidationResults:
    """Container for validation results"""
    horizon: int
    train_metrics: MetricResults
    valid_metrics: MetricResults
    best_iteration: int
    feature_count: int
    training_time: float
    model_path: str


class LGBMWithBNN:
    """
    LightGBM model using top 10 SHAP features + BNN predictions.
    Supports both SHAP-10 BNN and Aggregated BNN.
    Saves metrics to JSON and CSV files.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Paths
        project_root = Path(__file__).parent.parent.parent
        self.data_dir = project_root / 'data/processed/top_10'
        self.bnn_shap10_dir = project_root / 'results/predictions/bnn_shap10'
        self.bnn_agg_dir = project_root / 'results/predictions/bnn_aggregated'
        self.models_dir = project_root / 'results/models/lgbm_with_bnn'
        self.predictions_dir = project_root / 'results/predictions/lgbm_with_bnn'
        self.metrics_dir = project_root / 'results/metrics'

        # Create directories
        for dir_path in [self.models_dir, self.predictions_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Validation split (from notebook)
        self.train_split = 3000
        self.valid_start = 3001
        self.valid_end = 3600
        self.max_ts_train = 3601

        # Horizons
        self.horizons = [1, 3, 10, 25]
        self.seed = 42

        # LGBM parameters (same as SHAP-10 baseline)
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

        # Feature engineer for base features
        self.feature_engineer = SHAPFeatureEngineer()

    def load_engineered_data(self, horizon: int) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load engineered data (has top 10 SHAP + target encoding)."""
        train_path = self.data_dir / f'train_h{horizon}_engineered.parquet'
        test_path = self.data_dir / f'test_h{horizon}_engineered.parquet'

        train_df = pl.read_parquet(train_path)
        test_df = pl.read_parquet(test_path)

        return train_df, test_df

    def load_bnn_predictions(self, horizon: int, bnn_type: str = 'shap10') -> Tuple[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Load BNN predictions for both train and test data."""
        if bnn_type == 'shap10':
            pred_dir = self.bnn_shap10_dir
        elif bnn_type == 'aggregated':
            pred_dir = self.bnn_agg_dir
        else:
            raise ValueError(f"Unknown BNN type: {bnn_type}")

        # Fix naming inconsistency: BNN aggregated files are saved as 'bnn_agg' not 'bnn_aggregated'
        file_prefix = 'bnn_agg' if bnn_type == 'aggregated' else f'bnn_{bnn_type}'
        pred_path = pred_dir / f'{file_prefix}_h{horizon}_predictions.npz'

        if not pred_path.exists():
            self.logger.warning(f"BNN predictions not found: {pred_path}")
            return None, None

        data = np.load(pred_path)
        return (data['train_mean'], data['train_scale']), (data['test_mean'], data['test_scale'])

    def get_base_features(self, horizon: int, train_df: pl.DataFrame) -> List[str]:
        """Get base features (top 10 SHAP + target encoding)."""
        top_features = self.feature_engineer.get_top_features(horizon, n_features=10)
        target_cols = ['sub_category_te', 'sub_code_te']
        all_features = top_features + target_cols
        return [f for f in all_features if f in train_df.columns]

    def get_lgbm_params_dict(self, horizon: int) -> Dict[str, Any]:
        """Get LightGBM parameters."""
        hp = self.lgbm_params[horizon]

        return {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': hp['num_leaves'],
            'learning_rate': hp['learning_rate'],
            'n_estimators': 1000,
            'max_depth': hp['max_depth'],
            'min_child_samples': hp['min_child_samples'],
            'subsample': 0.85,
            'colsample_bytree': 0.75,
            'reg_lambda': hp['reg_lambda'],
            'random_state': self.seed,
            'verbose': -1
        }

    def _save_metrics(self, horizon: int, model_name: str,
                      train_metrics: MetricResults, valid_metrics: MetricResults,
                      best_iter: int, feature_count: int) -> None:
        """Save metrics to JSON and CSV files."""
        metrics_dict = {
            'horizon': horizon,
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'train': train_metrics.to_dict(),
            'valid': valid_metrics.to_dict(),
            'best_iteration': best_iter,
            'features_used': feature_count
        }

        # Save JSON
        json_path = self.metrics_dir / f'metrics_h{horizon}_{model_name}.json'
        TimeSeriesMetrics.save_metrics_to_json(metrics_dict, json_path)

        # Save CSV (appends to single file)
        csv_path = self.metrics_dir / 'all_metrics.csv'
        TimeSeriesMetrics.save_metrics_to_csv(metrics_dict, csv_path, 'train')
        TimeSeriesMetrics.save_metrics_to_csv(metrics_dict, csv_path, 'valid')

    def train_horizon(self, horizon: int, bnn_type: Optional[str] = None) -> ValidationResults:
        """Train model for single horizon with optional BNN features."""
        bnn_label = f"BNN-{bnn_type}" if bnn_type else "BASELINE"
        model_name = f"lgbm_{bnn_label.lower()}" if bnn_type else "lgbm_baseline"
        self.logger.info(f"Training H={horizon} ({bnn_label})...")

        # Load data
        train_df, test_df = self.load_engineered_data(horizon)

        # Get base features
        base_features = self.get_base_features(horizon, train_df)
        features = base_features.copy()

        # Add BNN features if requested
        if bnn_type:
            bnn_preds = self.load_bnn_predictions(horizon, bnn_type)
            if bnn_preds is not None:
                # Unpack predictions
                (train_mean, train_scale), (test_mean, test_scale) = bnn_preds
                # Add BNN predictions to train data
                train_df = train_df.with_columns([
                    pl.Series('bnn_mean', train_mean),
                    pl.Series('bnn_scale', train_scale)
                ])
                # Add BNN predictions to test data
                test_df = test_df.with_columns([
                    pl.Series('bnn_mean', test_mean),
                    pl.Series('bnn_scale', test_scale)
                ])
                features = base_features + ['bnn_mean', 'bnn_scale']
                self.logger.info(f"Added BNN features: bnn_mean, bnn_scale")
            else:
                self.logger.warning(f"BNN predictions not available for H={horizon}")

        # Time-based split
        train_split = train_df.filter(pl.col('ts_index') <= self.train_split)
        valid_split = train_df.filter(
            (pl.col('ts_index') >= self.valid_start) &
            (pl.col('ts_index') <= self.valid_end)
        )
        train_full = train_df.filter(pl.col('ts_index') <= self.max_ts_train)

        # Prepare data
        X_train = train_split.select(features).to_numpy()
        y_train = train_split['y_target'].to_numpy().ravel()
        w_train = train_split['weight'].to_numpy().ravel()

        X_valid = valid_split.select(features).to_numpy()
        y_valid = valid_split['y_target'].to_numpy().ravel()
        w_valid = valid_split['weight'].to_numpy().ravel()

        X_full = train_full.select(features).to_numpy()
        y_full = train_full['y_target'].to_numpy().ravel()
        w_full = train_full['weight'].to_numpy().ravel()

        X_test = test_df.select(features).to_numpy()

        # Train with early stopping
        params = self.get_lgbm_params_dict(horizon)
        model = lgb.LGBMRegressor(**params)

        model.fit(
            X_train, y_train, sample_weight=w_train,
            eval_set=[(X_valid, y_valid)], eval_sample_weight=[w_valid],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
        )

        best_iter = model.best_iteration_

        # Retrain on full data
        final_model = lgb.LGBMRegressor(**params)
        final_model.set_params(n_estimators=best_iter)
        final_model.fit(X_full, y_full, sample_weight=w_full)

        # Evaluate
        y_train_pred = final_model.predict(X_full)
        train_metrics = TimeSeriesMetrics.evaluate_all(y_full, y_train_pred, w_full, y_train=y_full)

        y_valid_pred = model.predict(X_valid)
        valid_metrics = TimeSeriesMetrics.evaluate_all(y_valid, y_valid_pred, w_valid, y_train=y_full)

        # Test predictions
        y_test_pred = final_model.predict(X_test)

        # Save metrics
        self._save_metrics(horizon, model_name, train_metrics, valid_metrics, best_iter, len(features))

        # Save model
        suffix = f"_{bnn_type}" if bnn_type else "_baseline"
        model_path = self.models_dir / f'lgbm_h{horizon}{suffix}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(final_model, f)

        # Save predictions
        pred_path = self.predictions_dir / f'lgbm_h{horizon}{suffix}_predictions.npz'
        np.savez(pred_path, train_pred=y_train_pred, test_pred=y_test_pred)

        # Print results
        print(f"\n  {'=' * 50}")
        print(f"  HORIZON {horizon} RESULTS ({bnn_label})")
        print(f"  {'=' * 50}")
        print(f"  Train Weighted RMSE: {train_metrics.weighted_rmse:.6f}")
        print(f"  Train Pearson:       {train_metrics.pearson:.6f}")
        print(f"  Train RMSE:          {train_metrics.rmse:.6f}")
        print(f"  Valid Weighted RMSE: {valid_metrics.weighted_rmse:.6f}")
        print(f"  Valid Pearson:       {valid_metrics.pearson:.6f}")
        print(f"  Valid RMSE:          {valid_metrics.rmse:.6f}")
        print(f"  Best iteration:      {best_iter}")
        print(f"  Features used:       {len(features)}")
        print(f"  Test predictions:    {len(y_test_pred):,}")
        print(f"  Test mean:           {np.mean(y_test_pred):.6f}")
        print(f"  Test std:            {np.std(y_test_pred):.6f}")
        print(f"  {'=' * 50}")

        return ValidationResults(
            horizon=horizon,
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            best_iteration=best_iter,
            feature_count=len(features),
            training_time=0,
            model_path=str(model_path)
        )

    def compare_all_variants(self) -> Dict[int, Dict[str, ValidationResults]]:
        """Compare baseline, BNN-SHAP10, and BNN-Aggregated."""
        self.logger.info("Comparing LGBM variants...")

        results = {}

        for horizon in self.horizons:
            print(f"\n{'=' * 80}")
            print(f"HORIZON: {horizon}")
            print(f"{'=' * 80}")

            # Baseline (no BNN)
            baseline = self.train_horizon(horizon, bnn_type=None)

            results[horizon] = {'baseline': baseline}

            # BNN SHAP-10
            bnn_shap10_preds = self.load_bnn_predictions(horizon, 'shap10')
            if bnn_shap10_preds is not None:
                with_bnn_shap10 = self.train_horizon(horizon, bnn_type='shap10')
                results[horizon]['bnn_shap10'] = with_bnn_shap10

            # BNN Aggregated
            bnn_agg_preds = self.load_bnn_predictions(horizon, 'aggregated')
            if bnn_agg_preds is not None:
                with_bnn_agg = self.train_horizon(horizon, bnn_type='aggregated')
                results[horizon]['bnn_aggregated'] = with_bnn_agg

        self._print_comparison(results)

        return results

    def _print_comparison(self, results: Dict[int, Dict[str, ValidationResults]]) -> None:
        """Print comparison between all variants."""
        print(f"\n{'=' * 100}")
        print("COMPARISON: BASELINE vs BNN-SHAP10 vs BNN-AGGREGATED")
        print(f"{'=' * 100}")

        print(f"\n{'Horizon':<8} {'Metric':<15} {'Baseline':<15} {'BNN-SHAP10':<15} {'BNN-Agg':<15} {'Best':<10}")
        print("-" * 85)

        for horizon in self.horizons:
            baseline = results[horizon]['baseline']
            bnn_shap10 = results[horizon].get('bnn_shap10')
            bnn_agg = results[horizon].get('bnn_aggregated')

            # Weighted RMSE (higher is better)
            b_shap10_str = f"{bnn_shap10.valid_metrics.weighted_rmse:.6f}" if bnn_shap10 else "N/A"
            b_agg_str = f"{bnn_agg.valid_metrics.weighted_rmse:.6f}" if bnn_agg else "N/A"

            scores = {'Baseline': baseline.valid_metrics.weighted_rmse}
            if bnn_shap10:
                scores['BNN-SHAP10'] = bnn_shap10.valid_metrics.weighted_rmse
            if bnn_agg:
                scores['BNN-Agg'] = bnn_agg.valid_metrics.weighted_rmse

            best = max(scores, key=scores.get)

            print(
                f"{horizon:<8} {'Weighted RMSE':<15} {baseline.valid_metrics.weighted_rmse:<15.6f} {b_shap10_str:<15} {b_agg_str:<15} {best}")

            # Pearson (higher is better)
            p_shap10_str = f"{bnn_shap10.valid_metrics.pearson:.6f}" if bnn_shap10 else "N/A"
            p_agg_str = f"{bnn_agg.valid_metrics.pearson:.6f}" if bnn_agg else "N/A"

            scores_p = {'Baseline': baseline.valid_metrics.pearson}
            if bnn_shap10:
                scores_p['BNN-SHAP10'] = bnn_shap10.valid_metrics.pearson
            if bnn_agg:
                scores_p['BNN-Agg'] = bnn_agg.valid_metrics.pearson

            best_p = max(scores_p, key=scores_p.get)

            print(
                f"{horizon:<8} {'Pearson':<15} {baseline.valid_metrics.pearson:<15.6f} {p_shap10_str:<15} {p_agg_str:<15} {best_p}")
            print("-" * 85)

    def generate_final_submission(self, results: Dict[int, Dict[str, ValidationResults]],
                                  variant: str = 'baseline') -> Path:
        """Generate final submission for specified variant."""
        all_ids = []
        all_preds = []

        for horizon in self.horizons:
            # Load test data
            _, test_df = self.load_engineered_data(horizon)
            ids = test_df['id'].to_numpy().ravel()
            all_ids.extend(ids)

            # Get predictions
            suffix = f"_{variant}" if variant != 'baseline' else "_baseline"
            pred_path = self.predictions_dir / f'lgbm_h{horizon}{suffix}_predictions.npz'

            if pred_path.exists():
                data = np.load(pred_path)
                all_preds.extend(data['test_pred'])
            else:
                self.logger.warning(f"Predictions not found for H={horizon}")
                all_preds.extend(np.zeros(len(ids)))

        submission_df = pl.DataFrame({'id': all_ids, 'prediction': all_preds})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{variant}" if variant != 'baseline' else "_baseline"
        submission_path = self.predictions_dir / f'lgbm{suffix}_submission_{timestamp}.csv'
        submission_df.write_csv(submission_path)

        self.logger.info(f"Submission saved: {submission_path}")
        self.logger.info(f"Shape: {submission_df.shape}")
        self.logger.info(f"Prediction range: [{np.min(all_preds):.4f}, {np.max(all_preds):.4f}]")

        return submission_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = LGBMWithBNN()
    results = model.compare_all_variants()

    # Generate submissions for all variants
    model.generate_final_submission(results, variant='baseline')
    model.generate_final_submission(results, variant='bnn_shap10')
    model.generate_final_submission(results, variant='bnn_aggregated')

    print("\n" + "=" * 60)
    print("LGBM with BNN comparison complete")
    print(f"✅ Metrics saved to: {model.metrics_dir}")
    print("=" * 60)