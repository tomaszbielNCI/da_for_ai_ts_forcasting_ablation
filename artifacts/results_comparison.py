#!/usr/bin/env python3
"""
LGBM Model with All Features + Top 10 SHAP Features

LightGBM model using:
- ALL original feature_* columns (raw, no engineering)
- PLUS top 10 SHAP features with FULL engineering (rolling, delta, lag)
- Target encoding and time features

Saves metrics to JSON and CSV files.
"""

import polars as pl
import numpy as np
import lightgbm as lgb
import logging
import time
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Add src to path and import metrics
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


class LGBM_All_Plus_SHAP:
    """
    LightGBM model using:
    - ALL original feature_* (raw, NO engineering)
    - PLUS top 10 SHAP features with FULL engineering (rolling, delta, lag)
    - Target encoding (sub_category_te, sub_code_te)
    - Time features
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LGBM All Plus SHAP model."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Paths
        project_root = Path(__file__).parent.parent.parent
        self.cleaned_dir = project_root / 'data/cleaned'
        self.processed_dir = project_root / 'data/processed/top_10'
        self.models_dir = project_root / 'results/models/lgbm_all_plus_shap'
        self.predictions_dir = project_root / 'results/predictions/lgbm_all_plus_shap'
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

        # Feature engineer for SHAP top features
        self.feature_engineer = SHAPFeatureEngineer()

        # Optimized horizon-specific parameters (from notebook experiments)
        self.horizon_params = {
            1: {'num_leaves': 240, 'learning_rate': 0.028, 'min_child_samples': 77,
                'reg_lambda': 3.89, 'max_depth': 10},
            3: {'num_leaves': 50, 'learning_rate': 0.05, 'min_child_samples': 20,
                'reg_lambda': 0.1, 'max_depth': 10},
            10: {'num_leaves': 50, 'learning_rate': 0.05, 'min_child_samples': 20,
                 'reg_lambda': 0.1, 'max_depth': 10},
            25: {'num_leaves': 350, 'learning_rate': 0.021, 'min_child_samples': 250,
                 'reg_lambda': 20.0, 'max_depth': 16}
        }

    def check_shap_results(self) -> bool:
        """Check if SHAP results exist."""
        if self.shap_results_path.exists():
            self.logger.info(f"Found SHAP results: {self.shap_results_path}")
            return True
        else:
            self.logger.error(f"SHAP results not found: {self.shap_results_path}")
            return False

    def load_cleaned_data(self, horizon: int) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load raw cleaned data (no engineering)."""
        train_path = self.cleaned_dir / 'train_clean.parquet'
        test_path = self.cleaned_dir / 'test_clean.parquet'

        train_df = pl.read_parquet(train_path)
        test_df = pl.read_parquet(test_path)

        # Filter by horizon
        train_df = train_df.filter(pl.col('horizon') == horizon)
        test_df = test_df.filter(pl.col('horizon') == horizon)

        return train_df, test_df

    def add_target_encoding(self, train_df: pl.DataFrame, test_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Add target encoding (sub_category_te, sub_code_te)."""
        self.logger.info("Adding target encoding...")

        # Compute on training data only (ts <= 3000) - no leakage
        te_train = train_df.filter(pl.col('ts_index') <= self.train_split)

        # Mean y_target per sub_category
        sub_category_means = te_train.group_by('sub_category').agg(
            pl.col('y_target').mean().alias('sub_category_te')
        )

        # Mean y_target per sub_code
        sub_code_means = te_train.group_by('sub_code').agg(
            pl.col('y_target').mean().alias('sub_code_te')
        )

        # Global mean for missing categories
        global_mean = te_train['y_target'].mean()

        # Add to train
        train_df = train_df.join(sub_category_means, on='sub_category', how='left')
        train_df = train_df.join(sub_code_means, on='sub_code', how='left')
        train_df = train_df.with_columns(
            pl.col('sub_category_te').fill_null(global_mean),
            pl.col('sub_code_te').fill_null(global_mean)
        )

        # Add to test
        test_df = test_df.join(sub_category_means, on='sub_category', how='left')
        test_df = test_df.join(sub_code_means, on='sub_code', how='left')
        test_df = test_df.with_columns(
            pl.col('sub_category_te').fill_null(global_mean),
            pl.col('sub_code_te').fill_null(global_mean)
        )

        return train_df, test_df

    def add_engineered_features_for_top10(self, horizon: int, train_df: pl.DataFrame, test_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, List[str]]:
        """
        Add engineered features (rolling, delta, lag) ONLY for top 10 SHAP features.
        Returns updated dataframes and list of engineered feature names.
        """
        self.logger.info(f"Adding engineered features for top 10 SHAP features (H={horizon})...")

        # Get top 10 SHAP features
        top10_features = self.feature_engineer.get_top_features(horizon, n_features=10)

        # Filter to existing columns
        top10_features = [f for f in top10_features if f in train_df.columns]

        engineered_features = []

        # Rolling mean/std (windows 3,5)
        for window in [3, 5]:
            for feat in top10_features:
                # Rolling mean
                col_mean = f'{feat}_roll_mean_{window}'
                train_df = train_df.with_columns(
                    pl.col(feat).rolling_mean(window_size=window, min_periods=1)
                    .over(['code', 'sub_code', 'sub_category', 'horizon'])
                    .alias(col_mean)
                ).fill_null(0)
                test_df = test_df.with_columns(
                    pl.col(feat).rolling_mean(window_size=window, min_periods=1)
                    .over(['code', 'sub_code', 'sub_category', 'horizon'])
                    .alias(col_mean)
                ).fill_null(0)
                engineered_features.append(col_mean)

                # Rolling std
                col_std = f'{feat}_roll_std_{window}'
                train_df = train_df.with_columns(
                    pl.col(feat).rolling_std(window_size=window, min_periods=1)
                    .over(['code', 'sub_code', 'sub_category', 'horizon'])
                    .alias(col_std)
                ).fill_null(0)
                test_df = test_df.with_columns(
                    pl.col(feat).rolling_std(window_size=window, min_periods=1)
                    .over(['code', 'sub_code', 'sub_category', 'horizon'])
                    .alias(col_std)
                ).fill_null(0)
                engineered_features.append(col_std)

        # Delta features
        for feat in top10_features:
            col_delta = f'{feat}_delta'
            train_df = train_df.with_columns(
                (pl.col(feat) - pl.col(feat).shift(1).over(['code', 'sub_code', 'sub_category', 'horizon']))
                .alias(col_delta)
            ).fill_null(0)
            test_df = test_df.with_columns(
                (pl.col(feat) - pl.col(feat).shift(1).over(['code', 'sub_code', 'sub_category', 'horizon']))
                .alias(col_delta)
            ).fill_null(0)
            engineered_features.append(col_delta)

        # Lag features (1,2,3)
        for lag in [1, 2, 3]:
            for feat in top10_features:
                col_lag = f'{feat}_lag_{lag}'
                train_df = train_df.with_columns(
                    pl.col(feat).shift(lag).over(['code', 'sub_code', 'sub_category', 'horizon'])
                    .alias(col_lag)
                ).fill_null(0)
                test_df = test_df.with_columns(
                    pl.col(feat).shift(lag).over(['code', 'sub_code', 'sub_category', 'horizon'])
                    .alias(col_lag)
                ).fill_null(0)
                engineered_features.append(col_lag)

        self.logger.info(f"Added {len(engineered_features)} engineered features for top 10 SHAP features")
        return train_df, test_df, engineered_features

    def add_time_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add time features."""
        df = df.with_columns([
            (pl.col('ts_index') % 200).alias('time_mod_200'),
            (pl.col('ts_index') % 50).alias('time_mod_50'),
            ((pl.col('ts_index') / 365.0 * np.pi * 2).sin()).alias('sin_year'),
            ((pl.col('ts_index') / 30.0 * np.pi * 2).sin()).alias('sin_month')
        ])
        return df

    def get_feature_columns(self, train_df: pl.DataFrame, test_df: pl.DataFrame,
                            engineered_features: List[str]) -> List[str]:
        """
        Get all feature columns:
        - ALL original feature_* (raw, no engineering)
        - Engineered features from top 10 SHAP
        - Target encoding
        - Time features
        """
        # All raw feature_* columns
        raw_features = [c for c in train_df.columns if c.startswith('feature_')]

        # Target encoding columns
        target_cols = ['sub_category_te', 'sub_code_te']

        # Time features
        time_features = ['time_mod_200', 'time_mod_50', 'sin_year', 'sin_month']

        # Combine all
        all_features = raw_features + engineered_features + target_cols + time_features

        # Remove duplicates and ensure they exist
        all_features = list(dict.fromkeys([f for f in all_features if f in train_df.columns]))

        self.logger.info(f"Total features: {len(all_features)}")
        self.logger.info(f"  - Raw features: {len(raw_features)}")
        self.logger.info(f"  - Engineered (top 10 SHAP): {len(engineered_features)}")
        self.logger.info(f"  - Target encoding: {len(target_cols)}")
        self.logger.info(f"  - Time features: {len(time_features)}")

        return all_features

    def get_lgbm_params(self, horizon: int) -> Dict[str, Any]:
        """Get LightGBM parameters for specific horizon."""
        hp = self.horizon_params[horizon]

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

    def _save_metrics(self, horizon: int, train_metrics: MetricResults,
                      valid_metrics: MetricResults, best_iter: int,
                      feature_count: int) -> None:
        """Save metrics to JSON and CSV files."""
        metrics_dict = {
            'horizon': horizon,
            'model': 'lgbm_all_plus_shap',
            'timestamp': datetime.now().isoformat(),
            'train': train_metrics.to_dict(),
            'valid': valid_metrics.to_dict(),
            'best_iteration': best_iter,
            'features_used': feature_count
        }

        # Save JSON
        json_path = self.metrics_dir / f'metrics_h{horizon}_lgbm_all_plus_shap.json'
        TimeSeriesMetrics.save_metrics_to_json(metrics_dict, json_path)

        # Save CSV (appends to single file)
        csv_path = self.metrics_dir / 'all_metrics.csv'
        TimeSeriesMetrics.save_metrics_to_csv(metrics_dict, csv_path, 'train')
        TimeSeriesMetrics.save_metrics_to_csv(metrics_dict, csv_path, 'valid')

    def train_horizon(self, horizon: int) -> ValidationResults:
        """Train model for single horizon."""
        self.logger.info(f"Training H={horizon}...")

        # Step 1: Load raw cleaned data
        train_df, test_df = self.load_cleaned_data(horizon)

        # Step 2: Add target encoding
        train_df, test_df = self.add_target_encoding(train_df, test_df)

        # Step 3: Add time features
        train_df = self.add_time_features(train_df)
        test_df = self.add_time_features(test_df)

        # Step 4: Add engineered features ONLY for top 10 SHAP
        train_df, test_df, engineered_features = self.add_engineered_features_for_top10(horizon, train_df, test_df)

        # Step 5: Get all feature columns
        feature_cols = self.get_feature_columns(train_df, test_df, engineered_features)

        # Time-based split (from notebook)
        train_split = train_df.filter(pl.col('ts_index') <= self.train_split)
        valid_split = train_df.filter(
            (pl.col('ts_index') >= self.valid_start) &
            (pl.col('ts_index') <= self.valid_end)
        )
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

        # Train with early stopping
        params = self.get_lgbm_params(horizon)
        model = lgb.LGBMRegressor(**params)

        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_valid, y_valid)],
            eval_sample_weight=[w_valid],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
        )

        best_iter = model.best_iteration_
        self.logger.info(f"Best iteration: {best_iter}")

        # Retrain on full data with best iterations
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
        self._save_metrics(horizon, train_metrics, valid_metrics, best_iter, len(feature_cols))

        # Save model
        model_path = self.models_dir / f'lgbm_all_plus_shap_h{horizon}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(final_model, f)

        # Save predictions
        pred_path = self.predictions_dir / f'lgbm_all_plus_shap_h{horizon}_predictions.npz'
        np.savez(pred_path, train_pred=y_train_pred, test_pred=y_test_pred)

        # Create submission
        self._save_submission(horizon, test_df, y_test_pred)

        # Print results
        print(f"\n  {'=' * 50}")
        print(f"  HORIZON {horizon} RESULTS (ALL RAW + SHAP-10 ENGINEERED)")
        print(f"  {'=' * 50}")
        print(f"  Train Weighted RMSE: {train_metrics.weighted_rmse:.6f}")
        print(f"  Train Pearson:       {train_metrics.pearson:.6f}")
        print(f"  Train RMSE:          {train_metrics.rmse:.6f}")
        print(f"  Valid Weighted RMSE: {valid_metrics.weighted_rmse:.6f}")
        print(f"  Valid Pearson:       {valid_metrics.pearson:.6f}")
        print(f"  Valid RMSE:          {valid_metrics.rmse:.6f}")
        print(f"  Best iteration:      {best_iter}")
        print(f"  Features used:       {len(feature_cols)}")
        print(f"    - Raw features: {len([c for c in feature_cols if c.startswith('feature_') and not any(x in c for x in ['roll', 'delta', 'lag'])])}")
        print(f"    - Engineered: {len([c for c in feature_cols if any(x in c for x in ['roll', 'delta', 'lag'])])}")
        print(f"  Test predictions:    {len(y_test_pred):,}")
        print(f"  Test mean:           {np.mean(y_test_pred):.6f}")
        print(f"  Test std:            {np.std(y_test_pred):.6f}")
        print(f"  {'=' * 50}")

        return ValidationResults(
            horizon=horizon,
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            best_iteration=best_iter,
            feature_count=len(feature_cols),
            training_time=0,
            model_path=str(model_path)
        )

    def train_all_horizons(self) -> Dict[int, ValidationResults]:
        """Train models for all horizons."""
        start_time = time.time()
        self.logger.info("Starting LGBM All+SHAP training...")

        # Check SHAP results exist
        if not self.check_shap_results():
            raise FileNotFoundError("SHAP results not found. Run SHAP analysis first.")

        results = {}
        for horizon in self.horizons:
            print(f"\n{'=' * 60}")
            print(f"HORIZON: {horizon}")
            print(f"{'=' * 60}")

            results[horizon] = self.train_horizon(horizon)

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
        submission_path = self.predictions_dir / f'lgbm_all_plus_shap_h{horizon}_submission_{timestamp}.csv'
        submission_df.write_csv(submission_path)
        self.logger.info(f"Submission saved: {submission_path}")

    def generate_final_submission(self, results: Dict[int, ValidationResults]) -> Path:
        """Generate final submission combining all horizons."""
        all_ids = []
        all_preds = []

        for horizon in self.horizons:
            # Load predictions from saved file
            pred_path = self.predictions_dir / f'lgbm_all_plus_shap_h{horizon}_predictions.npz'
            data = np.load(pred_path)
            predictions = data['test_pred']

            # Load test data to get IDs
            test_path = self.cleaned_dir / 'test_clean.parquet'
            test_df = pl.read_parquet(test_path)
            test_h = test_df.filter(pl.col('horizon') == horizon)
            ids = test_h['id'].to_numpy().ravel()

            all_ids.extend(ids)
            all_preds.extend(predictions)

        submission_df = pl.DataFrame({'id': all_ids, 'prediction': all_preds})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = self.predictions_dir / f'lgbm_all_plus_shap_submission_{timestamp}.csv'
        submission_df.write_csv(submission_path)

        self.logger.info(f"Final submission saved: {submission_path}")
        self.logger.info(f"Shape: {submission_df.shape}")
        self.logger.info(f"Prediction range: [{np.min(all_preds):.4f}, {np.max(all_preds):.4f}]")

        return submission_path

    def _print_summary(self, results: Dict[int, ValidationResults], total_time: float) -> None:
        """Print comprehensive summary."""
        print(f"\n{'=' * 80}")
        print("LGBM ALL+SHAP - FINAL SUMMARY")
        print(f"{'=' * 80}")

        print(f"\n{'Horizon':<8} {'Train WRMSE':<14} {'Valid WRMSE':<14} {'Train Pearson':<14} {'Valid Pearson':<14} {'Best Iter':<10} {'Features':<10}")
        print("-" * 90)

        for h in self.horizons:
            r = results[h]
            print(f"{h:<8} {r.train_metrics.weighted_rmse:<14.6f} {r.valid_metrics.weighted_rmse:<14.6f} "
                  f"{r.train_metrics.pearson:<14.6f} {r.valid_metrics.pearson:<14.6f} "
                  f"{r.best_iteration:<10} {r.feature_count:<10}")

        print(f"\n{'=' * 60}")
        print(f"Total training time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
        print(f"Models saved to: {self.models_dir}")
        print(f"Predictions saved to: {self.predictions_dir}")
        print(f"Metrics saved to: {self.metrics_dir}")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = LGBM_All_Plus_SHAP()
    results = model.train_all_horizons()
    model.generate_final_submission(results)