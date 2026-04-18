#!/usr/bin/env python3
"""
SHAP-Based Feature Engineering

Creates engineered features based on SHAP analysis results:
- Rolling features (window-based statistics)
- Delta features (change-based features)
- Lag features (temporal dependencies)
- Per-horizon feature selection using SHAP results

Uses existing SHAP results from results/shap/data/horizon_specific_top_features.json
"""

import polars as pl
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    rolling_windows: List[int] = field(default_factory=lambda: [3, 5])
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3])


class SHAPFeatureEngineer:
    """
    SHAP-based feature engineering for time series forecasting.

    Creates engineered features for top N SHAP features per horizon.
    Uses existing SHAP results from JSON file.
    """

    # Time features (from notebook)
    TIME_FEATURES = ['time_mod_50', 'time_mod_200', 'sin_year', 'sin_month']

    # Target encoding columns
    TARGET_ENCODING = ['sub_category_te', 'sub_code_te']

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature engineer."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Paths
        project_root = Path(__file__).parent.parent.parent
        self.processed_dir = project_root / 'data/processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # SHAP results path
        self.shap_results_path = project_root / 'results/shap/data/horizon_specific_top_features.json'

        # Horizons
        self.horizons = self.config.get('horizons', [1, 3, 10, 25])

        # Feature config
        self.fc = FeatureConfig(**self.config.get('feature_config', {}))

        # Load SHAP results
        self.shap_results = self._load_shap_results()

    def _load_shap_results(self) -> Dict[int, Dict[str, Any]]:
        """Load SHAP results from JSON file."""
        if not self.shap_results_path.exists():
            raise FileNotFoundError(
                f"SHAP results not found at {self.shap_results_path}\n"
                "Please run SHAP analysis first using src/models/shap_analyzer.py"
            )

        with open(self.shap_results_path, 'r') as f:
            shap_data = json.load(f)

        self.logger.info(f"Loaded SHAP results for horizons: {list(shap_data.keys())}")
        return {int(k): v for k, v in shap_data.items()}

    def get_top_features(self, horizon: int, n_features: int = 10) -> List[str]:
        """
        Get top N SHAP features for a specific horizon.

        Args:
            horizon: Forecast horizon (1, 3, 10, 25)
            n_features: Number of top features (10 or 20)
        """
        if horizon not in self.shap_results:
            raise ValueError(f"SHAP results not available for horizon {horizon}")

        horizon_data = self.shap_results[horizon]
        features = horizon_data.get('features', [])

        if len(features) < n_features:
            self.logger.warning(f"Only {len(features)} features available for H={horizon}, requested {n_features}")
            return features

        return features[:n_features]

    def _get_engineered_features(self, base_features: List[str]) -> List[str]:
        """Generate engineered feature names for given base features."""
        engineered = []
        for feat in base_features:
            for w in self.fc.rolling_windows:
                engineered.append(f'{feat}_roll_mean_{w}')
                engineered.append(f'{feat}_roll_std_{w}')
            engineered.append(f'{feat}_delta')
            for lag in self.fc.lag_periods:
                engineered.append(f'{feat}_lag_{lag}')
        return engineered

    def _create_rolling_features(self, df: pl.DataFrame, features: List[str]) -> pl.DataFrame:
        """Create rolling mean/std features with group-based processing."""
        group_cols = ['code', 'sub_code', 'sub_category', 'horizon']

        for window in self.fc.rolling_windows:
            for feat in features:
                if feat not in df.columns:
                    continue

                df = df.with_columns(
                    pl.col(feat).rolling_mean(window_size=window, min_periods=1)
                    .over(group_cols).alias(f'{feat}_roll_mean_{window}')
                ).fill_null(0)

                df = df.with_columns(
                    pl.col(feat).rolling_std(window_size=window, min_periods=1)
                    .over(group_cols).alias(f'{feat}_roll_std_{window}')
                ).fill_null(0)

        return df

    def _create_delta_features(self, df: pl.DataFrame, features: List[str]) -> pl.DataFrame:
        """Create delta (change) features."""
        group_cols = ['code', 'sub_code', 'sub_category', 'horizon']

        for feat in features:
            if feat not in df.columns:
                continue
            df = df.with_columns(
                (pl.col(feat) - pl.col(feat).shift(1).over(group_cols))
                .alias(f'{feat}_delta')
            ).fill_null(0)

        return df

    def _create_lag_features(self, df: pl.DataFrame, features: List[str]) -> pl.DataFrame:
        """Create lag features."""
        group_cols = ['code', 'sub_code', 'sub_category', 'horizon']

        for lag in self.fc.lag_periods:
            for feat in features:
                if feat not in df.columns:
                    continue
                df = df.with_columns(
                    pl.col(feat).shift(lag).over(group_cols)
                    .alias(f'{feat}_lag_{lag}')
                ).fill_null(0)

        return df

    def engineer_features_for_horizon(
            self,
            train_df: pl.DataFrame,
            test_df: pl.DataFrame,
            horizon: int,
            n_features: int = 10
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Engineer features for a specific horizon.

        Args:
            train_df: Training data
            test_df: Test data
            horizon: Forecast horizon
            n_features: Number of top SHAP features (10 or 20)

        Returns:
            Tuple of (engineered_train, engineered_test)
        """
        self.logger.info(f"Engineering features for H={horizon} (top {n_features} SHAP)")

        # Filter by horizon
        train_h = train_df.filter(pl.col('horizon') == horizon)
        test_h = test_df.filter(pl.col('horizon') == horizon)

        # Get base features from SHAP results
        base_features = self.get_top_features(horizon, n_features)
        available_base = [f for f in base_features if f in train_h.columns]

        self.logger.info(f"Base features: {len(available_base)}/{len(base_features)}")

        # Drop all non-SHAP feature_* columns
        all_feature_cols = [c for c in train_h.columns if c.startswith('feature_')]
        to_drop = [c for c in all_feature_cols if c not in available_base]

        if to_drop:
            train_h = train_h.drop(to_drop)
            test_h = test_h.drop(to_drop)
            self.logger.info(f"Dropped {len(to_drop)} non-SHAP features, kept {len(available_base)}")

        # Create engineered features
        train_h = self._create_rolling_features(train_h, available_base)
        test_h = self._create_rolling_features(test_h, available_base)

        train_h = self._create_delta_features(train_h, available_base)
        test_h = self._create_delta_features(test_h, available_base)

        train_h = self._create_lag_features(train_h, available_base)
        test_h = self._create_lag_features(test_h, available_base)

        # Ensure time features and target encoding exist
        for col in self.TIME_FEATURES + self.TARGET_ENCODING:
            if col in train_df.columns and col not in train_h.columns:
                train_h = train_h.with_columns(pl.col(col))
                test_h = test_h.with_columns(pl.col(col))

        self.logger.info(f"Final features: {len(train_h.columns)} columns")

        return train_h, test_h

    def process_all_horizons(
            self,
            train_df: pl.DataFrame,
            test_df: pl.DataFrame,
            n_features: int = 10
    ) -> Dict[int, Tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Process all horizons with specified number of features.

        Args:
            train_df: Training data
            test_df: Test data
            n_features: Number of top SHAP features (10 or 20)

        Returns:
            Dict mapping horizon -> (engineered_train, engineered_test)
        """
        self.logger.info(f"Processing all horizons with top {n_features} features")

        results = {}
        for horizon in self.horizons:
            train_eng, test_eng = self.engineer_features_for_horizon(
                train_df, test_df, horizon, n_features
            )
            results[horizon] = (train_eng, test_eng)

        return results

    def save_engineered_data(
            self,
            data: Dict[int, Tuple[pl.DataFrame, pl.DataFrame]],
            mode: str = 'top_10'
    ) -> None:
        """Save engineered data to parquet files."""
        mode_dir = self.processed_dir / mode
        mode_dir.mkdir(exist_ok=True)

        for horizon, (train_df, test_df) in data.items():
            train_path = mode_dir / f'train_h{horizon}_engineered.parquet'
            test_path = mode_dir / f'test_h{horizon}_engineered.parquet'

            train_df.write_parquet(train_path)
            test_df.write_parquet(test_path)

            self.logger.info(f"Saved {mode} H={horizon}: train {train_df.shape}, test {test_df.shape}")

    def run_full_pipeline(
            self,
            train_df: pl.DataFrame,
            test_df: pl.DataFrame,
            n_features: int = 10
    ) -> Dict[int, Tuple[pl.DataFrame, pl.DataFrame]]:
        """Run complete feature engineering pipeline."""
        self.logger.info("Starting feature engineering pipeline...")

        results = self.process_all_horizons(train_df, test_df, n_features)
        mode = 'top_10' if n_features == 10 else 'top_20'
        self.save_engineered_data(results, mode)

        self._print_summary(results, mode)
        return results

    def _print_summary(self, data: Dict[int, Tuple[pl.DataFrame, pl.DataFrame]], mode: str) -> None:
        """Print summary of engineered data."""
        print(f"\n{'=' * 60}")
        print(f"FEATURE ENGINEERING SUMMARY - {mode.upper()}")
        print(f"{'=' * 60}")

        for horizon, (train_df, test_df) in data.items():
            n_features = len([c for c in train_df.columns if c.startswith(('feature_', 'time_', 'sub_'))])
            print(f"H={horizon}: Train {train_df.shape}, Test {test_df.shape}, Features={n_features}")

        print(f"\nData saved to: {self.processed_dir / mode}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load cleaned data using absolute paths
    project_root = Path(__file__).parent.parent.parent
    train_df = pl.read_parquet(project_root / 'data/cleaned/train_clean.parquet')
    test_df = pl.read_parquet(project_root / 'data/cleaned/test_clean.parquet')

    # Run for top 10 features
    engineer = SHAPFeatureEngineer()
    results_top10 = engineer.run_full_pipeline(train_df, test_df, n_features=10)

    print("\n✅ Feature engineering complete!")
    print("Now run: cd src && python run_shap_10_direct.py")