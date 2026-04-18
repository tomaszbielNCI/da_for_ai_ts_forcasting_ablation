#!/usr/bin/env python3
"""
SHAP-Based Feature Engineering

Creates engineered features based on SHAP analysis results:
- Rolling features (window-based statistics)
- Delta features (change-based features)
- Lag features (temporal dependencies)
- Per-horizon feature selection using SHAP results

Modular design for use with multiple model types.
"""

import polars as pl
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    rolling_windows: List[int] = None
    lag_periods: List[int] = None
    delta_periods: List[int] = None
    
    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [3, 5]  # Notebook approach
        if self.lag_periods is None:
            self.lag_periods = [1, 2, 3]  # Notebook approach
        if self.delta_periods is None:
            self.delta_periods = [1]  # Notebook approach

class SHAPFeatureEngineer:
    """
    SHAP-based feature engineering for time series forecasting.
    
    Features:
    - Rolling statistics (mean, std, min, max)
    - Delta features (change over time)
    - Lag features (temporal dependencies)
    - SHAP-based feature selection
    - Per-horizon optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature engineer with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Feature engineering parameters
        self.feature_config = FeatureConfig(**self.config.get('feature_config', {}))
        
        # SHAP results path
        self.shap_results_path = Path('results/shap/data/horizon_specific_top_features.json')
        
        # Output paths
        self.processed_dir = Path('data/processed')
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load SHAP results
        self.shap_results = self._load_shap_results()
        
        # Horizons
        self.horizons = self.config.get('horizons', [1, 3, 10, 25])
    
    def _load_shap_results(self) -> Dict[int, Dict[str, Any]]:
        """Load SHAP results from JSON file."""
        if not self.shap_results_path.exists():
            raise FileNotFoundError(f"SHAP results not found at {self.shap_results_path}")
        
        with open(self.shap_results_path, 'r') as f:
            shap_data = json.load(f)
        
        # Convert string keys to int
        return {int(k): v for k, v in shap_data.items()}
    
    def get_top_features(self, horizon: int, n_features: int = 10) -> List[str]:
        """Get top N features for a specific horizon."""
        if horizon not in self.shap_results:
            raise ValueError(f"SHAP results not available for horizon {horizon}")
        
        horizon_data = self.shap_results[horizon]
        features = horizon_data['features'][:n_features]
        
        self.logger.info(f"Top {n_features} features for H={horizon}: {features}")
        return features
    
    def create_rolling_features(self, df: pl.DataFrame, feature_cols: List[str]) -> pl.DataFrame:
        """Create rolling window features with group-based processing."""
        self.logger.info(f"Creating group-based rolling features for {len(feature_cols)} features...")
        
        # Use notebook's group-based approach with windows [3, 5]
        windows = [3, 5]
        
        for window in windows:
            for feature in feature_cols:
                # Group-based rolling mean and std
                df = df.with_columns(
                    pl.col(feature).rolling_mean(window_size=window, min_periods=1)
                    .over(['code', 'sub_code', 'sub_category', 'horizon'])
                    .alias(f'{feature}_roll_mean_{window}')
                ).fill_null(0)
                
                df = df.with_columns(
                    pl.col(feature).rolling_std(window_size=window, min_periods=1)
                    .over(['code', 'sub_code', 'sub_category', 'horizon'])
                    .alias(f'{feature}_roll_std_{window}')
                ).fill_null(0)
        
        return df
    
    def create_delta_features(self, df: pl.DataFrame, feature_cols: List[str]) -> pl.DataFrame:
        """Create delta (change) features with group-based processing."""
        self.logger.info(f"Creating group-based delta features for {len(feature_cols)} features...")
        
        for feature in feature_cols:
            # Group-based delta (period 1) - notebook approach
            df = df.with_columns(
                (pl.col(feature) - pl.col(feature).shift(1)
                 .over(['code', 'sub_code', 'sub_category', 'horizon']))
                .alias(f'{feature}_delta')
            ).fill_null(0)
        
        return df
    
    def create_lag_features(self, df: pl.DataFrame, feature_cols: List[str]) -> pl.DataFrame:
        """Create lag features with group-based processing."""
        self.logger.info(f"Creating group-based lag features for {len(feature_cols)} features...")
        
        # Use notebook's lags [1, 2, 3]
        lags = [1, 2, 3]
        
        for lag in lags:
            for feature in feature_cols:
                # Group-based lag features
                df = df.with_columns(
                    pl.col(feature).shift(lag)
                     .over(['code', 'sub_code', 'sub_category', 'horizon'])
                     .alias(f'{feature}_lag_{lag}')
                ).fill_null(0)
        
        return df
    
    def engineer_features_for_horizon(self, 
                                  train_df: pl.DataFrame, 
                                  test_df: pl.DataFrame,
                                  horizon: int,
                                  feature_mode: str = 'top_10') -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Engineer features for a specific horizon using hybrid approach.
        
        Args:
            train_df: Training data
            test_df: Test data
            horizon: Forecast horizon
            feature_mode: 'top_10', 'top_20', or 'all_plus_10'
            
        Returns:
            Tuple of (engineered_train, engineered_test)
        """
        self.logger.info(f"Engineering features for H={horizon}, mode={feature_mode}")
        
        # Determine feature set based on mode using SHAP results
        if feature_mode == 'top_10':
            shap_features = self.get_top_features(horizon, 10)
            base_features = shap_features
        elif feature_mode == 'top_20':
            shap_features = self.get_top_features(horizon, 20)
            base_features = shap_features
        elif feature_mode == 'all_plus_10':
            # All original features + top 10 SHAP features
            all_features = [col for col in train_df.columns if col.startswith('feature_')]
            shap_features = self.get_top_features(horizon, 10)
            base_features = list(set(all_features + shap_features))
        else:
            raise ValueError(f"Unsupported feature mode: {feature_mode}")
        
        # Get feature columns that exist in data
        available_features = [f for f in base_features if f in train_df.columns]
        
        self.logger.info(f"Using {len(available_features)} base features for H={horizon}")
        
        # Filter data by horizon first (notebook approach)
        train_h = train_df.filter(pl.col('horizon') == horizon)
        test_h = test_df.filter(pl.col('horizon') == horizon)
        
        # Create engineered features using group-based processing
        # Rolling features (notebook: windows [3,5])
        train_h = self.create_rolling_features(train_h, available_features)
        test_h = self.create_rolling_features(test_h, available_features)
        
        # Delta features (notebook: period 1)
        train_h = self.create_delta_features(train_h, available_features)
        test_h = self.create_delta_features(test_h, available_features)
        
        # Lag features (notebook: lags [1,2,3])
        train_h = self.create_lag_features(train_h, available_features)
        test_h = self.create_lag_features(test_h, available_features)
        
        # Add time features (notebook approach)
        time_features = ['time_mod_50', 'time_mod_200', 'sin_year', 'sin_month']
        available_time_features = [f for f in time_features if f in train_h.columns]
        
        # Collect all engineered features
        engineered_feature_list = []
        for feat in available_features:
            engineered_feature_list.extend([
                f'{feat}_roll_mean_3', f'{feat}_roll_mean_5',
                f'{feat}_roll_std_3', f'{feat}_roll_std_5',
                f'{feat}_delta',
                f'{feat}_lag_1', f'{feat}_lag_2', f'{feat}_lag_3'
            ])
        
        all_final_features = available_features + engineered_feature_list + available_time_features
        all_final_features = [f for f in all_final_features if f in train_h.columns]
        all_final_features = list(dict.fromkeys(all_final_features))  # Remove duplicates
        
        self.logger.info(f"Final features for H={horizon}: {len(all_final_features)}")
        
        return train_h, test_h
    
    def process_all_horizons(self, 
                          train_df: pl.DataFrame, 
                          test_df: pl.DataFrame,
                          feature_modes: List[str] = None) -> Dict[str, Dict[int, Tuple[pl.DataFrame, pl.DataFrame]]]:
        """
        Process all horizons with different feature modes.
        
        Args:
            train_df: Training data
            test_df: Test data
            feature_modes: List of feature modes to process
            
        Returns:
            Dict with results per mode and horizon
        """
        if feature_modes is None:
            feature_modes = ['top_10', 'top_20', 'all_plus_10']
        
        results = {}
        
        for mode in feature_modes:
            self.logger.info(f"Processing feature mode: {mode}")
            results[mode] = {}
            
            for horizon in self.horizons:
                # Filter data by horizon
                train_h = train_df.filter(pl.col('horizon') == horizon)
                test_h = test_df.filter(pl.col('horizon') == horizon)
                
                # Engineer features
                engineered_train, engineered_test = self.engineer_features_for_horizon(
                    train_h, test_h, horizon, mode
                )
                
                results[mode][horizon] = (engineered_train, engineered_test)
        
        return results
    
    def save_engineered_data(self, 
                          engineered_data: Dict[str, Dict[int, Tuple[pl.DataFrame, pl.DataFrame]]]) -> None:
        """Save engineered data to files."""
        self.logger.info("Saving engineered features...")
        
        for mode, horizon_data in engineered_data.items():
            mode_dir = self.processed_dir / mode
            mode_dir.mkdir(exist_ok=True)
            
            for horizon, (train_df, test_df) in horizon_data.items():
                # Save train data
                train_path = mode_dir / f'train_h{horizon}_engineered.parquet'
                train_df.write_parquet(train_path)
                
                # Save test data
                test_path = mode_dir / f'test_h{horizon}_engineered.parquet'
                test_df.write_parquet(test_path)
                
                self.logger.info(f"Saved {mode} H={horizon}: train {train_df.shape}, test {test_df.shape}")
    
    def run_full_feature_engineering(self, 
                                  train_df: pl.DataFrame, 
                                  test_df: pl.DataFrame,
                                  feature_modes: List[str] = None) -> Dict[str, Dict[int, Tuple[pl.DataFrame, pl.DataFrame]]]:
        """
        Run complete feature engineering pipeline.
        
        Args:
            train_df: Training data
            test_df: Test data
            feature_modes: Feature modes to process
            
        Returns:
            Engineered data dictionary
        """
        self.logger.info("Starting full feature engineering pipeline...")
        
        # Process all horizons and modes
        engineered_data = self.process_all_horizons(train_df, test_df, feature_modes)
        
        # Save results
        self.save_engineered_data(engineered_data)
        
        # Print summary
        self._print_summary(engineered_data)
        
        return engineered_data
    
    def _print_summary(self, engineered_data: Dict[str, Dict[int, Tuple[pl.DataFrame, pl.DataFrame]]]) -> None:
        """Print feature engineering summary."""
        print(f"\n{'='*80}")
        print("FEATURE ENGINEERING SUMMARY")
        print(f"{'='*80}")
        
        for mode, horizon_data in engineered_data.items():
            print(f"\nFeature Mode: {mode}")
            print("-" * 40)
            
            total_train_features = 0
            total_test_features = 0
            
            for horizon, (train_df, test_df) in horizon_data.items():
                train_features = len([col for col in train_df.columns if col.startswith('feature_')])
                test_features = len([col for col in test_df.columns if col.startswith('feature_')])
                
                total_train_features = max(total_train_features, train_features)
                total_test_features = max(total_test_features, test_features)
                
                print(f"  H={horizon}: Train {train_df.shape} ({train_features} features), "
                      f"Test {test_df.shape} ({test_features} features)")
            
            print(f"  Max features per dataset: Train={total_train_features}, Test={total_test_features}")
        
        print(f"\nData saved to: {self.processed_dir}")
        print(f"{'='*80}")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load cleaned data
    import polars as pl
    train_df = pl.read_parquet('data/cleaned/train_clean.parquet')
    test_df = pl.read_parquet('data/cleaned/test_clean.parquet')
    
    # Run feature engineering
    engineer = SHAPFeatureEngineer()
    engineered_data = engineer.run_full_feature_engineering(train_df, test_df)
