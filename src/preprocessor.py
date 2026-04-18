#!/usr/bin/env python3
"""
Data Preprocessor Module

Handles data cleaning, type conversion, and causal imputation.
Saves processed data to data/cleaned/ directory.
"""

import polars as pl
import logging
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ProcessingStats:
    """Statistics for data processing"""
    original_size_mb: float
    processed_size_mb: float
    nan_count_before: int
    nan_count_after: int
    processing_time_seconds: float
    features_processed: int

class DataPreprocessor:
    """
    Data preprocessing with type conversion and causal imputation.
    
    Features:
    - Type conversion for memory optimization
    - Causal imputation using EWMA and forward fill
    - Data validation and sanity checks
    - Statistics tracking and reporting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize preprocessor with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Data paths
        self.data_dir = Path('data')
        self.raw_dir = self.data_dir
        self.cleaned_dir = self.data_dir / 'cleaned'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories
        self.cleaned_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # Column mappings
        self.categorical_cols = ['code', 'sub_code', 'sub_category']
        self.int_cols = ['horizon', 'ts_index']
        self.group_keys = ['code', 'sub_code', 'sub_category', 'horizon']
    
    def load_raw_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load raw train and test data."""
        self.logger.info("Loading raw data...")
        
        train_path = self.raw_dir / 'train.parquet'
        test_path = self.raw_dir / 'test.parquet'
        
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError("Raw data files not found in data/ directory")
        
        train_df = pl.read_parquet(train_path)
        test_df = pl.read_parquet(test_path)
        
        self.logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
        return train_df, test_df
    
    def get_data_statistics(self, df: pl.DataFrame, name: str) -> Dict[str, Any]:
        """Get comprehensive data statistics."""
        feature_cols = [c for c in df.columns if c.startswith('feature_')]
        
        stats = {
            'shape': df.shape,
            'memory_usage_mb': df.estimated_size('mb'),
            'feature_columns': len(feature_cols),
            'null_counts': {col: df[col].null_count() for col in feature_cols},
            'dtypes': {col: str(df[col].dtype) for col in df.columns[:10]},  # First 10 columns
            'ts_index_range': (df['ts_index'].min(), df['ts_index'].max()) if 'ts_index' in df.columns else None
        }
        
        # Calculate total NaN count
        total_nans = sum(df[col].null_count() for col in feature_cols)
        stats['total_nan_count'] = total_nans
        
        self.logger.info(f"{name} statistics: {stats['shape']}, "
                        f"Memory: {stats['memory_usage_mb']:.2f}MB, "
                        f"NaNs: {total_nans}")
        
        return stats
    
    def convert_types(self, train_df: pl.DataFrame, test_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Convert data types for memory optimization."""
        self.logger.info("Converting data types...")
        
        train_clean = train_df.clone()
        test_clean = test_df.clone()
        
        # Convert categorical columns
        for col in self.categorical_cols:
            if col in train_clean.columns:
                train_clean = train_clean.with_columns(pl.col(col).cast(pl.Categorical))
            if col in test_clean.columns:
                test_clean = test_clean.with_columns(pl.col(col).cast(pl.Categorical))
        
        # Convert integer columns
        for col in self.int_cols:
            if col in train_clean.columns:
                train_clean = train_clean.with_columns(pl.col(col).cast(pl.Int16))
            if col in test_clean.columns:
                test_clean = test_clean.with_columns(pl.col(col).cast(pl.Int16))
        
        # Convert feature columns to Float32
        feature_cols = [c for c in train_clean.columns if c.startswith('feature_')]
        for col in feature_cols:
            if train_clean[col].dtype == pl.Float64:
                train_clean = train_clean.with_columns(pl.col(col).cast(pl.Float32))
            if col in test_clean.columns and test_clean[col].dtype == pl.Float64:
                test_clean = test_clean.with_columns(pl.col(col).cast(pl.Float32))
        
        self.logger.info(f"Type conversion complete. Features: {len(feature_cols)}")
        return train_clean, test_clean
    
    def validate_data(self, train_df: pl.DataFrame, test_df: pl.DataFrame) -> bool:
        """Perform data validation and sanity checks."""
        self.logger.info("Performing data validation...")
        
        # Check feature columns consistency
        train_features = [c for c in train_df.columns if c.startswith('feature_')]
        test_features = [c for c in test_df.columns if c.startswith('feature_')]
        
        if len(test_features) != len(train_features):
            raise ValueError("Test data missing features compared to train data")
        
        # Check ts_index ranges
        train_ts_min, train_ts_max = train_df['ts_index'].min(), train_df['ts_index'].max()
        test_ts_min, test_ts_max = test_df['ts_index'].min(), test_df['ts_index'].max()
        
        self.logger.info(f"Train ts_index range: {train_ts_min} - {train_ts_max}")
        self.logger.info(f"Test ts_index range: {test_ts_min} - {test_ts_max}")
        
        # No future data in train
        if train_ts_max > 3601:
            raise ValueError(f"Train contains ts_index > 3601: {train_ts_max}")
        
        # Check temporal continuity
        if test_ts_min <= train_ts_max:
            self.logger.warning(f"Test data starts at {test_ts_min}, train ends at {train_ts_max}")
        
        self.logger.info("Data validation passed")
        return True
    
    def perform_causal_imputation(self, train_df: pl.DataFrame, test_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Perform causal imputation using EWMA and forward fill."""
        self.logger.info("Performing causal imputation...")
        
        train_imputed = train_df.clone()
        test_imputed = test_df.clone()
        
        feature_cols = [c for c in train_imputed.columns if c.startswith('feature_')]
        
        # Count NaNs before imputation
        null_counts_before = {c: train_imputed[c].null_count() for c in feature_cols if train_imputed[c].null_count() > 0}
        total_nans_before = sum(null_counts_before.values())
        
        self.logger.info(f"Features with nulls before imputation: {len(null_counts_before)}")
        self.logger.info(f"Total NaNs before imputation: {total_nans_before}")
        
        # Perform imputation for each feature
        for col in feature_cols:
            if train_imputed[col].null_count() > 0 or (col in test_imputed.columns and test_imputed[col].null_count() > 0):
                
                # Train imputation
                train_imputed = train_imputed.with_columns(
                    pl.when(pl.col(col).is_null())
                    .then(
                        pl.coalesce([
                            pl.col(col).ewm_mean(span=20, adjust=False).over(self.group_keys),
                            pl.col(col).forward_fill().over(self.group_keys),
                            pl.col(col).ewm_mean(span=10, adjust=False).over(self.group_keys),
                            pl.lit(0.0)
                        ])
                    )
                    .otherwise(pl.col(col))
                    .alias(col)
                )
                
                # Test imputation
                if col in test_imputed.columns:
                    test_imputed = test_imputed.with_columns(
                        pl.when(pl.col(col).is_null())
                        .then(
                            pl.coalesce([
                                pl.col(col).ewm_mean(span=20, adjust=False).over(self.group_keys),
                                pl.col(col).forward_fill().over(self.group_keys),
                                pl.col(col).ewm_mean(span=10, adjust=False).over(self.group_keys),
                                pl.lit(0.0)
                            ])
                        )
                        .otherwise(pl.col(col))
                        .alias(col)
                    )
        
        # Final fallback to 0
        for col in feature_cols:
            if train_imputed[col].null_count() > 0:
                train_imputed = train_imputed.with_columns(pl.col(col).fill_null(0).alias(col))
            if col in test_imputed.columns and test_imputed[col].null_count() > 0:
                test_imputed = test_imputed.with_columns(pl.col(col).fill_null(0).alias(col))
        
        # Count NaNs after imputation
        null_counts_after = {c: train_imputed[c].null_count() for c in feature_cols if train_imputed[c].null_count() > 0}
        total_nans_after = sum(null_counts_after.values())
        
        self.logger.info(f"Features with nulls after imputation: {len(null_counts_after)}")
        self.logger.info(f"Total NaNs after imputation: {total_nans_after}")
        
        if total_nans_after > 0:
            self.logger.warning(f"Still have {total_nans_after} NaNs after imputation")
        else:
            self.logger.info("All NaNs successfully imputed")
        
        return train_imputed, test_imputed
    
    def save_cleaned_data(self, train_df: pl.DataFrame, test_df: pl.DataFrame) -> Tuple[Path, Path]:
        """Save cleaned data to files."""
        self.logger.info("Saving cleaned data...")
        
        train_path = self.cleaned_dir / 'train_clean.parquet'
        test_path = self.cleaned_dir / 'test_clean.parquet'
        
        train_df.write_parquet(train_path)
        test_df.write_parquet(test_path)
        
        self.logger.info(f"Saved cleaned data: {train_path}, {test_path}")
        return train_path, test_path
    
    def process_data(self, force_reprocess: bool = False) -> ProcessingStats:
        """
        Complete data processing pipeline.
        
        Args:
            force_reprocess: If True, reprocess even if cleaned data exists
            
        Returns:
            ProcessingStats: Statistics about the processing
        """
        start_time = time.time()
        
        # Check if cleaned data already exists
        train_clean_path = self.cleaned_dir / 'train_clean.parquet'
        test_clean_path = self.cleaned_dir / 'test_clean.parquet'
        
        if not force_reprocess and train_clean_path.exists() and test_clean_path.exists():
            self.logger.info("Cleaned data already exists, loading from files...")
            train_clean = pl.read_parquet(train_clean_path)
            test_clean = pl.read_parquet(test_clean_path)
            
            # Get statistics
            train_stats = self.get_data_statistics(train_clean, "Cleaned Train")
            test_stats = self.get_data_statistics(test_clean, "Cleaned Test")
            
            return ProcessingStats(
                original_size_mb=train_stats['memory_usage_mb'] + test_stats['memory_usage_mb'],
                processed_size_mb=train_stats['memory_usage_mb'] + test_stats['memory_usage_mb'],
                nan_count_before=0,
                nan_count_after=0,
                processing_time_seconds=0,
                features_processed=train_stats['feature_columns']
            )
        
        # Load raw data
        train_raw, test_raw = self.load_raw_data()
        
        # Get original statistics
        train_stats_before = self.get_data_statistics(train_raw, "Raw Train")
        original_size_mb = train_stats_before['memory_usage_mb']
        original_nans = train_stats_before['total_nan_count']
        
        # Type conversion
        train_clean, test_clean = self.convert_types(train_raw, test_raw)
        
        # Data validation
        self.validate_data(train_clean, test_clean)
        
        # Causal imputation
        train_imputed, test_imputed = self.perform_causal_imputation(train_clean, test_clean)
        
        # Get final statistics
        train_stats_after = self.get_data_statistics(train_imputed, "Processed Train")
        processed_size_mb = train_stats_after['memory_usage_mb']
        final_nans = train_stats_after['total_nan_count']
        
        # Save cleaned data
        self.save_cleaned_data(train_imputed, test_imputed)
        
        processing_time = time.time() - start_time
        
        # Create stats object
        stats = ProcessingStats(
            original_size_mb=original_size_mb,
            processed_size_mb=processed_size_mb,
            nan_count_before=original_nans,
            nan_count_after=final_nans,
            processing_time_seconds=processing_time,
            features_processed=train_stats_after['feature_columns']
        )
        
        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("DATA PROCESSING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Original size: {stats.original_size_mb:.2f} MB")
        self.logger.info(f"Processed size: {stats.processed_size_mb:.2f} MB")
        self.logger.info(f"Size reduction: {(1 - stats.processed_size_mb/stats.original_size_mb)*100:.1f}%")
        self.logger.info(f"NaNs before: {stats.nan_count_before:,}")
        self.logger.info(f"NaNs after: {stats.nan_count_after:,}")
        self.logger.info(f"Processing time: {stats.processing_time_seconds:.2f} seconds")
        self.logger.info(f"Features processed: {stats.features_processed}")
        self.logger.info("=" * 60)
        
        return stats
    
    def load_cleaned_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load previously cleaned data."""
        train_path = self.cleaned_dir / 'train_clean.parquet'
        test_path = self.cleaned_dir / 'test_clean.parquet'
        
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError("Cleaned data not found. Run process_data() first.")
        
        train_df = pl.read_parquet(train_path)
        test_df = pl.read_parquet(test_path)
        
        self.logger.info(f"Loaded cleaned data: train {train_df.shape}, test {test_df.shape}")
        return train_df, test_df

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    preprocessor = DataPreprocessor()
    stats = preprocessor.process_data()
    
    print(f"Processing complete in {stats.processing_time_seconds:.2f} seconds")
    print(f"Features processed: {stats.features_processed}")
    print(f"NaNs removed: {stats.nan_count_before - stats.nan_count_after:,}")
