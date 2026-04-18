#!/usr/bin/env python3
"""
Evaluation Metrics Module

Comprehensive metrics for time series forecasting including
Kaggle Weighted RMSE and additional evaluation metrics.
"""

import numpy as np
import polars as pl
from typing import Union, Dict, Any, Optional
from dataclasses import dataclass
import warnings

@dataclass
class MetricResults:
    """Container for metric results"""
    rmse: float
    mae: float
    mape: float
    smape: float
    weighted_rmse: float
    pearson: float
    r2: float
    directional_accuracy: float
    weighted_mape: Optional[float] = None
    # Additional comparison metrics vs y_train
    y_mean_diff: Optional[float] = None
    y_std_ratio: Optional[float] = None
    y_range_coverage: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy serialization"""
        result = {
            'rmse': self.rmse,
            'mae': self.mae,
            'mape': self.mape,
            'smape': self.smape,
            'weighted_rmse': self.weighted_rmse,
            'pearson': self.pearson,
            'r2': self.r2,
            'directional_accuracy': self.directional_accuracy,
            'weighted_mape': self.weighted_mape
        }
        
        # Add comparison metrics if available
        if self.y_mean_diff is not None:
            result['y_mean_diff'] = self.y_mean_diff
        if self.y_std_ratio is not None:
            result['y_std_ratio'] = self.y_std_ratio
        if self.y_range_coverage is not None:
            result['y_range_coverage'] = self.y_range_coverage
            
        return result

class TimeSeriesMetrics:
    """
    Comprehensive evaluation metrics for time series forecasting.
    
    Features:
    - Kaggle Weighted RMSE (primary metric)
    - Standard metrics (RMSE, MAE, MAPE, SMAPE)
    - Correlation metrics (Pearson, R²)
    - Directional accuracy
    - Weighted MAPE (stable for small/large values)
    """
    
    @staticmethod
    def rmse(y_true: Union[np.ndarray, pl.Series], y_pred: Union[np.ndarray, pl.Series]) -> float:
        """Root Mean Square Error"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    @staticmethod
    def mae(y_true: Union[np.ndarray, pl.Series], y_pred: Union[np.ndarray, pl.Series]) -> float:
        """Mean Absolute Error"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def mape(y_true: Union[np.ndarray, pl.Series], y_pred: Union[np.ndarray, pl.Series]) -> float:
        """
        Mean Absolute Percentage Error
        Warning: Can explode with values near zero
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Handle zero values in y_true
        mask = np.abs(y_true) > 1e-8
        if np.sum(mask) == 0:
            return float(np.inf)
        
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    @staticmethod
    def weighted_mape(y_true: Union[np.ndarray, pl.Series], 
                     y_pred: Union[np.ndarray, pl.Series],
                     weights: Optional[Union[np.ndarray, pl.Series]] = None) -> float:
        """
        Weighted Mean Absolute Percentage Error
        More stable for small/large values
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if weights is None:
            weights = np.ones_like(y_true)
        weights = np.array(weights)
        
        # Handle zero values
        mask = np.abs(y_true) > 1e-8
        if np.sum(mask) == 0:
            return float(np.inf)
        
        weighted_errors = weights[mask] * np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        return float(np.sum(weighted_errors) / np.sum(weights[mask]) * 100)
    
    @staticmethod
    def smape(y_true: Union[np.ndarray, pl.Series], y_pred: Union[np.ndarray, pl.Series]) -> float:
        """Symmetric Mean Absolute Percentage Error"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator > 1e-8
        
        if np.sum(mask) == 0:
            return 0.0
        
        return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)
    
    @staticmethod
    def weighted_rmse_score(y_true: Union[np.ndarray, pl.Series], 
                           y_pred: Union[np.ndarray, pl.Series], 
                           weights: Union[np.ndarray, pl.Series]) -> float:
        """
        Kaggle Weighted RMSE (higher is better)
        
        Formula: sqrt(1 - sum(w * (y_true - y_pred)^2) / sum(w * y_true^2))
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        weights = np.array(weights)
        
        def _clip01(x: float) -> float:
            return float(np.minimum(np.maximum(x, 0.0), 1.0))
        
        denom = np.sum(weights * y_true ** 2)
        if denom == 0:
            return 0.0
        
        ratio = np.sum(weights * (y_true - y_pred) ** 2) / denom
        return float(np.sqrt(1.0 - _clip01(ratio)))
    
    @staticmethod
    def pearson_corr(y_true: Union[np.ndarray, pl.Series], 
                    y_pred: Union[np.ndarray, pl.Series]) -> float:
        """Pearson correlation coefficient"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if len(y_true) < 2:
            return 0.0
        
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        return float(correlation if not np.isnan(correlation) else 0.0)
    
    @staticmethod
    def r2_score(y_true: Union[np.ndarray, pl.Series], 
                y_pred: Union[np.ndarray, pl.Series]) -> float:
        """R-squared (coefficient of determination)"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return float(1 - (ss_res / (ss_tot + 1e-8)))
    
    @staticmethod
    def directional_accuracy(y_true: Union[np.ndarray, pl.Series], 
                           y_pred: Union[np.ndarray, pl.Series]) -> float:
        """
        Directional Accuracy - percentage of correct direction predictions
        Compares signs of differences between consecutive points
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if len(y_true) < 2:
            return 0.0
        
        # Calculate directional changes
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        # Remove zeros (no change)
        mask = (true_direction != 0) & (pred_direction != 0)
        
        if np.sum(mask) == 0:
            return 0.0
        
        correct_directions = np.sum(true_direction[mask] == pred_direction[mask])
        total_directions = np.sum(mask)
        
        return float(correct_directions / total_directions)
    
    @staticmethod
    def compare_with_train(y_train: Union[np.ndarray, pl.Series], 
                         y_pred: Union[np.ndarray, pl.Series]) -> Dict[str, float]:
        """
        Compare prediction statistics with training target statistics.
        
        Args:
            y_train: Training target values
            y_pred: Prediction values
            
        Returns:
            Dict with comparison metrics
        """
        y_train = np.array(y_train)
        y_pred = np.array(y_pred)
        
        # Calculate statistics
        train_mean = np.mean(y_train)
        train_std = np.std(y_train)
        train_min = np.min(y_train)
        train_max = np.max(y_train)
        
        pred_mean = np.mean(y_pred)
        pred_std = np.std(y_pred)
        pred_min = np.min(y_pred)
        pred_max = np.max(y_pred)
        
        # Comparison metrics
        mean_diff = float(pred_mean - train_mean)
        std_ratio = float(pred_std / train_std) if train_std > 0 else float('inf')
        
        # Range coverage: percentage of train range covered by predictions
        train_range = train_max - train_min
        if train_range > 0:
            coverage = float(max(0, min(pred_max, train_max) - max(pred_min, train_min)) / train_range)
        else:
            coverage = 0.0
        
        return {
            'y_mean_diff': mean_diff,
            'y_std_ratio': std_ratio,
            'y_range_coverage': coverage
        }
    
    @classmethod
    def evaluate_all(cls, 
                    y_true: Union[np.ndarray, pl.Series], 
                    y_pred: Union[np.ndarray, pl.Series],
                    weights: Optional[Union[np.ndarray, pl.Series]] = None,
                    include_weighted_mape: bool = True,
                    y_train: Optional[Union[np.ndarray, pl.Series]] = None) -> MetricResults:
        """
        Calculate all metrics at once.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            weights: Sample weights (required for weighted metrics)
            include_weighted_mape: Whether to calculate weighted MAPE
            y_train: Training target values for comparison (optional)
            
        Returns:
            MetricResults: All calculated metrics
        """
        if weights is None:
            weights = np.ones_like(y_true)
        
        # Calculate all metrics
        rmse_val = cls.rmse(y_true, y_pred)
        mae_val = cls.mae(y_true, y_pred)
        mape_val = cls.mape(y_true, y_pred)
        smape_val = cls.smape(y_true, y_pred)
        weighted_rmse_val = cls.weighted_rmse_score(y_true, y_pred, weights)
        pearson_val = cls.pearson_corr(y_true, y_pred)
        r2_val = cls.r2_score(y_true, y_pred)
        directional_acc_val = cls.directional_accuracy(y_true, y_pred)
        
        # Weighted MAPE (optional)
        weighted_mape_val = None
        if include_weighted_mape:
            weighted_mape_val = cls.weighted_mape(y_true, y_pred, weights)
        
        # Comparison metrics with training data (optional)
        y_mean_diff = None
        y_std_ratio = None
        y_range_coverage = None
        
        if y_train is not None:
            comparison = cls.compare_with_train(y_train, y_pred)
            y_mean_diff = comparison['y_mean_diff']
            y_std_ratio = comparison['y_std_ratio']
            y_range_coverage = comparison['y_range_coverage']
        
        return MetricResults(
            rmse=rmse_val,
            mae=mae_val,
            mape=mape_val,
            smape=smape_val,
            weighted_rmse=weighted_rmse_val,
            pearson=pearson_val,
            r2=r2_val,
            directional_accuracy=directional_acc_val,
            weighted_mape=weighted_mape_val,
            y_mean_diff=y_mean_diff,
            y_std_ratio=y_std_ratio,
            y_range_coverage=y_range_coverage
        )
    
    @staticmethod
    def print_metrics(metrics: MetricResults, title: str = "Metrics Results") -> None:
        """Print formatted metrics results"""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        
        print(f"  Primary Metrics:")
        print(f"    Weighted RMSE (Kaggle): {metrics.weighted_rmse:.6f}")
        print(f"    Pearson Correlation: {metrics.pearson:.6f}")
        print(f"    R²: {metrics.r2:.6f}")
        
        print(f"\n  Standard Metrics:")
        print(f"    RMSE: {metrics.rmse:.6f}")
        print(f"    MAE: {metrics.mae:.6f}")
        print(f"    MAPE: {metrics.mape:.4f}%")
        print(f"    SMAPE: {metrics.smape:.4f}%")
        
        if metrics.weighted_mape is not None:
            print(f"    Weighted MAPE: {metrics.weighted_mape:.4f}%")
        
        print(f"\n  Additional Metrics:")
        print(f"    Directional Accuracy: {metrics.directional_accuracy:.4f}")
        
        # Comparison metrics with training data
        if metrics.y_mean_diff is not None:
            print(f"\n  Comparison vs Training Data:")
            print(f"    Mean Difference: {metrics.y_mean_diff:.6f}")
            print(f"    Std Ratio (pred/train): {metrics.y_std_ratio:.6f}")
            print(f"    Range Coverage: {metrics.y_range_coverage:.4f}")
        
        print(f"{'='*60}")

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    y_true = np.random.randn(1000)
    y_pred = y_true + np.random.randn(1000) * 0.1
    weights = np.random.uniform(0.5, 1.5, 1000)
    
    # Calculate metrics
    metrics = TimeSeriesMetrics.evaluate_all(y_true, y_pred, weights)
    
    # Print results
    TimeSeriesMetrics.print_metrics(metrics, "Sample Metrics")
    
    # Show dictionary format
    print("\nMetrics as dictionary:")
    for key, value in metrics.to_dict().items():
        print(f"  {key}: {value:.6f}")
