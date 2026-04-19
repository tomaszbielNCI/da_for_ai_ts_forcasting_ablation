#!/usr/bin/env python3
"""
Evaluation Metrics Module

Comprehensive metrics for time series forecasting including
Kaggle Weighted RMSE and additional evaluation metrics.
"""

import numpy as np
import polars as pl
import json
import csv
from pathlib import Path
from typing import Union, Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
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
        }

        if self.weighted_mape is not None:
            result['weighted_mape'] = self.weighted_mape

        # Add comparison metrics if available
        if self.y_mean_diff is not None:
            result['y_mean_diff'] = self.y_mean_diff
        if self.y_std_ratio is not None:
            result['y_std_ratio'] = self.y_std_ratio
        if self.y_range_coverage is not None:
            result['y_range_coverage'] = self.y_range_coverage

        return result

    def to_list(self) -> List:
        """Convert to list for CSV row (ordered)"""
        return [
            self.weighted_rmse,
            self.pearson,
            self.rmse,
            self.mae,
            self.r2,
            self.directional_accuracy,
            self.mape,
            self.smape,
            self.weighted_mape if self.weighted_mape else 0.0,
            self.y_mean_diff if self.y_mean_diff else 0.0,
            self.y_std_ratio if self.y_std_ratio else 0.0,
            self.y_range_coverage if self.y_range_coverage else 0.0
        ]


class TimeSeriesMetrics:
    """
    Comprehensive evaluation metrics for time series forecasting.
    """

    # CSV header for metrics
    CSV_HEADER = [
        'weighted_rmse', 'pearson', 'rmse', 'mae', 'r2',
        'directional_accuracy', 'mape', 'smape', 'weighted_mape',
        'y_mean_diff', 'y_std_ratio', 'y_range_coverage'
    ]

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
        """Mean Absolute Percentage Error"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mask = np.abs(y_true) > 1e-8
        if np.sum(mask) == 0:
            return float(np.inf)

        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    @staticmethod
    def weighted_mape(y_true: Union[np.ndarray, pl.Series],
                      y_pred: Union[np.ndarray, pl.Series],
                      weights: Optional[Union[np.ndarray, pl.Series]] = None) -> float:
        """Weighted Mean Absolute Percentage Error"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if weights is None:
            weights = np.ones_like(y_true)
        weights = np.array(weights)

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
        """Kaggle Weighted RMSE (higher is better)"""
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
        """Directional Accuracy"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if len(y_true) < 2:
            return 0.0

        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))

        mask = (true_direction != 0) & (pred_direction != 0)

        if np.sum(mask) == 0:
            return 0.0

        correct_directions = np.sum(true_direction[mask] == pred_direction[mask])
        total_directions = np.sum(mask)

        return float(correct_directions / total_directions)

    @staticmethod
    def compare_with_train(y_train: Union[np.ndarray, pl.Series],
                           y_pred: Union[np.ndarray, pl.Series]) -> Dict[str, float]:
        """Compare prediction statistics with training target statistics."""
        y_train = np.array(y_train)
        y_pred = np.array(y_pred)

        train_mean = np.mean(y_train)
        train_std = np.std(y_train)
        train_min = np.min(y_train)
        train_max = np.max(y_train)

        pred_mean = np.mean(y_pred)
        pred_std = np.std(y_pred)
        pred_min = np.min(y_pred)
        pred_max = np.max(y_pred)

        mean_diff = float(pred_mean - train_mean)
        std_ratio = float(pred_std / train_std) if train_std > 0 else float('inf')

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
        """Calculate all metrics at once."""
        if weights is None:
            weights = np.ones_like(y_true)

        rmse_val = cls.rmse(y_true, y_pred)
        mae_val = cls.mae(y_true, y_pred)
        mape_val = cls.mape(y_true, y_pred)
        smape_val = cls.smape(y_true, y_pred)
        weighted_rmse_val = cls.weighted_rmse_score(y_true, y_pred, weights)
        pearson_val = cls.pearson_corr(y_true, y_pred)
        r2_val = cls.r2_score(y_true, y_pred)
        directional_acc_val = cls.directional_accuracy(y_true, y_pred)

        weighted_mape_val = None
        if include_weighted_mape:
            weighted_mape_val = cls.weighted_mape(y_true, y_pred, weights)

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
        print(f"\n{'=' * 60}")
        print(f"{title}")
        print(f"{'=' * 60}")

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

        if metrics.y_mean_diff is not None:
            print(f"\n  Comparison vs Training Data:")
            print(f"    Mean Difference: {metrics.y_mean_diff:.6f}")
            print(f"    Std Ratio (pred/train): {metrics.y_std_ratio:.6f}")
            print(f"    Range Coverage: {metrics.y_range_coverage:.4f}")

        print(f"{'=' * 60}")

    # ============================================
    # NEW: SAVE METRICS TO FILES
    # ============================================

    @staticmethod
    def save_metrics_to_json(metrics_dict: Dict[str, Any], filepath: Path) -> None:
        """
        Save metrics dictionary to JSON file.

        Args:
            metrics_dict: Dictionary containing metrics
            filepath: Path where to save the JSON file
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Add timestamp if not present
        if 'timestamp' not in metrics_dict:
            metrics_dict['timestamp'] = datetime.now().isoformat()

        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        print(f"  ✅ JSON saved: {filepath}")

    @staticmethod
    def save_metrics_to_csv(metrics_dict: Dict[str, Any], filepath: Path,
                            metrics_type: str = 'train') -> None:
        """
        Save metrics to CSV file (appends to existing file).

        Args:
            metrics_dict: Dictionary containing metrics (must have 'horizon' and metrics for type)
            filepath: Path where to save the CSV file
            metrics_type: 'train' or 'valid' - which metrics to extract
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Extract data
        horizon = metrics_dict.get('horizon')
        model = metrics_dict.get('model', 'unknown')
        metrics_data = metrics_dict.get(metrics_type, {})

        if not metrics_data:
            return

        # Prepare row
        row = {
            'timestamp': datetime.now().isoformat(),
            'horizon': horizon,
            'model': model,
            'type': metrics_type,
            'weighted_rmse': metrics_data.get('weighted_rmse', 0),
            'pearson': metrics_data.get('pearson', 0),
            'rmse': metrics_data.get('rmse', 0),
            'mae': metrics_data.get('mae', 0),
            'r2': metrics_data.get('r2', 0),
            'directional_accuracy': metrics_data.get('directional_accuracy', 0),
            'mape': metrics_data.get('mape', 0),
            'smape': metrics_data.get('smape', 0),
            'best_iteration': metrics_dict.get('best_iteration', 0),
            'features_used': metrics_dict.get('features_used', 0)
        }

        # Check if file exists to write header
        file_exists = filepath.exists()

        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        print(f"  ✅ CSV saved: {filepath}")

    @staticmethod
    def save_all_metrics(metrics_dict: Dict[str, Any],
                         json_path: Path,
                         csv_path: Path,
                         metrics_type: str = 'train') -> None:
        """
        Save metrics to both JSON and CSV.

        Args:
            metrics_dict: Dictionary containing metrics
            json_path: Path for JSON file
            csv_path: Path for CSV file
            metrics_type: 'train' or 'valid' - which metrics to extract for CSV
        """
        TimeSeriesMetrics.save_metrics_to_json(metrics_dict, json_path)
        TimeSeriesMetrics.save_metrics_to_csv(metrics_dict, csv_path, metrics_type)

    @staticmethod
    def aggregate_metrics(results_dir: Path, output_json: Path, output_csv: Path) -> None:
        """
        Aggregate all metrics JSON files into one summary file.

        Args:
            results_dir: Directory containing individual metrics JSON files
            output_json: Path for aggregated JSON
            output_csv: Path for aggregated CSV
        """
        all_metrics = {}
        all_rows = []

        for json_file in Path(results_dir).glob('metrics_*.json'):
            with open(json_file, 'r') as f:
                data = json.load(f)
                key = f"h{data['horizon']}_{data['model']}"
                all_metrics[key] = data

                # Add to CSV rows (train and valid)
                for mtype in ['train', 'valid']:
                    if mtype in data:
                        all_rows.append({
                            'horizon': data['horizon'],
                            'model': data['model'],
                            'type': mtype,
                            'weighted_rmse': data[mtype].get('weighted_rmse', 0),
                            'pearson': data[mtype].get('pearson', 0),
                            'rmse': data[mtype].get('rmse', 0),
                            'best_iteration': data.get('best_iteration', 0),
                            'features_used': data.get('features_used', 0)
                        })

        # Save JSON
        with open(output_json, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        # Save CSV
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['horizon', 'model', 'type', 'weighted_rmse', 'pearson', 'rmse',
                                                   'best_iteration', 'features_used'])
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"✅ Aggregated metrics saved to: {output_json}")
        print(f"✅ Aggregated CSV saved to: {output_csv}")


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    y_true = np.random.randn(1000)
    y_pred = y_true + np.random.randn(1000) * 0.1
    weights = np.random.uniform(0.5, 1.5, 1000)

    metrics = TimeSeriesMetrics.evaluate_all(y_true, y_pred, weights)
    TimeSeriesMetrics.print_metrics(metrics, "Sample Metrics")

    # Example save
    test_dict = {
        'horizon': 1,
        'model': 'test',
        'train': metrics.to_dict(),
        'valid': metrics.to_dict(),
        'best_iteration': 42,
        'features_used': 10
    }

    TimeSeriesMetrics.save_all_metrics(
        test_dict,
        Path('results/test_metrics.json'),
        Path('results/test_metrics.csv'),
        'train'
    )