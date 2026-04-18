#!/usr/bin/env python3
"""
Ensemble SHAP Models - With Kaggle Results Comparison

Creates weighted ensemble from saved SHAP-10 and SHAP-20 predictions.
Displays final comparison table for report.
"""

import polars as pl
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.metrics.evaluation import TimeSeriesMetrics


class EnsembleWithKaggleComparison:
    """
    Ensemble combining SHAP-10 and SHAP-20 with Kaggle results comparison.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Paths
        project_root = Path(__file__).parent.parent.parent
        self.shap10_dir = project_root / 'results/predictions/lgbm_shap_10'
        self.shap20_dir = project_root / 'results/predictions/lgbm_shap_20'
        self.output_dir = project_root / 'results/predictions/ensemble'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load engineered data for validation
        self.processed_dir = project_root / 'data/processed/top_10'

        # Horizons
        self.horizons = [1, 3, 10, 25]

        # Optimal weights from Kaggle (80% v3 + 20% top20)
        self.weights = {
            1: {'shap10': 0.80, 'shap20': 0.20},
            3: {'shap10': 0.80, 'shap20': 0.20},
            10: {'shap10': 0.80, 'shap20': 0.20},
            25: {'shap10': 0.80, 'shap20': 0.20}
        }

        # Validation split parameters
        self.valid_start = 3001
        self.valid_end = 3600

    def load_validation_data(self, horizon: int) -> tuple:
        """Load validation data for metrics calculation."""
        train_path = self.processed_dir / f'train_h{horizon}_engineered.parquet'
        train_df = pl.read_parquet(train_path)

        valid_df = train_df.filter(
            (pl.col('ts_index') >= self.valid_start) &
            (pl.col('ts_index') <= self.valid_end)
        )

        y_true = valid_df['y_target'].to_numpy().ravel()
        weights = valid_df['weight'].to_numpy().ravel()

        return y_true, weights

    def load_predictions(self) -> tuple:
        """Load test and validation predictions."""
        shap10_preds = {}
        shap20_preds = {}
        shap10_valid = {}
        shap20_valid = {}

        for h in self.horizons:
            # Load from .npz files
            shap10_npz = self.shap10_dir / f'lgbm_shap_10_h{h}_predictions.npz'
            shap20_npz = self.shap20_dir / f'lgbm_shap_20_h{h}_predictions.npz'

            if shap10_npz.exists():
                data10 = np.load(shap10_npz)
                shap10_valid[h] = data10['valid_pred']
                shap10_test = data10['test_pred']
            else:
                raise FileNotFoundError(f"No predictions for SHAP-10 H={h}")

            if shap20_npz.exists():
                data20 = np.load(shap20_npz)
                shap20_valid[h] = data20['valid_pred']
                shap20_test = data20['test_pred']
            else:
                raise FileNotFoundError(f"No predictions for SHAP-20 H={h}")

            shap10_preds[h] = shap10_test
            shap20_preds[h] = shap20_test

            self.logger.info(f"Loaded H={h}: SHAP-10, SHAP-20")

        return shap10_preds, shap20_preds, shap10_valid, shap20_valid

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> dict:
        """Compute metrics using TimeSeriesMetrics."""
        metrics = TimeSeriesMetrics.evaluate_all(y_true, y_pred, weights, y_train=y_true)
        return {
            'weighted_rmse': metrics.weighted_rmse,
            'pearson': metrics.pearson,
            'rmse': metrics.rmse,
            'r2': metrics.r2
        }

    def create_ensemble(self, shap10_preds: dict, shap20_preds: dict) -> dict:
        """Create weighted ensemble predictions."""
        ensemble_preds = {}

        for h in self.horizons:
            w10 = self.weights[h]['shap10']
            w20 = self.weights[h]['shap20']
            ensemble_preds[h] = w10 * shap10_preds[h] + w20 * shap20_preds[h]

        return ensemble_preds

    def compare_validation_metrics(self, shap10_valid: dict, shap20_valid: dict,
                                   ensemble_valid: dict) -> None:
        """Compare models on validation data."""
        print(f"\n{'=' * 100}")
        print("VALIDATION METRICS COMPARISON")
        print(f"{'=' * 100}")

        print(f"\n{'Horizon':<8} {'Model':<12} {'Weighted RMSE':<15} {'Pearson':<12} {'RMSE':<12} {'R²':<10}")
        print("-" * 80)

        for h in self.horizons:
            y_true, weights = self.load_validation_data(h)

            # SHAP-10
            if shap10_valid[h] is not None:
                m10 = self.compute_metrics(y_true, shap10_valid[h], weights)
                print(
                    f"{h:<8} {'SHAP-10':<12} {m10['weighted_rmse']:<15.6f} {m10['pearson']:<12.6f} {m10['rmse']:<12.6f} {m10['r2']:<10.6f}")

            # SHAP-20
            if shap20_valid[h] is not None:
                m20 = self.compute_metrics(y_true, shap20_valid[h], weights)
                print(
                    f"{h:<8} {'SHAP-20':<12} {m20['weighted_rmse']:<15.6f} {m20['pearson']:<12.6f} {m20['rmse']:<12.6f} {m20['r2']:<10.6f}")

            # Ensemble
            if ensemble_valid[h] is not None:
                m_ens = self.compute_metrics(y_true, ensemble_valid[h], weights)
                print(
                    f"{h:<8} {'Ensemble':<12} {m_ens['weighted_rmse']:<15.6f} {m_ens['pearson']:<12.6f} {m_ens['rmse']:<12.6f} {m_ens['r2']:<10.6f}")

            print("-" * 80)

    def print_kaggle_comparison(self) -> None:
        """Print Kaggle results comparison table for report."""
        print(f"\n{'=' * 60}")
        print("KAGGLE RESULTS - FINAL COMPARISON")
        print(f"{'=' * 60}")
        print(f"{'Model':<20} {'Private Score':<15} {'Public Score':<15} {'Improvement':<15}")
        print("-" * 65)
        print(f"{'SHAP-10 (v3_no_ultra)':<20} {0.1749:<15.6f} {0.1798:<15.6f} {'-':<15}")
        print(f"{'SHAP-20 (top20)':<20} {0.1475:<15.6f} {0.1469:<15.6f} {'-0.0274':<15}")
        print(f"{'Ensemble (80/20)':<20} {0.1805:<15.6f} {0.1802:<15.6f} {'+0.0056 ✅':<15}")
        print("=" * 60)

        # Additional insights
        print(f"\n📊 KEY INSIGHTS:")
        print(f"  • Ensemble improves SHAP-10 by +0.0056 (3.2%)")
        print(f"  • Ensemble is the ONLY model above 0.18")
        print(f"  • SHAP-20 alone performs worse than SHAP-10")
        print(f"  • Ensemble combines best of both models")

    def generate_submission(self, ensemble_preds: dict) -> Path:
        """Generate final submission file."""
        all_ids = []
        all_preds = []

        for h in self.horizons:
            test_path = self.processed_dir / f'test_h{h}_engineered.parquet'
            test_df = pl.read_parquet(test_path)
            ids = test_df['id'].to_numpy().ravel()

            all_ids.extend(ids)
            all_preds.extend(ensemble_preds[h])

        submission_df = pl.DataFrame({'id': all_ids, 'prediction': all_preds})

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = self.output_dir / f'ensemble_80v3_20top20_{timestamp}.csv'
        submission_df.write_csv(submission_path)

        self.logger.info(f"Ensemble submission saved: {submission_path}")
        self.logger.info(f"Shape: {submission_df.shape}")
        self.logger.info(f"Prediction range: [{np.min(all_preds):.4f}, {np.max(all_preds):.4f}]")

        return submission_path

    def run(self) -> Path:
        """Run ensemble pipeline with comparison."""
        self.logger.info("Creating ensemble from SHAP-10 and SHAP-20...")

        # Load predictions
        shap10_preds, shap20_preds, shap10_valid, shap20_valid = self.load_predictions()

        # Create ensemble predictions
        ensemble_test = self.create_ensemble(shap10_preds, shap20_preds)

        # Create ensemble validation predictions
        ensemble_valid = {}
        for h in self.horizons:
            if shap10_valid[h] is not None and shap20_valid[h] is not None:
                w10 = self.weights[h]['shap10']
                w20 = self.weights[h]['shap20']
                ensemble_valid[h] = w10 * shap10_valid[h] + w20 * shap20_valid[h]
            else:
                ensemble_valid[h] = None

        # Compare validation metrics
        self.compare_validation_metrics(shap10_valid, shap20_valid, ensemble_valid)

        # Print Kaggle comparison
        self.print_kaggle_comparison()

        # Generate submission
        submission_path = self.generate_submission(ensemble_test)

        print(f"\n{'=' * 60}")
        print("ENSEMBLE COMPLETE")
        print(f"{'=' * 60}")
        print(f"Weights: 80% SHAP-10, 20% SHAP-20")
        print(f"Submission: {submission_path}")
        print(f"{'=' * 60}")

        return submission_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ensemble = EnsembleWithKaggleComparison()
    ensemble.run()