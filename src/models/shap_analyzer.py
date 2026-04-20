#!/usr/bin/env python3
"""
SHAP Analyzer for Feature Importance Analysis

Performs comprehensive SHAP analysis on trained baseline LGBM models:
- Calculates SHAP values for each horizon
- Identifies top 20 features per horizon
- Generates visualizations (standard + balanced)
- Exports results for feature engineering

Faithfully copied from notebook with modular structure.
"""

import polars as pl
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from .baseline_lgbm import BaselineLGBM

@dataclass
class SHAPResults:
    """Container for SHAP analysis results"""
    horizon: int
    features: List[str]
    mean_shap: np.ndarray
    mean_shap_dir: np.ndarray
    top_features: List[str]
    top_shap_values: np.ndarray
    top_directions: np.ndarray
    sample_size: int
    validation_samples: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'horizon': self.horizon,
            'features': self.features,
            'mean_shap': self.mean_shap.tolist(),
            'mean_shap_dir': self.mean_shap_dir.tolist(),
            'top_features': self.top_features,
            'top_shap_values': self.top_shap_values.tolist(),
            'top_directions': self.top_directions.tolist(),
            'sample_size': self.sample_size,
            'validation_samples': self.validation_samples
        }

class SHAPAnalyzer:
    """
    Comprehensive SHAP analysis for baseline LGBM models.
    
    Features:
    - Per-horizon SHAP value calculation
    - Top 20 feature identification
    - Direction analysis (positive/negative impact)
    - Visualization generation
    - Results export for feature engineering
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, model_type: str = 'baseline_lgbm'):
        """Initialize SHAP analyzer with configuration.
        
        Args:
            config: Configuration dictionary
            model_type: Type of model to analyze ('baseline_lgbm', 'xgboost', 'catboost')
        """
        self.config = config or {}
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        
        # Analysis parameters
        self.horizons = self.config.get('horizons', [1, 3, 10, 25])
        self.shap_sample_size = self.config.get('shap_sample_size', 5000)
        self.top_n_features = self.config.get('top_n_features', 20)  # Changed from 15 to 20
        
        # Paths
        self.results_dir = Path('results') / 'shap'
        self.plots_dir = self.results_dir / 'plots'
        self.data_dir = self.results_dir / 'data'
        self.models_dir = self.results_dir / 'models'
        
        # Create directories
        for dir_path in [self.results_dir, self.plots_dir, self.data_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize model based on type
        self._initialize_model()
        self.models = {}
        self.shap_results = {}
    
    def _initialize_model(self) -> None:
        """Initialize model based on model_type parameter."""
        if self.model_type == 'baseline_lgbm':
            self.baseline_lgbm = BaselineLGBM(self.config)
        # Future extension possibilities (commented out - requires model implementations)
        # elif self.model_type == 'xgboost':
        #     from .xgb_model import XGBoostModel  # Requires implementation
        #     self.baseline_lgbm = XGBoostModel(self.config)
        # elif self.model_type == 'catboost':
        #     from .catboost_model import CatBoostModel  # Requires implementation
        #     self.baseline_lgbm = CatBoostModel(self.config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. "
                           f"Supported: 'baseline_lgbm'")
    
    def load_baseline_models(self) -> None:
        """Load trained baseline models for SHAP analysis."""
        model_name = self.model_type.replace('_', ' ').title()
        self.logger.info(f"Loading {model_name} models...")
        
        # Load cleaned data
        train_df, test_df = self.baseline_lgbm.load_cleaned_data()
        
        # Train models (reusing baseline logic)
        feature_cols = self.baseline_lgbm.get_feature_columns(train_df)
        
        for horizon in self.horizons:
            self.logger.info(f"Training {model_name} model for horizon {horizon}...")
            
            # Use full training data for SHAP analysis
            horizon_df = train_df.filter(pl.col('horizon') == horizon)
            train_full = horizon_df.filter(pl.col('ts_index') <= self.baseline_lgbm.max_ts_train)
            
            # Prepare data (no validation needed for SHAP)
            X_train, y_train, w_train = self.baseline_lgbm.prepare_data(train_full, feature_cols, is_test=False)
            
            # For SHAP we don't need validation - use dummy data
            X_valid, y_valid, w_valid = X_train[:1], y_train[:1], w_train[:1]
            
            # Train model
            model = self.baseline_lgbm.train_model(X_train, y_train, w_train, X_valid, y_valid, w_valid)
            self.models[horizon] = {
                'model': model,
                'features': feature_cols,
                'train_data': train_full
            }
            
            self.logger.info(f"{model_name} model trained for horizon {horizon}")
    
    def calculate_shap_values(self, horizon: int) -> SHAPResults:
        """Calculate SHAP values for a specific horizon."""
        self.logger.info(f"Calculating SHAP values for horizon {horizon}...")
        
        model_data = self.models[horizon]
        model = model_data['model']
        features = model_data['features']
        train_data = model_data['train_data']
        
        # Sample data for SHAP analysis
        if len(train_data) > self.shap_sample_size:
            sample_data = train_data.sample(self.shap_sample_size, seed=42)
        else:
            sample_data = train_data
        
        # Prepare data
        X_sample, _, _ = self.baseline_lgbm.prepare_data(sample_data, features, is_test=False)
        
        # SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Calculate mean |SHAP| (global importance)
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        mean_shap_dir = np.mean(shap_values, axis=0)  # Direction (positive/negative)
        
        # Get top features
        top_indices = np.argsort(mean_shap)[::-1][:self.top_n_features]
        top_features = [features[i] for i in top_indices]
        top_shap = mean_shap[top_indices]
        top_direction = mean_shap_dir[top_indices]
        
        # Validation samples count
        validation_samples = len(train_data.filter(
            (pl.col('ts_index') > self.baseline_lgbm.train_split) & 
            (pl.col('ts_index') <= self.baseline_lgbm.valid_end)
        ))
        
        return SHAPResults(
            horizon=horizon,
            features=features,
            mean_shap=mean_shap,
            mean_shap_dir=mean_shap_dir,
            top_features=top_features,
            top_shap_values=top_shap,
            top_directions=top_direction,
            sample_size=len(X_sample),
            validation_samples=validation_samples
        )
    
    def analyze_all_horizons(self) -> Dict[int, SHAPResults]:
        """Perform SHAP analysis for all horizons."""
        self.logger.info("Starting SHAP analysis for all horizons...")
        
        # Load models
        self.load_baseline_models()
        
        # Calculate SHAP values for each horizon
        for horizon in self.horizons:
            self.shap_results[horizon] = self.calculate_shap_values(horizon)
            
            # Print results
            self._print_horizon_results(horizon)
        
        # Save results for LGBM_SHAP_10
        self.save_shap_results()
        
        return self.shap_results
    
    def save_shap_results(self) -> None:
        """Save SHAP results to JSON file for LGBM_SHAP_10 to use."""
        self.logger.info("Saving SHAP results to JSON...")
        
        # Convert to serializable format
        shap_dict = {}
        for horizon, result in self.shap_results.items():
            shap_dict[str(horizon)] = {
                'features': result.top_features[:10],  # Top 10 for LGBM_SHAP_10
                'top_shap_values': result.top_shap_values[:10].tolist(),
                'top_directions': result.top_directions[:10].tolist()
            }
        
        # Save to expected location
        output_path = self.data_dir / 'horizon_specific_top_features.json'
        with open(output_path, 'w') as f:
            json.dump(shap_dict, f, indent=2)
        
        self.logger.info(f"SHAP results saved to: {output_path}")
    
    def _print_horizon_results(self, horizon: int) -> None:
        """Print SHAP results for a horizon."""
        result = self.shap_results[horizon]
        
        print(f"\n{'='*60}")
        print(f"HORIZON: {horizon}")
        print(f"{'='*60}")
        print(f"  Validation samples: {result.validation_samples:,}")
        print(f"  SHAP sample size: {result.sample_size}")
        print(f"\n  TOP {self.top_n_features} FEATURES (by |SHAP|):")
        
        for i, (feature, shap_val, direction) in enumerate(zip(
            result.top_features, result.top_shap_values, result.top_directions
        ), 1):
            direction_symbol = "POSITIVE" if direction > 0 else "NEGATIVE"
            print(f"     {i:2d}. {feature:<12} |SHAP|={shap_val:.6f}, direction={direction:+.6f} ({direction_symbol})")
    
    def generate_visualizations(self) -> None:
        """Generate SHAP visualizations for all horizons."""
        self.logger.info("Generating SHAP visualizations...")
        
        print(f"\n{'='*60}")
        print("SHAP VISUALIZATIONS")
        print(f"{'='*60}")
        
        for horizon in self.horizons:
            result = self.shap_results[horizon]
            
            # Standard importance plot
            self._create_standard_plot(horizon, result)
            
            # Balanced plot (negative left, positive right)
            self._create_balanced_plot(horizon, result)
    
    def _create_standard_plot(self, horizon: int, result: SHAPResults) -> None:
        """Create standard SHAP importance plot."""
        colors = ['red' if d < 0 else 'green' for d in result.top_directions]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(result.top_features)), result.top_shap_values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(result.top_features)))
        ax.set_yticklabels(result.top_features)
        ax.set_xlabel('Mean |SHAP|')
        ax.set_title(f'SHAP Feature Importance - H={horizon} (red=negative, green=positive)')
        ax.invert_yaxis()
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f'shap_h{horizon}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved SHAP plot for H={horizon}")
    
    def _create_balanced_plot(self, horizon: int, result: SHAPResults) -> None:
        """Create balanced SHAP plot (negative left, positive right)."""
        # Separate positive and negative
        pos_mask = result.top_directions > 0
        neg_mask = ~pos_mask
        
        pos_features = [result.top_features[i] for i in range(len(result.top_features)) if pos_mask[i]]
        pos_values = [result.top_shap_values[i] for i in range(len(result.top_shap_values)) if pos_mask[i]]
        
        neg_features = [result.top_features[i] for i in range(len(result.top_features)) if neg_mask[i]]
        neg_values = [result.top_shap_values[i] for i in range(len(result.top_shap_values)) if neg_mask[i]]
        
        # Sort by absolute value
        pos_sorted = np.argsort(pos_values)[::-1]
        neg_sorted = np.argsort(neg_values)[::-1]
        
        pos_features = [pos_features[i] for i in pos_sorted]
        pos_values = [pos_values[i] for i in pos_sorted]
        neg_features = [neg_features[i] for i in neg_sorted]
        neg_values = [neg_values[i] for i in neg_sorted]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Positive bars (right)
        y_pos = np.arange(len(pos_features))
        ax.barh(y_pos, pos_values, color='green', alpha=0.7, label='Positive Impact')
        
        # Negative bars (left)
        y_neg = np.arange(len(neg_features))
        ax.barh(y_neg, [-v for v in neg_values], color='red', alpha=0.7, label='Negative Impact')
        
        # Labels and formatting
        all_features = neg_features + pos_features
        ax.set_yticks(range(len(all_features)))
        ax.set_yticklabels(all_features)
        ax.set_xlabel('SHAP Value')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title(f'SHAP Feature Impact - H={horizon} (red=negative, green=positive)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f'shap_balanced_h{horizon}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved balanced SHAP plot for H={horizon}")
    
    def save_results(self) -> None:
        """Save SHAP analysis results."""
        self.logger.info("Saving SHAP results...")
        
        # Save detailed results
        shap_results_path = self.data_dir / 'shap_results.json'
        shap_results_dict = {
            str(h): result.to_dict() for h, result in self.shap_results.items()
        }
        
        with open(shap_results_path, 'w') as f:
            json.dump(shap_results_dict, f, indent=2)
        
        # Save top features per horizon (for feature engineering)
        top_features_path = self.data_dir / 'horizon_specific_top_features.json'
        top_features_dict = {}
        
        for horizon, result in self.shap_results.items():
            top_features_dict[str(horizon)] = {
                'features': result.top_features,
                'shap_values': result.top_shap_values.tolist(),
                'directions': result.top_directions.tolist()
            }
        
        with open(top_features_path, 'w') as f:
            json.dump(top_features_dict, f, indent=2)
        
        # Create summary table
        self._create_summary_table()
        
        self.logger.info(f"Saved SHAP results to {self.results_dir}")
        print(f"Saved horizon-specific top features to {top_features_path}")
    
    def _create_summary_table(self) -> None:
        """Create summary table of top features across horizons."""
        summary_data = []
        
        for horizon, result in self.shap_results.items():
            for i, (feature, shap_val, direction) in enumerate(zip(
                result.top_features, result.top_shap_values, result.top_directions
            ), 1):
                summary_data.append({
                    'horizon': horizon,
                    'rank': i,
                    'feature': feature,
                    'mean_shap': shap_val,
                    'direction': 'positive' if direction > 0 else 'negative'
                })
        
        # Convert to DataFrame and save
        summary_df = pl.DataFrame(summary_data)
        summary_path = self.data_dir / 'shap_summary.csv'
        summary_df.write_csv(summary_path)
        
        self.logger.info(f"Saved summary table to {summary_path}")
    
    def run_full_analysis(self) -> Dict[int, SHAPResults]:
        """Run complete SHAP analysis pipeline."""
        self.logger.info("Starting full SHAP analysis...")
        
        # Analyze all horizons
        self.analyze_all_horizons()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Save results
        self.save_results()
        
        # Print summary
        self._print_summary()
        
        return self.shap_results
    
    def _print_summary(self) -> None:
        """Print analysis summary."""
        print(f"\n{'='*80}")
        print("SHAP ANALYSIS - SUMMARY")
        print(f"{'='*80}")
        
        print(f"Horizons analyzed: {self.horizons}")
        print(f"Top features per horizon: {self.top_n_features}")
        print(f"SHAP sample size: {self.shap_sample_size}")
        print(f"Results saved to: {self.results_dir}")
        
        # Common features across horizons
        all_features = set()
        for result in self.shap_results.values():
            all_features.update(result.top_features[:10])  # Top 10 per horizon
        
        print(f"Total unique features in top 10: {len(all_features)}")
        print(f"{'='*80}")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Default: Baseline LGBM
    analyzer = SHAPAnalyzer()
    results = analyzer.run_full_analysis()
    
    # Future usage examples (commented out - requires model implementations):
    # analyzer_xgb = SHAPAnalyzer(model_type='xgboost')
    # results_xgb = analyzer_xgb.run_full_analysis()
    
    # analyzer_cat = SHAPAnalyzer(model_type='catboost')
    # results_cat = analyzer_cat.run_full_analysis()
