#!/usr/bin/env python3
"""
Submission Results Comparison

Creates comprehensive comparison tables for submission results across models,
including Kaggle public scores and prediction statistics.
"""

import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

class SubmissionComparisonAnalyzer:
    """Analyze and compare submission results across models"""
    
    def __init__(self):
        self.project_root = Path('.')
        self.predictions_dir = self.project_root / 'results/predictions'
        self.horizons = [1, 3, 10, 25]
        
        # Model naming for submissions
        self.model_names = {
            'lgbm_baseline': 'Baseline LGBM',
            'lgbm_shap_10': 'LGBM SHAP-10',
            'lgbm_all_plus_shap': 'LGBM All+SHAP',
            'lgbm_bnn-shap10': 'LGBM BNN-SHAP10',
            'lgbm_bnn-aggregated': 'LGBM BNN-Aggregated',
            'xgb_shap_10': 'XGBoost SHAP-10',
            'catboost_shap_10': 'CatBoost SHAP-10',
            'trio_lgbm': 'Trio LGBM (WF)',
            'trio_xgb': 'Trio XGBoost (WF)',
            'trio_catboost': 'Trio CatBoost (WF)'
        }
        
        # Known Kaggle scores (update with actual scores when available)
        self.kaggle_scores = {
            'lgbm_baseline': None,
            'lgbm_shap_10': None,
            'lgbm_all_plus_shap': None,  # Just completed
            'lgbm_bnn-shap10': None,
            'lgbm_bnn-aggregated': None,
            'xgb_shap_10': None,
            'catboost_shap_10': None,
            'trio_lgbm': None,
            'trio_xgb': None,
            'trio_catboost': None
        }
    
    def load_predictions(self, model_key: str, horizon: int) -> Optional[np.ndarray]:
        """Load predictions for specific model and horizon"""
        model_dir = self.predictions_dir / model_key
        if not model_dir.exists():
            return None
        
        pred_file = model_dir / f'{model_key}_h{horizon}_predictions.npz'
        if not pred_file.exists():
            return None
        
        try:
            data = np.load(pred_file)
            # Use test_pred as the key (test set predictions)
            if 'test_pred' in data:
                predictions = data['test_pred']
            elif 'predictions' in data:
                predictions = data['predictions']
            else:
                print(f"No prediction key found in {pred_file}. Available keys: {list(data.keys())}")
                return None
            return predictions
        except Exception as e:
            print(f"Error loading {pred_file}: {e}")
            return None
    
    def analyze_prediction_stats(self, predictions: np.ndarray) -> Dict:
        """Calculate statistics for predictions"""
        if predictions is None or len(predictions) == 0:
            return {}
        
        stats = {
            'count': len(predictions),
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions),
            'range': np.max(predictions) - np.min(predictions),
            'median': np.median(predictions),
            'q25': np.percentile(predictions, 25),
            'q75': np.percentile(predictions, 75),
            'skewness': self._calculate_skewness(predictions),
            'kurtosis': self._calculate_kurtosis(predictions)
        }
        
        return stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        if len(data) < 2:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        n = len(data)
        skew = (n / ((n-1) * (n-2))) * np.sum(((data - mean) / std) ** 3)
        return skew
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        if len(data) < 2:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        n = len(data)
        kurt = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((data - mean) / std) ** 4)
        kurt -= 3 * (n-1) ** 2 / ((n-2) * (n-3))
        return kurt
    
    def create_prediction_stats_table(self) -> pd.DataFrame:
        """Create comprehensive prediction statistics table"""
        all_stats = []
        
        for model_key in self.model_names.keys():
            for horizon in self.horizons:
                predictions = self.load_predictions(model_key, horizon)
                stats = self.analyze_prediction_stats(predictions)
                
                if stats:
                    row = {
                        'Model': self.model_names[model_key],
                        'Model_Key': model_key,
                        'Horizon': f'H{horizon}',
                        'Count': stats['count'],
                        'Mean': stats['mean'],
                        'Std': stats['std'],
                        'Min': stats['min'],
                        'Max': stats['max'],
                        'Range': stats['range'],
                        'Median': stats['median'],
                        'Q25': stats['q25'],
                        'Q75': stats['q75'],
                        'Skewness': stats['skewness'],
                        'Kurtosis': stats['kurtosis']
                    }
                    all_stats.append(row)
        
        return pd.DataFrame(all_stats)
    
    def format_prediction_stats_markdown(self, df: pd.DataFrame) -> str:
        """Format prediction statistics as markdown table"""
        lines = []
        lines.append("# Prediction Statistics Comparison")
        lines.append("")
        lines.append("Generated: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
        lines.append("")
        
        # Group by model for better organization
        for model_key in df['Model_Key'].unique():
            model_df = df[df['Model_Key'] == model_key].sort_values('Horizon')
            model_name = model_df['Model'].iloc[0]
            
            lines.append(f"## {model_name}")
            lines.append("")
            
            # Header
            header = "| Horizon | Count | Mean | Std | Min | Max | Range | Median | Skewness | Kurtosis |"
            separator = "|---------|-------|------|-----|-----|-----|-------|--------|----------|----------|"
            lines.append(header)
            lines.append(separator)
            
            # Data rows
            for _, row in model_df.iterrows():
                lines.append(f"| {row['Horizon']} | {row['Count']:,} | {row['Mean']:.6f} | {row['Std']:.6f} | {row['Min']:.6f} | {row['Max']:.6f} | {row['Range']:.6f} | {row['Median']:.6f} | {row['Skewness']:.4f} | {row['Kurtosis']:.4f} |")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def create_submission_comparison_table(self) -> pd.DataFrame:
        """Create submission results comparison table"""
        submission_data = []
        
        for model_key, model_name in self.model_names.items():
            # Check if submission exists
            model_dir = self.predictions_dir / model_key
            submission_files = list(model_dir.glob(f'{model_key}_submission_*.csv')) if model_dir.exists() else []
            
            has_submission = len(submission_files) > 0
            kaggle_score = self.kaggle_scores.get(model_key)
            
            # Get average prediction stats across horizons
            avg_stats = {}
            for horizon in self.horizons:
                predictions = self.load_predictions(model_key, horizon)
                if predictions is not None:
                    stats = self.analyze_prediction_stats(predictions)
                    if stats:
                        for key, value in stats.items():
                            if key not in avg_stats:
                                avg_stats[key] = []
                            avg_stats[key].append(value)
            
            # Calculate averages
            final_stats = {}
            for key, values in avg_stats.items():
                final_stats[f'avg_{key}'] = np.mean(values) if values else None
            
            row = {
                'Model': model_name,
                'Model_Key': model_key,
                'Has_Submission': has_submission,
                'Kaggle_Public_Score': kaggle_score,
                'Avg_Prediction_Mean': final_stats.get('avg_mean'),
                'Avg_Prediction_Std': final_stats.get('avg_std'),
                'Avg_Prediction_Range': final_stats.get('avg_range'),
                'Submission_Files': len(submission_files),
                'Submission_Status': 'Submitted' if has_submission else 'Not Submitted'
            }
            
            submission_data.append(row)
        
        return pd.DataFrame(submission_data)
    
    def format_submission_comparison_markdown(self, df: pd.DataFrame) -> str:
        """Format submission comparison as markdown table"""
        lines = []
        lines.append("# Submission Results Comparison")
        lines.append("")
        lines.append("Generated: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
        lines.append("")
        
        # Sort by Kaggle score (best first) or by model name
        submitted = df[df['Has_Submission'] == True].copy()
        not_submitted = df[df['Has_Submission'] == False].copy()
        
        if len(submitted) > 0:
            # Sort submitted models by score if available
            submitted_with_score = submitted[submitted['Kaggle_Public_Score'].notna()]
            submitted_without_score = submitted[submitted['Kaggle_Public_Score'].isna()]
            
            lines.append("## Submitted Models")
            lines.append("")
            
            if len(submitted_with_score) > 0:
                lines.append("### Models with Kaggle Scores")
                lines.append("")
                lines.append("| Model | Kaggle Public Score | Avg Pred Mean | Avg Pred Std | Avg Pred Range | Status |")
                lines.append("|-------|-------------------|---------------|--------------|----------------|--------|")
                
                submitted_with_score = submitted_with_score.sort_values('Kaggle_Public_Score')
                
                for _, row in submitted_with_score.iterrows():
                    score = f"**{row['Kaggle_Public_Score']:.6f}**" if row['Kaggle_Public_Score'] else "Pending"
                    mean_val = f"{row['Avg_Prediction_Mean']:.6f}" if pd.notna(row['Avg_Prediction_Mean']) else "N/A"
                    std_val = f"{row['Avg_Prediction_Std']:.6f}" if pd.notna(row['Avg_Prediction_Std']) else "N/A"
                    range_val = f"{row['Avg_Prediction_Range']:.6f}" if pd.notna(row['Avg_Prediction_Range']) else "N/A"
                    
                    lines.append(f"| {row['Model']} | {score} | {mean_val} | {std_val} | {range_val} | {row['Submission_Status']} |")
                
                lines.append("")
            
            if len(submitted_without_score) > 0:
                lines.append("### Models Submitted (Score Pending)")
                lines.append("")
                lines.append("| Model | Avg Pred Mean | Avg Pred Std | Avg Pred Range | Status |")
                lines.append("|-------|---------------|--------------|----------------|--------|")
                
                for _, row in submitted_without_score.iterrows():
                    mean_val = f"{row['Avg_Prediction_Mean']:.6f}" if pd.notna(row['Avg_Prediction_Mean']) else "N/A"
                    std_val = f"{row['Avg_Prediction_Std']:.6f}" if pd.notna(row['Avg_Prediction_Std']) else "N/A"
                    range_val = f"{row['Avg_Prediction_Range']:.6f}" if pd.notna(row['Avg_Prediction_Range']) else "N/A"
                    
                    lines.append(f"| {row['Model']} | {mean_val} | {std_val} | {range_val} | {row['Submission_Status']} |")
                
                lines.append("")
        
        if len(not_submitted) > 0:
            lines.append("## Models Not Submitted")
            lines.append("")
            lines.append("| Model | Status |")
            lines.append("|-------|--------|")
            
            for _, row in not_submitted.iterrows():
                lines.append(f"| {row['Model']} | {row['Submission_Status']} |")
            
            lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Total models: {len(df)}")
        lines.append(f"- Submitted models: {len(submitted)}")
        lines.append(f"- Models with scores: {len(df[df['Kaggle_Public_Score'].notna()])}")
        
        if len(submitted_with_score) > 0:
            best_model = submitted_with_score.iloc[0]
            lines.append(f"- Best performing model: **{best_model['Model']}** ({best_model['Kaggle_Public_Score']:.6f})")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def generate_comparison_report(self):
        """Generate complete comparison report"""
        print("Generating submission comparison report...")
        
        # Generate prediction statistics
        print("Analyzing prediction statistics...")
        pred_stats_df = self.create_prediction_stats_table()
        pred_stats_md = self.format_prediction_stats_markdown(pred_stats_df)
        
        # Generate submission comparison
        print("Creating submission comparison...")
        submission_df = self.create_submission_comparison_table()
        submission_md = self.format_submission_comparison_markdown(submission_df)
        
        # Combine into single report
        full_report = pred_stats_md + "\n" + submission_md
        
        # Save report
        output_file = self.project_root / 'artifacts' / 'submission_comparison_report.md'
        with open(output_file, 'w') as f:
            f.write(full_report)
        
        print(f"Submission comparison report saved to: {output_file}")
        
        # Save CSV data for further analysis
        pred_stats_csv = self.project_root / 'artifacts' / 'prediction_statistics.csv'
        submission_csv = self.project_root / 'artifacts' / 'submission_comparison.csv'
        
        pred_stats_df.to_csv(pred_stats_csv, index=False)
        submission_df.to_csv(submission_csv, index=False)
        
        print(f"Data saved to:")
        print(f"- {pred_stats_csv}")
        print(f"- {submission_csv}")
        
        return output_file, pred_stats_df, submission_df


if __name__ == "__main__":
    analyzer = SubmissionComparisonAnalyzer()
    analyzer.generate_comparison_report()
    
    print("\n" + "="*60)
    print("SUBMISSION COMPARISON REPORT GENERATED")
    print("="*60)
    print("Files created:")
    print("- artifacts/submission_comparison_report.md")
    print("- artifacts/prediction_statistics.csv")
    print("- artifacts/submission_comparison.csv")
    print("="*60)
