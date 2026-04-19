#!/usr/bin/env python3
"""
Horizon-by-Horizon Model Comparison Tables

Creates comprehensive comparison tables for each horizon showing:
- All model approaches with clear naming
- Bold formatting for best results
- Scientific comparison across methodologies
"""

import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple

class HorizonComparisonAnalyzer:
    """Generate horizon-by-horizon comparison tables"""
    
    def __init__(self):
        self.project_root = Path('.')
        self.metrics_dir = self.project_root / 'results/metrics'
        self.horizons = [1, 3, 10, 25]
        
        # Clear model naming mapping
        self.model_names = {
            'lgbm_baseline': 'Baseline LGBM (Raw Features)',
            'baseline_lgbm': 'Baseline LGBM (Main)',
            'baseline_lgbm_raw': 'Baseline LGBM (Raw)',
            'lgbm_shap_10': 'LGBM SHAP-10',
            'lgbm_shap_20': 'LGBM SHAP-20',
            'lgbm_all_plus_shap': 'LGBM All+SHAP (Raw + SHAP-10 Engineered)',
            'lgbm_bnn-shap10': 'LGBM BNN-SHAP10',
            'lgbm_bnn-aggregated': 'LGBM BNN-Aggregated',
            'xgb_shap_10': 'XGBoost SHAP-10',
            'catboost_shap_10': 'CatBoost SHAP-10',
            'cat': 'Trio CatBoost (Walk-Forward)',
            'lgbm': 'Trio LGBM (Walk-Forward)',
            'xgb': 'Trio XGBoost (Walk-Forward)',
            'lgbm_with_bnn': 'LGBM with BNN Features'
        }
        
        # Model categories for organization
        self.model_categories = {
            'baseline': ['lgbm_baseline', 'baseline_lgbm', 'baseline_lgbm_raw'],
            'shap_enhanced': ['lgbm_shap_10', 'lgbm_shap_20', 'lgbm_all_plus_shap'],
            'bnn_enhanced': ['lgbm_bnn-shap10', 'lgbm_bnn-aggregated', 'lgbm_with_bnn'],
            'algorithm_variants': ['xgb_shap_10', 'catboost_shap_10'],
            'walk_forward': ['cat', 'lgbm', 'xgb']
        }
    
    def load_all_metrics(self) -> Dict:
        """Load all available metrics"""
        all_metrics = {}
        
        # Load regular metrics
        for json_file in self.metrics_dir.glob('metrics_h*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    horizon = data['horizon']
                    model = data['model']
                    
                    # Skip walk-forward window files for main comparison
                    if 'Window' in json_file.name:
                        continue
                    
                    key = f"h{horizon}_{model}"
                    all_metrics[key] = data
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        # Load walk-forward metrics (average across windows)
        for json_file in self.metrics_dir.glob('metrics_wf_h*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    horizon = data['horizon']
                    model = data['model']
                    
                    key = f"h{horizon}_{model}"
                    
                    # If we already have this model/horizon, average the metrics
                    if key in all_metrics:
                        existing = all_metrics[key]
                        # Average the validation metrics
                        for metric in ['weighted_rmse', 'pearson', 'rmse', 'mae', 'r2', 'directional_accuracy']:
                            if metric in existing.get('valid', {}) and metric in data.get('valid', {}):
                                existing['valid'][metric] = (existing['valid'][metric] + data['valid'][metric]) / 2
                    else:
                        all_metrics[key] = data
            except Exception as e:
                print(f"Error loading walk-forward {json_file}: {e}")
                
        return all_metrics
    
    def load_walkforward_metrics(self) -> Dict:
        """Load walk-forward metrics (average across windows)"""
        wf_metrics = {}
        
        for horizon in self.horizons:
            # Collect all windows for this horizon
            window_data = {}
            
            for json_file in self.metrics_dir.glob(f'metrics_wf_h{horizon}_Window*.json'):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        model = data['model']
                        
                        if model not in window_data:
                            window_data[model] = []
                        window_data[model].append(data)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
            
            # Average across windows
            for model, windows in window_data.items():
                if len(windows) > 0:
                    avg_metrics = self._average_window_metrics(windows)
                    wf_metrics[f"h{horizon}_trio_{model}"] = {
                        'horizon': horizon,
                        'model': f'trio_{model}',
                        'train': avg_metrics,
                        'valid': avg_metrics,  # Use same for walk-forward
                        'best_iteration': np.mean([w.get('best_iteration', 0) for w in windows]),
                        'features_used': windows[0].get('features_used', 0)
                    }
        
        return wf_metrics
    
    def _average_window_metrics(self, windows: List[Dict]) -> Dict:
        """Average metrics across multiple windows"""
        metrics_keys = windows[0]['train'].keys()
        avg_metrics = {}
        
        for key in metrics_keys:
            values = [w['train'][key] for w in windows if key in w['train']]
            if values:
                avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def create_horizon_table(self, horizon: int) -> pd.DataFrame:
        """Create comparison table for specific horizon"""
        all_metrics = self.load_all_metrics()
        wf_metrics = self.load_walkforward_metrics()
        
        # Combine all metrics
        combined_metrics = {**all_metrics, **wf_metrics}
        
        # Filter for this horizon
        horizon_data = []
        for key, data in combined_metrics.items():
            if data['horizon'] == horizon:
                horizon_data.append(data)
        
        # Create DataFrame
        df_data = []
        for data in horizon_data:
            model_key = data['model']
            model_name = self.model_names.get(model_key, model_key)
            
            valid_metrics = data['valid']
            
            df_data.append({
                'Model': model_name,
                'Model_Key': model_key,
                'Category': self._get_model_category(model_key),
                'Weighted RMSE': valid_metrics.get('weighted_rmse', 0),
                'Pearson': valid_metrics.get('pearson', 0),
                'RMSE': valid_metrics.get('rmse', 0),
                'MAE': valid_metrics.get('mae', 0),
                'R²': valid_metrics.get('r2', 0),
                'Directional Accuracy': valid_metrics.get('directional_accuracy', 0),
                'Best Iteration': data.get('best_iteration', 0),
                'Features Used': data.get('features_used', 0)
            })
        
        df = pd.DataFrame(df_data)
        
        # Sort by category and then by Weighted RMSE
        df = df.sort_values(['Category', 'Weighted RMSE'])
        
        return df
    
    def _get_model_category(self, model_key: str) -> str:
        """Get category for model"""
        for category, models in self.model_categories.items():
            if model_key in models:
                return category.replace('_', ' ').title()
        return 'Other'
    
    def format_markdown_table(self, df: pd.DataFrame, horizon: int) -> str:
        """Format DataFrame as markdown table with bold best results"""
        # Find best values for each metric
        best_metrics = {}
        metrics_to_bold = ['Weighted RMSE', 'Pearson', 'RMSE', 'MAE', 'R²', 'Directional Accuracy']
        
        for metric in metrics_to_bold:
            if metric in ['RMSE', 'MAE']:
                # Lower is better
                best_metrics[metric] = df[metric].min()
            else:
                # Higher is better (Weighted RMSE, Pearson, R², Directional Accuracy)
                best_metrics[metric] = df[metric].max()
        
        # Format table
        table_lines = []
        table_lines.append(f"### Horizon {horizon} - Model Comparison")
        table_lines.append("")
        
        # Header
        header = "| Model | Category | Weighted RMSE | Pearson | RMSE | MAE | R² | Dir. Acc. | Iter. | Features |"
        separator = "|-------|----------|---------------|---------|------|-----|----|-----------|-------|----------|"
        table_lines.append(header)
        table_lines.append(separator)
        
        # Data rows
        for _, row in df.iterrows():
            model_name = row['Model']
            category = row['Category']
            
            # Format metrics with bold for best values
            wrmse = f"**{row['Weighted RMSE']:.6f}**" if row['Weighted RMSE'] == best_metrics['Weighted RMSE'] else f"{row['Weighted RMSE']:.6f}"
            pearson = f"**{row['Pearson']:.6f}**" if row['Pearson'] == best_metrics['Pearson'] else f"{row['Pearson']:.6f}"
            rmse = f"**{row['RMSE']:.6f}**" if row['RMSE'] == best_metrics['RMSE'] else f"{row['RMSE']:.6f}"
            mae = f"**{row['MAE']:.6f}**" if row['MAE'] == best_metrics['MAE'] else f"{row['MAE']:.6f}"
            r2 = f"**{row['R²']:.6f}**" if row['R²'] == best_metrics['R²'] else f"{row['R²']:.6f}"
            dir_acc = f"**{row['Directional Accuracy']:.6f}**" if row['Directional Accuracy'] == best_metrics['Directional Accuracy'] else f"{row['Directional Accuracy']:.6f}"
            
            iter_val = int(row['Best Iteration']) if row['Best Iteration'] > 0 else "N/A"
            features = int(row['Features Used']) if row['Features Used'] > 0 else "N/A"
            
            row_line = f"| {model_name} | {category} | {wrmse} | {pearson} | {rmse} | {mae} | {r2} | {dir_acc} | {iter_val} | {features} |"
            table_lines.append(row_line)
        
        table_lines.append("")
        
        # Add insights
        best_model = df.loc[df['Weighted RMSE'].idxmax(), 'Model']
        best_pearson = df.loc[df['Pearson'].idxmax(), 'Model']
        
        insights = f"**Key Insights for H{horizon}:**\n"
        insights += f"- Best Weighted RMSE: **{best_model}** ({df['Weighted RMSE'].max():.6f})\n"
        insights += f"- Best Pearson: **{best_pearson}** ({df['Pearson'].max():.6f})\n"
        insights += f"- Total models compared: {len(df)}\n"
        
        table_lines.append(insights)
        table_lines.append("")
        
        return "\n".join(table_lines)
    
    def generate_all_horizon_tables(self) -> str:
        """Generate comparison tables for all horizons"""
        all_tables = []
        
        # Header
        all_tables.append("# Horizon-by-Horizon Model Comparison")
        all_tables.append("")
        all_tables.append("Generated: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
        all_tables.append("")
        
        # Model categories explanation
        all_tables.append("## Model Categories")
        all_tables.append("")
        all_tables.append("- **Baseline**: Raw features without SHAP enhancement")
        all_tables.append("- **Shap Enhanced**: SHAP feature selection and engineering")
        all_tables.append("- **Bnn Enhanced**: BNN predictions as additional features")
        all_tables.append("- **Algorithm Variants**: XGBoost and CatBoost implementations")
        all_tables.append("- **Walk Forward**: Time-based cross-validation approach")
        all_tables.append("")
        
        # Generate table for each horizon
        for horizon in self.horizons:
            print(f"Creating H{horizon} comparison table...")
            df = self.create_horizon_table(horizon)
            table_md = self.format_markdown_table(df, horizon)
            all_tables.append(table_md)
        
        # Summary section
        all_tables.append("## Overall Summary")
        all_tables.append("")
        
        # Collect best models across all horizons
        summary_data = []
        for horizon in self.horizons:
            df = self.create_horizon_table(horizon)
            best_model = df.loc[df['Weighted RMSE'].idxmax(), 'Model']
            best_score = df['Weighted RMSE'].max()
            summary_data.append({
                'Horizon': f'H{horizon}',
                'Best Model': best_model,
                'Best Weighted RMSE': best_score
            })
        
        summary_df = pd.DataFrame(summary_data)
        all_tables.append("### Best Models by Horizon")
        all_tables.append("")
        all_tables.append("| Horizon | Best Model | Weighted RMSE |")
        all_tables.append("|---------|------------|---------------|")
        
        for _, row in summary_df.iterrows():
            all_tables.append(f"| {row['Horizon']} | {row['Best Model']} | **{row['Best Weighted RMSE']:.6f}** |")
        
        all_tables.append("")
        
        # Overall best model
        overall_best = summary_df.loc[summary_df['Best Weighted RMSE'].idxmax(), 'Best Model']
        overall_score = summary_df['Best Weighted RMSE'].max()
        
        all_tables.append(f"### Overall Best Performance")
        all_tables.append("")
        all_tables.append(f"**Best Model Overall**: {overall_best}")
        all_tables.append(f"**Best Weighted RMSE**: {overall_score:.6f}")
        all_tables.append("")
        
        return "\n".join(all_tables)
    
    def save_comparison_tables(self):
        """Save all comparison tables to markdown file"""
        print("Generating horizon-by-horizon comparison tables...")
        
        tables_md = self.generate_all_horizon_tables()
        
        output_file = self.project_root / 'artifacts' / 'horizon_comparison_tables.md'
        with open(output_file, 'w') as f:
            f.write(tables_md)
        
        print(f"Comparison tables saved to: {output_file}")
        
        # Also create a summary of available models
        self.create_model_summary()
        
        return output_file
    
    def create_model_summary(self):
        """Create summary of all available models"""
        all_metrics = self.load_all_metrics()
        wf_metrics = self.load_walkforward_metrics()
        
        # Count models by category
        model_counts = {}
        for category, models in self.model_categories.items():
            available = []
            for model in models:
                if any(model in key for key in all_metrics.keys()) or any(f'trio_{model}' in key for key in wf_metrics.keys()):
                    available.append(self.model_names.get(model, model))
            
            model_counts[category.replace('_', ' ').title()] = {
                'total': len(models),
                'available': len(available),
                'models': available
            }
        
        summary_md = "# Model Availability Summary\n\n"
        summary_md += "Generated: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n"
        
        for category, info in model_counts.items():
            summary_md += f"## {category}\n"
            summary_md += f"Available: {info['available']}/{info['total']}\n"
            if info['models']:
                summary_md += "\nModels:\n"
                for model in info['models']:
                    summary_md += f"- {model}\n"
            summary_md += "\n"
        
        output_file = self.project_root / 'artifacts' / 'model_availability_summary.md'
        with open(output_file, 'w') as f:
            f.write(summary_md)
        
        print(f"Model availability summary saved to: {output_file}")


if __name__ == "__main__":
    analyzer = HorizonComparisonAnalyzer()
    analyzer.save_comparison_tables()
    
    print("\n" + "="*60)
    print("HORIZON COMPARISON TABLES GENERATED")
    print("="*60)
    print("Files created:")
    print("- artifacts/horizon_comparison_tables.md")
    print("- artifacts/model_availability_summary.md")
    print("="*60)
