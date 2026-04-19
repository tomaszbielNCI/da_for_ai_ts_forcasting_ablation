#!/usr/bin/env python3
"""
Scientific Visualizations for Model Analysis

Creates comprehensive visualizations for academic submission including:
- Learning curves and convergence analysis
- Prediction distribution comparisons
- Model performance heatmaps
- Feature importance visualizations
- Target convergence analysis
"""

import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for scientific papers
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ScientificVisualizationGenerator:
    """Generate scientific visualizations for model analysis"""
    
    def __init__(self):
        self.project_root = Path('.')
        self.metrics_dir = self.project_root / 'results/metrics'
        self.predictions_dir = self.project_root / 'results/predictions'
        self.output_dir = self.project_root / 'artifacts' / 'visualizations'
        self.output_dir.mkdir(exist_ok=True)
        
        self.horizons = [1, 3, 10, 25]
        
        # Model naming for plots
        self.model_names = {
            'baseline_lgbm': 'Baseline LGBM (Main)',
            'baseline_lgbm_raw': 'Baseline LGBM (Raw)',
            'lgbm_baseline': 'Baseline LGBM',
            'lgbm_shap_10': 'LGBM SHAP-10 (BEST)',
            'lgbm_shap_20': 'LGBM SHAP-20',
            'lgbm_all_plus_shap': 'LGBM All+SHAP',
            'lgbm_bnn-shap10': 'LGBM BNN-SHAP10',
            'lgbm_bnn-aggregated': 'LGBM BNN-Aggregated',
            'xgb_shap_10': 'XGBoost SHAP-10',
            'catboost_shap_10': 'CatBoost SHAP-10',
            'cat': 'Trio CatBoost (WF)',
            'lgbm': 'Trio LGBM (WF)',
            'xgb': 'Trio XGBoost (WF)'
        }
        
        # Color palette for models - highlight LGBM SHAP-10 as best
        self.colors = {
            'baseline_lgbm': '#1f77b4',
            'baseline_lgbm_raw': '#1f77b4',
            'lgbm_baseline': '#1f77b4',
            'lgbm_shap_10': '#FF6B35',  # Bright orange for best model
            'lgbm_shap_20': '#2ca02c',
            'lgbm_all_plus_shap': '#2ca02c',
            'lgbm_bnn-shap10': '#d62728',
            'lgbm_bnn-aggregated': '#9467bd',
            'xgb_shap_10': '#8c564b',
            'catboost_shap_10': '#e377c2',
            'cat': '#7f7f7f',
            'lgbm': '#7f7f7f',
            'xgb': '#7f7f7f'
        }
    
    def load_all_metrics(self) -> Dict:
        """Load all available metrics including walk-forward"""
        all_metrics = {}
        
        # Load regular metrics
        for json_file in self.metrics_dir.glob('metrics_h*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    horizon = data['horizon']
                    model = data['model']
                    
                    # Skip walk-forward window files for now
                    if 'Window' in json_file.name:
                        continue
                    
                    key = f"h{horizon}_{model}"
                    all_metrics[key] = data
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        # Load walk-forward metrics (average across windows)
        wf_models = {}
        for json_file in self.metrics_dir.glob('metrics_wf_h*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    horizon = data['horizon']
                    model = data['model']
                    
                    key = f"h{horizon}_{model}"
                    
                    # Average across windows
                    if key not in wf_models:
                        wf_models[key] = []
                    wf_models[key].append(data)
            except Exception as e:
                print(f"Error loading walk-forward {json_file}: {e}")
        
        # Add averaged walk-forward metrics
        for key, model_data_list in wf_models.items():
            if len(model_data_list) > 0:
                # Average the validation metrics
                avg_data = model_data_list[0].copy()
                for metric in ['weighted_rmse', 'pearson', 'rmse', 'mae', 'r2', 'directional_accuracy']:
                    if metric in avg_data.get('valid', {}):
                        values = [d['valid'].get(metric, 0) for d in model_data_list]
                        avg_data['valid'][metric] = sum(values) / len(values)
                
                all_metrics[key] = avg_data
                
        return all_metrics
    
    def create_performance_heatmap(self) -> str:
        """Create heatmap of model performance across metrics and horizons"""
        metrics = self.load_all_metrics()
        
        # Prepare data for heatmap
        heatmap_data = []
        for key, data in metrics.items():
            horizon = data['horizon']
            model = data['model']
            model_name = self.model_names.get(model, model)
            
            valid_metrics = data['valid']
            
            heatmap_data.append({
                'Model': model_name,
                'Horizon': f'H{horizon}',
                'Weighted RMSE': valid_metrics.get('weighted_rmse', np.nan),
                'Pearson': valid_metrics.get('pearson', np.nan),
                'RMSE': valid_metrics.get('rmse', np.nan),
                'MAE': valid_metrics.get('mae', np.nan),
                'R²': valid_metrics.get('r2', np.nan),
                'Directional Accuracy': valid_metrics.get('directional_accuracy', np.nan)
            })
        
        df = pd.DataFrame(heatmap_data)
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Heatmap Across Horizons and Metrics', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['Weighted RMSE', 'Pearson', 'RMSE', 'MAE', 'R²', 'Directional Accuracy']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i // 3, i % 3]
            
            # Pivot data for heatmap
            pivot_df = df.pivot(index='Model', columns='Horizon', values=metric)
            
            # Create heatmap
            if metric in ['Weighted RMSE', 'RMSE', 'MAE']:
                # Lower is better - use reversed colormap
                sns.heatmap(pivot_df, annot=True, fmt='.6f', cmap='RdYlBu_r', ax=ax, cbar_kws={'label': metric})
            else:
                # Higher is better
                sns.heatmap(pivot_df, annot=True, fmt='.6f', cmap='RdYlBu', ax=ax, cbar_kws={'label': metric})
            
            ax.set_title(f'{metric}', fontweight='bold')
            ax.set_xlabel('Horizon')
            ax.set_ylabel('Model')
        
        plt.tight_layout()
        output_file = self.output_dir / 'performance_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance heatmap saved to: {output_file}")
        return str(output_file)
    
    def create_metric_comparison_bars(self) -> str:
        """Create bar charts comparing metrics across models"""
        metrics = self.load_all_metrics()
        
        # Prepare data
        comparison_data = []
        for key, data in metrics.items():
            horizon = data['horizon']
            model = data['model']
            model_name = self.model_names.get(model, model)
            
            valid_metrics = data['valid']
            
            comparison_data.append({
                'Model': model_name,
                'Model_Key': model,
                'Horizon': f'H{horizon}',
                'Weighted RMSE': valid_metrics.get('weighted_rmse', np.nan),
                'Pearson': valid_metrics.get('pearson', np.nan),
                'R²': valid_metrics.get('r2', np.nan),
                'Directional Accuracy': valid_metrics.get('directional_accuracy', np.nan)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Create subplots for each horizon
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Metric Comparison Across Models by Horizon', fontsize=16, fontweight='bold')
        
        horizons = ['H1', 'H3', 'H10', 'H25']
        metrics_to_plot = ['Weighted RMSE', 'Pearson', 'R²', 'Directional Accuracy']
        
        for i, horizon in enumerate(horizons):
            ax = axes[i // 2, i % 2]
            horizon_df = df[df['Horizon'] == horizon].sort_values('Weighted RMSE')
            
            # Create grouped bar chart
            x = np.arange(len(horizon_df))
            width = 0.2
            
            for j, metric in enumerate(metrics_to_plot):
                values = horizon_df[metric].fillna(0)
                color = self.colors.get(horizon_df['Model_Key'].iloc[j], f'C{j}')
                
                if metric in ['Weighted RMSE']:
                    # Lower is better - invert for visual consistency
                    values = -values
                
                ax.bar(x + j * width, values, width, label=metric, alpha=0.8)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Metric Value')
            ax.set_title(f'{horizon} Comparison', fontweight='bold')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(horizon_df['Model'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / 'metric_comparison_bars.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Metric comparison bars saved to: {output_file}")
        return str(output_file)
    
    def create_prediction_distribution_plots(self) -> str:
        """Create distribution plots for predictions"""
        # Load predictions for available models
        prediction_data = []
        
        for model_key in ['lgbm_baseline', 'lgbm_shap_10', 'lgbm_all_plus_shap', 'xgb_shap_10']:
            model_dir = self.predictions_dir / model_key
            if not model_dir.exists():
                continue
            
            for horizon in self.horizons:
                pred_file = model_dir / f'{model_key}_h{horizon}_predictions.npz'
                if pred_file.exists():
                    try:
                        data = np.load(pred_file)
                        if 'test_pred' in data:
                            predictions = data['test_pred']
                            model_name = self.model_names.get(model_key, model_key)
                            
                            # Sample for efficiency (max 10000 points)
                            if len(predictions) > 10000:
                                sample_indices = np.random.choice(len(predictions), 10000, replace=False)
                                predictions = predictions[sample_indices]
                            
                            prediction_data.extend([{
                                'Model': model_name,
                                'Horizon': f'H{horizon}',
                                'Prediction': pred
                            } for pred in predictions])
                    except Exception as e:
                        print(f"Error loading {pred_file}: {e}")
        
        if not prediction_data:
            print("No prediction data available for distribution plots")
            return None
        
        df = pd.DataFrame(prediction_data)
        
        # Create distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Prediction Distributions Across Models and Horizons', fontsize=16, fontweight='bold')
        
        horizons = ['H1', 'H3', 'H10', 'H25']
        
        for i, horizon in enumerate(horizons):
            ax = axes[i // 2, i % 2]
            horizon_df = df[df['Horizon'] == horizon]
            
            # Create violin plots
            models = horizon_df['Model'].unique()
            data_to_plot = [horizon_df[horizon_df['Model'] == model]['Prediction'].values for model in models]
            
            parts = ax.violinplot(data_to_plot, positions=range(len(models)), showmeans=True, showmedians=True)
            
            # Color the violins
            for j, (pc, model) in enumerate(zip(parts['bodies'], models)):
                model_key = [k for k, v in self.model_names.items() if v == model][0]
                pc.set_facecolor(self.colors.get(model_key, f'C{j}'))
                pc.set_alpha(0.7)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Prediction Values')
            ax.set_title(f'{horizon} Prediction Distributions', fontweight='bold')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / 'prediction_distributions.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Prediction distribution plots saved to: {output_file}")
        return str(output_file)
    
    def create_feature_count_analysis(self) -> str:
        """Create visualization of feature counts vs performance"""
        metrics = self.load_all_metrics()
        
        # Prepare data
        feature_data = []
        for key, data in metrics.items():
            horizon = data['horizon']
            model = data['model']
            model_name = self.model_names.get(model, model)
            
            valid_metrics = data['valid']
            features_used = data.get('features_used', 0)
            
            feature_data.append({
                'Model': model_name,
                'Model_Key': model,
                'Horizon': f'H{horizon}',
                'Features Used': features_used,
                'Weighted RMSE': valid_metrics.get('weighted_rmse', np.nan),
                'Pearson': valid_metrics.get('pearson', np.nan)
            })
        
        df = pd.DataFrame(feature_data)
        
        # Create scatter plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Feature Count vs Performance Analysis', fontsize=16, fontweight='bold')
        
        # Weighted RMSE vs Features
        for model_key in df['Model_Key'].unique():
            model_df = df[df['Model_Key'] == model_key]
            color = self.colors.get(model_key, 'gray')
            ax1.scatter(model_df['Features Used'], model_df['Weighted RMSE'], 
                       label=self.model_names.get(model_key, model_key), 
                       color=color, alpha=0.7, s=100)
        
        ax1.set_xlabel('Number of Features Used')
        ax1.set_ylabel('Weighted RMSE')
        ax1.set_title('Features vs Weighted RMSE', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Pearson vs Features
        for model_key in df['Model_Key'].unique():
            model_df = df[df['Model_Key'] == model_key]
            color = self.colors.get(model_key, 'gray')
            ax2.scatter(model_df['Features Used'], model_df['Pearson'], 
                       label=self.model_names.get(model_key, model_key), 
                       color=color, alpha=0.7, s=100)
        
        ax2.set_xlabel('Number of Features Used')
        ax2.set_ylabel('Pearson Correlation')
        ax2.set_title('Features vs Pearson Correlation', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / 'feature_count_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature count analysis saved to: {output_file}")
        return str(output_file)
    
    def create_training_efficiency_plot(self) -> str:
        """Create visualization of training efficiency"""
        metrics = self.load_all_metrics()
        
        # Prepare data
        efficiency_data = []
        for key, data in metrics.items():
            horizon = data['horizon']
            model = data['model']
            model_name = self.model_names.get(model, model)
            
            valid_metrics = data['valid']
            best_iteration = data.get('best_iteration', 0)
            features_used = data.get('features_used', 0)
            
            efficiency_data.append({
                'Model': model_name,
                'Model_Key': model,
                'Horizon': f'H{horizon}',
                'Best Iteration': best_iteration,
                'Features Used': features_used,
                'Weighted RMSE': valid_metrics.get('weighted_rmse', np.nan),
                'Efficiency_Score': features_used / best_iteration if best_iteration > 0 else np.nan
            })
        
        df = pd.DataFrame(efficiency_data)
        
        # Create efficiency plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Training Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # Iterations vs Performance
        for model_key in df['Model_Key'].unique():
            model_df = df[df['Model_Key'] == model_key]
            color = self.colors.get(model_key, 'gray')
            ax1.scatter(model_df['Best Iteration'], model_df['Weighted RMSE'], 
                       label=self.model_names.get(model_key, model_key), 
                       color=color, alpha=0.7, s=100)
        
        ax1.set_xlabel('Best Iteration')
        ax1.set_ylabel('Weighted RMSE')
        ax1.set_title('Training Iterations vs Performance', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Efficiency score
        efficiency_df = df.dropna(subset=['Efficiency_Score'])
        for model_key in efficiency_df['Model_Key'].unique():
            model_df = efficiency_df[efficiency_df['Model_Key'] == model_key]
            color = self.colors.get(model_key, 'gray')
            ax2.scatter(model_df['Efficiency_Score'], model_df['Weighted RMSE'], 
                       label=self.model_names.get(model_key, model_key), 
                       color=color, alpha=0.7, s=100)
        
        ax2.set_xlabel('Efficiency Score (Features / Iterations)')
        ax2.set_ylabel('Weighted RMSE')
        ax2.set_title('Efficiency vs Performance', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / 'training_efficiency.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training efficiency plot saved to: {output_file}")
        return str(output_file)
    
    def generate_all_visualizations(self) -> List[str]:
        """Generate all scientific visualizations"""
        print("Generating scientific visualizations...")
        
        generated_files = []
        
        # Generate each visualization
        try:
            file_path = self.create_performance_heatmap()
            generated_files.append(file_path)
        except Exception as e:
            print(f"Error creating performance heatmap: {e}")
        
        try:
            file_path = self.create_metric_comparison_bars()
            generated_files.append(file_path)
        except Exception as e:
            print(f"Error creating metric comparison bars: {e}")
        
        try:
            file_path = self.create_prediction_distribution_plots()
            if file_path:
                generated_files.append(file_path)
        except Exception as e:
            print(f"Error creating prediction distribution plots: {e}")
        
        try:
            file_path = self.create_feature_count_analysis()
            generated_files.append(file_path)
        except Exception as e:
            print(f"Error creating feature count analysis: {e}")
        
        try:
            file_path = self.create_training_efficiency_plot()
            generated_files.append(file_path)
        except Exception as e:
            print(f"Error creating training efficiency plot: {e}")
        
        return generated_files


if __name__ == "__main__":
    generator = ScientificVisualizationGenerator()
    files = generator.generate_all_visualizations()
    
    print("\n" + "="*60)
    print("SCIENTIFIC VISUALIZATIONS GENERATED")
    print("="*60)
    print("Files created:")
    for file_path in files:
        if file_path:
            print(f"- {file_path}")
    print("="*60)
    print(f"Output directory: artifacts/visualizations/")
    print("="*60)
