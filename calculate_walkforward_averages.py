#!/usr/bin/env python3
"""
Calculate average validation metrics for Walk Forward Trio Models
"""

import polars as pl

def calculate_walkforward_averages():
    # Load walkforward metrics
    df = pl.read_csv('results/metrics/all_metrics_walkforward.csv')
    
    # Filter for validation results only
    valid_df = df.filter(pl.col('type') == 'valid')
    
    # Group by horizon and model to get averages
    averages = valid_df.group_by(['horizon', 'model']).agg([
        pl.col('weighted_rmse').mean().alias('avg_weighted_rmse'),
        pl.col('pearson').mean().alias('avg_pearson'),
        pl.col('rmse').mean().alias('avg_rmse'),
        pl.col('mae').mean().alias('avg_mae'),
        pl.col('r2').mean().alias('avg_r2'),
        pl.col('directional_accuracy').mean().alias('avg_directional_accuracy'),
        pl.col('mape').mean().alias('avg_mape'),
        pl.col('smape').mean().alias('avg_smape'),
        pl.col('best_iteration').mean().alias('avg_best_iteration'),
        pl.col('features_used').mean().alias('avg_features_used')
    ])
    
    # Sort by horizon and model
    result = averages.sort(['horizon', 'model'])
    
    # Print results for table
    print('Walk Forward Trio Models - Average Validation Metrics Across Windows')
    print('=' * 80)
    for row in result.iter_rows(named=True):
        horizon = row['horizon']
        model = row['model'].upper()
        print(f'| {horizon} | {model} | {row["avg_weighted_rmse"]:.6f} | {row["avg_pearson"]:.6f} | {row["avg_rmse"]:.6f} | {row["avg_mae"]:.6f} | {row["avg_r2"]:.6f} | {row["avg_directional_accuracy"]:.6f} | {row["avg_mape"]:.6f} | {row["avg_smape"]:.6f} | {row["avg_best_iteration"]:.1f} | {row["avg_features_used"]:.0f} |')

if __name__ == "__main__":
    calculate_walkforward_averages()
