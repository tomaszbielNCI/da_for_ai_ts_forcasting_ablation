#!/usr/bin/env python3
"""
BNN Aggregated Features

Creates aggregated features by grouping correlated features.
Based on hybrid_bnn_lightgbm_features.py - only mean + std per group.
"""

import polars as pl
import logging
from typing import Dict, List, Tuple, Optional, Any


class BNNAggregatedFeatures:
    """
    Create BNN features by aggregating correlated feature groups.

    Features:
    - 9 groups based on correlation analysis
    - 2 aggregated features per group (mean, std)
    - Total: 18 BNN features
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize BNN aggregated features."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Define 9 groups (from notebook analysis)
        self.bnn_groups = {
            'ultra_group': ['feature_bz', 'feature_aw', 'feature_cc'],
            'perfect_negative_group': ['feature_aq', 'feature_ai', 'feature_ar', 'feature_as', 'feature_at'],
            'perfect_positive_group': ['feature_bp', 'feature_br', 'feature_bd', 'feature_be', 'feature_bf'],
            'strong_consistent_group': ['feature_cg', 'feature_cf', 'feature_v', 'feature_h'],
            'moderate_stable_group': ['feature_n', 'feature_l', 'feature_r', 'feature_o', 'feature_p'],
            'high_variance_group': ['feature_q', 'feature_s', 'feature_t', 'feature_u'],
            'horizon_specialists_group': ['feature_w', 'feature_x', 'feature_y', 'feature_z', 'feature_aa'],
            'momentum_like_group': ['feature_ab', 'feature_ac', 'feature_ad', 'feature_ae', 'feature_af'],
            'statistical_moments_group': ['feature_ag', 'feature_ah', 'feature_aj', 'feature_ak'],
            'remaining_group': [
                'feature_a', 'feature_b', 'feature_c', 'feature_d', 'feature_e', 'feature_f', 'feature_g',
                'feature_i', 'feature_j', 'feature_k', 'feature_m', 'feature_al', 'feature_am', 'feature_an',
                'feature_ao', 'feature_ap', 'feature_az', 'feature_bb', 'feature_bh', 'feature_bl',
                'feature_bm', 'feature_bn', 'feature_bo', 'feature_bq', 'feature_bs', 'feature_bt',
                'feature_bu', 'feature_bv', 'feature_bw', 'feature_bx', 'feature_by', 'feature_ca',
                'feature_cb', 'feature_cd', 'feature_ch'
            ]
        }

    def create_aggregated_features(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, List[str]]:
        """
        Create aggregated features (mean + std) for each group.

        Returns:
            Tuple of (df_with_features, list_of_new_feature_names)
        """
        self.logger.info("Creating BNN aggregated features (mean + std per group)")

        aggregated_features = []

        for group_name, features in self.bnn_groups.items():
            existing = [f for f in features if f in df.columns]
            if len(existing) < 2:
                self.logger.warning(f"Group {group_name}: only {len(existing)} features, skipping")
                continue

            # Mean
            df = df.with_columns(
                pl.mean_horizontal([pl.col(f) for f in existing]).alias(f'{group_name}_mean')
            )

            # Standard deviation
            mean_expr = pl.col(f'{group_name}_mean')
            variance_expr = pl.sum_horizontal([(pl.col(f) - mean_expr) ** 2 for f in existing]) / len(existing)
            df = df.with_columns(
                variance_expr.sqrt().alias(f'{group_name}_std')
            )

            aggregated_features.append(f'{group_name}_mean')
            aggregated_features.append(f'{group_name}_std')

            self.logger.info(f"  {group_name}: {len(existing)} features -> mean + std")

        self.logger.info(f"Created {len(aggregated_features)} aggregated features")
        return df, aggregated_features

    def create_all_bnn_features(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, List[str]]:
        """Create all BNN features (aggregated groups only)."""
        self.logger.info("Creating all BNN features...")

        df, bnn_features = self.create_aggregated_features(df)

        self.logger.info(f"Total BNN features created: {len(bnn_features)}")
        return df, bnn_features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load data - use absolute path from project root
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data/cleaned/train_clean.parquet'
    
    print(f"Loading data from: {data_path}")
    df = pl.read_parquet(data_path)

    # Create BNN features
    bnn = BNNAggregatedFeatures()
    df_with_bnn, bnn_cols = bnn.create_all_bnn_features(df)

    print(f"\nFinal shape: {df_with_bnn.shape}")
    print(f"BNN features: {len(bnn_cols)}")
    print(f"First 5 BNN features: {bnn_cols[:5]}")
    
    # Save BNN features
    output_dir = project_root / 'data/processed/bnn_features'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'train_with_bnn_features.parquet'
    df_with_bnn.write_parquet(output_path)
    print(f"BNN features saved to: {output_path}")
    
    # Save feature list
    feature_list_path = output_dir / 'bnn_feature_list.txt'
    with open(feature_list_path, 'w') as f:
        f.write(f"Total BNN features: {len(bnn_cols)}\n\n")
        for i, feature in enumerate(bnn_cols, 1):
            f.write(f"{i}. {feature}\n")
    print(f"BNN feature list saved to: {feature_list_path}")