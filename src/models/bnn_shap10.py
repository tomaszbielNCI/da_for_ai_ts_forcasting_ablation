#!/usr/bin/env python3
"""
BNN Model on Top 10 SHAP Features

Bayesian Neural Network using top 10 SHAP features + target encoding.
Dynamically adds target encoding (no hardcoding).
Uses features from shap_features.py.
Saves BOTH train and test predictions for later use in LGBM.
"""

import numpy as np
import polars as pl
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, models, callbacks
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

tfd = tfp.distributions

# Force legacy Keras
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Import features from shap_features
import sys
sys.path.append(str(Path(__file__).parent.parent))
from features.shap_features import SHAPFeatureEngineer


class BNNShap10:
    """
    Bayesian Neural Network using top 10 SHAP features + target encoding.
    Dynamically adds target encoding (no hardcoding).
    Saves train and test predictions separately.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Paths
        project_root = Path(__file__).parent.parent.parent
        self.data_dir = project_root / 'data/cleaned'
        self.models_dir = project_root / 'results/models/bnn_shap10'
        self.predictions_dir = project_root / 'results/predictions/bnn_shap10'

        for dir_path in [self.models_dir, self.predictions_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Configuration (from notebook)
        self.window_size = {1: 10, 3: 15, 10: 20, 25: 25}
        self.epochs = 30
        self.batch_size = 128
        self.patience = 10
        self.learning_rate = 0.0005

        self.horizons = [1, 3, 10, 25]
        self.seed = 42

        # Validation split (from notebook)
        self.train_split = 3000
        self.valid_start = 3001
        self.valid_end = 3600
        self.max_ts_train = 3601

        # Feature engineer (for top 10 SHAP features)
        self.feature_engineer = SHAPFeatureEngineer()

        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

    def add_target_encoding(self, train_df: pl.DataFrame, test_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Add target encoding (sub_category_te, sub_code_te) - from notebook."""
        self.logger.info("Adding target encoding...")

        # Compute on training data only (ts <= 3000) - no leakage
        te_train = train_df.filter(pl.col('ts_index') <= self.train_split)

        # Mean y_target per sub_category
        sub_category_means = te_train.group_by('sub_category').agg(
            pl.col('y_target').mean().alias('sub_category_te')
        )

        # Mean y_target per sub_code
        sub_code_means = te_train.group_by('sub_code').agg(
            pl.col('y_target').mean().alias('sub_code_te')
        )

        # Global mean for missing categories
        global_mean = te_train['y_target'].mean()

        # Add to train
        train_df = train_df.join(sub_category_means, on='sub_category', how='left')
        train_df = train_df.join(sub_code_means, on='sub_code', how='left')
        train_df = train_df.with_columns(
            pl.col('sub_category_te').fill_null(global_mean),
            pl.col('sub_code_te').fill_null(global_mean)
        )

        # Add to test
        test_df = test_df.join(sub_category_means, on='sub_category', how='left')
        test_df = test_df.join(sub_code_means, on='sub_code', how='left')
        test_df = test_df.with_columns(
            pl.col('sub_category_te').fill_null(global_mean),
            pl.col('sub_code_te').fill_null(global_mean)
        )

        return train_df, test_df

    def load_data(self, horizon: int) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load cleaned data and add target encoding."""
        train_path = self.data_dir / 'train_clean.parquet'
        test_path = self.data_dir / 'test_clean.parquet'

        train_df = pl.read_parquet(train_path)
        test_df = pl.read_parquet(test_path)

        # Filter by horizon
        train_df = train_df.filter(pl.col('horizon') == horizon)
        test_df = test_df.filter(pl.col('horizon') == horizon)

        # Add target encoding
        train_df, test_df = self.add_target_encoding(train_df, test_df)

        self.logger.info(f"Loaded H={horizon}: train {train_df.shape}, test {test_df.shape}")
        return train_df, test_df

    def get_bnn_features(self, horizon: int, train_df: pl.DataFrame) -> List[str]:
        """Get BNN numeric features (top 10 SHAP + target encoding)."""
        # Get top 10 SHAP features from feature_engineer (no hardcoding!)
        top_features = self.feature_engineer.get_top_features(horizon, n_features=10)

        # Target encoding columns (added dynamically)
        target_cols = ['sub_category_te', 'sub_code_te']

        # Combine and filter existing
        all_features = top_features + target_cols
        existing = [f for f in all_features if f in train_df.columns]

        self.logger.info(f"H={horizon}: using {len(existing)} features")
        return existing

    def create_mappings(self, df: pl.DataFrame) -> Tuple[Dict, Dict, Dict, int, int, int]:
        """Create mappings for categorical features."""
        code_mapping = {v: i for i, v in enumerate(df['code'].unique().to_list())}
        subcode_mapping = {v: i for i, v in enumerate(df['sub_code'].unique().to_list())}
        subcat_mapping = {v: i for i, v in enumerate(df['sub_category'].unique().to_list())}

        return code_mapping, subcode_mapping, subcat_mapping, len(code_mapping), len(subcode_mapping), len(subcat_mapping)

    def build_model(self, window: int, n_features: int, n_code: int, n_subcode: int, n_subcat: int) -> tf.keras.Model:
        """Build BNN model (from notebook - NO CHANGES)."""
        numeric_input = layers.Input(shape=(window, n_features), name='numeric')

        code_input = layers.Input(shape=(1,), name='code', dtype=tf.int32)
        subcode_input = layers.Input(shape=(1,), name='sub_code', dtype=tf.int32)
        subcat_input = layers.Input(shape=(1,), name='sub_category', dtype=tf.int32)

        code_embed = layers.Embedding(n_code, 8, name='code_embed')(code_input)
        code_embed = layers.Flatten()(code_embed)

        subcode_embed = layers.Embedding(n_subcode, 12, name='subcode_embed')(subcode_input)
        subcode_embed = layers.Flatten()(subcode_embed)

        subcat_embed = layers.Embedding(n_subcat, 4, name='subcat_embed')(subcat_input)
        subcat_embed = layers.Flatten()(subcat_embed)

        x = layers.BatchNormalization()(numeric_input)
        x = layers.GlobalAveragePooling1D()(x)

        concat = layers.Concatenate()([x, code_embed, subcode_embed, subcat_embed])

        x = layers.Dense(64, activation='relu')(concat)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        params = layers.Dense(2)(x)
        mu = layers.Lambda(lambda p: p[..., 0:1])(params)
        log_scale = layers.Lambda(lambda p: p[..., 1:2])(params)
        log_scale = layers.Lambda(lambda s: s - 2.3)(log_scale)
        scale = layers.Lambda(lambda s: tf.nn.softplus(s) + 1e-4)(log_scale)

        outputs = layers.Concatenate()([mu, scale])

        model = models.Model(
            inputs=[numeric_input, code_input, subcode_input, subcat_input],
            outputs=outputs
        )
        return model

    def nll(self, y_true, y_pred):
        mu = y_pred[..., 0:1]
        scale = y_pred[..., 1:2]
        scale = tf.maximum(scale, 1e-4)
        mu = tf.clip_by_value(mu, -100.0, 100.0)
        scale = tf.clip_by_value(scale, 1e-4, 100.0)
        dist = tfd.Laplace(loc=mu, scale=scale)
        return -dist.log_prob(y_true)

    def create_windows(self, X, y, window):
        n = len(X) - window
        X_w, y_w = [], []
        for i in range(n):
            X_w.append(X[i:i + window])
            y_w.append(y[i + window])
        return np.array(X_w, dtype=np.float32), np.array(y_w, dtype=np.float32)

    def train_horizon(self, horizon, train_df, test_df, code_mapping, subcode_mapping, subcat_mapping,
                      n_code, n_subcode, n_subcat):
        window = self.window_size[horizon]
        feature_cols = self.get_bnn_features(horizon, train_df)

        if not feature_cols:
            self.logger.warning(f"No features for H={horizon}")
            return np.zeros(len(test_df)), np.zeros(len(test_df))

        # Prepare data
        X = train_df.select(feature_cols).to_numpy().astype(np.float32)
        y = train_df['y_target'].to_numpy().astype(np.float32)
        y = np.clip(y, -100.0, 100.0)

        codes = np.array([code_mapping.get(c, 0) for c in train_df['code'].to_list()]).reshape(-1, 1)
        subcodes = np.array([subcode_mapping.get(c, 0) for c in train_df['sub_code'].to_list()]).reshape(-1, 1)
        subcats = np.array([subcat_mapping.get(c, 0) for c in train_df['sub_category'].to_list()]).reshape(-1, 1)

        # Normalize
        mean_X = np.mean(X, axis=0, keepdims=True)
        std_X = np.std(X, axis=0, keepdims=True) + 1e-6
        X = (X - mean_X) / std_X

        # Time split
        ts = train_df['ts_index'].to_numpy()
        train_mask = ts <= self.train_split
        valid_mask = (ts >= self.valid_start) & (ts <= self.valid_end)

        X_train, y_train = X[train_mask], y[train_mask]
        X_valid, y_valid = X[valid_mask], y[valid_mask]

        codes_train = codes[train_mask]
        codes_valid = codes[valid_mask]
        subcodes_train = subcodes[train_mask]
        subcodes_valid = subcodes[valid_mask]
        subcats_train = subcats[train_mask]
        subcats_valid = subcats[valid_mask]

        # Windows
        X_tr, y_tr = self.create_windows(X_train, y_train, window)
        X_val, y_val = self.create_windows(X_valid, y_valid, window)

        if len(X_tr) == 0 or len(X_val) == 0:
            self.logger.warning(f"Not enough windows for H={horizon}")
            return np.zeros(len(test_df)), np.zeros(len(test_df))

        # Code windows
        code_tr = np.array([codes_train[i + window] for i in range(len(X_tr))], dtype=np.int32)
        code_val = np.array([codes_valid[i + window] for i in range(len(X_val))], dtype=np.int32)
        subcode_tr = np.array([subcodes_train[i + window] for i in range(len(X_tr))], dtype=np.int32)
        subcode_val = np.array([subcodes_valid[i + window] for i in range(len(X_val))], dtype=np.int32)
        subcat_tr = np.array([subcats_train[i + window] for i in range(len(X_tr))], dtype=np.int32)
        subcat_val = np.array([subcats_valid[i + window] for i in range(len(X_val))], dtype=np.int32)

        # Build and train
        model = self.build_model(window, len(feature_cols), n_code, n_subcode, n_subcat)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss=self.nll)

        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True, verbose=1)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

        model.fit([X_tr, code_tr, subcode_tr, subcat_tr], y_tr,
                  validation_data=([X_val, code_val, subcode_val, subcat_val], y_val),
                  batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                  callbacks=[early_stop, reduce_lr])

        # ========== PREDICT ON ALL TRAINING DATA ==========
        X_full_all = train_df.select(feature_cols).to_numpy().astype(np.float32)
        X_full_all = (X_full_all - mean_X) / std_X

        codes_full = np.array([code_mapping.get(c, 0) for c in train_df['code'].to_list()]).reshape(-1, 1)
        subcodes_full = np.array([subcode_mapping.get(c, 0) for c in train_df['sub_code'].to_list()]).reshape(-1, 1)
        subcats_full = np.array([subcat_mapping.get(c, 0) for c in train_df['sub_category'].to_list()]).reshape(-1, 1)

        n_full = len(X_full_all) - window
        if n_full > 0:
            X_full_windows = []
            codes_full_windows = []
            subcodes_full_windows = []
            subcats_full_windows = []

            for i in range(n_full):
                X_full_windows.append(X_full_all[i:i+window])
                codes_full_windows.append(codes_full[i+window])
                subcodes_full_windows.append(subcodes_full[i+window])
                subcats_full_windows.append(subcats_full[i+window])

            X_full_windows = np.array(X_full_windows, dtype=np.float32)
            codes_full_windows = np.array(codes_full_windows, dtype=np.int32)
            subcodes_full_windows = np.array(subcodes_full_windows, dtype=np.int32)
            subcats_full_windows = np.array(subcats_full_windows, dtype=np.int32)

            pred_full = model.predict([X_full_windows, codes_full_windows, subcodes_full_windows, subcats_full_windows])
            train_mean_pred = pred_full[..., 0].flatten()
            train_scale_pred = pred_full[..., 1].flatten()

            train_mean = np.full(len(train_df), 0.0, dtype=np.float64)
            train_scale = np.full(len(train_df), 0.0, dtype=np.float64)
            train_mean[window:] = train_mean_pred
            train_scale[window:] = train_scale_pred
        else:
            train_mean = np.zeros(len(train_df))
            train_scale = np.zeros(len(train_df))

        # ========== PREDICT ON TEST DATA ==========
        X_test = test_df.select(feature_cols).to_numpy().astype(np.float32)
        X_test = (X_test - mean_X) / std_X

        codes_test = np.array([code_mapping.get(c, 0) for c in test_df['code'].to_list()]).reshape(-1, 1)
        subcodes_test = np.array([subcode_mapping.get(c, 0) for c in test_df['sub_code'].to_list()]).reshape(-1, 1)
        subcats_test = np.array([subcat_mapping.get(c, 0) for c in test_df['sub_category'].to_list()]).reshape(-1, 1)

        n_test = len(X_test) - window
        if n_test > 0:
            X_test_windows = []
            codes_test_windows = []
            subcodes_test_windows = []
            subcats_test_windows = []

            for i in range(n_test):
                X_test_windows.append(X_test[i:i+window])
                codes_test_windows.append(codes_test[i+window])
                subcodes_test_windows.append(subcodes_test[i+window])
                subcats_test_windows.append(subcats_test[i+window])

            X_test_windows = np.array(X_test_windows, dtype=np.float32)
            codes_test_windows = np.array(codes_test_windows, dtype=np.int32)
            subcodes_test_windows = np.array(subcodes_test_windows, dtype=np.int32)
            subcats_test_windows = np.array(subcats_test_windows, dtype=np.int32)

            pred_test = model.predict([X_test_windows, codes_test_windows, subcodes_test_windows, subcats_test_windows])
            test_mean_pred = pred_test[..., 0].flatten()
            test_scale_pred = pred_test[..., 1].flatten()

            test_mean = np.full(len(test_df), 0.0, dtype=np.float64)
            test_scale = np.full(len(test_df), 0.0, dtype=np.float64)
            test_mean[window:] = test_mean_pred
            test_scale[window:] = test_scale_pred
        else:
            test_mean = np.zeros(len(test_df))
            test_scale = np.zeros(len(test_df))

        # Save model
        model.save(self.models_dir / f'bnn_shap10_h{horizon}.keras')

        # Save BOTH train and test predictions
        np.savez(self.predictions_dir / f'bnn_shap10_h{horizon}_predictions.npz',
                 train_mean=train_mean, train_scale=train_scale,
                 test_mean=test_mean, test_scale=test_scale)

        # Print diagnostics
        print(f"\n  H={horizon} BNN Results:")
        print(f"    Train mean range: [{np.min(train_mean):.4f}, {np.max(train_mean):.4f}]")
        print(f"    Train scale range: [{np.min(train_scale):.4f}, {np.max(train_scale):.4f}]")
        print(f"    Test mean range: [{np.min(test_mean):.4f}, {np.max(test_mean):.4f}]")
        print(f"    Test scale range: [{np.min(test_scale):.4f}, {np.max(test_scale):.4f}]")

        return test_mean, test_scale

    def train_all_horizons(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Train BNN for all horizons."""
        self.logger.info("Starting BNN SHAP-10 training...")

        # Load first horizon to get mappings
        first_train, _ = self.load_data(1)
        code_mapping, subcode_mapping, subcat_mapping, n_code, n_subcode, n_subcat = self.create_mappings(first_train)

        results = {}
        for horizon in self.horizons:
            print(f"\n{'=' * 60}")
            print(f"HORIZON: {horizon}")
            print(f"{'=' * 60}")

            train_df, test_df = self.load_data(horizon)

            mean, scale = self.train_horizon(horizon, train_df, test_df,
                                             code_mapping, subcode_mapping, subcat_mapping,
                                             n_code, n_subcode, n_subcat)
            results[horizon] = (mean, scale)

        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    bnn = BNNShap10()
    predictions = bnn.train_all_horizons()

    print("\n" + "=" * 60)
    print("BNN SHAP-10 training complete")
    print("=" * 60)