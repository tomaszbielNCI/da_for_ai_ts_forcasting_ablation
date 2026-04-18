#!/usr/bin/env python3
"""
BNN Model on Aggregated Features

Bayesian Neural Network using aggregated features from BNNAggregatedFeatures.
Based on notebook architecture - STABLE VERSION.
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

# Import aggregated features builder
import sys
sys.path.append(str(Path(__file__).parent.parent))
from features.bnn_aggregated_features import BNNAggregatedFeatures


class BNNAggregated:
    """
    Bayesian Neural Network using aggregated features.
    Uses BNNAggregatedFeatures to create features if needed.
    Saves train and test predictions separately.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Paths
        project_root = Path(__file__).parent.parent.parent
        self.models_dir = project_root / 'results/models/bnn_aggregated'
        self.predictions_dir = project_root / 'results/predictions/bnn_aggregated'

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

        # Validation split
        self.train_split = 3000
        self.valid_start = 3001
        self.valid_end = 3600

        # Feature builder
        self.feature_builder = BNNAggregatedFeatures()

        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

    def get_aggregated_features(self, df: pl.DataFrame) -> List[str]:
        """Get aggregated feature columns from the builder."""
        df_with_features, feature_cols = self.feature_builder.create_all_bnn_features(df)
        return feature_cols

    def add_aggregated_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add aggregated features to dataframe."""
        df_with_features, _ = self.feature_builder.create_all_bnn_features(df)
        return df_with_features

    def build_model(self, window: int, n_features: int) -> tf.keras.Model:
        """Build BNN model (same architecture as notebook)."""
        inputs = layers.Input(shape=(window, n_features))

        x = layers.BatchNormalization()(inputs)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
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
        return models.Model(inputs=inputs, outputs=outputs)

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

    def train_horizon(self, horizon, train_df, test_df):
        window = self.window_size[horizon]

        # Add aggregated features
        train_df = self.add_aggregated_features(train_df)
        test_df = self.add_aggregated_features(test_df)

        # Get feature columns
        _, feature_cols = self.feature_builder.create_all_bnn_features(train_df)

        train_h = train_df.filter(pl.col('horizon') == horizon).sort('ts_index')
        test_h = test_df.filter(pl.col('horizon') == horizon).sort('ts_index')

        X = train_h.select(feature_cols).to_numpy().astype(np.float32)
        y = train_h['y_target'].to_numpy().astype(np.float32)
        y = np.clip(y, -100.0, 100.0)

        # Normalize
        mean_X = np.mean(X, axis=0, keepdims=True)
        std_X = np.std(X, axis=0, keepdims=True) + 1e-6
        X = (X - mean_X) / std_X

        # Time split
        ts = train_h['ts_index'].to_numpy()
        train_mask = ts <= self.train_split
        valid_mask = (ts >= self.valid_start) & (ts <= self.valid_end)

        X_train, y_train = X[train_mask], y[train_mask]
        X_valid, y_valid = X[valid_mask], y[valid_mask]

        # Windows
        X_tr, y_tr = self.create_windows(X_train, y_train, window)
        X_val, y_val = self.create_windows(X_valid, y_valid, window)

        # Build and train
        model = self.build_model(window, len(feature_cols))
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss=self.nll)

        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=self.patience,
                                             restore_best_weights=True, verbose=1)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

        model.fit(X_tr, y_tr, validation_data=(X_val, y_val), batch_size=self.batch_size,
                  epochs=self.epochs, verbose=1, callbacks=[early_stop, reduce_lr])

        # ========== PREDICT ON ALL TRAINING DATA ==========
        X_full_all = train_h.select(feature_cols).to_numpy().astype(np.float32)
        X_full_all = (X_full_all - mean_X) / std_X

        n_full = len(X_full_all) - window
        if n_full > 0:
            X_full_windows = np.array([X_full_all[i:i+window] for i in range(n_full)], dtype=np.float32)
            pred_full = model.predict(X_full_windows)
            train_mean_pred = pred_full[..., 0].flatten()
            train_scale_pred = pred_full[..., 1].flatten()

            train_mean = np.full(len(train_h), 0.0, dtype=np.float64)
            train_scale = np.full(len(train_h), 0.0, dtype=np.float64)
            train_mean[window:] = train_mean_pred
            train_scale[window:] = train_scale_pred
        else:
            train_mean = np.zeros(len(train_h))
            train_scale = np.zeros(len(train_h))

        # ========== PREDICT ON TEST DATA ==========
        X_test = test_h.select(feature_cols).to_numpy().astype(np.float32)
        X_test = (X_test - mean_X) / std_X

        n_test = len(X_test) - window
        if n_test > 0:
            X_te = np.array([X_test[i:i + window] for i in range(n_test)], dtype=np.float32)
            pred_test = model.predict(X_te)
            test_mean_pred = pred_test[..., 0].flatten()
            test_scale_pred = pred_test[..., 1].flatten()

            test_mean = np.full(len(test_h), 0.0, dtype=np.float64)
            test_scale = np.full(len(test_h), 0.0, dtype=np.float64)
            test_mean[window:] = test_mean_pred
            test_scale[window:] = test_scale_pred
        else:
            test_mean = np.zeros(len(test_h))
            test_scale = np.zeros(len(test_h))

        # Save model
        model.save(self.models_dir / f'bnn_agg_h{horizon}.keras')

        # Save BOTH train and test predictions
        np.savez(self.predictions_dir / f'bnn_agg_h{horizon}_predictions.npz',
                 train_mean=train_mean, train_scale=train_scale,
                 test_mean=test_mean, test_scale=test_scale)

        # Print diagnostics
        print(f"\n  H={horizon} BNN Results:")
        print(f"    Train mean range: [{np.min(train_mean):.4f}, {np.max(train_mean):.4f}]")
        print(f"    Train scale range: [{np.min(train_scale):.4f}, {np.max(train_scale):.4f}]")
        print(f"    Test mean range: [{np.min(test_mean):.4f}, {np.max(test_mean):.4f}]")
        print(f"    Test scale range: [{np.min(test_scale):.4f}, {np.max(test_scale):.4f}]")

        return test_mean, test_scale

    def train_all_horizons(self, train_df: pl.DataFrame, test_df: pl.DataFrame) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Train BNN for all horizons."""
        self.logger.info("Starting BNN aggregated training...")

        results = {}
        for horizon in self.horizons:
            print(f"\n{'=' * 60}")
            print(f"HORIZON: {horizon}")
            print(f"{'=' * 60}")

            mean, scale = self.train_horizon(horizon, train_df, test_df)
            results[horizon] = (mean, scale)

        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    project_root = Path(__file__).parent.parent.parent
    train_path = project_root / 'data/cleaned/train_clean.parquet'
    test_path = project_root / 'data/cleaned/test_clean.parquet'

    train_df = pl.read_parquet(train_path)
    test_df = pl.read_parquet(test_path)

    bnn = BNNAggregated()
    predictions = bnn.train_all_horizons(train_df, test_df)

    print("\n" + "=" * 60)
    print("BNN Aggregated training complete")
    print("=" * 60)