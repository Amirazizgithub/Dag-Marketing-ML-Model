"""
Edge Model — Per-edge LightGBM wrapper
========================================
Each edge in the Causal DAG is a separate lightweight model.
Supports: LightGBM (default), Linear Regression (fallback for small data).

Features per edge model:
  - Parent KPI values (the causal inputs)
  - Temporal features: day_of_week, month, is_weekend
  - Lag features: 7-day rolling mean of the target (optional)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from utils.logger import log
from typing import Dict, List, Optional
from dataclasses import dataclass
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from components.storage_manager import StorageManager


@dataclass
class EdgeMetrics:
    """Metrics for a single edge model."""

    r2_train: float
    r2_test: float
    mae_train: float
    mae_test: float
    mape_test: float
    smape_test: float
    feature_importances: Dict[str, float]


class EdgeModel(StorageManager):
    """
    Wrapper for a single edge in the Causal DAG.

    Each edge predicts a child KPI from its parent KPIs + temporal features.
    Uses LightGBM by default, falls back to LinearRegression for sparse data.
    """

    def __init__(
        self,
        data: Dict,
        child_id: str = None,
        parent_ids: List[str] = None,
        use_temporal: bool = True,
        use_lags: bool = True,
        model_type: str = "lightgbm",  # "lightgbm" or "linear"
    ) -> None:
        super().__init__(data=data)
        self.child_id = child_id
        self.parent_ids = parent_ids or []
        self.use_temporal = use_temporal
        self.use_lags = use_lags
        self.model_type = model_type
        self.model = None
        self.metrics: Optional[EdgeMetrics] = None
        self.feature_names: List[str] = []
        self.is_trained = False

    def symmetric_mean_absolute_percentage_error(self, y_true, y_pred):
        return (
            100
            / len(y_true)
            * np.sum(
                2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)
            )
        )

    def _build_features(
        self, df: pd.DataFrame, is_training: bool = True
    ) -> pd.DataFrame:
        """Construct feature matrix for this edge."""
        features = pd.DataFrame(index=df.index)

        # Parent KPI values (primary causal inputs)
        for pid in self.parent_ids:
            if pid in df.columns:
                features[pid] = df[pid].values

        # Temporal features
        if self.use_temporal:
            if "day_of_week" in df.columns:
                features["day_of_week"] = df["day_of_week"].values
            if "month" in df.columns:
                features["month"] = df["month"].values
            if "is_weekend" in df.columns:
                features["is_weekend"] = df["is_weekend"].values

        # Lag features (rolling averages of the target — only for training/batch)
        if self.use_lags and is_training and self.child_id in df.columns:
            features[f"{self.child_id}_lag7_mean"] = (
                df[self.child_id].rolling(window=7, min_periods=1).mean().values
            )
            features[f"{self.child_id}_lag14_mean"] = (
                df[self.child_id].rolling(window=14, min_periods=1).mean().values
            )

        self.feature_names = list(features.columns)
        return features

    def train(self, df: pd.DataFrame, test_size: int) -> EdgeMetrics:
        """
        Train the edge model on historical data.

        Args:
            df: Full historical DataFrame with all KPI columns
            test_size: Number of days to hold out for testing (from the end)

        Returns:
            EdgeMetrics with train/test scores and feature importances
        """
        # Split chronologically
        train_df = df.iloc[:-test_size].copy()
        test_df = df.iloc[-test_size:].copy()

        X_train = self._build_features(train_df, is_training=True)
        y_train = train_df[self.child_id].values

        X_test = self._build_features(test_df, is_training=True)
        y_test = test_df[self.child_id].values

        # Drop NaN rows (from lag features)
        mask_train = ~(X_train.isna().any(axis=1) | np.isnan(y_train))
        X_train = X_train[mask_train]
        y_train = y_train[mask_train]

        mask_test = ~(X_test.isna().any(axis=1) | np.isnan(y_test))
        X_test = X_test[mask_test]
        y_test = y_test[mask_test]
        log.debug(f"  Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

        if self.model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                verbose=-1,
                random_state=42,
            )
            log.debug("  Using LightGBM model for edge training.")
        else:
            self.model = Ridge(alpha=1.0)
            log.debug("  Using Ridge Regression model for edge training.")

        self.model.fit(X_train, y_train)
        self.is_trained = True
        log.debug("  Model training completed.")

        # Evaluate
        pred_train = self.model.predict(X_train)
        pred_test = self.model.predict(X_test)

        # Feature importances
        if self.model_type == "lightgbm":
            importances = dict(zip(self.feature_names, self.model.feature_importances_))
        else:
            importances = dict(zip(self.feature_names, np.abs(self.model.coef_)))
        log.debug(f"Raw feature importances computed for edge '{self.child_id}'.")

        # Normalize importances
        total = sum(importances.values()) or 1
        importances = {k: round(v / total, 4) for k, v in importances.items()}

        self.metrics = EdgeMetrics(
            r2_train=round(r2_score(y_train, pred_train), 4),
            r2_test=round(r2_score(y_test, pred_test), 4),
            mae_train=round(mean_absolute_error(y_train, pred_train), 2),
            mae_test=round(mean_absolute_error(y_test, pred_test), 2),
            mape_test=round(
                mean_absolute_percentage_error(y_test + 1e-4, pred_test) * 100, 2
            ),
            smape_test=round(
                self.symmetric_mean_absolute_percentage_error(y_test, pred_test), 2
            ),
            feature_importances=importances,
        )
        log.info(
            f"  Edge '{self.child_id}' trained. R2 test: {self.metrics.r2_test}, MAE test: {self.metrics.mae_test}, MAPE test: {self.metrics.mape_test}%"
        )
        return self.metrics

    def predict(
        self,
        parent_values: Dict[str, float],
        temporal: Optional[Dict] = None,
        lag_baseline: Optional[float] = None,
    ) -> float:
        """
        Predict the child KPI value from parent KPI values.

        Args:
            parent_values: Dict of {parent_id: value} for all parents
            temporal: Optional dict with day_of_week, month, is_weekend
            lag_baseline: Baseline value of the child KPI (used as lag proxy)

        Returns:
            Predicted value for the child KPI
        """
        if not self.is_trained:
            raise RuntimeError(f"Edge model for '{self.child_id}' not trained yet")

        row = {}
        for pid in self.parent_ids:
            row[pid] = parent_values.get(pid, 0)

        if self.use_temporal and temporal:
            row["day_of_week"] = temporal.get("day_of_week", 3)
            row["month"] = temporal.get("month", 6)
            row["is_weekend"] = temporal.get("is_weekend", 0)

        # For lag features during inference, use the baseline value of the child
        # This tells the model "historically, this KPI has been around X"
        if self.use_lags:
            for fname in self.feature_names:
                if fname not in row and lag_baseline is not None:
                    row[fname] = lag_baseline
                elif fname not in row:
                    row[fname] = 0

        # Build as DataFrame to preserve feature names (avoids sklearn warning)
        X = pd.DataFrame([row], columns=self.feature_names)
        pred = self.model.predict(X)[0]
        return max(0, pred)  # Clamp to non-negative

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Predict for a full DataFrame (used for backtesting)."""
        X = self._build_features(df, is_training=True)
        X = X.fillna(0)
        return self.model.predict(X)

    def save(self):
        """Save model and metadata to disk."""

        model_name = f"edge_{self.child_id}.pkl"
        metadata_name = f"edge_{self.child_id}_meta.json"

        meta = {
            "child_id": self.child_id,
            "parent_ids": self.parent_ids,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
            "use_temporal": self.use_temporal,
            "use_lags": self.use_lags,
            "metrics": (
                {
                    "r2_train": self.metrics.r2_train,
                    "r2_test": self.metrics.r2_test,
                    "mae_test": self.metrics.mae_test,
                    "mape_test": self.metrics.mape_test,
                    "smape_test": self.metrics.smape_test,
                    "feature_importances": self.metrics.feature_importances,
                }
                if self.metrics
                else None
            ),
        }
        self.save_model_and_metadata(
            model=self.model,
            model_name=model_name,
            metadata=meta,
            metadata_name=metadata_name,
        )

    def load(self, child_id: str) -> "EdgeModel":
        """Load a saved edge model."""
        model, meta = self.load_model_and_metadata(
            model_name=f"edge_{child_id}.pkl",
            metadata_name=f"edge_{child_id}_meta.json",
        )
        if not meta:
            raise FileNotFoundError(f"Metadata for edge_{child_id} not found in GCS.")

        edge = self.__class__(
            data=self.data,
            child_id=meta["child_id"],
            parent_ids=meta["parent_ids"],
            model_type=meta["model_type"],
            use_temporal=meta["use_temporal"],
            use_lags=meta["use_lags"],
        )
        edge.feature_names = meta["feature_names"]
        edge.model = model
        edge.is_trained = True

        if meta.get("metrics"):
            m = meta["metrics"]
            edge.metrics = EdgeMetrics(
                r2_train=m["r2_train"],
                r2_test=m["r2_test"],
                mae_train=m.get("mae_train", 0),
                mae_test=m["mae_test"],
                mape_test=m["mape_test"],
                smape_test=m.get("smape_test", 0),
                feature_importances=m["feature_importances"],
            )

        return edge
