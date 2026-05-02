"""
Train Pipeline — Causal DAG
=============================
Generates synthetic data (replace with your DB connector),
trains all edge models, evaluates, and saves to disk.

Usage:
    python train.py                         # Train with defaults
    python train.py --days 730 --model linear  # 2 years, linear models
    python train.py --data-path real_data.csv  # Use real data
"""

import sys
import os
import time
from typing import Dict
from models.dag_engine import CausalDAGEngine
from components.data_loading import DataLoader
from components.dag_definition import validate_dag
import pandas as pd
from fastapi.encoders import jsonable_encoder
from utils.logger import log

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ModelTrainer(DataLoader):
    """Configuration for training pipeline."""

    def __init__(self, data: Dict):
        self.data = data
        super().__init__(data=self.data)

    def normalize_metrics(self, metrics_dict) -> Dict:
        """
        Converts a dictionary of EdgeMetrics objects into a
        standard nested dictionary.
        """
        normalized = {}

        for key, value in metrics_dict.items():
            # Check if the value is an EdgeMetrics object
            # We use vars() to extract all attributes like r2_train, mae_test, etc.
            if hasattr(value, "__dict__"):
                normalized[key] = vars(value)
            else:
                # If it's already a dict or other type, keep as is
                normalized[key] = value
        log.info("Metrics normalized for JSON serialization.")
        return normalized

    def model_training(self) -> Dict:

        log.info("\n" + "=" * 60)
        log.info("CAUSAL DAG — TRAINING PIPELINE")
        log.info("=" * 60)

        # Step 1: Validate DAG
        log.info("\n[1/4] Validating DAG topology...")
        is_valid, errors = validate_dag()
        if not is_valid:
            log.error(f"  FATAL: DAG validation failed:")
            for e in errors:
                log.error(f"    - {e}")
            sys.exit(1)
        log.info("  DAG is valid (no cycles, all references resolved)")

        # Step 2: Load data
        log.info("Loading data...")
        df = self.load_data()
        # Ensure temporal features exist
        if "day_of_week" not in df.columns:
            df["day_of_week"] = pd.to_datetime(df[self.DATE_VARIABLE]).dt.weekday
        if "month" not in df.columns:
            df["month"] = pd.to_datetime(df[self.DATE_VARIABLE]).dt.month
        if "is_weekend" not in df.columns:
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        log.debug(f"  Data shape: {df.shape}")
        log.debug(
            f"  Date range: {df[self.DATE_VARIABLE].min().date()} to {df[self.DATE_VARIABLE].max().date()}"
        )
        log.debug(f"  Columns: {list(df.columns)}")

        # Step 3: Train
        log.info(
            f"Training edge models (type={self.MODEL_TYPE}, test_size={self.TEST_SIZE})..."
        )
        start_time = time.time()

        engine = CausalDAGEngine(data=self.data, baseline_window=self.BASELINE_WINDOW)
        all_metrics = engine.train(
            df=df,
            test_size=self.TEST_SIZE,
            model_type=self.MODEL_TYPE,
            verbose=True,
        )

        train_time = time.time() - start_time
        log.info(f"\n  Training time: {train_time:.1f}s")

        # Step 4: Save
        log.info(f"Saving models to {self.OUTPUT_DIR}/...")
        engine.save()

        # Quick sanity check: simulate baseline scenario
        log.info("\n" + "-" * 40)
        log.info("SANITY CHECK — Baseline scenario:")
        baseline = engine.simulate({})
        for nid in [
            "ad_spend",
            "impressions",
            "clicks",
            "conversions",
            "revenue",
            "roas",
        ]:
            val = baseline["values"].get(nid, 0)
            src = baseline["sources"].get(nid, "?")
            log.info(f"  {nid:>20s} = {val:>12,.2f}  ({src})")

        # Quick sanity check: ad_spend +50%
        log.info("\nSANITY CHECK — Ad Spend +50%:")
        boosted = engine.simulate({"ad_spend": engine.baselines["ad_spend"] * 1.5})
        for nid in [
            "ad_spend",
            "impressions",
            "clicks",
            "conversions",
            "revenue",
            "roas",
        ]:
            base_val = baseline["values"].get(nid, 0)
            new_val = boosted["values"].get(nid, 0)
            delta = ((new_val - base_val) / base_val * 100) if base_val else 0
            log.info(f"  {nid:>20s} = {new_val:>12,.2f}  ({delta:+.1f}%)")

        # Model report
        report = engine.get_model_report()
        if report["summary"].get("weak_edges"):
            log.warning(
                f"\n  WARNING: Weak edges (R² < 0.7): {report['summary']['weak_edges']}"
            )
            log.info(
                "  Consider: more data, different features, or hyperparameter tuning"
            )

        log.info("\n" + "=" * 60)
        log.info("TRAINING COMPLETE")
        normalized_metrics = self.normalize_metrics(all_metrics)
        serializable_metrics = jsonable_encoder(normalized_metrics)
        # print(f"Normalized Metrics: {serializable_metrics}")
        return serializable_metrics
