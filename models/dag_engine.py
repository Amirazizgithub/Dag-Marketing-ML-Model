"""
Causal DAG Engine
==================
The core engine that:
  1. Trains all edge models from historical data
  2. Propagates user tweaks through the DAG (top-down)
  3. Handles manual overrides at any node
  4. Generates time-series forecasts with confidence intervals
  5. Provides SHAP explanations per edge

Usage:
    engine = CausalDAGEngine()
    engine.train(historical_df)

    # Single-point what-if
    result = engine.simulate({"ad_spend": 25000, "bounce_rate": 35})

    # Multi-day forecast
    forecast = engine.forecast({"ad_spend": 25000}, days=10)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from utils.logger import log
from components.dag_definition import (
    DAG_NODES,
    get_topological_order,
    validate_dag,
)
from models.edge_model import EdgeModel, EdgeMetrics
from components.storage_manager import StorageManager


class CausalDAGEngine:
    """
    Main engine for the Causal DAG prediction system.
    """

    def __init__(self, data: Dict, baseline_window: str = None) -> None:
        self.data = data
        self.baseline_window: str = baseline_window
        self.edge_models: Dict[str, EdgeModel] = {}
        self.baselines: Dict[str, float] = {}
        self.historical_std: Dict[str, float] = {}
        self.topo_order = get_topological_order()
        self.is_trained = False
        self.storage_manager = StorageManager(data=self.data)

        # Validate DAG on init
        is_valid, errors = validate_dag()
        if not is_valid:
            log.error(f"Invalid DAG topology: {errors}")
            raise ValueError(f"Invalid DAG: {errors}")

    def train(
        self,
        df: pd.DataFrame,
        test_size: int,
        model_type: str,
        verbose: bool = True,
    ) -> Dict[str, EdgeMetrics]:
        """
        Train all edge models from historical data.

        Args:
            df: Historical DataFrame with all KPI columns + temporal features
            test_size: Holdout days for evaluation
            model_type: "lightgbm" or "linear"
            verbose: Print training progress

        Returns:
            Dict of {child_id: EdgeMetrics} for all trained edges
        """
        if verbose:
            log.info("=" * 60)
            log.info("TRAINING CAUSAL DAG — EDGE MODELS")
            log.info("=" * 60)

        all_metrics = {}

        # Compute baselines (last 30 days mean)
        df_baseline_window = df.iloc[-self.baseline_window :]
        for nid in DAG_NODES:
            if nid in df.columns:
                self.baselines[nid] = round(df_baseline_window[nid].mean(), 4)
                self.historical_std[nid] = round(df[nid].std(), 4)

        # Train edge model for each non-root, non-derived node
        for nid in self.topo_order:
            node = DAG_NODES[nid]

            if node.is_root:
                if verbose:
                    log.info(f"[{nid}] ROOT — no model needed (user input)")
                continue

            if node.is_derived:
                if verbose:
                    log.info(f"[{nid}] DERIVED — computed via formula")
                continue

            if verbose:
                log.info(f"[{nid}] Training edge: {node.parents} → {nid}")

            edge = EdgeModel(
                data=self.data,
                child_id=nid,
                parent_ids=node.parents,
                use_temporal=True,
                use_lags=True,
                model_type=model_type,
            )

            metrics = edge.train(df, test_size=test_size)
            self.edge_models[nid] = edge
            all_metrics[nid] = metrics

            if verbose:
                log.info(
                    f"    R² train={metrics.r2_train:.3f}  test={metrics.r2_test:.3f}"
                )
                log.info(
                    f"    MAE test={metrics.mae_test:.1f}  MAPE={metrics.mape_test:.1f}%   SMAPE={metrics.smape_test:.1f}%"
                )
                top_feats = sorted(
                    metrics.feature_importances.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]
                feat_str = ", ".join(f"{k}={v:.2f}" for k, v in top_feats)
                log.info(f"    Top features: {feat_str}")

        self.is_trained = True

        if verbose:
            log.info("\n" + "=" * 60)
            log.info("TRAINING COMPLETE")
            log.info(f"  Edges trained: {len(self.edge_models)}")
            avg_r2 = round(np.mean([m.r2_test for m in all_metrics.values()]), 3)
            avg_mape = round(np.mean([m.mape_test for m in all_metrics.values()]), 2)
            avg_smape = round(np.mean([m.smape_test for m in all_metrics.values()]), 2)
            log.info(f"  Avg R² (test): {avg_r2:.3f}")
            log.info(f"  Avg MAPE (test): {avg_mape:.1f}%")
            log.info(f"  Avg SMAPE (test): {avg_smape:.1f}%")
            log.info("=" * 60)

        return all_metrics

    def simulate(
        self,
        manual_overrides: Dict[str, float],
        locked_kpis: Optional[List[str]] = None,
        temporal: Optional[Dict] = None,
    ) -> Dict:
        """
        Run a what-if scenario through the DAG.

        Args:
            manual_overrides: {kpi_id: value} for user-tweaked KPIs
            locked_kpis: KPI IDs to hold at baseline (immune to cascade)
            temporal: {day_of_week, month, is_weekend} for temporal context

        Returns:
            Dict with:
              - values: {kpi_id: predicted_value}
              - sources: {kpi_id: "manual" | "predicted" | "locked" | "baseline" | "derived"}
              - deltas: {kpi_id: {baseline, predicted, change_pct}}
        """
        if not self.is_trained:
            log.error("Engine not trained. Call train() before simulate().")
            raise RuntimeError("Engine not trained. Call train() first.")

        locked_kpis = set(locked_kpis or [])
        values = {}
        sources = {}

        # Default temporal context (Wednesday, June, weekday)
        if temporal is None:
            now = datetime.now()
            temporal = {
                "day_of_week": now.weekday(),
                "month": now.month,
                "is_weekend": int(now.weekday() >= 5),
            }

        # Propagate through DAG in topological order
        for nid in self.topo_order:
            node = DAG_NODES[nid]

            # Case 1: User manually set this value
            if nid in manual_overrides:
                values[nid] = manual_overrides[nid]
                sources[nid] = "manual"
                continue

            # Case 2: Locked at baseline
            if nid in locked_kpis:
                values[nid] = self.baselines.get(nid, 0)
                sources[nid] = "locked"
                continue

            # Case 3: Root node with no override → use baseline
            if node.is_root:
                values[nid] = self.baselines.get(nid, 0)
                sources[nid] = "baseline"
                continue

            # Case 4: Derived node → compute from formula
            if node.is_derived:
                if nid == "roas":
                    ad_spend = values.get("ad_spend", 1)
                    revenue = values.get("revenue", 0)
                    values[nid] = round(revenue / max(ad_spend, 1), 2)
                else:
                    values[nid] = 0
                sources[nid] = "derived"
                continue

            # Case 5: Predict using edge model
            if nid in self.edge_models:
                parent_values = {pid: values.get(pid, 0) for pid in node.parents}
                lag_baseline = self.baselines.get(nid, 0)
                predicted = self.edge_models[nid].predict(
                    parent_values, temporal, lag_baseline=lag_baseline
                )

                # Clamp to valid range
                predicted = np.clip(predicted, node.min_val, node.max_val)
                values[nid] = round(predicted, 4)
                sources[nid] = "predicted"
            else:
                values[nid] = self.baselines.get(nid, 0)
                sources[nid] = "baseline"

        # Compute deltas vs baseline
        deltas = {}
        for nid, val in values.items():
            baseline = self.baselines.get(nid, 0)
            change_pct = ((val - baseline) / baseline * 100) if baseline else 0
            deltas[nid] = {
                "baseline": round(baseline, 2),
                "predicted": round(val, 2),
                "change_pct": round(change_pct, 2),
            }

        return {
            "values": {k: round(v, 2) for k, v in values.items()},
            "sources": sources,
            "deltas": deltas,
        }

    def forecast(
        self,
        manual_overrides: Dict[str, float],
        days: int = 10,
        locked_kpis: Optional[List[str]] = None,
        ci_level: float = 0.95,
    ) -> List[Dict]:
        """
        Generate multi-day time-series forecast.

        Args:
            manual_overrides: {kpi_id: daily_value} for user-set KPIs
            days: Number of forecast days
            locked_kpis: KPIs to hold at baseline
            ci_level: Confidence interval level (0.80 or 0.95)

        Returns:
            List of daily forecast dicts with predicted, upper, lower CI
        """
        z_score = 1.96 if ci_level >= 0.95 else 1.28  # 95% or 80%
        last_train_date = (
            datetime.now()
        )  # In real case, should be last date in training data
        forecasts = []

        for i in range(1, days + 1):
            forecast_date = last_train_date + timedelta(days=i)
            temporal = {
                "day_of_week": forecast_date.weekday(),
                "month": forecast_date.month,
                "is_weekend": int(forecast_date.weekday() >= 5),
            }

            # Run DAG simulation for this day
            result = self.simulate(
                manual_overrides=manual_overrides,
                locked_kpis=locked_kpis,
                temporal=temporal,
            )

            # Compute CI for revenue (widens with horizon)
            revenue = result["values"].get("revenue", 0)
            rev_std = self.historical_std.get("revenue", revenue * 0.1)
            horizon_factor = np.sqrt(i / days)  # CI widens over time
            ci_width = z_score * rev_std * horizon_factor * 0.3  # Dampened

            # forecasts.append(
            #     {
            #         "day": i,
            #         "date": forecast_date.strftime("%Y-%m-%d"),
            #         "date_label": forecast_date.strftime("%d/%m"),
            #         "day_of_week": temporal["day_of_week"],
            #         "values": result["values"],
            #         "sources": result["sources"],
            #         "revenue": {
            #             "predicted": round(revenue, 2),
            #             "upper": round(revenue + ci_width, 2),
            #             "lower": round(max(0, revenue - ci_width), 2),
            #             "ci_level": ci_level,
            #         },
            #     }
            # )
            forecasts.append(
                {
                    "Date": forecast_date.strftime("%Y-%m-%d"),
                    "Revenue": {
                        "predicted": round(revenue, 2),
                        "upper": round(revenue + ci_width, 2),
                        "lower": round(max(0, revenue - ci_width), 2),
                        "ci_level": ci_level,
                    },
                }
            )

        return forecasts

    def explain_scenario(self, manual_overrides: Dict[str, float]) -> Dict:
        """
        SHAP-style explanation of how each manual tweak impacts revenue.

        Returns a waterfall: baseline → tweak1 contribution → ... → final revenue.
        """
        # Get baseline scenario (no tweaks)
        baseline_result = self.simulate({})
        baseline_revenue = baseline_result["values"].get("revenue", 0)

        # Get tweaked scenario
        tweaked_result = self.simulate(manual_overrides)
        tweaked_revenue = tweaked_result["values"].get("revenue", 0)

        # Marginal contribution: add one tweak at a time
        contributions = {}
        for kpi_id, value in manual_overrides.items():
            partial_overrides = {kpi_id: value}
            partial_result = self.simulate(partial_overrides)
            partial_revenue = partial_result["values"].get("revenue", 0)
            contributions[kpi_id] = {
                "tweak_value": value,
                "baseline_value": self.baselines.get(kpi_id, 0),
                "revenue_impact": round(partial_revenue - baseline_revenue, 2),
                "revenue_impact_pct": round(
                    (partial_revenue - baseline_revenue)
                    / max(baseline_revenue, 1)
                    * 100,
                    2,
                ),
            }

        # Interaction effect (total != sum of parts)
        sum_individual = sum(c["revenue_impact"] for c in contributions.values())
        interaction = tweaked_revenue - baseline_revenue - sum_individual

        return {
            "baseline_revenue": round(baseline_revenue, 2),
            "predicted_revenue": round(tweaked_revenue, 2),
            "total_impact": round(tweaked_revenue - baseline_revenue, 2),
            "total_impact_pct": round(
                (tweaked_revenue - baseline_revenue) / max(baseline_revenue, 1) * 100, 2
            ),
            "per_tweak_contributions": contributions,
            "interaction_effect": round(interaction, 2),
            "interaction_note": (
                "Positive interaction = tweaks amplify each other. "
                "Negative = diminishing returns when combined."
            ),
        }

    def get_model_report(self) -> Dict:
        """Full model health report for all edges."""
        report = {"edges": {}, "summary": {}}

        for nid, edge in self.edge_models.items():
            if edge.metrics:
                report["edges"][nid] = {
                    "parents": edge.parent_ids,
                    "r2_test": edge.metrics.r2_test,
                    "mape_test": edge.metrics.mape_test,
                    "smape_test": edge.metrics.smape_test,
                    "mae_test": edge.metrics.mae_test,
                    "top_features": dict(
                        sorted(
                            edge.metrics.feature_importances.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:5]
                    ),
                }

        # Summary stats
        r2_scores = [e["r2_test"] for e in report["edges"].values()]
        mape_scores = [e["mape_test"] for e in report["edges"].values()]
        smape_scores = [e["smape_test"] for e in report["edges"].values()]

        report["summary"] = {
            "total_edges": len(self.edge_models),
            "avg_r2": round(np.mean(r2_scores), 3) if r2_scores else 0,
            "min_r2": round(min(r2_scores), 3) if r2_scores else 0,
            "avg_mape": round(np.mean(mape_scores), 2) if mape_scores else 0,
            "max_mape": round(max(mape_scores), 2) if mape_scores else 0,
            "avg_smape": round(np.mean(smape_scores), 2) if smape_scores else 0,
            "max_smape": round(max(smape_scores), 2) if smape_scores else 0,
            "weak_edges": [
                nid for nid, e in report["edges"].items() if e["r2_test"] < 0.7
            ],
        }

        return report

    def save(self):
        """Save all edge models and engine state to disk."""

        for nid, edge in self.edge_models.items():
            edge.save()

        state = {
            "baselines": self.baselines,
            "historical_std": self.historical_std,
            "topo_order": self.topo_order,
        }
        self.storage_manager.save_metadata(
            metadata=state, file_name="engine_state.json"
        )

        log.info(f"Engine saved to GCS Bucket: {self.storage_manager.bucket_name}")

    def load(self):
        """Load all edge models and engine state from disk."""
        state_file = "engine_state.json"
        state = self.storage_manager.load_metadata(file_name=state_file)

        if not state:
            raise FileNotFoundError(
                f"Engine state file '{state_file}' not found in GCS. "
                "Train and save the engine first."
            )

        self.baselines = state["baselines"]
        self.historical_std = state["historical_std"]
        self.topo_order = state["topo_order"]

        for nid in self.topo_order:
            node = DAG_NODES[nid]
            if node.is_root or node.is_derived:
                continue
            try:
                self.edge_models[nid] = EdgeModel(data=self.data).load(child_id=nid)
            except FileNotFoundError:
                log.error(f"  WARNING: No saved model for edge '{nid}'")

        self.is_trained = True
        log.info(
            f"Engine loaded from {self.storage_manager.bucket_name}/ ({len(self.edge_models)} edges)"
        )
