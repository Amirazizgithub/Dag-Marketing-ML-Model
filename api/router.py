"""
FastAPI — Causal DAG Prediction API
=====================================
Endpoints:
  POST /model-train       — Train models for all causal relationships
  POST /simulate          — Single-point what-if scenario
  POST /forecast          — Multi-day time-series forecast
  POST /explain           — SHAP-style revenue impact breakdown
  GET  /baselines         — Current baseline values for all KPIs
  GET  /dag               — DAG topology (nodes, edges, layers)
  GET  /model-report      — Model health metrics for all edges
  GET  /health            — Health check

All endpoints return JSON. The React frontend calls these on every slider change.
"""

import sys
import os
from fastapi import Request
from pydantic import BaseModel
from typing import Dict, List, Optional
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from components.model_manager import model_manager
from components.model_trainer import ModelTrainer
from models.dag_engine import CausalDAGEngine
from components.dag_definition import (
    DAG_NODES,
    get_layers,
    get_topological_order,
    get_children,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


app_router = APIRouter()

# ── Global engine instance (loaded at startup) ──
engine: Optional[CausalDAGEngine] = None

# ── Request / Response Models ──


class SimulateRequest(BaseModel):
    """Single-point what-if simulation."""

    client_number: int
    manual_overrides: Dict[str, float]
    locked_kpis: Optional[List[str]] = None
    temporal: Optional[Dict] = None

    class Config:
        json_schema_extra = {
            "example": {
                "client_number": 00000000000000,
                "manual_overrides": {"ad_spend": 25000, "bounce_rate": 35},
                "locked_kpis": ["cpc"],
                "temporal": {"day_of_week": 4, "month": 10, "is_weekend": 0},
            }
        }


# class TrainingRequest(BaseModel):
#     """Trigger model training with optional parameters."""
#     client_number: int
#     project_id: str
#     table_id: str
#     start_date: Optional[str]
#     end_date: Optional[str]

#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "client_number": 00000000000000,
#                 "project_id": "project_123",
#                 "table_id": "table_01",
#                 "start_date": "2026-01-01",
#                 "end_date": "2026-12-31"
#             }
#         }


class ForecastRequest(BaseModel):
    """Multi-day forecast with manual overrides."""

    client_number: int
    manual_overrides: Dict[str, float]
    days: int = 10
    locked_kpis: Optional[List[str]] = None
    ci_level: float = 0.95

    class Config:
        json_schema_extra = {
            "example": {
                "client_number": 00000000000000,
                "manual_overrides": {"ad_spend": 20000},
                "days": 10,
                "locked_kpis": [],
                "ci_level": 0.95,
            }
        }


class ExplainRequest(BaseModel):
    """Explain revenue impact of manual tweaks."""

    client_number: int
    manual_overrides: Dict[str, float]

    class Config:
        json_schema_extra = {
            "example": {
                "client_number": 00000000000000,
                "manual_overrides": {"ad_spend": 30000, "ctr": 3.5},
            }
        }


# ── Endpoints ──


@app_router.get("/health")
async def health(client_number: int = None):
    """Health check endpoint to verify if models are loaded and ready."""
    try:
        if not client_number:
            return JSONResponse(
                status_code=400,
                content={"message": "client_number query parameter is required"},
            )
        try:
            engine = model_manager.get_engine(client_number)
        except Exception as e:
            return JSONResponse(
                status_code=200,
                content={
                    "message": f"Model not trained yet. Please train the model first. Error details: {str(e)}"
                },
            )
        return {
            "status": "healthy" if engine and engine.is_trained else "not_trained",
            "edges_loaded": len(engine.edge_models) if engine else 0,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": f"Health check error: {str(e)}"}
        )


@app_router.get("/baselines")
async def get_baselines(client_number: int = None):
    """Return current baseline values for all KPIs (last 30-day average)."""
    try:
        if not client_number:
            return JSONResponse(
                status_code=400,
                content={"message": "client_number query parameter is required"},
            )
        try:
            engine = model_manager.get_engine(client_number)
        except Exception as e:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Model not trained yet. Please train the model first."
                },
            )
        if not engine or not engine.is_trained:
            return JSONResponse(
                status_code=200,
                content={"message": "Models not trained. Trained Model first."},
            )

        result = {}
        for nid, node in DAG_NODES.items():
            result[nid] = {
                "label": node.label,
                "unit": node.unit,
                "baseline": engine.baselines.get(nid, 0),
                "layer": node.layer,
                "is_root": node.is_root,
                "is_derived": node.is_derived,
                "is_target": node.is_target,
                "parents": node.parents,
                "children": get_children(nid),
                "min_val": node.min_val,
                "max_val": node.max_val if node.max_val != float("inf") else None,
            }

        return {"baselines": result}
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": f"Error fetching baselines: {str(e)}"}
        )


@app_router.get("/dag")
async def get_dag_topology():
    """Return the full DAG structure for frontend rendering."""
    try:
        nodes = {}
        edges = []

        for nid, node in DAG_NODES.items():
            nodes[nid] = {
                "id": nid,
                "label": node.label,
                "layer": node.layer,
                "is_root": node.is_root,
                "is_derived": node.is_derived,
                "is_target": node.is_target,
                "parents": node.parents,
            }
            for pid in node.parents:
                edges.append({"from": pid, "to": nid})

        return {
            "nodes": nodes,
            "edges": edges,
            "layers": get_layers(),
            "topological_order": get_topological_order(),
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error fetching DAG topology: {str(e)}"},
        )


@app_router.post("/training")
async def model_training(request: Request):
    """
    Trigger model training with optional parameters.

    This endpoint allows retraining the models with new data or parameters.
    """
    # Here you would implement the logic to retrain your models based on the parameters in req
    # For example, you could call a function like train_models(req) that encapsulates your training logic
    # Make sure to handle any exceptions and return appropriate responses
    try:
        # Placeholder for training logic
        # You would replace this with your actual training code
        data = await request.json()
        if data.get("client_number") is None:
            return JSONResponse(
                status_code=400,
                content={"message": "client_number is required in the request body"},
            )
        if data.get("project_id") is None:
            return JSONResponse(
                status_code=400,
                content={"message": "project_id is required in the request body"},
            )
        trainer = ModelTrainer(data=data)
        all_metrics = trainer.model_training()
        # 2. PROACTIVE CACHING (The "Warming" Step)
        # Immediately pull those new files from GCS into RAM
        client_number = data["client_number"]
        model_manager.warm_cache(client_number=client_number)

        return JSONResponse(
            content={
                "status": "success",
                "message": "Model trained and cache warmed",
                "metrics": all_metrics,
            },
            status_code=200,
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": f"Training failed: {str(e)}"}
        )


@app_router.post("/simulate")
async def simulate(request: SimulateRequest):
    """
    Run a single-point what-if scenario.

    User sets values for any KPIs → DAG propagates downstream →
    returns all KPI values, sources, and deltas vs baseline.
    """
    try:
        if not request.client_number:
            return JSONResponse(
                status_code=400,
                content={"message": "client_number is required in the request body"},
            )
        try:
            engine = model_manager.get_engine(request.client_number)
        except Exception as e:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Model not trained yet. Please train the model first."
                },
            )
        if not engine or not engine.is_trained:
            return JSONResponse(
                status_code=200,
                content={"message": "Models not trained. Trained Model first."},
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error initializing simulation: {str(e)}"},
        )

    try:
        result = engine.simulate(
            manual_overrides=request.manual_overrides,
            locked_kpis=request.locked_kpis,
            temporal=request.temporal,
        )
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": f"Simulation error: {str(e)}"}
        )


@app_router.post("/forecast")
async def forecast(request: ForecastRequest):
    """
    Generate multi-day time-series forecast.

    Returns daily predictions with confidence intervals.
    This is what powers the line chart in the React frontend.
    """
    try:
        if request.client_number is None:
            return JSONResponse(
                status_code=400,
                content={"message": "client_number is required in the request body"},
            )
        try:
            engine = model_manager.get_engine(request.client_number)
        except Exception as e:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Model not trained yet. Please train the model first."
                },
            )
        if not engine or not engine.is_trained:
            return JSONResponse(
                status_code=200,
                content={"message": "Models not trained. Trained Model first."},
            )

        if request.days < 1 or request.days > 90:
            return JSONResponse(
                status_code=400,
                content={"message": "Forecast horizon must be between 1 and 90 days"},
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error initializing forecast: {str(e)}"},
        )

    try:
        daily_forecasts = engine.forecast(
            manual_overrides=request.manual_overrides,
            days=request.days,
            locked_kpis=request.locked_kpis,
            ci_level=request.ci_level,
        )

        # Aggregate summary
        total_revenue = sum(d["Revenue"]["predicted"] for d in daily_forecasts)
        total_upper = sum(d["Revenue"]["upper"] for d in daily_forecasts)
        total_lower = sum(d["Revenue"]["lower"] for d in daily_forecasts)
        baseline_revenue = engine.baselines.get("revenue", 0) * request.days
        filtered_keys = [
            nid
            for nid, node in DAG_NODES.items()
            if nid not in ["roas", "revenue", "Date"]
        ]
        independent_features = ", ".join(filtered_keys)

        # insight = f"Model Configuration: The machine learning models were trained on select key performance indicators (KPIs) and their interdependencies, using the following independent features: {", ".join(filtered_keys)}."
        # # Append revenue projections to the insight
        # insight += f" Revenue Projections: The forecasted revenue for the upcoming {request.days}-day period is {round(total_revenue, 2)}₹, reflecting a {round((total_revenue - baseline_revenue) / max(baseline_revenue, 1) * 100, 2)}% lift over the baseline of {round(baseline_revenue, 2)}₹."

        return {
            "daily": daily_forecasts,
            "independent_features": independent_features,
            "summary": {
                "total_predicted_revenue": round(total_revenue, 2),
                "total_upper_ci": round(total_upper, 2),
                "total_lower_ci": round(total_lower, 2),
                "baseline_total": round(baseline_revenue, 2),
                "lift_pct": round(
                    (total_revenue - baseline_revenue) / max(baseline_revenue, 1) * 100,
                    2,
                ),
                "forecast_days": request.days,
                "ci_level": request.ci_level,
            },
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": f"Forecast error: {str(e)}"}
        )


@app_router.post("/explain")
async def explain(request: ExplainRequest):
    """
    Explain how each manual tweak impacts revenue.

    Returns a SHAP-style waterfall breakdown:
    baseline → tweak1 impact → tweak2 impact → interaction → final.
    """
    try:
        if request.client_number is None:
            return JSONResponse(
                status_code=400,
                content={"message": "client_number is required in the request body"},
            )
        try:
            engine = model_manager.get_engine(request.client_number)
        except Exception as e:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Model not trained yet. Please train the model first."
                },
            )
        if not engine or not engine.is_trained:
            return JSONResponse(
                status_code=200,
                content={"message": "Models not trained. Trained Model first."},
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error initializing explanation: {str(e)}"},
        )

    try:
        explanation = engine.explain_scenario(request.manual_overrides)
        return explanation
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": f"Explanation error: {str(e)}"}
        )


@app_router.get("/model-report")
async def model_report(client_number: int = None):
    """Full model health report — R², MAPE, feature importances per edge."""
    try:
        if client_number is None:
            return JSONResponse(
                status_code=400,
                content={"message": "client_number is required in the request body"},
            )
        try:
            engine = model_manager.get_engine(client_number)
        except Exception as e:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Model not trained yet. Please train the model first."
                },
            )
        if not engine or not engine.is_trained:
            return JSONResponse(
                status_code=200,
                content={"message": "Models not trained. Trained Model first."},
            )
        return engine.get_model_report()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error fetching model report: {str(e)}"},
        )


# ── Convenience: slider panel endpoint for React frontend ──


@app_router.get("/ui/slider-config")
async def slider_config(client_number: int = None):
    """
    Returns everything the React frontend needs to render the KPI sliders.
    Includes baseline values, min/max ranges, units, layer grouping.
    """
    try:
        if client_number is None:
            return JSONResponse(
                status_code=400,
                content={"message": "client_number is required in the request body"},
            )
        try:
            engine = model_manager.get_engine(client_number)
        except Exception as e:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Model not trained yet. Please train the model first."
                },
            )
        if not engine or not engine.is_trained:
            return JSONResponse(
                status_code=200,
                content={"message": "Models not trained. Trained Model first."},
            )

        sliders = []
        for nid in get_topological_order():
            node = DAG_NODES[nid]
            if node.is_derived:
                continue
            sliders.append(
                {
                    "id": nid,
                    "label": node.label,
                    "unit": node.unit,
                    "layer": node.layer,
                    "is_root": node.is_root,
                    "baseline": engine.baselines.get(nid, 0),
                    "min": node.min_val,
                    "max": (
                        node.max_val
                        if node.max_val != float("inf")
                        else engine.baselines.get(nid, 0) * 5
                    ),
                    "parents": node.parents,
                    "children": get_children(nid),
                }
            )

        return {"sliders": sliders}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error fetching slider config: {str(e)}"},
        )
