from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from api.router import app_router

app = FastAPI(
    title="Causal DAG Prediction API",
    description="What-if simulator for ecommerce KPI forecasting",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    try:
        return JSONResponse(
            content={"message": "Welcome to the Causal DAG Prediction API"},
            status_code=200,
        )
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)


app.include_router(app_router, prefix="/api/v1")
