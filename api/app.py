"""
FastAPI Serving Application
Owned by: Person 2 (Backend Developer)
"""

import sys
from pathlib import Path

# Add project root to sys.path so ml module can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from api.schemas import PredictRequest, PredictResponse, HealthResponse
from ml.inference import predict_window

app = FastAPI(
    title="Telecom QoS Anomaly Detection API",
    description="Real-time 4G/5G cellular network anomaly detection using Computer Vision (GAF + ResNet-18)",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint to verify API readiness."""
    return HealthResponse()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Evaluates a 16-step sliding window (4 hours) of 3 cellular KPIs.
    Returns anomaly classification, confidence, and inference latency.
    """
    try:
        # Convert Pydantic models to dict list for inference module
        kpi_list = [reading.model_dump() for reading in request.window]
        result = predict_window(kpi_list)

        return PredictResponse(
            cell_name=request.cell_name,
            status=result["status"],
            anomaly_label=result["anomaly_label"],
            confidence=result["confidence"],
            inference_time_ms=result["inference_time_ms"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
