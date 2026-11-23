"""
FastAPI Serving Application
Owned by: Person 2 (Backend Developer & API Engineer)

Production-grade REST microservice providing:
- Real-time 4G/5G cellular base station anomaly detection
- 16-step sliding window validation
- Dynamic operational threshold tuning (0.50 Safety Net vs 0.66 Balanced)
- High-performance singleton model execution (sub-15ms inference)
- Multi-cell batch prediction endpoint
- Structured CORS, latency measurement, and health monitoring
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to sys.path so ml module can be imported cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from ml.inference import load_artifacts, predict_window


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Warms up the ML model and scaler on startup so live API requests
    experience zero cold-start latency.
    """
    print("[API Startup] Pre-loading ResNet-18 model and MinMaxScaler into memory...")
    try:
        model, scaler, device, _ = load_artifacts()
        print(f"[API Startup] Model & scaler successfully initialized on device: {device}")
    except Exception as e:
        print(f"[API Startup ERROR] Failed to pre-load artifacts: {e}")
    yield
    print("[API Shutdown] Releasing resources and shutting down server.")


app = FastAPI(
    title="Telecom QoS Anomaly Detection API",
    description="""
### Real-Time Cellular Network Anomaly Detection Platform

This microservice ingests 16-step (4-hour) chronological windows of 3 key cellular metrics:
- **PRBUsageDL (Red Channel):** Downlink Physical Resource Block radio congestion %
- **meanThr_DL (Green Channel):** Delivered downlink throughput speed (Mbps)
- **meanUE_DL (Blue Channel):** Active User Equipment connected count

The telemetry is converted on-the-fly into **2D Gramian Angular Summation Field (GASF)** RGB matrices
and classified by a fine-tuned **PyTorch ResNet-18** deep learning backbone.

#### Operational Threshold Modes:
- **0.50 (Default / Safety Net):** Maximizes Anomaly Recall (70.7%) for mission-critical outage prevention.
- **0.66 (Optimal F1 / Balanced):** Cuts false alarms by 45% and raises precision to 48.7%.
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware Configuration
# ---------------------------------------------------------------------------

# 1. CORS Middleware (allows web dashboards and external clients to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 2. Performance & Request Timing Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handles window length and KPI validation errors with HTTP 422."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc), "error_type": "ValidationError"},
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["General"])
def root():
    """Root metadata endpoint providing service overview and documentation links."""
    return {
        "service": "Telecom QoS Anomaly Detection API",
        "version": "1.0.0",
        "owner": "Person 2 (Backend Developer)",
        "docs_url": "/docs",
        "health_url": "/health",
        "status": "online",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """
    Health check endpoint to verify API readiness and loaded neural network weights.
    Used by Docker healthcheck probes and Kubernetes readiness checks.
    """
    try:
        model, scaler, _, _ = load_artifacts()
        is_ready = model is not None and scaler is not None
    except Exception:
        is_ready = False

    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        model_architecture="ResNet-18 (GASF 3-Channel RGB)",
        service="Telecom QoS Anomaly Detection API",
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    """
    Evaluates a 16-step sliding window (4 hours) of 3 cellular KPIs.
    Transforms data into 3-channel GAF images and classifies via PyTorch ResNet-18.
    Supports dynamic operational decision threshold tuning.
    """
    try:
        kpi_list = [reading.model_dump() for reading in request.window]
        threshold = request.threshold if request.threshold is not None else 0.50

        # Execute ResNet-18 inference pipeline
        result = predict_window(kpi_list, threshold=threshold)

        return PredictResponse(
            cell_name=request.cell_name,
            status=result["status"],
            anomaly_label=result["anomaly_label"],
            confidence=result["confidence"],
            threshold_applied=threshold,
            inference_time_ms=result["inference_time_ms"],
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(batch_request: BatchPredictRequest):
    """
    Evaluates multiple base stations simultaneously in a single batch request.
    Ideal for Network Operations Center (NOC) periodic sweeps across dozens of antennas.
    """
    batch_start = time.perf_counter()
    results = []
    anomalies_count = 0

    try:
        for item in batch_request.requests:
            kpi_list = [reading.model_dump() for reading in item.window]
            threshold = item.threshold if item.threshold is not None else 0.50

            result = predict_window(kpi_list, threshold=threshold)
            if result["anomaly_label"] == 1:
                anomalies_count += 1

            results.append(
                PredictResponse(
                    cell_name=item.cell_name,
                    status=result["status"],
                    anomaly_label=result["anomaly_label"],
                    confidence=result["confidence"],
                    threshold_applied=threshold,
                    inference_time_ms=result["inference_time_ms"],
                )
            )

        total_elapsed_ms = (time.perf_counter() - batch_start) * 1000.0

        return BatchPredictResponse(
            total_cells=len(results),
            anomalies_detected=anomalies_count,
            results=results,
            total_batch_time_ms=round(total_elapsed_ms, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Batch error: {str(e)}")
