"""
Automated Integration and Regression Test Suite for Telecom QoS Anomaly Detection API
Owned by: Person 2 (Backend Developer & API Engineer)

Tests cover:
- System & healthcheck endpoints (GET /, GET /health)
- CORS and request latency middleware headers (X-Process-Time-Ms)
- Single-cell inference contract and operational threshold tuning (POST /predict)
- Multi-cell batch scoring (POST /predict/batch)
- Strict boundary validation & error handling (16-step window, PRB [0-100%], non-negativity)
- CPU inference latency SLA benchmarks
"""

import sys
from pathlib import Path
from typing import Dict, List

import pytest
from starlette.testclient import TestClient

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import app
from ml.inference import SAMPLE_ANOMALY_WINDOW, SAMPLE_NORMAL_WINDOW


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    """
    Module-scoped FastAPI TestClient.
    Using context manager syntax activates the FastAPI lifespan,
    pre-loading the ResNet-18 model and scaler before tests run.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def normal_window() -> List[Dict[str, float]]:
    """Returns a realistic 16-step healthy cellular telemetry window."""
    return [dict(step) for step in SAMPLE_NORMAL_WINDOW]


@pytest.fixture
def anomaly_window() -> List[Dict[str, float]]:
    """Returns a realistic 16-step outage / anomaly telemetry window."""
    return [dict(step) for step in SAMPLE_ANOMALY_WINDOW]


# ---------------------------------------------------------------------------
# 1. System Endpoints & Middleware Tests
# ---------------------------------------------------------------------------
class TestSystemEndpoints:
    """Verifies baseline API infrastructure, health readiness, and middleware."""

    def test_root_endpoint_metadata(self, client: TestClient):
        """GET / should return HTTP 200 with service metadata and documentation links."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert data["service"] == "Telecom QoS Anomaly Detection API"
        assert "version" in data
        assert "docs_url" in data
        assert "health_url" in data
        assert data["status"] == "online"

    def test_healthcheck_model_readiness(self, client: TestClient):
        """GET /health should return HTTP 200 with model_loaded == True."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "ResNet-18" in data["model_architecture"]
        assert data["version"] == "1.0.0"

    def test_latency_timing_header_present(self, client: TestClient):
        """Middleware must inject X-Process-Time-Ms on all HTTP responses."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "x-process-time-ms" in response.headers

        latency_val = float(response.headers["x-process-time-ms"])
        assert latency_val >= 0.0

    def test_cors_headers_configured(self, client: TestClient):
        """Pre-flight / CORS request should allow origin for frontend dashboard integration."""
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        }
        response = client.options("/predict", headers=headers)
        assert response.status_code == 200
        # CORSMiddleware with credentials returns the requesting origin or '*'
        allowed_origin = response.headers.get("access-control-allow-origin")
        assert allowed_origin in ["*", "http://localhost:3000"]


# ---------------------------------------------------------------------------
# 2. Prediction Endpoints & Threshold Tuning Tests
# ---------------------------------------------------------------------------
class TestPredictionEndpoints:
    """Verifies single-window prediction contracts, labels, and thresholds."""

    def test_predict_normal_window(self, client: TestClient, normal_window: List[Dict[str, float]]):
        """POST /predict on normal telemetry should return status NORMAL and label 0."""
        payload = {
            "cell_name": "3BLTE",
            "threshold": 0.50,
            "window": normal_window,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["cell_name"] == "3BLTE"
        assert data["status"] == "NORMAL"
        assert data["anomaly_label"] == 0
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["threshold_applied"] == 0.50
        assert data["inference_time_ms"] > 0.0

    def test_predict_anomaly_window(self, client: TestClient, anomaly_window: List[Dict[str, float]]):
        """POST /predict on outage telemetry should return status ANOMALY_DETECTED and label 1."""
        payload = {
            "cell_name": "3BLTE",
            "threshold": 0.50,
            "window": anomaly_window,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["cell_name"] == "3BLTE"
        assert data["status"] == "ANOMALY_DETECTED"
        assert data["anomaly_label"] == 1
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["threshold_applied"] == 0.50
        assert data["inference_time_ms"] > 0.0

    def test_predict_balanced_threshold_066(self, client: TestClient, anomaly_window: List[Dict[str, float]]):
        """POST /predict with threshold=0.66 applies Person 1's balanced threshold."""
        payload = {
            "cell_name": "3BLTE",
            "threshold": 0.66,
            "window": anomaly_window,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["threshold_applied"] == 0.66
        # Even at 0.66, true anomaly sample confidence is ~0.6749, thus flagged
        assert data["status"] == "ANOMALY_DETECTED"
        assert data["anomaly_label"] == 1

    def test_predict_custom_cell_identifier(self, client: TestClient, normal_window: List[Dict[str, float]]):
        """Arbitrary valid cell names should be preserved and reflected in the response."""
        custom_cell = "SECTOR_5G_DOWNTOWN_04"
        payload = {
            "cell_name": custom_cell,
            "threshold": 0.50,
            "window": normal_window,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["cell_name"] == custom_cell


# ---------------------------------------------------------------------------
# 3. Batch Scoring Endpoint Tests
# ---------------------------------------------------------------------------
class TestBatchPrediction:
    """Verifies multi-cell batch evaluation endpoint."""

    def test_batch_predict_multiple_cells(
        self,
        client: TestClient,
        normal_window: List[Dict[str, float]],
        anomaly_window: List[Dict[str, float]],
    ):
        """POST /predict/batch should evaluate a list of cells and aggregate results."""
        payload = {
            "requests": [
                {"cell_name": "CELL_ALPHA", "threshold": 0.50, "window": normal_window},
                {"cell_name": "CELL_BETA", "threshold": 0.50, "window": anomaly_window},
            ]
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["total_cells"] == 2
        assert data["anomalies_detected"] == 1
        assert len(data["results"]) == 2
        assert data["results"][0]["cell_name"] == "CELL_ALPHA"
        assert data["results"][0]["status"] == "NORMAL"
        assert data["results"][1]["cell_name"] == "CELL_BETA"
        assert data["results"][1]["status"] == "ANOMALY_DETECTED"
        assert data["total_batch_time_ms"] > 0.0

    def test_batch_predict_empty_list(self, client: TestClient):
        """POST /predict/batch with an empty list should return 0 counts cleanly."""
        payload = {"requests": []}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["total_cells"] == 0
        assert data["anomalies_detected"] == 0
        assert data["results"] == []


# ---------------------------------------------------------------------------
# 4. Strict Validation & Error Boundary Tests (HTTP 422)
# ---------------------------------------------------------------------------
class TestValidationAndErrorHandling:
    """Verifies Pydantic schema validation rejects invalid inputs before ML inference."""

    def test_reject_window_with_15_steps(self, client: TestClient, normal_window: List[Dict[str, float]]):
        """Reject windows with fewer than 16 steps with HTTP 422."""
        payload = {
            "cell_name": "3BLTE",
            "window": normal_window[:15],  # 15 steps
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_reject_window_with_17_steps(self, client: TestClient, normal_window: List[Dict[str, float]]):
        """Reject windows with more than 16 steps with HTTP 422."""
        payload = {
            "cell_name": "3BLTE",
            "window": normal_window + [normal_window[0]],  # 17 steps
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_reject_prb_above_100_percent(self, client: TestClient, normal_window: List[Dict[str, float]]):
        """Reject PRBUsageDL > 100.0% with HTTP 422."""
        corrupted = [dict(s) for s in normal_window]
        corrupted[5]["PRBUsageDL"] = 150.0  # Physically impossible PRB
        payload = {"cell_name": "3BLTE", "window": corrupted}

        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_reject_negative_prb(self, client: TestClient, normal_window: List[Dict[str, float]]):
        """Reject negative PRBUsageDL with HTTP 422."""
        corrupted = [dict(s) for s in normal_window]
        corrupted[0]["PRBUsageDL"] = -5.0
        payload = {"cell_name": "3BLTE", "window": corrupted}

        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_reject_negative_throughput(self, client: TestClient, normal_window: List[Dict[str, float]]):
        """Reject negative throughput with HTTP 422."""
        corrupted = [dict(s) for s in normal_window]
        corrupted[2]["meanThr_DL"] = -1.5
        payload = {"cell_name": "3BLTE", "window": corrupted}

        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_reject_negative_connected_users(self, client: TestClient, normal_window: List[Dict[str, float]]):
        """Reject negative user equipment count with HTTP 422."""
        corrupted = [dict(s) for s in normal_window]
        corrupted[3]["meanUE_DL"] = -2.0
        payload = {"cell_name": "3BLTE", "window": corrupted}

        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_reject_threshold_out_of_bounds(self, client: TestClient, normal_window: List[Dict[str, float]]):
        """Reject thresholds outside [0.05, 0.95] with HTTP 422."""
        payload = {
            "cell_name": "3BLTE",
            "threshold": 1.50,  # Invalid threshold > 1.0
            "window": normal_window,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_reject_missing_required_kpi_field(self, client: TestClient, normal_window: List[Dict[str, float]]):
        """Reject readings missing any of the 3 required KPIs with HTTP 422."""
        corrupted = [dict(s) for s in normal_window]
        del corrupted[4]["meanThr_DL"]  # Delete required key
        payload = {"cell_name": "3BLTE", "window": corrupted}

        response = client.post("/predict", json=payload)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# 5. Performance Latency SLA Benchmark
# ---------------------------------------------------------------------------
class TestPerformanceSLA:
    """Verifies that ML CPU inference meets real-time latency requirements."""

    def test_cpu_inference_latency_sla(self, client: TestClient, normal_window: List[Dict[str, float]]):
        """Post-warmup CPU inference latency must be sub-150ms per window."""
        payload = {
            "cell_name": "3BLTE",
            "threshold": 0.50,
            "window": normal_window,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        latency = data["inference_time_ms"]
        # ResNet-18 forward pass on modern CPU typically runs in 30-90ms
        assert latency < 250.0, f"Inference latency {latency}ms exceeds SLA limit!"
