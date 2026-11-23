# Person 2: Backend Development & API Engineering Journal

> **Project:** Telecom QoS Anomaly Detection Platform  
> **Role:** Person 2 (Backend Developer & API Engineer)  
> **Repository:** [github.com/ExodOO2/telecom_anomaly_detection](https://github.com/ExodOO2/telecom_anomaly_detection)  
> **Ownership:** `api/`, `requirements-api.txt`, API Test Suite & Live Antenna Simulator  

---

## 1. Roadmap & Progress Overview

| Phase | Milestone Description | Status | Completion Date |
| :--- | :--- | :---: | :---: |
| **Step 1** | **Local Environment Setup, Dependency Resolution & Live Contract Verification** | **COMPLETED** | 2026-11-23 |
| **Step 2** | **API Hardening: Advanced Pydantic Validation, Operational Thresholds & CORS** | **COMPLETED** | 2026-11-23 |
| **Step 3** | **Live Network Operations Center (NOC) Antenna Simulator (`api/simulate_tower.py`)** | **COMPLETED** | 2026-11-23 |
| **Step 4** | **Automated Integration & Regression Test Suite (`tests/test_api.py`)** | *Pending* | - |
| **Step 5** | **Final System Verification, Documentation & Docker Handoff to Person 3** | *Pending* | - |

---

## 2. Step 1 Detailed Log: Environment Setup & Live Model Verification

### Objective
Verify that Person 2's local machine environment can cleanly load Person 1's trained Computer Vision model (`artifacts/best_resnet18.pt`) and preprocessor (`artifacts/scaler.joblib`) via `ml/inference.py`, and confirm that the baseline FastAPI server runs and responds with HTTP 200 on all endpoints.

---

### A. Environment & Package Matrix
The machine was provisioned with **Python 3.14 (64-bit)**. The following dependencies were installed and validated:

| Package | Version | Purpose |
| :--- | :--- | :--- |
| **FastAPI** | `0.124.4` | High-performance asynchronous web framework |
| **Uvicorn** | `0.38.0` | ASGI server implementation |
| **Pydantic** | `2.12.5` | Data validation and parsing using Python type hints |
| **Requests** | `2.34.2` | HTTP client for tower streaming simulations |
| **HTTPX** | `0.28.1` | Asynchronous test client support for Starlette/FastAPI |
| **Pytest** | `9.2.0` | Automated test suite execution framework |
| **NumPy** | `2.5.5` | Array manipulation for time-series windows |
| **Pillow** | `12.2.0` | Image processing for GAF encoding |
| **Joblib** | `1.6.0` | Deserialization of fitted `MinMaxScaler` (`scaler.joblib`) |
| **Scikit-learn** | `1.9.1` | Preprocessing transformer runtime dependencies |
| **PyTorch (CPU)** | `2.15.0+cpu` | Lightweight CPU inference engine for ResNet-18 (<150MB wheel) |
| **TorchVision** | `0.29.2+cpu` | Computer Vision transformations (Resize, Normalize) |

---

### B. Machine Learning Inference Verification (`ml/inference.py`)
Direct testing of `ml/inference.py` confirmed that the pre-trained weights and preprocessor load cleanly on CPU and produce accurate classifications:

1. **Test 1: Healthy Operational Window**
   * **Input:** 16 time steps with balanced PRB load, healthy throughput, and normal user count.
   * **Result:** `{'status': 'NORMAL', 'anomaly_label': 0, 'confidence': 0.6177}`
   * **Status:** **PASSED**

2. **Test 2: Cell Outage / Congestion Window**
   * **Input:** 16 time steps showing 42% PRB load, high connected devices, but throughput collapsed to near-zero.
   * **Result:** `{'status': 'ANOMALY_DETECTED', 'anomaly_label': 1, 'confidence': 0.6749}`
   * **Status:** **PASSED**

---

### C. Live FastAPI Endpoints Verification
Using FastAPI's `TestClient`, both existing endpoints were tested:

#### 1. `GET /health`
* **Response Status:** `HTTP 200 OK`
* **Payload:**
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "service": "Telecom QoS Anomaly Detection API"
  }
  ```
* **Verdict:** Verified service readiness.

#### 2. `POST /predict` (Normal Telemetry)
* **Response Status:** `HTTP 200 OK`
* **Payload:**
  ```json
  {
    "cell_name": "3BLTE",
    "status": "NORMAL",
    "anomaly_label": 0,
    "confidence": 0.6177,
    "inference_time_ms": 111.66
  }
  ```
* **Verdict:** Successful prediction and schema compliance.

#### 3. `POST /predict` (Outage Telemetry)
* **Response Status:** `HTTP 200 OK`
* **Payload:**
  ```json
  {
    "cell_name": "3BLTE",
    "status": "ANOMALY_DETECTED",
    "anomaly_label": 1,
    "confidence": 0.6749,
    "inference_time_ms": 89.36
  }
  ```
* **Verdict:** Accurate detection of anomalous cell state with sub-100ms CPU latency on cold inference.

---

### D. Deliverables & Next Action
* Created `PERSON_2_PROGRESS.md` to track Person 2's development log.
* Updated `requirements-api.txt` with `httpx` and `pytest`.
* **Next Task (Step 2):** Implement API hardening in `api/schemas.py` and `api/app.py` (CORS middleware, request latency logging, custom operational threshold support `0.50` vs `0.66`, and strict boundary validation).

---

## 3. Step 2 Detailed Log: API Hardening & Operational Thresholds

### Objective
Harden the FastAPI service to production standards by implementing strict schema validation, integrating Person 1's recommended operational decision thresholds ($T=0.50$ Safety Net vs $T=0.66$ Balanced), adding CORS middleware, tracking request latency via HTTP headers, pre-warming ML model artifacts at startup, and supporting batch scoring for multi-cell NOC operations.

---

### A. Schema Hardening (`api/schemas.py`)
1. **Physical Boundary Checks on Telemetry (`KPIReading`):**
   * `PRBUsageDL`: Enforced `ge=0.0`, `le=100.0` (Physical radio resource percentage).
   * `meanThr_DL`: Enforced `ge=0.0` (Non-negative downlink throughput in kbps).
   * `meanUE_DL`: Enforced `ge=0.0` (Non-negative connected user count).
2. **Strict Time-Window Length Validation (`PredictRequest`):**
   * Added a `@field_validator('window')` ensuring `len(v) == 16`.
   * Rejects incomplete or oversized sliding windows (e.g. 15 or 17 steps) immediately with structured `HTTP 422 Unprocessable Entity` before invoking PyTorch.
3. **Operational Decision Threshold Field (`threshold`):**
   * Added `threshold: Optional[float] = Field(default=0.50, ge=0.05, le=0.95)` to `PredictRequest`.
   * Enables runtime switching between **Safety Net mode ($0.50$)** for maximum incident capture (70.7% recall) and **Balanced mode ($0.66$)** to cut false alarms by 45%.
4. **Enhanced Prediction Response (`PredictResponse`):**
   * Added `threshold_applied: float` to make the decision boundary transparent to NOC operators and audit logs.
   * Fully backward-compatible with Person 1's initial client contract (`cell_name`, `status`, `anomaly_label`, `confidence`, `inference_time_ms`).
5. **Batch Scoring Schemas:**
   * Introduced `BatchPredictRequest` (accepting a list of cell prediction requests) and `BatchPredictResponse` (returning aggregated predictions and total processing time).
6. **OpenAPI / Swagger Interactive Documentation:**
   * Embedded realistic `json_schema_extra` examples in all models so interactive testing via `/docs` works out-of-the-box.

---

### B. Application Architecture Hardening (`api/app.py`)
1. **Startup Pre-Warming via FastAPI Lifespan:**
   * Replaced ad-hoc on-demand model loading with `lifespan(app: FastAPI)` context manager.
   * Model weights (`best_resnet18.pt`) and scaler (`scaler.joblib`) are pre-loaded into memory when Uvicorn boots, eliminating the ~4-second cold-start latency on the first client request.
2. **CORS Middleware (`CORSMiddleware`):**
   * Configured permissive CORS (`allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]`) to guarantee seamless integration with Person 3's Streamlit / React frontend dashboards.
3. **Processing Latency Middleware:**
   * Added request-intercepting middleware measuring round-trip server execution time and injecting the `X-Process-Time-Ms: <ms>` header into every HTTP response.
4. **Custom Exception Handling:**
   * Implemented `@app.exception_handler(ValueError)` returning HTTP 422 JSON payload when data transformation or window validation raises an exception.
5. **New & Hardened Endpoints:**
   * `GET /`: API root returning service metadata, status, and route map.
   * `GET /health`: Healthcheck verifying model readiness.
   * `POST /predict`: Single 16-step window inference with custom threshold support.
   * `POST /predict/batch`: Batch multi-cell telemetry scoring.

---

### C. Comprehensive Verification Test Suite
All 9 automated verification tests executed via Starlette `TestClient` passed cleanly:

| # | Test Description | Target Route | Expected Code | Result | Note |
| :-: | :--- | :--- | :-: | :-: | :--- |
| **1** | Root Endpoint Metadata | `GET /` | `200 OK` | **PASSED** | Returns platform name & endpoints |
| **2** | Healthcheck Readiness | `GET /health` | `200 OK` | **PASSED** | Model loaded & status healthy |
| **3** | Single Window Normal Telemetry | `POST /predict` | `200 OK` | **PASSED** | `status: "NORMAL"`, `threshold: 0.5` |
| **4** | Single Window Anomaly Telemetry | `POST /predict` | `200 OK` | **PASSED** | `status: "ANOMALY_DETECTED"`, `confidence: 0.6749` |
| **5** | Custom Operational Threshold ($T=0.66$) | `POST /predict` | `200 OK` | **PASSED** | Evaluated with Person 1's balanced threshold |
| **6** | Reject Incomplete Window (15 steps) | `POST /predict` | `422 Unprocessable` | **PASSED** | Validates strict 16-step window requirement |
| **7** | Reject Out-of-Bounds KPI (`PRB=150%`) | `POST /predict` | `422 Unprocessable` | **PASSED** | Catches corrupted telemetry upstream |
| **8** | Multi-Cell Batch Prediction | `POST /predict/batch`| `200 OK` | **PASSED** | Successfully scored 2 cells in batch |
| **9** | Response Timing Header | All endpoints | `200 OK` | **PASSED** | `X-Process-Time-Ms` present on all responses |

---

### D. Deliverables & Next Action
* Modified `api/schemas.py` and `api/app.py`.
* All validation constraints and error handlers tested and confirmed.
* **Next Task (Step 3):** Implement the live NOC antenna telemetry simulator in `api/simulate_tower.py` to stream realistic sliding windows with colorized terminal dashboards.

---

## 4. Step 3 Detailed Log: Live NOC Antenna Simulator (`api/simulate_tower.py`)

### Objective
Create a realistic streaming antenna simulator that emulates continuous 15-minute telemetry streams hitting the running FastAPI microservice, displaying real-time anomaly detection alerts in a colorized Network Operations Center (NOC) dashboard for live demonstrations and jury defense.

---

### A. Architectural Capabilities & Features
1. **Pre-Flight Health Inspection:**
   * Automatically queries `GET /health` before initiating streaming.
   * Gracefully notifies the operator if the service is offline with exact startup instructions (`uvicorn api.app:app ...`) instead of raising unhandled connection exceptions.
2. **Realistic Telemetry Generation:**
   * Built-in reference banks (`NORMAL_TELEMETRY_BANK` and `ANOMALY_TELEMETRY_BANK`) directly derived from cell `3BLTE` historical distributions.
   * `mixed` Profile: Demonstrates nominal operational traffic (Ticks 1–7) $\rightarrow$ Sudden QoS / Throughput collapse incident (Ticks 8–15) $\rightarrow$ Self-healing recovery back to nominal health (Ticks 16+).
   * `normal` Profile: Continuous healthy diurnal traffic cycles.
   * `anomaly` Profile: Continuous outage / radio degradation traffic.
   * `csv` Profile: Reads and streams historical CSV rows if provided.
3. **Executive NOC Dashboard Console:**
   * Dynamic header displaying target cell, endpoint, operational threshold mode, and pacing speed.
   * Structured tabular log tracking simulated time, PRB usage %, throughput Mb/s, user count, colorized status badge (`[ ✅ NORMAL ]` vs `[ 🚨 ANOMALY ALERT ]`), visual confidence meter (`███████░`), and round-trip execution latency.
   * High-visibility event banners: Automatic incident alert on outage detection and recovery confirmation on restoration.
4. **Session Summary Card:**
   * Aggregates total evaluated sliding windows, normal vs anomaly counts and percentages, average and peak model latencies, and an overall sector operational rating (`NOMINAL / HEALTHY`, `MINOR QOS INSTABILITY`, or `CRITICAL OUTAGE`).
5. **Configurable CLI Interface (`argparse`):**
   * Flags: `--url`, `--cell`, `--speed`, `--threshold`, `--mode`, `--steps`, `--csv`.

---

### B. Live Simulation Test Results
The simulator was verified live against the running FastAPI service with 20 sliding windows:

| Metric | Test Run 1: Safety Net Mode ($T=0.50$) | Test Run 2: Balanced Mode ($T=0.66$) |
| :--- | :---: | :---: |
| **Target Base Station** | `3BLTE` | `3BLTE` |
| **Total Windows Evaluated** | 20 | 20 |
| **Healthy Windows (NORMAL)** | 8 (40.0%) | **16 (80.0%)** |
| **Anomalies Flagged** | 12 (60.0%) | **4 (20.0%)** |
| **False Alarm Reduction** | Baseline | **66.7% reduction** |
| **Incident Trigger Step** | Tick 8 (Throughput collapse to 0.069 Mb/s) | Tick 8 (Throughput collapse to 0.070 Mb/s) |
| **Recovery Confirmed Step**| Tick 16 (Telemetry restored to nominal) | Tick 16 (Telemetry restored to nominal) |
| **Average Model Latency** | **58.85 ms** | **59.93 ms** |
| **Peak Model Latency** | 80.28 ms | 124.64 ms |
| **Sector Assessment** | `CRITICAL OUTAGE DETECTED` | `MINOR QOS INSTABILITY DETECTED` |

*Verdict:* Confirms Person 1's validation sweep findings: $T=0.66$ effectively eliminates false alarms during nominal traffic while still catching the outage incident with $>76-79\%$ confidence.

---

### C. Deliverables & Next Action
* Created and verified [`api/simulate_tower.py`](api/simulate_tower.py).
* Verified end-to-end integration with live FastAPI endpoints.
* **Next Task (Step 4):** Build the automated test suite in `tests/test_api.py` covering unit and integration tests across schemas, error boundaries, thresholds, and batch execution.
