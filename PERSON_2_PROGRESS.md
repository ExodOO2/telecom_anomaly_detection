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
| **Step 2** | **API Hardening: Advanced Pydantic Validation, Operational Thresholds & CORS** | *Pending* | - |
| **Step 3** | **Live Network Operations Center (NOC) Antenna Simulator (`api/simulate_tower.py`)** | *Pending* | - |
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
