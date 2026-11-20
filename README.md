# Telecom QoS Anomaly Detection Platform
> **Computer Vision + MLOps MVP** for automated cellular base station (eNodeB) anomaly detection using Gramian Angular Fields (GAF) and PyTorch ResNet-18.

---

## 1. Project Overview

Mobile network operators monitor thousands of 4G/5G antennas generating continuous Key Performance Indicators (KPIs). This platform converts 4-hour chronological windows of 3 key cellular metrics (`PRBUsageDL`, `meanThr_DL`, `meanUE_DL`) into 3-channel 2D RGB-like images via **Gramian Angular Summation Fields (GASF)**, classifying them with **ResNet-18** to flag cell degradation before outages occur.

---

## 2. Team Work Split & Ownership

| Member | Role | Folder Ownership | Deliverables |
| :--- | :--- | :--- | :--- |
| **Person 1** | **ML & Computer Vision** | `ml/`, `artifacts/` | `data_prep.py`, `train.py`, `evaluate.py`, `inference.py`, `best_resnet18.pt`, `scaler.joblib`, Grad-CAM overlays |
| **Person 2** | **Backend Developer** | `api/` | `app.py`, `schemas.py`, `simulate_tower.py`, Swagger UI testing |
| **Person 3** | **DevOps & MLOps** | `docker/`, CI/CD | `Dockerfile`, `docker-compose.yml`, MLflow setup, packaging |

---

## 3. Repository Structure

```text
telecom-qos-mlops/
├── data/
│   ├── raw/                  # Kaggle CSV (git-ignored, kept via .gitkeep)
│   └── gaf_images/           # Generated GAF PNGs (git-ignored, train/val/test)
├── ml/                       # OWNED BY PERSON 1
│   ├── data_prep.py          # Cleaning, split, scaling, GAF image export
│   ├── train.py              # PyTorch ResNet-18 training + MLflow logging
│   ├── evaluate.py           # Metrics (F1, ROC-AUC), Confusion Matrix, Grad-CAM
│   └── inference.py          # Self-contained prediction function handed to Person 2
├── artifacts/                # Model artifacts (best_resnet18.pt, scaler.joblib)
├── api/                      # OWNED BY PERSON 2
│   ├── app.py                # FastAPI service (/health, /predict)
│   ├── schemas.py            # Pydantic request/response validation schemas
│   └── simulate_tower.py     # Live tower KPI telemetry simulation script
├── docker/                   # OWNED BY PERSON 3
│   ├── Dockerfile            # CPU-optimized Python + PyTorch container
│   └── docker-compose.yml    # Single-command stack launch
├── requirements-ml.txt       # ML & Computer Vision dependencies
├── requirements-api.txt      # API & serving dependencies
└── README.md
```

---

## 4. The Day-1 Integration Contract

Person 2 (API) and Person 3 (Docker) can run and test immediately against `ml/inference.py`:

### Endpoint: `POST /predict`

#### Request Body
```json
{
  "cell_name": "3BLTE",
  "window": [
    {"PRBUsageDL": 42.5, "meanThr_DL": 38.2, "meanUE_DL": 14.0},
    {"PRBUsageDL": 44.1, "meanThr_DL": 37.9, "meanUE_DL": 15.0}
    // ... exactly 16 chronological time steps total
  ]
}
```

#### Response Body
```json
{
  "cell_name": "3BLTE",
  "status": "ANOMALY_DETECTED",
  "anomaly_label": 1,
  "confidence": 0.924,
  "inference_time_ms": 15.2
}
```

---

## 5. Quick Start Instructions

### For Person 1 (ML Development)
```bash
pip install -r requirements-ml.txt
# Step 1 & 2: Prepare data and generate GAF images
python ml/data_prep.py
# Step 3: Train ResNet-18 with MLflow tracking
python ml/train.py
# Step 4: Evaluate and generate Grad-CAM heatmaps
python ml/evaluate.py
```

### For Person 2 (Local API Development)
```bash
pip install -r requirements-api.txt
# Start local FastAPI server with hot-reload
uvicorn api.app:app --reload --port 8000
# Open Swagger UI in browser: http://localhost:8000/docs
```

### For Person 3 (Docker & Deployment)
```bash
# Build and run the CPU-optimized container
docker-compose -f docker/docker-compose.yml up --build
```
