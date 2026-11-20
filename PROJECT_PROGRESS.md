# Telecom QoS Anomaly Detection - Project Progress & Engineering Journal

> **Project Goal:** End-to-end containerized MLOps service detecting 4G/5G cellular base station (eNodeB) anomalies by encoding multi-KPI time-series windows into 2D Gramian Angular Field (GAF) RGB images, classified using PyTorch ResNet-18.  
> **Team Role:** Person 1 (Machine Learning & Computer Vision Engineer)  
> **Repository:** [github.com/ExodOO2/telecom_anomaly_detection](https://github.com/ExodOO2/telecom_anomaly_detection)

---

## Progress Overview

| Phase | Description | Owner | Status | Completed At |
| :--- | :--- | :--- | :--- | :--- |
| **Step 0** | **Repository Setup, Day-1 Contract & Teammate Unblocking** | Person 1 | **COMPLETED** | 2026-11-20 |
| **Step 1** | **Dataset Acquisition & Exploratory Data Analysis (EDA)** | Person 1 | *In Progress* | - |
| **Step 2** | **Data Preparation, Chronological Splitting & GAF Encoding** | Person 1 | *Pending* | - |
| **Step 3** | **ResNet-18 Transfer Learning & MLflow Experiment Tracking** | Person 1 | *Pending* | - |
| **Step 4** | **Evaluation, Decision Threshold Tuning & Offline Grad-CAM** | Person 1 | *Pending* | - |
| **Step 5** | **Production Inference Wrapper & Artifact Handoff** | Person 1 | *Pending* | - |
| **Step 6** | **End-to-End Integration, Docker Verification & Jury Defense** | All Team | *Pending* | - |

---

## Step 0 Detailed Log: Architecture, Files Created & Design Rationale

In Step 0, we established the full project repository, isolated team ownership boundaries, created the Day-1 API contract, and implemented a functional mock inference engine. This enabled Person 2 (Backend) and Person 3 (DevOps) to build and containerize immediately without waiting for model training.

### 1. Directory Structure

```text
d:\telecom_anomaly_detection\
├── .gitignore                # Git exclusion rules for large datasets, artifacts, and cache
├── README.md                 # Public project documentation & team onboarding guide
├── PROJECT_PROGRESS.md       # Living engineering log updated at each development step
├── requirements-ml.txt       # Dependencies for ML/CV training (Person 1)
├── requirements-api.txt      # Dependencies for production serving (Persons 2 & 3)
├── data/
│   ├── raw/.gitkeep          # Stores raw Kaggle CSVs (ignored by Git)
│   └── gaf_images/.gitkeep   # Stores generated 3-channel PNGs (ignored by Git)
├── artifacts/.gitkeep        # Stores exported model weights and scaler
├── ml/                       # [OWNED BY PERSON 1 - ML & CV]
│   ├── __init__.py           # Declares 'ml' as a Python package
│   ├── inference.py          # Day-1 mock prediction engine complying with API contract
│   ├── data_prep.py          # Data cleaning, chronological split & GAF conversion skeleton
│   ├── train.py              # ResNet-18 training & MLflow tracking skeleton
│   └── evaluate.py           # Test set evaluation, metrics & Grad-CAM skeleton
├── api/                      # [OWNED BY PERSON 2 - BACKEND]
│   ├── schemas.py            # Pydantic schemas validating window length and KPI types
│   ├── app.py                # FastAPI server exposing /health and /predict
│   └── simulate_tower.py     # Live streaming telemetry simulator placeholder
└── docker/                   # [OWNED BY PERSON 3 - DEVOPS & MLOPS]
    ├── Dockerfile            # CPU-optimized PyTorch container (< 800MB)
    └── docker-compose.yml    # Single-command local deployment
```

---

### 2. Files Created, Code Written & Why

#### A. Repository Governance & Environment

##### 1. `.gitignore`
* **Why Created:** Machine learning projects generate massive files (raw CSVs, tens of thousands of PNG images, virtual environments, and MLflow run logs) that must **never** be committed to Git. Doing so would cause repository bloat, slow clone times, and merge conflicts.
* **Key Implementation Details:**
  * Uses the exclusion inversion pattern (`data/raw/*` and `!data/raw/.gitkeep`) so Git tracks the folder structure without tracking the gigabytes of data inside.
  * Ignores Python bytecode (`__pycache__`, `*.pyc`), virtual environments (`.venv/`, `venv/`), MLflow artifacts (`mlruns/`), and OS-specific files (`Thumbs.db`, `.DS_Store`).

##### 2. `README.md`
* **Why Created:** Serves as the central orientation document for teammates and jury evaluators.
* **Key Implementation Details:** Defines the business problem, team ownership matrix, repository tree, Day-1 API schema, and exact quickstart bash commands for Persons 1, 2, and 3.

##### 3. `requirements-ml.txt`
* **Why Created:** Captures all dependencies needed for Person 1 to run offline training on Google Colab or a local GPU workstation.
* **Packages Included:** `torch`, `torchvision`, `pyts` (Gramian Angular Field), `pandas`, `numpy`, `scikit-learn`, `Pillow`, `joblib`, `mlflow` (experiment tracking), `pytorch-grad-cam` (explainability), `matplotlib`, `seaborn`.

##### 4. `requirements-api.txt`
* **Why Created:** A lean dependency file strictly for production serving in Docker. It omits heavy training packages (like `mlflow`, `matplotlib`, `pytorch-grad-cam`) to keep container build times fast and images lightweight.

---

#### B. Machine Learning & Computer Vision (`ml/`) — Owned by Person 1

##### 5. `ml/__init__.py`
* **Why Created:** Marks the `ml/` directory as an importable Python package. This enables `api/app.py` to cleanly execute `from ml.inference import predict_window`.

##### 6. `ml/inference.py` (The Day-1 Contract Mock)
* **Why Created:** Solves the core bottleneck of ML projects: **preventing the backend and DevOps engineers from waiting 7 days for the ML model**. By providing a working mock that adheres strictly to the contract on Day 1, Person 2 and Person 3 can build their code against it immediately.
* **Code & Logic Breakdown:**
  * `predict_window(kpi_window: List[Dict[str, float]]) -> Dict[str, Any]`:
    1. **Input Validation:** Verifies that `len(kpi_window) == 16`. If not, raises a `ValueError`.
    2. **Mock Logic:** Inspects the last time step ($t_{16}$). If `PRBUsageDL > 80%` or (`meanThr_DL < 5 Mbps` and `meanUE_DL > 20`), it simulates an anomaly; otherwise returns normal. This allows Person 2 to test both application branches in Swagger UI immediately.
    3. **Latency Simulation:** Executes `time.sleep(0.015)` to mimic real ResNet-18 CPU inference latency (~15ms) and records exact execution time using `time.perf_counter()`.
    4. **Day 9 Ready:** When real model training finishes, only the interior of this function changes (loading `best_resnet18.pt` and `scaler.joblib`). The function signature remains identical, meaning zero code breaks for Person 2.

##### 7. `ml/data_prep.py` (Skeleton)
* **Why Created:** Defines the module structure for Step 2: reading the Kaggle CSV, executing cell-grouped chronological splitting, fitting `MinMaxScaler([-1, 1])` on training data only, computing 3-channel Gramian Angular Fields via `pyts`, and writing PNGs into `data/gaf_images/`.

##### 8. `ml/train.py` (Skeleton)
* **Why Created:** Defines the module structure for Step 3: loading images with PyTorch `ImageFolder`, modifying pretrained ResNet-18, computing inverse class weights to handle anomaly imbalance, training with AdamW, and logging loss and validation F1-scores to local MLflow.

##### 9. `ml/evaluate.py` (Skeleton)
* **Why Created:** Defines the module structure for Step 4: computing held-out Test set metrics (Macro F1, Precision, Recall, ROC-AUC), sweeping the classification decision threshold to maximize F1, and generating publication-ready Grad-CAM attention maps on `model.layer4[-1]`.

---

#### C. Backend API (`api/`) — Owned by Person 2

##### 10. `api/schemas.py`
* **Why Created:** Implements strict data validation at the API boundary using Pydantic. If an external client sends malformed JSON, invalid data types, or a window length other than 16, FastAPI returns an informative HTTP 422 error before reaching the ML model.
* **Code & Logic Breakdown:**
  * `KPIReading`: Validates that `PRBUsageDL` is a float between 0 and 100, and throughput/UE counts are non-negative.
  * `PredictRequest`: Enforces that `window` is a list of exactly 16 `KPIReading` objects via a `@field_validator`.
  * `PredictResponse`: Enforces typed fields: `cell_name` (str), `status` (str), `anomaly_label` (int), `confidence` (float), and `inference_time_ms` (float).
  * `HealthResponse`: Standard status payload for Docker and load-balancer probes.

##### 11. `api/app.py`
* **Why Created:** The FastAPI serving application exposing the REST interface.
* **Code & Logic Breakdown:**
  * Exposes `GET /health` for container orchestration health checks.
  * Exposes `POST /predict`, which takes `PredictRequest`, extracts the KPI dictionaries, calls `ml.inference.predict_window()`, and returns `PredictResponse`.
  * Automatically generates interactive Swagger documentation at `http://localhost:8000/docs`.

##### 12. `api/simulate_tower.py`
* **Why Created:** Starter script for Person 2 to stream chronological sliding windows against `POST /predict`, creating a live NOC dashboard demo during the jury defense.

---

#### D. Containerization & Deployment (`docker/`) — Owned by Person 3

##### 13. `docker/Dockerfile`
* **Why Created:** Standardizes execution across Windows, Linux, and macOS.
* **Key Engineering Optimization:** 
  * Standard PyTorch wheels package over 2.5 GB of NVIDIA CUDA binaries, resulting in image sizes exceeding 3.5 GB.
  * Because real-time inference on a single 16-step window takes under 20ms on a standard CPU, this Dockerfile specifies the **CPU-only PyTorch wheel**:
    ```dockerfile
    RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
    ```
  * Keeps final image size under 800 MB and ensures seamless deployment on any student or production laptop.

##### 14. `docker/docker-compose.yml`
* **Why Created:** Allows single-command boot (`docker-compose up --build`) binding container port 8000 to the host, configured with built-in healthchecks.

---

### 3. Git Operations Performed

1. Initialized local repository on `main` branch.
2. Added `.gitignore` and all 16 starter files.
3. Created initial commit: `feat: initial project structure with Day-1 contract and dummy inference` (`b4daefc`).
4. Configured remote: `https://github.com/ExodOO2/telecom_anomaly_detection.git`.
5. Pushed `main` branch to remote GitHub repository.

---

## Next Action: Step 1 (Dataset Acquisition & EDA)
* **Input:** Real-world Kaggle 4G LTE CSV (Josep Vidal competition).
* **Deliverable:** `notebooks/eda.ipynb` or `ml/eda_report.py` analyzing class balance, temporal continuity, and KPI correlations.
