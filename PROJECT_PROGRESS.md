# Telecom QoS Anomaly Detection - Project Progress & Engineering Journal

> **Project Goal:** End-to-end containerized MLOps service detecting 4G/5G cellular base station (eNodeB) anomalies by encoding multi-KPI time-series windows into 2D Gramian Angular Field (GAF) RGB images, classified using PyTorch ResNet-18.  
> **Team Role:** Person 1 (Machine Learning & Computer Vision Engineer)  
> **Repository:** [github.com/ExodOO2/telecom_anomaly_detection](https://github.com/ExodOO2/telecom_anomaly_detection)

---

## Progress Overview

| Phase | Description | Owner | Status | Completed At |
| :--- | :--- | :--- | :--- | :--- |
| **Step 0** | **Repository Setup, Day-1 Contract & Teammate Unblocking** | Person 1 | **COMPLETED** | 2026-11-20 |
| **Step 1** | **Dataset Acquisition & Exploratory Data Analysis (EDA)** | Person 1 | **COMPLETED** | 2026-11-20 |
| **Step 2** | **Data Preparation, Chronological Splitting & GAF Encoding** | Person 1 | *In Progress* | - |
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

## Step 1 Detailed Log: Dataset Acquisition & Exploratory Data Analysis (EDA)

In Step 1, we ingested the official competition dataset files from Josep Vidal's *Anomaly Detection in 4G Cellular Networks* competition and executed automated Exploratory Data Analysis via `ml/eda.py`.

### 1. Dataset Files Placed in `data/raw/`
* `ML-MATT-CompetitionQT2021_train.csv` (2.87 MB): 36,904 rows, 14 columns, labeled with `Unusual`.
* `ML-MATT-CompetitionQT2021_test.csv` (630 KB): 9,158 rows, 13 columns, unlabeled test set.

*(Both files are git-ignored via `.gitignore` to preserve repository portability).*

### 2. Key EDA Findings & Mathematical Properties

1. **Delimiter & Encoding:** Semicolon-delimited (`;`), UTF-8 encoded.
2. **Data Integrity:** **Zero missing values (NaNs)** across all 36,904 rows and 14 features. No imputation needed.
3. **Target Class Distribution (`Unusual`):**
   * **Normal (`0`):** 26,721 samples (**72.41%**)
   * **Anomaly / Unusual (`1`):** 10,183 samples (**27.59%**)
   * *Engineering takeaway:* Moderately imbalanced (ratio ~2.6:1). Perfectly manageable using class-weighted loss `CrossEntropyLoss(weight=[1.0, 2.62])` and decision threshold tuning.
4. **Cell Tower Distribution:**
   * 33 unique base station sectors (`CellName`, e.g., `5ALTE`, `4CLTE`, `3BLTE`).
   * Average of ~1,118 samples per tower (spanning ~12-14 days of 15-minute telemetry).
5. **The 3 Core Computer Vision "RGB" KPIs:**
   * **Red Channel (`PRBUsageDL`):** Radio resource block usage % (Min: 0.0, Mean: 2.12%, Max: 77.51%).
   * **Green Channel (`meanThr_DL`):** Mean user throughput (Min: 0.0, Mean: 0.57 Mbps, Max: 19.60 Mbps).
   * **Blue Channel (`meanUE_DL`):** Active connected devices (Min: 0.0, Mean: 1.18 devices, Max: 3.04 devices).

### 3. Visual Analysis Plots & Artifacts Created

The refactored `ml/eda.py` pipeline generates 4 publication-quality visualizations using `matplotlib` and `seaborn`, saved in `artifacts/`:

1. **`artifacts/eda_correlation_matrix.png`**: Full 14-feature correlation heatmap proving extreme multicollinearity between mean and peak metrics:
   * $\text{corr}(\text{meanThr\_DL}, \text{maxThr\_DL}) = 0.63$ (sustained speed captures QoS better than short-lived bursts).
   * $\text{corr}(\text{meanUE\_DL}, \text{maxUE\_DL}) = 0.85$ (85% redundant user metric).
   * $\text{corr}(\text{maxUE\_UL}, \text{maxUE\_UL+DL}) = 0.95$ (95% redundant composite metric).
2. **`artifacts/eda_feature_importance.png`**: Ranked correlation barplot with `Unusual`, showing how downlink metrics directly impact anomaly states compared to noisy uplink counters.
3. **`artifacts/eda_diurnal_traffic.png`**: 24-hour diurnal cycle across the 96 timestamps, demonstrating traffic collapse at 04:00 AM and peak demand at 20:00 PM (proves why static thresholds trigger false alarms).
4. **`artifacts/eda_distributions.png`**: Boxplots comparing Normal vs. Anomaly for the core KPIs, visually demonstrating range overlap and the need for multi-metric temporal CNN inspection.

### 4. Mathematical Feature Selection Justification

| Metric | Selected For | Role / Channel | Justification |
| :--- | :--- | :--- | :--- |
| **`PRBUsageDL`** | **Selected** | **Red Channel (R)** | Measures physical radio spectrum load and base station resource saturation. |
| **`meanThr_DL`** | **Selected** | **Green Channel (G)** | Measures delivered downlink data rate (perceived customer QoS). |
| **`meanUE_DL`** | **Selected** | **Blue Channel (B)** | Measures active connected user volume (traffic demand). |
| `maxThr_DL`, `maxUE_DL` | *Discarded* | - | Redundant with mean metrics; high multicollinearity adds noise. |
| Uplink KPIs (`*UL`) | *Discarded* | - | Mobile traffic is ~85-90% downlink-dominated. QoS degradation is predominantly felt in downlink throughput. |

**Cross-Channel Orthogonality:**
* $\text{corr}(\text{PRBUsageDL}, \text{meanThr\_DL}) = 0.79$
* $\text{corr}(\text{PRBUsageDL}, \text{meanUE\_DL}) = 0.60$
* $\text{corr}(\text{meanThr\_DL}, \text{meanUE\_DL}) = 0.49$

Moderate cross-correlations ($0.49 - 0.79$) guarantee non-redundant, complementary features across the 3 RGB channels.

### 5. Files Created / Updated in Step 1
* `ml/eda.py`: Modular EDA pipeline with dedicated plotting functions for all 4 visual figures and feature selection defense reporting.
* `artifacts/eda_correlation_matrix.png`
* `artifacts/eda_feature_importance.png`
* `artifacts/eda_diurnal_traffic.png`
* `artifacts/eda_distributions.png`

---

## Next Action: Step 2 (Data Preparation & GAF Encoding Pipeline)
* **Script:** `ml/data_prep.py`
* **Tasks:**
  1. Split dataset into Train (70%), Validation (15%), and Test (15%) splits (using Cell-Holdout or chronological per-cell partitioning to eliminate leakage).
  2. Fit `MinMaxScaler([-1, 1])` on Train only $\rightarrow$ save `artifacts/scaler.joblib`.
  3. Generate sliding windows ($W=16$, stride $S=2$).
  4. Compute 3-channel Gramian Angular Field (GAF) images via `pyts`.
  5. Save PNGs in standard PyTorch `ImageFolder` structure under `data/gaf_images/`.

