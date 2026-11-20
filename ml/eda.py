"""
Exploratory Data Analysis (EDA) Script
Owned by: Person 1 (ML & Computer Vision Engineer)

Analyzes the raw Kaggle 4G LTE dataset in data/raw/:
- Schema, data types, missing values
- Cell tower distribution (33 unique cells)
- Class balance of 'Unusual' (Normal vs Anomaly)
- Statistical distribution of the 3 target KPIs:
  1. PRBUsageDL (Radio Resource Block usage % -> Red channel)
  2. meanThr_DL (Throughput Mbps -> Green channel)
  3. meanUE_DL  (Active User Equipment count -> Blue channel)
- Correlation between KPIs and Anomaly status
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
TRAIN_CSV = DATA_DIR / "ML-MATT-CompetitionQT2021_train.csv"
TEST_CSV = DATA_DIR / "ML-MATT-CompetitionQT2021_test.csv"

def run_eda():
    print("=" * 70)
    print("TELECOM 4G LTE QOS DATASET - EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    if not TRAIN_CSV.exists():
        print(f"[ERROR] Training file not found at: {TRAIN_CSV}")
        return

    # Load with semicolon delimiter
    train_df = pd.read_csv(TRAIN_CSV, sep=";")
    print(f"\n[1] DATASET DIMENSIONS:")
    print(f"  - Train samples: {train_df.shape[0]:,} rows")
    print(f"  - Features:      {train_df.shape[1]} columns")

    if TEST_CSV.exists():
        test_df = pd.read_csv(TEST_CSV, sep=";")
        print(f"  - Test samples:  {test_df.shape[0]:,} rows (unlabeled Kaggle test set)")

    # Missing values
    missing = train_df.isnull().sum()
    print(f"\n[2] MISSING VALUES CHECK:")
    if missing.sum() == 0:
        print("  - Zero missing values detected across all columns. Data is 100% complete!")
    else:
        print(missing[missing > 0])

    # Target variable distribution
    print(f"\n[3] GROUND TRUTH TARGET ('Unusual') DISTRIBUTION:")
    counts = train_df["Unusual"].value_counts()
    percentages = train_df["Unusual"].value_counts(normalize=True) * 100
    for label, count in counts.items():
        name = "Normal (0)" if label == 0 else "Anomaly / Unusual (1)"
        print(f"  - {name:<25}: {count:>6,} ({percentages[label]:.2f}%)")

    # Base Station / Cell Towers
    cells = train_df["CellName"].nunique()
    print(f"\n[4] BASE STATIONS / CELL SECTORS:")
    print(f"  - Total unique cells: {cells} towers")
    print(f"  - Average samples per cell: {len(train_df) / cells:.1f}")

    # Core 3 KPIs for Computer Vision RGB mapping
    kpis = ["PRBUsageDL", "meanThr_DL", "meanUE_DL"]
    print(f"\n[5] STATISTICAL SUMMARY FOR 3 CORE 'RGB' KPIS:")
    summary = train_df[kpis].describe().T[["min", "mean", "std", "50%", "max"]]
    summary.columns = ["Min", "Mean", "Std", "Median", "Max"]
    print(summary.to_string())

    # Comparison Normal vs Anomaly
    print(f"\n[6] KPI BEHAVIOR: NORMAL vs ANOMALOUS CELLS:")
    comparison = train_df.groupby("Unusual")[kpis].mean().T
    comparison.columns = ["Normal Mean", "Anomaly Mean"]
    comparison["% Difference"] = (
        (comparison["Anomaly Mean"] - comparison["Normal Mean"]) / comparison["Normal Mean"] * 100
    )
    print(comparison.to_string())

    print("\n" + "=" * 70)
    print("EDA COMPLETE - INSIGHTS READY FOR DATA PREPARATION (STEP 2)")
    print("=" * 70)

if __name__ == "__main__":
    run_eda()
