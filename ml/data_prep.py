"""
Data Preparation & Gramian Angular Field (GAF) Pipeline
Owned by: Person 1 (ML & Computer Vision Engineer)

Responsibilities:
1. Load 4G LTE Kaggle dataset from data/raw/
2. Clean NaNs per cell
3. Chronological/Cell split (Train 70%, Val 15%, Test 15%)
4. Fit MinMaxScaler([-1, 1]) on Train only -> save artifacts/scaler.joblib
5. Generate sliding windows (W=16, stride=2) across 3 KPIs:
   - PRBUsageDL -> Red Channel
   - meanThr_DL -> Green Channel
   - meanUE_DL  -> Blue Channel
6. Encode to Gramian Angular Summation Field (GASF) via pyts
7. Export 3-channel PNGs into data/gaf_images/{train,val,test}/{normal,anomaly}/
"""

import sys
from pathlib import Path

def main():
    print("[ml/data_prep.py] Pipeline initialized. Ready for Step 1 & 2 implementation.")

if __name__ == "__main__":
    main()
