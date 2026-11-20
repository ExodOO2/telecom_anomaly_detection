"""
Data Preparation & Gramian Angular Field (GAF) Pipeline
Owned by: Person 1 (ML & Computer Vision Engineer)

This script performs the full transformation from raw 4G tabular telemetry
into a partitioned Computer Vision dataset:
1. Reconstructs chronological time sequences per cell tower.
2. Performs a Cell-Holdout Split (23 Train, 5 Val, 5 Test) to eliminate data leakage.
3. Fits MinMaxScaler([-1, 1]) on the Train split ONLY and saves artifacts/scaler.joblib.
4. Slices 16-step sliding windows (4 hours) with stride S=2 across 3 core KPIs:
   - PRBUsageDL -> Red Channel
   - meanThr_DL -> Green Channel
   - meanUE_DL  -> Blue Channel
5. Encodes windows into 3-channel GAF RGB PNG images.
6. Organizes images into PyTorch ImageFolder format (train/val/test -> normal/anomaly).
"""

import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "ML-MATT-CompetitionQT2021_train.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
IMAGES_DIR = PROJECT_ROOT / "data" / "gaf_images"

# The 3 core Downlink KPIs mathematically proven in EDA
KPIS = ["PRBUsageDL", "meanThr_DL", "meanUE_DL"]
TARGET_COL = "Unusual"

# Sliding window parameters
WINDOW_SIZE = 16  # 16 steps * 15 mins = 4 hours
STRIDE = 2        # Slide forward by 2 steps (30 mins) to avoid redundant frames
IMAGE_SIZE = 64   # Native 16x16 upscaled smoothly to 64x64 PNG for PyTorch


def load_and_order_telemetry() -> pd.DataFrame:
    """
    Loads raw CSV and reconstructs chronological day tracks for each cell tower.
    """
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"Raw CSV not found at: {RAW_CSV_PATH}")

    df = pd.read_csv(RAW_CSV_PATH, sep=";")

    # Convert HH:MM time into minutes from midnight (0 to 1425)
    def to_minutes(t_str: str) -> int:
        h, m = t_str.split(":")
        return int(h) * 60 + int(m)

    df["mins"] = df["Time"].apply(to_minutes)

    # Sort each cell by minutes, then assign day index via occurrence count
    ordered_cells = []
    for cell_name, group in df.groupby("CellName"):
        group = group.sort_values(by="mins").reset_index(drop=True)
        # Cumulative count of each timestamp represents the continuous day
        group["day"] = group.groupby("mins").cumcount()
        # Sort chronologically by day, then time of day
        group = group.sort_values(by=["day", "mins"]).reset_index(drop=True)
        ordered_cells.append(group)

    ordered_df = pd.concat(ordered_cells, ignore_index=True)
    return ordered_df


def perform_cell_holdout_split(df: pd.DataFrame):
    """
    Splits the 33 base stations into 23 Train (70%), 5 Val (15%), 5 Test (15%).
    Prevents temporal data leakage and proves generalization to unseen cell towers.
    """
    unique_cells = sorted(df["CellName"].unique())
    train_cells = unique_cells[:23]
    val_cells = unique_cells[23:28]
    test_cells = unique_cells[28:]

    train_df = df[df["CellName"].isin(train_cells)].reset_index(drop=True)
    val_df = df[df["CellName"].isin(val_cells)].reset_index(drop=True)
    test_df = df[df["CellName"].isin(test_cells)].reset_index(drop=True)

    print("\n" + "=" * 65)
    print("CELL-HOLDOUT SPLIT SUMMARY (LEAKAGE-FREE)")
    print("=" * 65)
    print(f"Train Set : {len(train_cells)} towers | {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val Set   : {len(val_cells)} towers  | {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test Set  : {len(test_cells)} towers  | {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def fit_and_save_scaler(train_df: pd.DataFrame) -> MinMaxScaler:
    """
    Fits MinMaxScaler([-1, 1]) on Train split ONLY and saves to artifacts/scaler.joblib.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = ARTIFACTS_DIR / "scaler.joblib"

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_df[KPIS].values)

    joblib.dump(scaler, scaler_path)
    print(f"\n[+] Fitted MinMaxScaler([-1, 1]) on Train ONLY -> Saved to: {scaler_path.name}")
    return scaler


def compute_gasf(series_1d: np.ndarray) -> np.ndarray:
    """
    Computes Gramian Angular Summation Field (GASF) for a 1D scaled sequence in [-1, 1].
    G[i, j] = cos(phi_i + phi_j)
    """
    # Safety clamping to prevent floating-point precision NaN in arccos
    clamped = np.clip(series_1d, -1.0 + 1e-7, 1.0 - 1e-7)
    phi = np.arccos(clamped)
    # 2D outer angle summation: shape (16, 16)
    return np.cos(phi[:, None] + phi[None, :])


def encode_window_to_rgb_image(window_3kpi_scaled: np.ndarray) -> Image.Image:
    """
    Converts a (16, 3) matrix of scaled KPIs into a 3-channel RGB PIL Image:
    - Channel 0 (Red)   : PRBUsageDL GAF
    - Channel 1 (Green) : meanThr_DL GAF
    - Channel 2 (Blue)  : meanUE_DL GAF
    """
    gaf_r = compute_gasf(window_3kpi_scaled[:, 0])
    gaf_g = compute_gasf(window_3kpi_scaled[:, 1])
    gaf_b = compute_gasf(window_3kpi_scaled[:, 2])

    # Stack into (16, 16, 3) and map [-1, 1] to uint8 pixel range [0, 255]
    rgb_gaf = np.stack([gaf_r, gaf_g, gaf_b], axis=-1)
    rgb_uint8 = np.clip((rgb_gaf + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)

    # Smooth bilinear upscaling to 64x64 PNG
    pil_img = Image.fromarray(rgb_uint8).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return pil_img


def generate_split_images(split_df: pd.DataFrame, split_name: str, scaler: MinMaxScaler):
    """
    Generates sliding window GAF images for a split and saves into PyTorch ImageFolder:
    data/gaf_images/{split_name}/{normal, anomaly}/
    """
    split_dir = IMAGES_DIR / split_name
    normal_dir = split_dir / "normal"
    anomaly_dir = split_dir / "anomaly"

    normal_dir.mkdir(parents=True, exist_ok=True)
    anomaly_dir.mkdir(parents=True, exist_ok=True)

    counts = {"normal": 0, "anomaly": 0}
    start_time = time.perf_counter()

    for cell_name, cell_group in split_df.groupby("CellName"):
        # Scale the 3 KPIs using the training scaler
        scaled_kpis = scaler.transform(cell_group[KPIS].values)
        scaled_kpis = np.clip(scaled_kpis, -1.0 + 1e-7, 1.0 - 1e-7)
        labels = cell_group[TARGET_COL].values

        n_samples = len(cell_group)
        for i in range(0, n_samples - WINDOW_SIZE + 1, STRIDE):
            window_scaled = scaled_kpis[i : i + WINDOW_SIZE]
            # Window label is determined by the final timestamp of the 4-hour window
            target_label = labels[i + WINDOW_SIZE - 1]

            img = encode_window_to_rgb_image(window_scaled)

            category = "anomaly" if target_label == 1 else "normal"
            filename = f"{cell_name}_win{i:04d}_{category}.png"

            if category == "anomaly":
                img.save(anomaly_dir / filename)
                counts["anomaly"] += 1
            else:
                img.save(normal_dir / filename)
                counts["normal"] += 1

    elapsed = time.perf_counter() - start_time
    total = counts["normal"] + counts["anomaly"]
    print(f"\n[+] {split_name.upper():<5} Done in {elapsed:.1f}s | Total: {total:,} images "
          f"(Normal: {counts['normal']:,}, Anomaly: {counts['anomaly']:,} - {counts['anomaly']/total*100:.1f}%)")
    return counts


def main():
    print("=" * 65)
    print("STEP 2: DATA PREPARATION & GAF IMAGE GENERATION PIPELINE")
    print("=" * 65)

    # 1. Load and order data
    print("[1] Loading raw telemetry and reconstructing day tracks...")
    df = load_and_order_telemetry()
    print(f"    Loaded {len(df):,} chronological samples across {df['CellName'].nunique()} cells.")

    # 2. Cell-Holdout Split
    print("[2] Executing Cell-Holdout Split...")
    train_df, val_df, test_df = perform_cell_holdout_split(df)

    # 3. Fit scaler on Train only
    print("[3] Fitting MinMaxScaler([-1, 1]) on Train split...")
    scaler = fit_and_save_scaler(train_df)

    # 4. Generate GAF PNGs for Train, Val, and Test
    print("[4] Generating 3-channel GAF images into data/gaf_images/...")
    train_counts = generate_split_images(train_df, "train", scaler)
    val_counts = generate_split_images(val_df, "val", scaler)
    test_counts = generate_split_images(test_df, "test", scaler)

    total_images = (
        train_counts["normal"] + train_counts["anomaly"] +
        val_counts["normal"] + val_counts["anomaly"] +
        test_counts["normal"] + test_counts["anomaly"]
    )

    print("\n" + "=" * 65)
    print(f"[SUCCESS] Step 2 Complete! Generated {total_images:,} GAF images.")
    print("PyTorch ImageFolder directory is ready for ResNet-18 training.")
    print("=" * 65)


if __name__ == "__main__":
    main()
