"""
GAF Visual Representation Generator
Demonstrates how 16-step chronological windows of 3 telecom KPIs
are converted into 2D Gramian Angular Field (GAF) RGB images for ResNet-18.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

# Output paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ML-MATT-CompetitionQT2021_train.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = ARTIFACTS_DIR / "gaf_sample_visualization.png"


def compute_gasf_numpy(series_1d: np.ndarray) -> np.ndarray:
    """
    Computes Gramian Angular Summation Field (GASF) for a 1D sequence in [-1, 1].
    G[i, j] = cos(phi_i + phi_j)
    """
    # Ensure safe numerical range for arccos
    clamped = np.clip(series_1d, -1.0 + 1e-7, 1.0 - 1e-7)
    phi = np.arccos(clamped)
    # 2D outer summation of angles
    phi_sum = phi[:, None] + phi[None, :]
    return np.cos(phi_sum)


def series_to_rgb_gaf(window_3kpi: np.ndarray, scaler: MinMaxScaler, image_size: int = 224):
    """
    Transforms a (16, 3) matrix of KPIs into:
    - Scaled signals
    - Individual GAF channels (R, G, B)
    - Merged RGB image (resized to image_size)
    """
    # 1. Scale to [-1, 1]
    scaled = scaler.transform(window_3kpi)
    scaled = np.clip(scaled, -1.0 + 1e-7, 1.0 - 1e-7)

    # 2. Compute GAF for each of the 3 channels
    gaf_r = compute_gasf_numpy(scaled[:, 0])  # PRBUsageDL -> Red
    gaf_g = compute_gasf_numpy(scaled[:, 1])  # meanThr_DL -> Green
    gaf_b = compute_gasf_numpy(scaled[:, 2])  # meanUE_DL  -> Blue

    # 3. Map [-1, 1] to uint8 pixel values [0, 255]
    r_uint8 = np.clip((gaf_r + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
    g_uint8 = np.clip((gaf_g + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
    b_uint8 = np.clip((gaf_b + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)

    # 4. Stack into (16, 16, 3) RGB image
    rgb_16 = np.stack([r_uint8, g_uint8, b_uint8], axis=-1)

    # 5. Smoothly upscale to 224x224 (as ResNet-18 expects)
    pil_img = Image.fromarray(rgb_16).resize((image_size, image_size), Image.Resampling.BILINEAR)
    rgb_224 = np.array(pil_img)

    return scaled, (gaf_r, gaf_g, gaf_b), rgb_224


def main():
    print("[1] Loading Kaggle 4G LTE dataset...")
    df = pd.read_csv(DATA_PATH, sep=";")
    kpis = ["PRBUsageDL", "meanThr_DL", "meanUE_DL"]

    # Fit MinMaxScaler on full training dataset
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(df[kpis].values)

    # Find a representative cell tower (e.g. 3BLTE)
    cell = df[df["CellName"] == "3BLTE"].reset_index(drop=True)

    # Pick a 16-step window with mostly Normal (0) and one with Anomaly (1)
    # Search for an anomaly window
    anomaly_idx = None
    normal_idx = None
    W = 16

    for i in range(len(cell) - W):
        window = cell.iloc[i : i + W]
        # Check if window ends in an anomaly vs completely normal
        if window["Unusual"].iloc[-1] == 1 and anomaly_idx is None:
            # Prefer a window with a noticeable spike/dip
            anomaly_idx = i
        elif (window["Unusual"] == 0).all() and normal_idx is None:
            normal_idx = i

        if anomaly_idx is not None and normal_idx is not None:
            break

    if normal_idx is None:
        normal_idx = 0
    if anomaly_idx is None:
        anomaly_idx = 100

    print(f"[2] Selected Normal Window: rows {normal_idx} to {normal_idx+W}")
    print(f"[3] Selected Anomaly Window: rows {anomaly_idx} to {anomaly_idx+W}")

    norm_window = cell.iloc[normal_idx : normal_idx + W]
    anom_window = cell.iloc[anomaly_idx : anomaly_idx + W]

    # Convert to GAF
    norm_scaled, norm_gafs, norm_rgb = series_to_rgb_gaf(norm_window[kpis].values, scaler)
    anom_scaled, anom_gafs, anom_rgb = series_to_rgb_gaf(anom_window[kpis].values, scaler)

    # Create publication-quality figure
    fig = plt.figure(figsize=(15, 9), facecolor="white")
    fig.suptitle(
        "Time-Series to Computer Vision: 16-Step Telecom KPI Windows Encoded into 3-Channel GAF RGB Images",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    time_steps = np.arange(1, 17)

    # ROW 1: NORMAL WINDOW
    # 1. 1D Line Plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(time_steps, norm_window["PRBUsageDL"].values, "r-o", label="PRBUsageDL (Radio %)", linewidth=2)
    ax1.plot(time_steps, norm_window["meanThr_DL"].values, "g-s", label="meanThr_DL (Mbps)", linewidth=2)
    ax1.plot(time_steps, norm_window["meanUE_DL"].values, "b-^", label="meanUE_DL (Users)", linewidth=2)
    ax1.set_title("A. Normal Window: 1D Telemetry\n(4 Hours = 16 Steps, 15-min intervals)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time Steps (t1 to t16)")
    ax1.set_ylabel("Metric Values")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8)

    # 2. Individual GAF Channels
    ax2 = plt.subplot(2, 3, 2)
    # Combine the 3 16x16 GAF matrices side-by-side
    combined_channels = np.hstack([norm_gafs[0], norm_gafs[1], norm_gafs[2]])
    im2 = ax2.imshow(combined_channels, cmap="coolwarm", vmin=-1, vmax=1)
    ax2.set_title("B. Normal: 3 Individual GAF Matrices\n[ Red: PRB  |  Green: Throughput  |  Blue: Users ]", fontsize=11, fontweight="bold")
    ax2.set_xticks([8, 24, 40])
    ax2.set_xticklabels(["R Channel", "G Channel", "B Channel"])
    ax2.set_yticks([])
    plt.colorbar(im2, ax=ax2, orientation="horizontal", fraction=0.046, pad=0.15, label="Gramian Correlation cos(φi + φj)")

    # 3. Merged RGB Image for ResNet-18
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(norm_rgb)
    ax3.set_title("C. Normal: Merged 3-Channel RGB Image\n(Fed into PyTorch ResNet-18)", fontsize=11, fontweight="bold", color="darkgreen")
    ax3.set_xlabel("Width: 224 pixels")
    ax3.set_ylabel("Height: 224 pixels")
    ax3.text(10, 210, "LABEL: NORMAL (0)\nSmooth, balanced colors", color="white", fontsize=10, weight="bold",
             bbox=dict(boxstyle="round", facecolor="darkgreen", alpha=0.8))

    # ROW 2: ANOMALOUS WINDOW
    # 4. 1D Line Plot
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(time_steps, anom_window["PRBUsageDL"].values, "r-o", label="PRBUsageDL (Radio %)", linewidth=2)
    ax4.plot(time_steps, anom_window["meanThr_DL"].values, "g-s", label="meanThr_DL (Mbps)", linewidth=2)
    ax4.plot(time_steps, anom_window["meanUE_DL"].values, "b-^", label="meanUE_DL (Users)", linewidth=2)
    ax4.set_title("D. Anomaly Window: 1D Telemetry\n(Severe Congestion / Outage Pattern)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Time Steps (t1 to t16)")
    ax4.set_ylabel("Metric Values")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper left", fontsize=8)

    # 5. Individual GAF Channels
    ax5 = plt.subplot(2, 3, 5)
    combined_channels_anom = np.hstack([anom_gafs[0], anom_gafs[1], anom_gafs[2]])
    im5 = ax5.imshow(combined_channels_anom, cmap="coolwarm", vmin=-1, vmax=1)
    ax5.set_title("E. Anomaly: 3 Individual GAF Matrices\n[ Red: PRB  |  Green: Throughput  |  Blue: Users ]", fontsize=11, fontweight="bold")
    ax5.set_xticks([8, 24, 40])
    ax5.set_xticklabels(["R Channel", "G Channel", "B Channel"])
    ax5.set_yticks([])
    plt.colorbar(im5, ax=ax5, orientation="horizontal", fraction=0.046, pad=0.15, label="Gramian Correlation cos(φi + φj)")

    # 6. Merged RGB Image for ResNet-18
    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(anom_rgb)
    ax6.set_title("F. Anomaly: Merged 3-Channel RGB Image\n(Fed into PyTorch ResNet-18)", fontsize=11, fontweight="bold", color="crimson")
    ax6.set_xlabel("Width: 224 pixels")
    ax6.set_ylabel("Height: 224 pixels")
    ax6.text(10, 210, "LABEL: ANOMALY (1)\nSharp color tear & disruption", color="white", fontsize=10, weight="bold",
             bbox=dict(boxstyle="round", facecolor="crimson", alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"\n[SUCCESS] Generated visual representation saved to:\n  -> {OUT_PNG}")

    # Copy to brain artifact directory for user inspection
    brain_artifact = Path("C:/Users/mehdi/.gemini/antigravity/brain/6e8d6089-867b-44c9-9ac5-23b4d9999a0d/gaf_sample_visualization.png")
    try:
        import shutil
        shutil.copy(OUT_PNG, brain_artifact)
        print(f"  -> Copied to brain artifact directory: {brain_artifact}")
    except Exception as e:
        print(f"  -> Notice: Could not copy to brain dir: {e}")


if __name__ == "__main__":
    main()
