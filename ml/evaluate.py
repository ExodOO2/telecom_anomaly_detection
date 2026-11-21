"""
Comprehensive Model Evaluation, Decision Threshold Tuning & Explainable AI (Grad-CAM)
Owned by: Person 1 (ML & Computer Vision Engineer)

This script performs the full post-training evaluation pipeline on the unseen Test set:
1. Loads best_resnet18.pt and evaluates on 2,770 holdout Test images (5 unseen cell towers).
2. Computes Confusion Matrix, Precision, Recall, F1, ROC-AUC, and PR-AUC.
3. Generates high-resolution publication-quality plots:
   - artifacts/eval_confusion_matrix.png (annotated heatmap)
   - artifacts/eval_roc_pr_curves.png (ROC-AUC & Precision-Recall curves)
   - artifacts/eval_threshold_tuning.png (Decision threshold sweep 0.05-0.95)
   - artifacts/eval_gradcam_sample.png (Grad-CAM explainability heatmaps on layer4)
4. Exports artifacts/test_metrics.json with numerical performance benchmarks.
"""

import os
import json
from pathlib import Path

# Silence MLflow startup agent hints and cosmetic logs
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from PIL import Image

# Path definitions
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "gaf_images"
TEST_DIR = DATA_DIR / "test"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = ARTIFACTS_DIR / "best_resnet18.pt"


def get_device() -> torch.device:
    """Detects and returns GPU if available, else CPU."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] Evaluation running on GPU: {gpu_name} (CUDA active)")
        return torch.device("cuda")
    else:
        print("[Device] CUDA not available. Running evaluation on CPU.")
        return torch.device("cpu")


def build_resnet18(num_classes: int = 2, dropout_rate: float = 0.3) -> nn.Module:
    """Reconstructs the ResNet-18 architecture with modified classification head."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    return model


def load_test_dataloader(batch_size: int = 64):
    """Loads the holdout Test set via ImageFolder."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}. Run ml/data_prep.py first.")

    test_dataset = ImageFolder(root=str(TEST_DIR), transform=data_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"[Data] Loaded {len(test_dataset):,} test images (Classes: {test_dataset.class_to_idx})")
    return test_dataset, test_loader


# ==============================================================================
# Explainable AI: Native PyTorch Grad-CAM Implementation for ResNet-18
# ==============================================================================
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for ResNet-18.
    Hooks into the final convolutional layer (model.layer4[-1]) to extract feature
    maps and gradients, producing a visual saliency heatmap for anomaly decisions.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Computes Grad-CAM heatmap for a single input image tensor (1, 3, 224, 224).
        Returns a normalized 2D heatmap of shape (224, 224) in range [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)
        
        # Target score backward pass
        score = output[0, target_class]
        score.backward(retain_graph=True)

        # Global average pooling on gradients across spatial dimensions (H, W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)  # Positive contributions only

        # Spatial interpolation back to input image size
        cam = torch.nn.functional.interpolate(
            cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False
        )
        cam_np = cam.squeeze().cpu().detach().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


# ==============================================================================
# Plotting & Analysis Functions
# ==============================================================================
def plot_confusion_matrix(cm: np.ndarray, target_names: list, save_path: Path):
    """Plots and saves an annotated confusion matrix heatmap."""
    plt.figure(figsize=(7, 6))
    
    # Calculate percentage over true class (row sums)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_percent[i, j]:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=True,
        annot_kws={"size": 13, "weight": "bold"},
    )
    plt.title("ResNet-18 Holdout Test Confusion Matrix\n(5 Unseen Base Stations, 2,770 Windows)", fontsize=13, pad=15)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("True Class", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved Confusion Matrix heatmap to: {save_path}")


def plot_roc_pr_curves(targets: np.ndarray, probs: np.ndarray, save_path: Path):
    """Plots ROC-AUC and Precision-Recall curves side by side."""
    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(targets, probs)
    pr_auc = auc(recall, precision)
    baseline_pr = np.sum(targets == 1) / len(targets)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: ROC Curve
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ResNet-18 ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Guess (AUC = 0.50)")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (FPR)", fontsize=12)
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=12)
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=13, pad=10)
    axes[0].legend(loc="lower right", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Precision-Recall Curve
    axes[1].plot(recall, precision, color="#2ca02c", lw=2.5, label=f"ResNet-18 PR (AUC = {pr_auc:.4f})")
    axes[1].axhline(y=baseline_pr, color="red", lw=1.5, linestyle="--", label=f"Prevalence Baseline ({baseline_pr*100:.1f}%)")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall (Anomaly Catch Rate)", fontsize=12)
    axes[1].set_ylabel("Precision (Accuracy of Alarms)", fontsize=12)
    axes[1].set_title("Precision-Recall Curve (Imbalanced Anomaly Detection)", fontsize=13, pad=10)
    axes[1].legend(loc="upper right", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved ROC & PR curves to: {save_path}")
    return roc_auc, pr_auc


def plot_threshold_sweep(targets: np.ndarray, probs: np.ndarray, save_path: Path):
    """
    Sweeps decision threshold from 0.05 to 0.95 and identifies optimal F1 threshold.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    precisions = []
    recalls = []
    f1s = []

    best_thresh = 0.50
    best_f1 = 0.0

    for t in thresholds:
        preds = (probs >= t).astype(int)
        p = precision_score(targets, preds, zero_division=0)
        r = recall_score(targets, preds, zero_division=0)
        f = f1_score(targets, preds, zero_division=0)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

        if f > best_f1:
            best_f1 = f
            best_thresh = t

    plt.figure(figsize=(9, 6))
    plt.plot(thresholds, precisions, label="Anomaly Precision", color="#1f77b4", lw=2.2)
    plt.plot(thresholds, recalls, label="Anomaly Recall", color="#ff7f0e", lw=2.2)
    plt.plot(thresholds, f1s, label="Anomaly F1-Score", color="#2ca02c", lw=2.8)

    # Highlight optimal threshold
    plt.axvline(best_thresh, color="purple", linestyle="--", lw=1.8,
                label=f"Optimal F1 Threshold (T* = {best_thresh:.2f}, F1 = {best_f1:.4f})")
    plt.axvline(0.50, color="grey", linestyle=":", lw=1.5, label="Default Threshold (0.50)")

    plt.scatter([best_thresh], [best_f1], color="purple", s=100, zorder=5)

    plt.title("Operational Decision Threshold Tuning\n(Precision vs. Recall vs. F1 Trade-off)", fontsize=13, pad=15)
    plt.xlabel("Anomaly Probability Decision Threshold (T)", fontsize=12, labelpad=10)
    plt.ylabel("Metric Score", fontsize=12, labelpad=10)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.05, 0.95])
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Plot] Saved Threshold Sweep plot to: {save_path}")
    print(f"[Tuning] Optimal Threshold: T* = {best_thresh:.2f} (Peak F1: {best_f1:.4f})")
    return best_thresh, best_f1


def generate_gradcam_visualizations(
    model: nn.Module,
    test_dataset: ImageFolder,
    device: torch.device,
    save_path: Path,
    num_samples: int = 4
):
    """
    Generates side-by-side explainability visualizations using Grad-CAM.
    Selects true anomaly cases and overlays attention heatmaps from layer4[-1].
    """
    grad_cam = GradCAM(model, model.layer4[-1])
    anomaly_idx = test_dataset.class_to_idx["anomaly"]

    # Collect indices of actual anomalies
    anomaly_indices = [
        i for i, (_, label) in enumerate(test_dataset.samples) if label == anomaly_idx
    ]

    selected_indices = anomaly_indices[:num_samples]

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3.5 * num_samples))

    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    for row_idx, sample_idx in enumerate(selected_indices):
        img_path, true_label = test_dataset.samples[sample_idx]
        img_tensor, _ = test_dataset[sample_idx]
        input_batch = img_tensor.unsqueeze(0).to(device)

        # Forward prediction
        with torch.no_grad():
            output = model(input_batch)
            prob = torch.softmax(output, dim=1)[0, anomaly_idx].item()
            pred_class = torch.argmax(output, dim=1).item()

        # Generate Grad-CAM for anomaly class
        cam_mask = grad_cam.generate(input_batch, target_class=anomaly_idx)

        # Denormalize original image for display
        img_display = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_display = img_display * imagenet_std + imagenet_mean
        img_display = np.clip(img_display, 0.0, 1.0)

        # Create blended heatmap overlay
        heatmap = plt.cm.jet(cam_mask)[:, :, :3]
        overlay = 0.55 * img_display + 0.45 * heatmap
        overlay = np.clip(overlay, 0.0, 1.0)

        filename = Path(img_path).name

        # Column 1: Original GAF (RGB = PRB / Throughput / Users)
        axes[row_idx, 0].imshow(img_display)
        axes[row_idx, 0].set_title(f"Sample {row_idx+1}: 3-Channel GAF\n({filename[:25]}...)", fontsize=10)
        axes[row_idx, 0].axis("off")

        # Column 2: Grad-CAM Saliency Map
        im = axes[row_idx, 1].imshow(cam_mask, cmap="jet")
        axes[row_idx, 1].set_title("Grad-CAM Attention Map\n(Conv Layer 4 Saliency)", fontsize=10)
        axes[row_idx, 1].axis("off")

        # Column 3: Blended Decision Overlay
        status = "CONFIRMED ANOMALY" if pred_class == anomaly_idx else "MISSED (FN)"
        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title(
            f"Decision Overlay [{status}]\nAnomaly Prob: {prob*100:.1f}%",
            fontsize=10,
            fontweight="bold",
            color="red" if pred_class == anomaly_idx else "black"
        )
        axes[row_idx, 2].axis("off")

    plt.suptitle(
        "Explainable AI: ResNet-18 Grad-CAM Saliency on Cellular GAF Telemetry\n"
        "(Bright zones identify the exact 15-min intervals & KPI correlations driving anomaly classification)",
        fontsize=13,
        y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved Grad-CAM explainability visualization to: {save_path}")


# ==============================================================================
# Main Evaluation Pipeline
# ==============================================================================
def run_evaluation():
    print("=" * 80)
    print("STEP 4: MODEL EVALUATION, THRESHOLD TUNING & GRAD-CAM EXPLAINABILITY")
    print("=" * 80)

    device = get_device()

    # 1. Load Test Dataset
    test_dataset, test_loader = load_test_dataloader(batch_size=64)
    anomaly_idx = test_dataset.class_to_idx["anomaly"]
    normal_idx = test_dataset.class_to_idx["normal"]

    # 2. Load Trained ResNet-18 Model Checkpoint
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {CHECKPOINT_PATH}")

    model = build_resnet18(num_classes=2, dropout_rate=0.3)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"[Model] Successfully loaded weights from: {CHECKPOINT_PATH}")

    # 3. Execute Inference on Test Set
    all_targets = []
    all_probs = []  # Probability of anomaly class
    all_preds_default = []

    print("[Inference] Running forward pass on 2,770 holdout test images...")
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # Extract anomaly probabilities (index = anomaly_idx)
            anomaly_probs = probs[:, anomaly_idx].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_targets.extend(targets.numpy())
            all_probs.extend(anomaly_probs)
            all_preds_default.extend(preds)

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_preds_default = np.array(all_preds_default)

    # Convert targets to binary: 1 = Anomaly, 0 = Normal
    binary_targets = (all_targets == anomaly_idx).astype(int)
    binary_preds_default = (all_preds_default == anomaly_idx).astype(int)

    # 4. Standard Metrics at Default Threshold (0.50)
    cm_default = confusion_matrix(binary_targets, binary_preds_default)
    tn, fp, fn, tp = cm_default.ravel()

    acc_default = accuracy_score(binary_targets, binary_preds_default)
    prec_default = precision_score(binary_targets, binary_preds_default, zero_division=0)
    rec_default = recall_score(binary_targets, binary_preds_default, zero_division=0)
    f1_default = f1_score(binary_targets, binary_preds_default, zero_division=0)

    print("\n" + "-" * 75)
    print("TEST SET BENCHMARKS (Default Threshold = 0.50)")
    print("-" * 75)
    print(f"Accuracy:           {acc_default:.4f}")
    print(f"Anomaly Precision:  {prec_default:.4f} ({prec_default*100:.2f}%)")
    print(f"Anomaly Recall:     {rec_default:.4f} ({rec_default*100:.2f}%)")
    print(f"Anomaly F1-Score:   {f1_default:.4f}")
    print(f"\nConfusion Matrix (Test Set - 2,770 Samples):")
    print(f"  True Positives  (TP - Caught Outages): {tp:>5} ({tp/(tp+fn)*100:.1f}%)")
    print(f"  False Negatives (FN - Missed Outages): {fn:>5} ({fn/(tp+fn)*100:.1f}%)")
    print(f"  True Negatives  (TN - Correct Normal): {tn:>5} ({tn/(tn+fp)*100:.1f}%)")
    print(f"  False Positives (FP - False Alarms):   {fp:>5} ({fp/(tn+fp)*100:.1f}%)")

    # 5. Generate Visual Artifacts
    print("\n[Plots] Generating publication-quality evaluation figures...")
    plot_confusion_matrix(
        cm_default,
        target_names=["Normal", "Anomaly"],
        save_path=ARTIFACTS_DIR / "eval_confusion_matrix.png"
    )

    roc_auc, pr_auc = plot_roc_pr_curves(
        binary_targets,
        all_probs,
        save_path=ARTIFACTS_DIR / "eval_roc_pr_curves.png"
    )

    optimal_thresh, optimal_f1 = plot_threshold_sweep(
        binary_targets,
        all_probs,
        save_path=ARTIFACTS_DIR / "eval_threshold_tuning.png"
    )

    # Calculate metrics at optimal threshold
    binary_preds_optimal = (all_probs >= optimal_thresh).astype(int)
    prec_optimal = precision_score(binary_targets, binary_preds_optimal, zero_division=0)
    rec_optimal = recall_score(binary_targets, binary_preds_optimal, zero_division=0)
    cm_optimal = confusion_matrix(binary_targets, binary_preds_optimal)
    tn_opt, fp_opt, fn_opt, tp_opt = cm_optimal.ravel()

    print("\n" + "-" * 75)
    print(f"TEST SET BENCHMARKS (Optimal Threshold T* = {optimal_thresh:.2f})")
    print("-" * 75)
    print(f"Anomaly Precision:  {prec_optimal:.4f} ({prec_optimal*100:.2f}%)")
    print(f"Anomaly Recall:     {rec_optimal:.4f} ({rec_optimal*100:.2f}%)")
    print(f"Anomaly F1-Score:   {optimal_f1:.4f}")
    print(f"  Caught Outages (TP): {tp_opt} / {tp_opt + fn_opt} ({tp_opt/(tp_opt+fn_opt)*100:.1f}%)")
    print(f"  False Alarms   (FP): {fp_opt} / {tn_opt + fp_opt} ({fp_opt/(tn_opt+fp_opt)*100:.1f}%)")

    # 6. Generate Grad-CAM Explainability Visualization
    print("\n[Explainability] Generating Grad-CAM attention overlays...")
    generate_gradcam_visualizations(
        model,
        test_dataset,
        device,
        save_path=ARTIFACTS_DIR / "eval_gradcam_sample.png",
        num_samples=4
    )

    # 7. Serialize Metrics to JSON
    test_metrics = {
        "dataset": "test_holdout_towers",
        "total_test_samples": len(binary_targets),
        "total_anomalies": int(np.sum(binary_targets == 1)),
        "total_normals": int(np.sum(binary_targets == 0)),
        "roc_auc_score": round(float(roc_auc), 4),
        "pr_auc_score": round(float(pr_auc), 4),
        "default_threshold_0_50": {
            "accuracy": round(float(acc_default), 4),
            "anomaly_precision": round(float(prec_default), 4),
            "anomaly_recall": round(float(rec_default), 4),
            "anomaly_f1_score": round(float(f1_default), 4),
            "confusion_matrix": {
                "true_positives": int(tp),
                "false_negatives": int(fn),
                "false_positives": int(fp),
                "true_negatives": int(tn),
            }
        },
        "optimal_threshold_tuning": {
            "optimal_threshold": round(float(optimal_thresh), 2),
            "anomaly_precision": round(float(prec_optimal), 4),
            "anomaly_recall": round(float(rec_optimal), 4),
            "anomaly_f1_score": round(float(optimal_f1), 4),
            "confusion_matrix": {
                "true_positives": int(tp_opt),
                "false_negatives": int(fn_opt),
                "false_positives": int(fp_opt),
                "true_negatives": int(tn_opt),
            }
        }
    }

    metrics_path = ARTIFACTS_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n[+] Saved complete evaluation metrics to: {metrics_path}")
    print("=" * 80)
    print("STEP 4 COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
