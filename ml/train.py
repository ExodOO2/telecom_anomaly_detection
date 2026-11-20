"""
ResNet-18 Transfer Learning & MLflow Experiment Tracking
Owned by: Person 1 (ML & Computer Vision Engineer)

This script trains our Computer Vision binary classification model on 3-channel
Gramian Angular Field (GAF) images to detect 4G LTE cellular anomalies:
1. PyTorch ImageFolder data loading (train: 12,745 images, val: 2,698 images).
2. Input transforms (Bilinear resize 224x224, ImageNet normalization; NO flips/rotations).
3. Pretrained ResNet-18 transfer learning with custom classification head.
4. Inverse class-weighted loss (CrossEntropyLoss) mapped dynamically to class indices.
5. AdamW optimizer + ReduceLROnPlateau learning rate scheduler.
6. Epoch-by-epoch evaluation tracking (Loss, Accuracy, Anomaly Precision, Recall, F1).
7. Checkpoint saving: artifacts/best_resnet18.pt saved on peak Anomaly F1-Score.
8. Post-training validation confusion matrix and classification report.
"""

import os
import json
import time
from pathlib import Path

# Silence MLflow startup agent hints and cosmetic logs
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

# Try importing MLflow gracefully
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Path definitions
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "gaf_images"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters
HYPERPARAMS = {
    "model_name": "resnet18",
    "image_size": 224,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "epochs": 10,
    "dropout_rate": 0.3,
    "patience": 3,  # Early stopping patience
}


def get_device() -> torch.device:
    """Detects and returns GPU (CUDA) if available, otherwise CPU."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] Accelerated training on GPU: {gpu_name} (CUDA active)")
        return torch.device("cuda")
    else:
        print("[Device] CUDA not available. Training on CPU.")
        return torch.device("cpu")


def build_dataloaders(batch_size: int = 64):
    """
    Constructs PyTorch ImageFolder datasets and DataLoaders for train and validation.
    Note: num_workers=0 is set for Windows to avoid multiprocessing spawn overhead.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    data_transforms = transforms.Compose([
        transforms.Resize((HYPERPARAMS["image_size"], HYPERPARAMS["image_size"]), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise FileNotFoundError(
            f"ImageFolder datasets not found in {DATA_DIR}. Please run ml/data_prep.py first."
        )

    train_dataset = ImageFolder(root=str(TRAIN_DIR), transform=data_transforms)
    val_dataset = ImageFolder(root=str(VAL_DIR), transform=data_transforms)

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )

    print(f"[Data] Loaded {len(train_dataset):,} train images (Classes: {train_dataset.class_to_idx})")
    print(f"[Data] Loaded {len(val_dataset):,} val images (Classes: {val_dataset.class_to_idx})")

    return train_dataset, train_loader, val_loader


def compute_class_weights(dataset: ImageFolder, device: torch.device):
    """
    Computes inverse class weights to handle anomaly class imbalance (~27.8% anomaly).
    Dynamically maps class indices from dataset.class_to_idx to ensure correct weighting.
    """
    anomaly_idx = dataset.class_to_idx["anomaly"]
    normal_idx = dataset.class_to_idx["normal"]

    targets = np.array(dataset.targets)
    count_normal = np.sum(targets == normal_idx)
    count_anomaly = np.sum(targets == anomaly_idx)

    weight_normal = 1.0
    weight_anomaly = float(count_normal) / float(count_anomaly)

    class_weights = torch.zeros(2, dtype=torch.float, device=device)
    class_weights[normal_idx] = weight_normal
    class_weights[anomaly_idx] = weight_anomaly

    print(f"[Loss] Class mapping: 'anomaly' -> index {anomaly_idx} ({count_anomaly:,} samples)")
    print(f"[Loss] Class mapping: 'normal'  -> index {normal_idx} ({count_normal:,} samples)")
    print(f"[Loss] Computed class weights: [Normal (idx {normal_idx}): {class_weights[normal_idx]:.2f}, Anomaly (idx {anomaly_idx}): {class_weights[anomaly_idx]:.2f}]")
    return class_weights, anomaly_idx, normal_idx


def build_resnet18(num_classes: int = 2, dropout_rate: float = 0.3) -> nn.Module:
    """
    Loads pretrained ResNet-18 weights and adapts the classification head:
    Replaces model.fc with Dropout(0.3) -> Linear(512, 2).
    """
    print("[Model] Loading pretrained torchvision.models.resnet18(weights='DEFAULT')...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    return model


def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device, anomaly_idx: int):
    """
    Evaluates model on validation data under torch.no_grad().
    Returns loss, accuracy, and Anomaly-specific precision, recall, and F1-score.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    total_samples = len(all_targets)
    val_loss = running_loss / total_samples
    acc = accuracy_score(all_targets, all_preds)
    
    # Explicitly calculate precision, recall, and F1 for the Anomaly class
    precision = precision_score(all_targets, all_preds, pos_label=anomaly_idx, zero_division=0)
    recall = recall_score(all_targets, all_preds, pos_label=anomaly_idx, zero_division=0)
    f1 = f1_score(all_targets, all_preds, pos_label=anomaly_idx, zero_division=0)

    return val_loss, acc, precision, recall, f1


def train_pipeline():
    print("=" * 80)
    print("STEP 3: RESNET-18 TRANSFER LEARNING & EXPERIMENT TRACKING")
    print("=" * 80)

    device = get_device()

    # 1. Prepare DataLoaders
    train_dataset, train_loader, val_loader = build_dataloaders(batch_size=HYPERPARAMS["batch_size"])

    # 2. Compute Class Weights for Imbalanced Loss (Dynamically mapped)
    class_weights, anomaly_idx, normal_idx = compute_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3. Instantiate Model
    model = build_resnet18(num_classes=2, dropout_rate=HYPERPARAMS["dropout_rate"]).to(device)

    # 4. Configure Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HYPERPARAMS["learning_rate"],
        weight_decay=HYPERPARAMS["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # 5. Initialize MLflow Run
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment("telecom_qos_anomaly_detection")
            mlflow.start_run(run_name="resnet18_transfer_learning")
            mlflow.log_params(HYPERPARAMS)
            print("[MLflow] Logging run to experiment: telecom_qos_anomaly_detection")
        except Exception as e:
            print(f"[MLflow Notice] Could not start MLflow run: {e}")

    best_val_f1 = 0.0
    patience_counter = 0
    checkpoint_path = ARTIFACTS_DIR / "best_resnet18.pt"

    print("\n" + "-" * 82)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Val Loss':^10} | {'Val Acc':^9} | {'Anom Prec':^9} | {'Anom Rec':^8} | {'Anom F1':^8} | {'Status'}")
    print("-" * 82)

    start_training_time = time.perf_counter()
    num_batches = len(train_loader)

    for epoch in range(1, HYPERPARAMS["epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_train_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

            # Print batch progress every 50 batches
            if batch_idx % 50 == 0 or batch_idx == num_batches:
                print(f"  [Epoch {epoch:02d}/{HYPERPARAMS['epochs']:02d}] Batch {batch_idx:03d}/{num_batches:03d} - Batch Loss: {loss.item():.4f}", flush=True)

        train_loss = running_train_loss / len(train_dataset)

        # Validation Phase (pos_label = anomaly_idx)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
            model, val_loader, criterion, device, anomaly_idx=anomaly_idx
        )
        scheduler.step(val_f1)
        epoch_sec = time.perf_counter() - epoch_start

        # Log metrics to MLflow
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_anomaly_precision": val_prec,
                "val_anomaly_recall": val_rec,
                "val_anomaly_f1": val_f1,
            }, step=epoch)

        # Checkpoint Saving & Early Stopping based on Anomaly F1
        status = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)
            status = f"* BEST ({epoch_sec:.1f}s)"
            patience_counter = 0
        else:
            patience_counter += 1
            status = f"patience {patience_counter}/{HYPERPARAMS['patience']} ({epoch_sec:.1f}s)"

        print(
            f"{epoch:^7} | {train_loss:^10.4f} | {val_loss:^10.4f} | {val_acc:^9.4f} | "
            f"{val_prec:^9.4f} | {val_rec:^8.4f} | {val_f1:^8.4f} | {status}",
            flush=True
        )

        if patience_counter >= HYPERPARAMS["patience"]:
            print(f"\n[Early Stopping] No improvement in Anomaly F1 for {HYPERPARAMS['patience']} epochs. Stopping.")
            break

    total_time = time.perf_counter() - start_training_time
    print("-" * 82)
    print(f"\n[+] Training Complete in {total_time/60.0:.2f} minutes.")
    print(f"[+] Best Validation Anomaly F1-Score: {best_val_f1:.4f}")
    print(f"[+] Saved Best Model Weights to: {checkpoint_path}")

    # Load best checkpoint and evaluate confusion matrix & classification report
    print("\n" + "=" * 80)
    print("BEST MODEL VALIDATION CONFUSION MATRIX & EVALUATION REPORT")
    print("=" * 80)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(targets.cpu().numpy())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Class-specific counts
    tp = int(np.sum((val_targets == anomaly_idx) & (val_preds == anomaly_idx)))
    fn = int(np.sum((val_targets == anomaly_idx) & (val_preds == normal_idx)))
    fp = int(np.sum((val_targets == normal_idx) & (val_preds == anomaly_idx)))
    tn = int(np.sum((val_targets == normal_idx) & (val_preds == normal_idx)))

    total_anomalies = tp + fn
    total_normals = fp + tn

    print(f"Confusion Matrix on Validation Set ({len(val_targets):,} samples):")
    print(f"  True Anomalies ({total_anomalies:,}):")
    print(f"    - True Positives  (TP - Caught Anomalies): {tp:>5} ({tp/total_anomalies*100:.1f}%)")
    print(f"    - False Negatives (FN - Missed Anomalies): {fn:>5} ({fn/total_anomalies*100:.1f}%)")
    print(f"  True Normals ({total_normals:,}):")
    print(f"    - True Negatives  (TN - Correct Normal):   {tn:>5} ({tn/total_normals*100:.1f}%)")
    print(f"    - False Positives (FP - False Alarms):     {fp:>5} ({fp/total_normals*100:.1f}%)")

    target_names = ["anomaly" if i == anomaly_idx else "normal" for i in range(2)]
    print("\nClassification Report (Validation Set):")
    print(classification_report(val_targets, val_preds, target_names=target_names, digits=4))

    # Save training summary JSON
    summary = {
        "model_name": HYPERPARAMS["model_name"],
        "class_mapping": train_dataset.class_to_idx,
        "best_val_anomaly_f1": round(best_val_f1, 4),
        "total_training_minutes": round(total_time / 60.0, 2),
        "epochs_completed": epoch,
        "device": str(device),
        "checkpoint_file": checkpoint_path.name,
        "validation_metrics": {
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "anomaly_recall": round(float(tp / total_anomalies), 4) if total_anomalies > 0 else 0.0,
            "anomaly_precision": round(float(tp / (tp + fp)), 4) if (tp + fp) > 0 else 0.0,
        }
    }
    with open(ARTIFACTS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if MLFLOW_AVAILABLE and mlflow.active_run():
        try:
            mlflow.log_artifact(str(checkpoint_path))
            mlflow.end_run()
            print("[MLflow] Run finalized and logged successfully.")
        except Exception:
            pass


if __name__ == "__main__":
    train_pipeline()
