"""
ResNet-18 Transfer Learning & MLflow Experiment Tracking
Owned by: Person 1 (ML & Computer Vision Engineer)

Responsibilities:
1. Load GAF PNG dataset via torchvision.datasets.ImageFolder
2. Apply PyTorch transforms (Resize 224x224, ImageNet normalize, strictly NO flips/rotations)
3. Instantiate ResNet-18 (pretrained ImageNet weights) + 2-class head
4. Compute inverse class weights for CrossEntropyLoss
5. Train with AdamW optimizer, tracking loss & validation F1 in MLflow
6. Save best checkpoint to artifacts/best_resnet18.pt
"""

import sys
from pathlib import Path

def main():
    print("[ml/train.py] Training pipeline initialized. Ready for Step 3 implementation.")

if __name__ == "__main__":
    main()
