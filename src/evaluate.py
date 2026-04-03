"""
src/evaluate.py — Post-training evaluation tools for LeafScan.

WHY this file is separate from train.py:
Training runs hundreds of batches per epoch. Evaluation tools like confusion
matrices and classification reports are heavy, one-time operations run AFTER
training completes. Mixing them into train.py would bloat and slow the hot path.

Outputs are saved to experiments/ so they survive terminal sessions and can be
included in a portfolio or report.
"""

import logging
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_names: List[str],
    device: torch.device,
) -> tuple[float, dict]:
    """
    Run full evaluation on a dataset split and collect per-class accuracy.

    WHY collect all predictions first, then compute metrics:
    Avoids accumulating running means which can introduce floating-point errors
    in per-class accuracy calculations on unbalanced batches.

    Args:
        model:       Trained model in eval mode.
        loader:      DataLoader for the split to evaluate (val or test).
        class_names: Ordered list of class label strings.
        device:      Compute device.

    Returns:
        Tuple of (overall_accuracy, per_class_accuracy_dict).
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            preds  = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    overall_acc = (all_preds == all_labels).mean()

    # Per-class accuracy
    per_class = {}
    for idx, name in enumerate(class_names):
        mask = all_labels == idx
        per_class[name] = (all_preds[mask] == all_labels[mask]).mean() if mask.sum() > 0 else 0.0

    logging.info(f"Overall accuracy: {overall_acc:.4f}")
    return overall_acc, per_class


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str = os.path.join(config.CHECKPOINT_DIR, "confusion_matrix.png"),
) -> None:
    """
    Generate and save a 38×38 confusion matrix heatmap.

    WHY a confusion matrix matters:
    Overall accuracy hides class-level failures. A confusion matrix shows
    WHICH diseases the model confuses with each other — far more actionable
    than a single number. E.g., if the model confuses Early Blight with
    Late Blight, we can collect more training examples of those specific classes.

    Args:
        y_true:      Ground truth integer labels.
        y_pred:      Predicted integer labels.
        class_names: Ordered list of class strings for axis labels.
        save_path:   File path to save the PNG.
    """
    cm = confusion_matrix(y_true, y_pred)

    # WHY normalize: raw counts are meaningless when class sizes differ.
    # Row-wise normalization shows prediction accuracy per true class.
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(22, 20))
    sns.heatmap(
        cm_norm,
        annot=False,    # 38×38 = 1444 cells — annotations would be illegible
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.3,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("LeafScan — Confusion Matrix (row-normalised)", fontsize=14)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logging.info(f"Confusion matrix saved → {save_path}")


def get_classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str = os.path.join(config.CHECKPOINT_DIR, "classification_report.txt"),
) -> str:
    """
    Generate a per-class precision / recall / F1 report and save it to disk.

    WHY precision, recall, and F1 instead of just accuracy:
    - Precision: of all times we predicted 'Apple Scab', how often were we right?
    - Recall:    of all actual 'Apple Scab' images, how many did we catch?
    - F1:        harmonic mean — punishes models that sacrifice one for the other.
    Essential for a portfolio: shows understanding of evaluation beyond raw accuracy.

    Args:
        y_true:     Ground truth labels.
        y_pred:     Predicted labels.
        class_names: Class strings for readable output.
        save_path:  File path to save the text report.

    Returns:
        The full classification report string.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(report)

    logging.info(f"Classification report saved → {save_path}")
    return report

if __name__ == "__main__":
    from src.dataset import get_dataloaders
    from src.model import build_model
    from src.utils import load_checkpoint, get_device, setup_logging
    
    setup_logging()
    device = get_device()
    _, val_loader, class_names = get_dataloaders(config.TRAIN_DIR, config.VALID_DIR)
    
    model = build_model(config.NUM_CLASSES, freeze_backbone=False)
    load_checkpoint(model, config.BEST_MODEL_PATH)
    model = model.to(device)
    model.eval()
    
    logging.info("Collecting predictions for evaluation...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            
    logging.info("Generating Matrix & Report...")
    plot_confusion_matrix(all_labels, all_preds, class_names)
    get_classification_report(all_labels, all_preds, class_names)
