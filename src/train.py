"""
src/train.py — Training loop for LeafScan.

WHY three separate functions (train_one_epoch, validate, train):
- Single Responsibility Principle: each function does exactly one thing.
- train_one_epoch and validate can be unit tested independently.
- train() is the orchestrator — it reads like a recipe, not spaghetti.

Advanced techniques used:
- Mixed precision (torch.cuda.amp): ~2x faster on modern NVIDIA GPUs.
- ReduceLROnPlateau: automatically lowers learning rate when validation
  loss stalls — less manual tuning required.
- Early stopping: prevents wasted compute and overfitting when the model
  has converged.
"""

import logging
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.utils import save_checkpoint, setup_logging, set_seed, get_device
from src.dataset import get_dataloaders
from src.model import build_model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler=None,
) -> tuple[float, float]:
    """
    Execute one full pass over the training dataset.

    WHY support scaler (mixed precision):
    On CUDA GPUs, torch.cuda.amp runs forward passes in float16 and keeps
    master weights in float32. This halves memory usage and speeds up training
    ~2x on compatible hardware. scaler=None safely disables it on CPU/MPS.

    Args:
        model:     Model in train mode.
        loader:    Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function (CrossEntropyLoss).
        device:    Compute device.
        scaler:    GradScaler for mixed precision, or None.

    Returns:
        Tuple of (avg_loss, accuracy) for this epoch.
    """
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds         = logits.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on the validation set.

    WHY torch.no_grad():
    During validation we never call .backward(). Disabling gradient tracking
    saves memory (no computation graph is built) and speeds up the forward pass.

    Args:
        model:     Model to evaluate.
        loader:    Validation DataLoader.
        criterion: Loss function.
        device:    Compute device.

    Returns:
        Tuple of (avg_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="  Val  ", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds         = logits.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train() -> None:
    """
    Main training orchestrator for LeafScan.

    WHY this is a standalone callable (not inline in __main__):
    Makes it importable for other scripts, experiments, or tests without
    re-executing the whole training pipeline.

    Training strategy:
    1. Adam optimizer — adaptive learning rate, works well out of the box.
    2. ReduceLROnPlateau — halves LR when val_loss stops improving (patience=3).
       WHY: prevents getting stuck in a local minimum due to a learning rate
       that is too large for the current loss landscape.
    3. Early stopping — halts training if val_acc doesn't improve for
       EARLY_STOPPING_PATIENCE consecutive epochs.
       WHY: saves compute, avoids overfitting the validation set.
    4. Best-checkpoint saving — only saves when val_acc improves.
    """
    setup_logging(config.LOG_FILE)
    set_seed(42)
    device = get_device()

    logging.info("=" * 60)
    logging.info("LeafScan — Training Start")
    logging.info(f"Epochs: {config.NUM_EPOCHS} | Batch: {config.BATCH_SIZE} | LR: {config.LEARNING_RATE}")

    # --- Data ---
    train_loader, val_loader, class_names = get_dataloaders(
        config.TRAIN_DIR,
        config.VALID_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    logging.info(f"Classes: {len(class_names)} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # --- Model ---
    model = build_model(num_classes=config.NUM_CLASSES, freeze_backbone=config.FREEZE_BACKBONE)
    model = model.to(device)

    # --- Loss, optimiser, scheduler ---
    criterion = nn.CrossEntropyLoss()
    # WHY filter requires_grad: when backbone is frozen, Adam should only update
    # the head. Passing all parameters would waste memory on frozen gradients.
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    # WHY ReduceLROnPlateau on val_loss (not val_acc):
    # Loss is a smoother signal than accuracy for small val sets.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # Mixed precision only makes sense on CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_acc    = 0.0
    epochs_no_improve = 0

    # --- Main loop ---
    for epoch in range(1, config.NUM_EPOCHS + 1):
        logging.info(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        logging.info(
            f"  Train → loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
            f"Val → loss: {val_loss:.4f}  acc: {val_acc:.4f}"
        )

        # --- Checkpoint on improvement ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_acc, config.BEST_MODEL_PATH)
            logging.info(f"  ✓ New best model saved (val_acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            logging.info(f"  No improvement for {epochs_no_improve}/{config.EARLY_STOPPING_PATIENCE} epochs")

        # --- Early stopping ---
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            logging.info(f"Early stopping triggered after {epoch} epochs.")
            break

    logging.info("=" * 60)
    logging.info(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
    logging.info(f"Best model saved at: {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    train()
