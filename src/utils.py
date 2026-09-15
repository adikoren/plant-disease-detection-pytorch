"""
src/utils.py — Shared helper utilities for LeafScan.

WHY this file exists: Training runs for hours. We need reproducibility,
device-agnostic code, reliable checkpointing, and persistent logging.
These helpers are imported by every other module — built first, depended upon by all.
"""

import json
import os
import random
import logging
from typing import List

import torch
import numpy as np


def get_class_names(class_names_path: str, train_dir: str) -> List[str]:
    """
    Resolve the ordered list of class names the checkpoint was trained with.

    WHY a JSON sidecar first: the training dataset (train_dir) is intentionally
    never shipped to production, so serving can't depend on scanning it via
    ImageFolder. class_names_path is a small, git-tracked file generated once
    from the real training data, kept in lockstep with the checkpoint.

    Falls back to scanning train_dir via ImageFolder when the sidecar isn't
    present, so local development still works before the file is generated.
    """
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            return json.load(f)

    if os.path.exists(train_dir):
        from torchvision import datasets
        return datasets.ImageFolder(train_dir).classes

    raise FileNotFoundError(
        f"No class names available: neither '{class_names_path}' nor "
        f"'{train_dir}' exists."
    )


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for torch, numpy, and Python's random module.

    WHY: Deep learning has many sources of randomness (weight init, data shuffling,
    dropout). Setting a fixed seed means two runs with identical code produce
    identical results — essential for debugging and reproducible experiments.

    Args:
        seed: Integer seed value. Default 42 is conventional.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # WHY deterministic=True: eliminates non-deterministic CUDA operations.
    # Trade-off: slightly slower, but results are fully reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Detect and return the best available compute device.

    WHY: Hardcoding 'cuda' breaks the code on machines without a GPU.
    This function gracefully falls back: CUDA → Apple MPS → CPU.
    The same codebase runs on a cloud GPU, a Mac, or a basic laptop.

    Returns:
        torch.device: The selected device object.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[LeafScan] Using device: {device}")
    return device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    path: str,
) -> None:
    """
    Save model and optimizer state to disk.

    WHY: Saving both model AND optimizer state allows us to resume training
    exactly where we left off if a run is interrupted. Saving only model
    weights would force restarting from scratch.

    Args:
        model:     The PyTorch model whose weights to save.
        optimizer: The optimizer whose state to save.
        epoch:     Current epoch number (for bookkeeping).
        val_acc:   Validation accuracy at this checkpoint (for bookkeeping).
        path:      Full file path where the checkpoint will be written.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_acc": val_acc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    logging.info(f"Checkpoint saved → {path}  (epoch={epoch}, val_acc={val_acc:.4f})")


_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_MIN_CHECKPOINT_BYTES = 10 * 1024 * 1024  # real checkpoint is 100MB+; an LFS pointer is ~130 bytes.


def validate_checkpoint_file(path: str) -> None:
    """
    Guard against a missing/corrupted/unresolved checkpoint before torch.load() runs.

    WHY: DigitalOcean App Platform's git fetch does not resolve Git LFS, so a
    naive copy of this file into a container build can silently ship a ~130-byte
    LFS pointer text file instead of the real 100MB+ weights. torch.load() on
    that pointer fails deep inside model loading with a cryptic UnpicklingError —
    this check catches the real problem immediately with an actionable message.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{path}'. "
            "Run train.py first to generate a model checkpoint."
        )

    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(64)

    if head.startswith(_LFS_POINTER_PREFIX):
        raise RuntimeError(
            f"'{path}' is a Git LFS pointer file ({size} bytes), not the real "
            "checkpoint — the LFS object was never downloaded. "
            "See 'Checkpoint delivery' in README.md."
        )

    if size < _MIN_CHECKPOINT_BYTES:
        raise RuntimeError(
            f"'{path}' is suspiciously small ({size} bytes) to be a real model "
            "checkpoint (expected 100MB+). It may be truncated or corrupted."
        )


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: torch.optim.Optimizer = None,
) -> tuple[int, float]:
    """
    Load a checkpoint into an existing model (and optionally optimizer).

    WHY two modes (with/without optimizer):
    - inference.py needs ONLY the model weights — no optimizer required.
    - train.py needs BOTH model weights and optimizer state to correctly resume.

    Args:
        model:     Model instance to load weights into.
        path:      Path to the .pth checkpoint file.
        optimizer: Optional optimizer to restore state into.

    Returns:
        Tuple of (epoch, val_acc) recorded at checkpoint time.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If the file exists but is an LFS pointer or looks truncated.
    """
    validate_checkpoint_file(path)

    # WHY map_location='cpu': loads safely on any device, then .to(device) moves it.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch   = checkpoint.get("epoch", 0)
    val_acc = checkpoint.get("val_acc", 0.0)
    logging.info(f"Checkpoint loaded from '{path}'  (epoch={epoch}, val_acc={val_acc:.4f})")
    return epoch, val_acc


def setup_logging(log_file: str = "experiments/training.log") -> None:
    """
    Configure logging to write to both the console and a persistent log file.

    WHY: When training runs for 30 epochs overnight, the terminal is gone by morning.
    Logging to a file captures the entire history — loss curves, checkpoints,
    and any warnings — for post-run analysis.

    Args:
        log_file: Path to the log file. Created along with any parent directories.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    log_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),                        # console
            logging.FileHandler(log_file, mode="a"),        # file (append)
        ],
    )
    logging.info(f"Logging initialised. Log file: {log_file}")
