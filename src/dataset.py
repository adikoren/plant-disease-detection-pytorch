"""
src/dataset.py — Data loading and augmentation for LeafScan.

WHY this file has exactly two responsibilities:
1. get_transforms — defines the image preprocessing pipeline per mode.
2. get_dataloaders — wraps the filesystem into PyTorch DataLoaders.

Keeping data logic isolated means train.py and inference.py never touch
raw file paths or PIL images directly.
"""

import torch
from typing import Tuple, List
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import config


def get_transforms(mode: str) -> transforms.Compose:
    """
    Build the image transformation pipeline for a given dataset split.

    WHY separate train vs val transforms:
    - Train: heavy augmentation forces the model to learn from varied views,
      reducing overfitting on the training set.
    - Val/Inference: deterministic transforms ensure fair, reproducible evaluation.
      inference.py reuses 'val' exactly so predictions match training assumptions.

    Args:
        mode: Either 'train' or 'val'. Raises ValueError for anything else.

    Returns:
        A torchvision.transforms.Compose pipeline.
    """
    if mode not in ("train", "val"):
        raise ValueError(f"mode must be 'train' or 'val', got '{mode}'")

    normalize = transforms.Normalize(mean=config.MEAN, std=config.STD)

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            # WHY RandomHorizontalFlip: a diseased leaf looks the same mirrored.
            transforms.RandomHorizontalFlip(p=0.5),
            # WHY RandomVerticalFlip: less common in real photos, but adds diversity.
            transforms.RandomVerticalFlip(p=0.3),
            # WHY RandomRotation: leaves appear at any angle in field photos.
            transforms.RandomRotation(degrees=15),
            # WHY ColorJitter: lighting conditions vary wildly in the real world.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:  # val / inference
        return transforms.Compose([
            # WHY resize to IMG_SIZE + 32 then crop: standard practice from ResNet paper.
            # Slightly larger resize followed by center crop avoids border artifacts.
            transforms.Resize(config.IMG_SIZE + 32),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            normalize,
        ])


def get_dataloaders(
    train_dir: str,
    valid_dir: str,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Load the train and validation splits from disk into PyTorch DataLoaders.

    WHY ImageFolder: our dataset is already organised as class subfolders
    (e.g. data/train/Tomato___Early_blight/). ImageFolder reads this structure
    automatically and assigns integer labels sorted alphabetically — no manual
    label mapping needed.

    Args:
        train_dir:   Path to the training images root (contains class subfolders).
        valid_dir:   Path to the validation images root.
        batch_size:  Number of images per batch. From config.
        num_workers: Parallel CPU workers for data loading. From config.

    Returns:
        Tuple of (train_loader, val_loader, class_names).
        class_names is a sorted list of the 38 string labels.
    """
    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms("train"))
    valid_dataset = datasets.ImageFolder(valid_dir, transform=get_transforms("val"))

    # WHY check class consistency: a mismatch between train and valid class lists
    # would silently corrupt labels and produce meaningless evaluation results.
    if train_dataset.classes != valid_dataset.classes:
        raise ValueError(
            "Train and validation class lists do not match. "
            "Check that both directories have identical subfolders."
        )

    # WHY pin_memory: when using CUDA, pinned memory enables faster GPU transfers.
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # WHY shuffle=True: prevents order-based overfitting.
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,          # WHY shuffle=False: evaluation must be deterministic.
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader, val_loader, train_dataset.classes
