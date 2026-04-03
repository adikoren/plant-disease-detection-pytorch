"""
src/model.py — Neural network architecture for LeafScan.

WHY ResNet50 + custom head:
- ResNet50 backbone: pre-trained on ImageNet, expert at extracting visual features
  (edges, textures, shapes). We get this expertise for free via transfer learning.
- Custom Sequential head: replaces the original 1000-class ImageNet classifier
  with our own 3-layer head tuned for 38 plant disease classes.
- Dropout in the head: regularises the newly trained layers to prevent overfitting
  since the head is trained from scratch on a relatively small dataset.

WHY build_model() wrapper: hides the class name from all callers. If we later
switch from ResNet50 to EfficientNet, only this file changes — inference.py,
train.py, etc. remain untouched.
"""

import logging
import torch
import torch.nn as nn
from torchvision import models

import config


class PlantDiseaseModel(nn.Module):
    """
    ResNet50 backbone with a custom classification head for 38 plant disease classes.

    Architecture:
        ResNet50 (ImageNet pretrained) → custom FC head → 38 logits

    The backbone can be fully frozen (feature extractor mode) or left unfrozen
    (fine-tuning mode), controlled by freeze_backbone.
    """

    def __init__(self, num_classes: int, freeze_backbone: bool = True) -> None:
        """
        Initialise the model: load backbone, optionally freeze it, attach custom head.

        Args:
            num_classes:     Number of output classes (38 for LeafScan).
            freeze_backbone: If True, backbone weights are frozen and only the
                             head is trained. Set False for full fine-tuning.
        """
        super().__init__()

        # Load ResNet50 with ImageNet-pretrained weights.
        # WHY pretrained weights: the backbone already knows how to detect edges,
        # textures, and shapes. We build on top of that — no need to learn from zero.
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            # WHY freeze: when data is limited relative to model size, freezing the
            # backbone prevents it from "forgetting" its ImageNet knowledge
            # (catastrophic forgetting). Only the head learns new plant-specific features.
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace the original 1000-class FC layer with our custom head.
        # model.fc.in_features = 2048 for ResNet50.
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            # WHY Dropout: randomly zeroes 30% of activations during training,
            # forcing the head to learn redundant representations → less overfitting.
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.ReLU(),
            # WHY no Dropout before final layer: the last layer should be stable
            # to produce consistent class scores.
            nn.Linear(256, num_classes),
        )
        # WHY: the custom head's requires_grad defaults to True even when the
        # backbone is frozen, because new nn.Sequential layers are always trainable.

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        logging.info(
            f"PlantDiseaseModel | trainable params: {trainable:,} / {total:,} "
            f"({'frozen backbone' if freeze_backbone else 'full fine-tune'})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full model.

        WHY no Softmax here: CrossEntropyLoss expects raw logits and applies
        log-softmax internally. Adding Softmax here would double-apply it
        and corrupt gradients.

        Args:
            x: Input image tensor of shape [B, 3, H, W].

        Returns:
            Raw logits tensor of shape [B, num_classes].
        """
        return self.model(x)


def build_model(num_classes: int = config.NUM_CLASSES,
                freeze_backbone: bool = config.FREEZE_BACKBONE) -> PlantDiseaseModel:
    """
    Factory function — the single point of entry for creating the model.

    WHY a factory wrapper: all callers (train.py, inference.py) import
    build_model, NOT PlantDiseaseModel directly. If we swap to EfficientNet,
    we change only this function. All downstream code stays the same.

    Args:
        num_classes:     Number of disease classes to classify.
        freeze_backbone: Whether to freeze the ResNet50 backbone.

    Returns:
        An initialised PlantDiseaseModel instance (not yet moved to a device).
    """
    return PlantDiseaseModel(num_classes=num_classes, freeze_backbone=freeze_backbone)
