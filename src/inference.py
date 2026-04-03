"""
src/inference.py — Single-image prediction engine for LeafScan.

WHY this file is separate from evaluate.py:
- evaluate.py: batch processing, metrics, charts — used once after training.
- inference.py: single-image, real-time predictions — called by the API on
  every user request. Keeping them separate means the API never imports
  matplotlib or sklearn, keeping the server lean.

The predict() function is the "product" — everything we built during training
is distilled into this one callable.
"""

import sys
import os
import logging
from io import BytesIO
from typing import List, Union

import torch
import torch.nn as nn
from PIL import Image
import torchvision.models as models

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.dataset import get_transforms
from src.model import build_model
from src.utils import load_checkpoint


def load_model(
    checkpoint_path: str,
    num_classes: int = config.NUM_CLASSES,
    device: torch.device = None,
) -> tuple[nn.Module, List[str]]:
    """
    Load a trained model from a checkpoint file and set it to eval mode.

    WHY eval() is CRITICAL:
    Without model.eval(), Dropout layers remain active (randomly zeroing neurons)
    and BatchNorm uses batch statistics instead of the learned running stats.
    This introduces randomness and degraded accuracy in every prediction.

    Args:
        checkpoint_path: Path to the .pth checkpoint produced by train.py.
        num_classes:     Must match what the model was trained with.
        device:          Target device. Auto-detected if None.

    Returns:
        Tuple of (model_in_eval_mode, class_names_list).
    """
    if device is None:
        from src.utils import get_device
        device = get_device()

    model = build_model(num_classes=num_classes, freeze_backbone=False)

    # load_checkpoint handles FileNotFoundError with a helpful message
    epoch, val_acc = load_checkpoint(model, checkpoint_path)
    logging.info(f"Model loaded (trained for {epoch} epochs, best val_acc={val_acc:.4f})")

    model = model.to(device)
    model.eval()  # CRITICAL — see docstring above
    return model


class OODDetector:
    """
    Out-Of-Distribution (OOD) Detector using MobileNetV3.
    
    WHY: Our LeafScan model only knows 38 diseases. If fed a dog, it will confidently 
    call it a diseased leaf. This pre-filter catches standard objects.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=self.weights).to(device)
        self.model.eval()
        self.transforms = self.weights.transforms()
        self.categories = self.weights.meta["categories"]
        
        # ImageNet classes that are plausibly in our photos
        # 300-320: Insects (often on leaves)
        # 580: greenhouse, 738: pot, 935-957: fruits/veg, 985-998: plants/fungi
        self.whitelist = set(list(range(300, 321)) + [580, 738] + list(range(935, 999)))

    def is_leaf(self, pil_image: Image.Image) -> tuple[bool, str]:
        """
        Returns (True/False, reason)
        If MobileNet is highly confident it's a random object (e.g. dog, car), 
        we reject it. We allow it if MobileNet is 'confused' (since ImageNet lacks 
        a generic 'leaf' class) or if the confident prediction is in the whitelist.
        """
        tensor = self.transforms(pil_image).unsqueeze(0).to(self.device, non_blocking=True)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(self.model(tensor)[0], dim=0)
            max_prob, max_idx = probs.max(0)
            
        prob_val = max_prob.item()
        idx = max_idx.item()
        predicted_object = self.categories[idx]
        
        # If the model is 60%+ sure it's a random object (not in whitelist), reject it.
        if prob_val > 0.60 and idx not in self.whitelist:
            return False, predicted_object
            
        return True, ""

# Global OOD instance to avoid reloading
_ood_detector = None


def predict(
    model: nn.Module,
    image: Union[str, bytes, Image.Image],
    class_names: List[str],
    device: torch.device,
) -> dict:
    """
    Run inference on a single image and return the top-3 predictions.

    WHY accept str | bytes | PIL.Image:
    - str:       CLI scripts pass file paths.
    - bytes:     FastAPI receives uploaded files as bytes.
    - PIL.Image: Gradio passes PIL images directly.
    Centralising all input handling here means the API code stays clean.

    WHY CONFIDENCE_THRESHOLD:
    Without a threshold, the model always outputs a prediction even for a photo
    of a car or a sunset. Below 30% confidence we refuse to diagnose — this
    prevents the app from giving confidently wrong disease predictions to users.

    Args:
        model:        Loaded, eval-mode PlantDiseaseModel.
        image:        Input image as file path, raw bytes, or PIL Image.
        class_names:  Ordered list of 38 class strings.
        device:       Compute device.

    Returns:
        dict with keys:
          'success' (bool), 'disease' (str), 'confidence' (float),
          'top_3' (list of {disease, confidence}), or 'error' (str).
    """
    # --- Normalise input to PIL.Image ---
    try:
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            pil_image = Image.open(BytesIO(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
    except Exception as exc:
        return {"success": False, "error": f"Image loading failed: {exc}"}

    # --- OOD Detection (Smart Pre-filter) ---
    global _ood_detector
    if _ood_detector is None:
        _ood_detector = OODDetector(device)
        
    is_valid, detected_obj = _ood_detector.is_leaf(pil_image)
    if not is_valid:
        return {
            "success": False,
            "error": f"This does not look like a plant leaf. (Detected: {detected_obj})"
        }

    # --- Preprocess ---
    # WHY val transforms: must match the transforms used during validation.
    # Using train transforms (with random augmentation) at inference would
    # produce different results on every call for the same image.
    val_tf = get_transforms("val")
    tensor = val_tf(pil_image)             # shape: [3, H, W]
    tensor = tensor.unsqueeze(0)           # → [1, 3, H, W] (batch dimension)
    tensor = tensor.to(device, non_blocking=True)

    # --- Forward pass ---
    with torch.no_grad():
        logits = model(tensor)                        # [1, 38]
        # WHY softmax here (not in model.forward):
        # During training, CrossEntropyLoss applies log-softmax internally.
        # At inference we need actual probabilities (sum to 1) for thresholding
        # and display — so we apply softmax explicitly here.
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze()  # [38]

    # --- Extract results ---
    top3_probs, top3_indices = torch.topk(probs, k=3)
    top3 = [
        {"disease": class_names[idx.item()], "confidence": round(prob.item(), 4)}
        for prob, idx in zip(top3_probs, top3_indices)
    ]

    best_confidence = top3[0]["confidence"]
    best_disease    = top3[0]["disease"]

    # --- Confidence gate ---
    if best_confidence < config.CONFIDENCE_THRESHOLD:
        return {
            "success": False,
            "error": (
                f"Low confidence ({best_confidence:.2%}). "
                "Please upload a clear, close-up photo of a plant leaf."
            ),
        }

    return {
        "success":    True,
        "disease":    best_disease,
        "confidence": best_confidence,
        "top_3":      top3,
    }
