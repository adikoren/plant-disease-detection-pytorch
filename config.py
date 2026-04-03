"""
config.py — Central configuration for LeafScan.

WHY: Every hyperparameter and path lives here and ONLY here.
This means changing batch size, learning rate, or any setting
requires editing exactly one file. No magic numbers scattered
across the codebase.
"""

import os

# ---------------------------------------------------------------------------
# Paths — all relative to project root
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR  = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR  = os.path.join(DATA_DIR, "test")

CHECKPOINT_DIR  = os.path.join(BASE_DIR, "experiments")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LOG_FILE        = os.path.join(CHECKPOINT_DIR, "training.log")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
NUM_CLASSES      = 38    # Fixed: 38 crop/disease classes from PlantVillage
FREEZE_BACKBONE  = True  # True = only train the custom head; False = fine-tune all

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE               = 32
NUM_EPOCHS               = 30
LEARNING_RATE            = 1e-3
WEIGHT_DECAY             = 1e-4   # L2 regularisation to reduce overfitting
DROPOUT_RATE             = 0.3    # Dropout inside the custom FC head
EARLY_STOPPING_PATIENCE  = 5      # Stop if val_acc doesn't improve for 5 epochs
NUM_WORKERS              = 4      # DataLoader parallel workers

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
IMG_SIZE = 224  # ResNet50 standard input size

# ---------------------------------------------------------------------------
# Normalization — ImageNet defaults (updated after EDA if needed)
# WHY ImageNet values: our backbone was pre-trained on ImageNet, so we must
# normalise inputs the same way it expects them.
# ---------------------------------------------------------------------------
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.30  # Below this probability → reject as "not a plant leaf"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
APP_NAME = "LeafScan"
APP_HOST = "0.0.0.0"
APP_PORT = 8000
