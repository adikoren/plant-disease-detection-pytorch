"""
src/__init__.py — Clean public API for the src package.

WHY explicit imports here:
- External callers (app/main.py, notebooks, tests) can write
  `from src import build_model` instead of `from src.model import build_model`.
- Controls exactly what is "public" from this package.
- One direction only: app imports src, src does NOT import app.

No logic lives here — imports only.
"""

from src.model    import build_model
from src.dataset  import get_dataloaders, get_transforms
from src.train    import train
from src.evaluate import evaluate
from src.utils    import set_seed, get_device, save_checkpoint, load_checkpoint
from src.inference import predict, load_model

__all__ = [
    "build_model",
    "get_dataloaders",
    "get_transforms",
    "train",
    "evaluate",
    "set_seed",
    "get_device",
    "save_checkpoint",
    "load_checkpoint",
    "predict",
    "load_model",
]
