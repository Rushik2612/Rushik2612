"""Shared settings for EdgeHRC: paths, class names, image size and seeding.

All paths hang off ROOT. By default ROOT is the project folder, but it can be
overridden with the EDGEHRC_ROOT environment variable (used for the dummy-data
smoke test so real data/models/results are never touched).
"""
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("EDGEHRC_ROOT", Path(__file__).resolve().parent.parent))

DATA_RAW = ROOT / "data" / "raw"
SPLITS_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Class index -> name. The order matters: it is the model's output order.
CLASSES = ["stop", "resume", "pause", "none"]
# Class name -> robot command shown at inference time.
COMMANDS = {"stop": "STOP", "resume": "RESUME", "pause": "PAUSE", "none": "no command"}

IMG_SIZE = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SEED = 42

# Filenames look like "<session>__<class>__<index>.jpg", so the session id can
# always be recovered from the file name (needed for splitting by session).
SEP = "__"


def set_seed(seed: int = SEED) -> None:
    """Fix every random number generator we use so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def ensure_dirs() -> None:
    for d in (DATA_RAW, SPLITS_DIR, MODELS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
