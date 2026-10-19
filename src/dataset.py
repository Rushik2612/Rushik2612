"""PyTorch Dataset over the split CSV files.

Train split: random resized crop, horizontal flip, color jitter, small rotation.
Val/test:    no augmentation (just the shared preprocess()).
Both paths end in the same preprocess() function used at inference time.
"""
import csv

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMG_SIZE, ROOT, SPLITS_DIR
from preprocess import load_image, preprocess


def read_split(split: str):
    """Return a list of (absolute_path, label) for 'train', 'val' or 'test'."""
    with open(SPLITS_DIR / f"{split}.csv", newline="") as f:
        return [(ROOT / row["path"], int(row["label"])) for row in csv.DictReader(f)]


# Augmentations work on PIL RGB images and output a 128x128 PIL image.
TRAIN_AUGMENT = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomRotation(10),
])


class GestureDataset(Dataset):
    def __init__(self, split: str, augment: bool = False):
        self.items = read_split(split)
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = load_image(path)                       # BGR uint8
        if self.augment:
            rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            rgb = np.asarray(TRAIN_AUGMENT(rgb))
            x = preprocess(rgb, bgr=False)
        else:
            x = preprocess(img, bgr=True)
        return torch.from_numpy(x), label
