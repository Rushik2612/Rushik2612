"""The ONE preprocessing function used everywhere (training, evaluation,
export check, quantization calibration, benchmark and webcam inference).

Image in (uint8, HxWx3) -> float32 tensor-like array (3, 128, 128),
resized to 128x128, scaled to [0, 1] and normalized with ImageNet mean/std.
"""
import cv2
import numpy as np

from config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_STD = np.array(IMAGENET_STD, dtype=np.float32)


def preprocess(img: np.ndarray, bgr: bool = True) -> np.ndarray:
    """Convert an image into the model's input format.

    img: uint8 array of shape (H, W, 3). OpenCV images are BGR (bgr=True);
         pass bgr=False if the image is already RGB (e.g. after augmentation).
    returns: float32 array of shape (3, IMG_SIZE, IMG_SIZE), RGB, normalized.
    """
    if bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))  # HWC -> CHW


def load_image(path) -> np.ndarray:
    """Read an image file as BGR uint8. Works with non-ASCII paths on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img
