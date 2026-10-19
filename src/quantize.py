"""Post-training static INT8 quantization with ONNX Runtime.

1. quant_pre_process: shape inference + graph optimizations (e.g. folding
   BatchNorm into Conv) so quantization works on a clean graph.
2. quantize_static: run ~300 training images through the model (calibration)
   to record the value range of every activation, then store weights and
   activations as 8-bit integers. QDQ format, per-channel weights.
Output: models/int8.onnx
"""
import random

import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from config import MODELS_DIR, SEED, ensure_dirs
from dataset import read_split
from preprocess import load_image, preprocess

NUM_CALIB = 300


class GestureCalibrationReader(CalibrationDataReader):
    """Feeds calibration images one at a time, preprocessed exactly like inference
    (no augmentation)."""

    def __init__(self, input_name: str, num_images: int = NUM_CALIB):
        items = read_split("train")
        random.Random(SEED).shuffle(items)
        self.paths = [p for p, _ in items[:num_images]]
        self.input_name = input_name
        self.it = iter(self.paths)
        print(f"Calibration set: {len(self.paths)} train images")

    def get_next(self):
        path = next(self.it, None)
        if path is None:
            return None
        return {self.input_name: preprocess(load_image(path))[None].astype(np.float32)}

    def rewind(self):
        self.it = iter(self.paths)


def main():
    ensure_dirs()
    fp32 = MODELS_DIR / "fp32.onnx"
    prep = MODELS_DIR / "fp32_prep.onnx"
    int8 = MODELS_DIR / "int8.onnx"

    quant_pre_process(str(fp32), str(prep))
    print(f"Pre-processed model -> {prep}")

    quantize_static(
        model_input=str(prep),
        model_output=str(int8),
        calibration_data_reader=GestureCalibrationReader("input"),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,   # activations: unsigned 8-bit
        weight_type=QuantType.QInt8,        # weights: signed 8-bit
    )
    print(f"Saved {int8}")


if __name__ == "__main__":
    main()
