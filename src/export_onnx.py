"""Export models/fp32.pth to models/fp32.onnx (fixed input 1x3x128x128) and
check that ONNX Runtime gives the same outputs as PyTorch on 100 val images.
Writes results/export_check.json.
"""
import json
import warnings

import numpy as np
import torch

from config import IMG_SIZE, MODELS_DIR, RESULTS_DIR, ensure_dirs
from dataset import read_split
from model import build_model
from preprocess import load_image, preprocess
from runtime import load_predictor

NUM_CHECK = 100
OPSET = 17


def main():
    ensure_dirs()
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(MODELS_DIR / "fp32.pth", map_location="cpu"))
    model.eval()

    onnx_path = MODELS_DIR / "fp32.onnx"
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    # No dynamic axes: a fixed shape is simplest for quantization and deployment.
    export_kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=OPSET)
    try:
        # Newer torch: dynamo=False picks the classic TorchScript exporter
        # (stable, no extra dependencies). It prints a deprecation warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch (< 2.5) has no `dynamo` argument; classic exporter is the default.
        torch.onnx.export(model, dummy, str(onnx_path), **export_kwargs)
    print(f"Exported {onnx_path}")

    # Compare PyTorch vs ONNX Runtime on up to 100 validation images.
    onnx_predict = load_predictor(onnx_path)
    items = read_split("val")[:NUM_CHECK]
    max_diff, agree = 0.0, 0
    for path, _ in items:
        x = preprocess(load_image(path))[None]
        with torch.inference_mode():
            pt = model(torch.from_numpy(x)).numpy()
        ox = onnx_predict(x)
        max_diff = max(max_diff, float(np.abs(pt - ox).max()))
        agree += int(pt.argmax() == ox.argmax())

    result = {"num_images": len(items), "max_abs_logit_diff": max_diff,
              "top1_agreement": agree / len(items)}
    with open(RESULTS_DIR / "export_check.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Checked {len(items)} val images: max |logit diff| = {max_diff:.2e}, "
          f"top-1 agreement = {agree}/{len(items)}")
    if max_diff > 1e-3 or agree != len(items):
        raise SystemExit("ONNX output does not match PyTorch closely enough!")
    print("OK: ONNX matches PyTorch.")


if __name__ == "__main__":
    main()
