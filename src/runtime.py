"""Load a model (.pth or .onnx) as a simple predict function for CPU.

predict(x) takes a float32 numpy array of shape (1, 3, 128, 128) and returns
logits as a numpy array of shape (1, 4).
"""
from pathlib import Path

import numpy as np


def load_predictor(model_path, threads: int = None):
    model_path = Path(model_path)
    if model_path.suffix == ".pth":
        import torch
        from model import build_model

        if threads:
            torch.set_num_threads(threads)
        model = build_model(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        def predict(x: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                return model(torch.from_numpy(x)).numpy()
        return predict

    if model_path.suffix == ".onnx":
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        sess = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        def predict(x: np.ndarray) -> np.ndarray:
            return sess.run(None, {input_name: x})[0]
        return predict

    raise ValueError(f"Unsupported model type: {model_path}")


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
