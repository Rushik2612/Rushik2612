"""Evaluate a .pth or .onnx model on a split (default: test).

Reports accuracy, per-class precision/recall/F1 and a confusion matrix.
Writes results/eval_<model>_<split>.json and results/confusion_<model>_<split>.png.

Usage: python src/evaluate.py --model models/int8.onnx --split test
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from config import CLASSES, RESULTS_DIR, ROOT, ensure_dirs
from dataset import read_split
from preprocess import load_image, preprocess
from runtime import load_predictor


def predict_split(model_path, split: str):
    """Run the model image-by-image (batch 1, like deployment). Returns y_true, y_pred."""
    predict = load_predictor(model_path)
    y_true, y_pred = [], []
    for path, label in read_split(split):
        x = preprocess(load_image(path))[None]  # add batch dim -> (1, 3, 128, 128)
        y_pred.append(int(predict(x).argmax()))
        y_true.append(label)
    return np.array(y_true), np.array(y_pred)


def compute_metrics(y_true, y_pred):
    labels = list(range(len(CLASSES)))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {
        "num_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {c: {"precision": float(prec[i]), "recall": float(rec[i]),
                          "f1": float(f1[i]), "support": int(support[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_confusion(cm, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to .pth or .onnx")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    ensure_dirs()
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    tag = f"{model_path.stem}{model_path.suffix.replace('.', '_')}_{args.split}"

    y_true, y_pred = predict_split(model_path, args.split)
    m = compute_metrics(y_true, y_pred)
    m.update({"model": model_path.name, "split": args.split})

    print(f"{model_path.name} on {args.split}: accuracy {m['accuracy']:.4f} "
          f"({m['num_images']} images)")
    print(f"{'class':<8} {'precision':>9} {'recall':>7} {'f1':>6} {'support':>8}")
    for c, v in m["per_class"].items():
        print(f"{c:<8} {v['precision']:>9.4f} {v['recall']:>7.4f} {v['f1']:>6.4f} {v['support']:>8}")
    print("confusion matrix (rows=true, cols=predicted):")
    for c, row in zip(CLASSES, m["confusion_matrix"]):
        print(f"  {c:<8} {row}")

    with open(RESULTS_DIR / f"eval_{tag}.json", "w") as f:
        json.dump(m, f, indent=2)
    plot_confusion(m["confusion_matrix"], f"{model_path.name} ({args.split})",
                   RESULTS_DIR / f"confusion_{tag}.png")
    print(f"Saved results/eval_{tag}.json and results/confusion_{tag}.png")


if __name__ == "__main__":
    main()
