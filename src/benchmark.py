"""Compare PyTorch FP32, ONNX Runtime FP32 and ONNX Runtime INT8 on this CPU.

For each model: batch size 1, 50 warm-up runs, then N timed runs (default 500)
on the same preprocessed test image. Reports median and p95 latency, file size,
test-set accuracy and speedup. Records CPU, OS and thread count.
Writes results/benchmark.json and results/benchmark.md.
"""
import argparse
import json
import os
import platform
import time
from datetime import datetime

import numpy as np

from config import MODELS_DIR, RESULTS_DIR, ensure_dirs
from dataset import read_split
from evaluate import compute_metrics, predict_split
from preprocess import load_image, preprocess
from runtime import load_predictor

MODELS = [
    ("PyTorch FP32", MODELS_DIR / "fp32.pth"),
    ("ONNX FP32", MODELS_DIR / "fp32.onnx"),
    ("ONNX INT8", MODELS_DIR / "int8.onnx"),
]


def cpu_name() -> str:
    """Best-effort human-readable CPU name on Linux and Windows."""
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def time_model(predict, x, warmup: int, runs: int):
    for _ in range(warmup):
        predict(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        predict(x)
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
    t = np.array(times)
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
            "mean_ms": float(t.mean()), "min_ms": float(t.min())}


def write_markdown(report, path):
    s = report["system"]
    lines = [
        "# EdgeHRC benchmark",
        "",
        f"- CPU: {s['cpu']}",
        f"- OS: {s['os']}",
        f"- Threads: {s['threads']} (logical CPUs: {s['logical_cpus']})",
        f"- Python {s['python']}, torch {s['torch']}, onnxruntime {s['onnxruntime']}",
        f"- Batch size 1, {report['warmup']} warm-up runs, {report['runs']} timed runs",
        f"- Test images: {report['test_images']}",
        f"- Date: {report['date']}",
        "",
        "| Model | Median latency (ms) | p95 latency (ms) | Size (MB) | Test accuracy "
        "| Speedup vs PyTorch FP32 | Speedup vs ONNX FP32 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["results"]:
        lines.append(
            f"| {r['name']} | {r['median_ms']:.2f} | {r['p95_ms']:.2f} | {r['size_mb']:.2f} "
            f"| {r['test_accuracy']:.4f} | {r['speedup_vs_pytorch_fp32']:.2f}x "
            f"| {r['speedup_vs_onnx_fp32']:.2f}x |")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--threads", type=int, default=os.cpu_count(),
                    help="CPU threads for both PyTorch and ONNX Runtime")
    args = ap.parse_args()
    ensure_dirs()

    import onnxruntime
    import torch

    # One fixed, real test image used for all latency measurements.
    test_items = read_split("test")
    x = preprocess(load_image(test_items[0][0]))[None]

    results = []
    for name, path in MODELS:
        print(f"Benchmarking {name} ({path.name}) ...")
        predict = load_predictor(path, threads=args.threads)
        timing = time_model(predict, x, args.warmup, args.runs)
        y_true, y_pred = predict_split(path, "test")
        acc = compute_metrics(y_true, y_pred)["accuracy"]
        results.append({"name": name, "file": path.name,
                        "size_mb": os.path.getsize(path) / 1e6,
                        "test_accuracy": acc, **timing})
        print(f"  median {timing['median_ms']:.2f} ms, p95 {timing['p95_ms']:.2f} ms, "
              f"test acc {acc:.4f}")

    base_pt = results[0]["median_ms"]
    base_onnx = results[1]["median_ms"]
    for r in results:
        r["speedup_vs_pytorch_fp32"] = base_pt / r["median_ms"]
        r["speedup_vs_onnx_fp32"] = base_onnx / r["median_ms"]

    report = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "system": {"cpu": cpu_name(), "os": platform.platform(),
                   "threads": args.threads, "logical_cpus": os.cpu_count(),
                   "python": platform.python_version(), "torch": torch.__version__,
                   "onnxruntime": onnxruntime.__version__},
        "batch_size": 1, "warmup": args.warmup, "runs": args.runs,
        "test_images": len(test_items),
        "results": results,
    }
    with open(RESULTS_DIR / "benchmark.json", "w") as f:
        json.dump(report, f, indent=2)
    write_markdown(report, RESULTS_DIR / "benchmark.md")
    print((RESULTS_DIR / "benchmark.md").read_text())
    print("Saved results/benchmark.json and results/benchmark.md")


if __name__ == "__main__":
    main()
