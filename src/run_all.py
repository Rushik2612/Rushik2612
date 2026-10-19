"""Run the whole pipeline in order:
splits -> train -> evaluate FP32 -> export ONNX -> quantize INT8 -> evaluate INT8
-> benchmark -> update README results.

Usage: python src/run_all.py [--epochs 20] [--freeze-epochs 5] [--runs 500]
Stops at the first step that fails.
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent


def run(script, *args):
    cmd = [sys.executable, str(SRC / script), *map(str, args)]
    print(f"\n=== {' '.join(cmd[1:])} ===", flush=True)
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"Step failed: {script}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--freeze-epochs", type=int, default=5)
    ap.add_argument("--runs", type=int, default=500, help="timed benchmark runs")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random init (only for offline smoke tests)")
    args = ap.parse_args()

    run("make_splits.py")
    train_args = ["--epochs", args.epochs, "--freeze-epochs", args.freeze_epochs]
    if args.no_pretrained:
        train_args.append("--no-pretrained")
    run("train.py", *train_args)
    run("evaluate.py", "--model", "models/fp32.pth", "--split", "test")
    run("export_onnx.py")
    run("quantize.py")
    run("evaluate.py", "--model", "models/int8.onnx", "--split", "test")
    run("benchmark.py", "--runs", args.runs)
    run("update_readme.py")
    print("\nAll steps finished. See results/ and README.md.")


if __name__ == "__main__":
    main()
