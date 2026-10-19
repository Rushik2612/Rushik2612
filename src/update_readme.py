"""Fill the Results section of README.md from files in results/.

Only copies numbers that the scripts wrote; nothing is typed in by hand.
The section lives between the RESULTS:START and RESULTS:END markers.
"""
import json

from config import CLASSES, RESULTS_DIR, ROOT

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def build_section() -> str:
    parts = []
    bench = RESULTS_DIR / "benchmark.md"
    if bench.exists():
        # Drop the "# EdgeHRC benchmark" title line; README has its own heading.
        body = bench.read_text().split("\n", 1)[1].strip()
        parts += ["### Speed, size and accuracy (from `results/benchmark.md`)", "", body, ""]

    for tag, title in [("fp32_pth_test", "PyTorch FP32"), ("int8_onnx_test", "ONNX INT8")]:
        p = RESULTS_DIR / f"eval_{tag}.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        parts += [f"### Per-class test results: {title} (from `results/eval_{tag}.json`)", "",
                  f"Accuracy: {m['accuracy']:.4f} on {m['num_images']} test images", "",
                  "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for c in CLASSES:
            v = m["per_class"][c]
            parts.append(f"| {c} | {v['precision']:.4f} | {v['recall']:.4f} "
                         f"| {v['f1']:.4f} | {v['support']} |")
        parts += ["", f"![confusion matrix {title}](results/confusion_{tag}.png)", ""]

    if (RESULTS_DIR / "train_curves.png").exists():
        parts += ["### Training curves", "", "![training curves](results/train_curves.png)", ""]

    if not parts:
        return "_No results yet. Run `python src/run_all.py` to generate them._"
    return "\n".join(parts).strip()


def main():
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit("README.md has no RESULTS markers; nothing updated.")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{build_section()}\n{END}{tail}", encoding="utf-8")
    print("README.md results section updated from results/.")


if __name__ == "__main__":
    main()
