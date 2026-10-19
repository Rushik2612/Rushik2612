# EdgeHRC — Edge Hand-gesture Robot Control

Real-time hand gesture recognition for touchless robot control, running fully on a
laptop CPU (no cloud). A lightweight CNN (MobileNetV2) is fine-tuned on a small
self-recorded dataset, exported to ONNX and quantized to INT8 with ONNX Runtime.
The project measures how much faster and smaller the INT8 model is, and how much
accuracy it loses.

| Class | Gesture | Robot command |
|---|---|---|
| 0 `stop` | open palm | STOP |
| 1 `resume` | thumbs up | RESUME |
| 2 `pause` | fist | PAUSE |
| 3 `none` | no hand / other gesture | no command |

## Pipeline

```
prepare_hagrid.py -> data/raw/<class>/     open HaGRID dataset -> our 4 classes (by person)
capture.py -> data/raw/<class>/            or record your own webcam frames per session
make_splits.py -> data/splits/*.csv         70/15/15 split BY SESSION
train.py -> models/fp32.pth                 MobileNetV2, frozen then fine-tuned
evaluate.py                                 accuracy, precision/recall, confusion matrix
export_onnx.py -> models/fp32.onnx          fixed 1x3x128x128 input, checked vs PyTorch
quantize.py -> models/int8.onnx             static INT8, QDQ, per-channel, 300 calib images
benchmark.py -> results/benchmark.{json,md} PyTorch FP32 vs ONNX FP32 vs ONNX INT8
infer_webcam.py                             live demo with INT8 (or --fp32)
run_all.py                                  runs make_splits ... benchmark in one go
```

All scripts share one preprocessing function (`src/preprocess.py`: 128x128 RGB,
ImageNet normalization), so training, quantization calibration, benchmarking and the
live demo all see images in exactly the same format.

## Folder layout

```
data/raw/<class>/   recorded images: <session>__<class>__<index>.jpg
data/splits/        train.csv, val.csv, test.csv
src/                all code
models/             fp32.pth, fp32.onnx, int8.onnx
results/            logs, metrics, plots, benchmark
docs/notes.md       plain-English explanation of every phase and key concepts
```

## Setup (Windows or Linux, Python 3.10+)

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Run every command from the project folder (the one containing `src/`).

## 1. Get data

There are two options, and you can combine them (both write to `data/raw/`).

### Option A: open dataset (HaGRID), no recording needed

[HaGRID](https://github.com/hukenovs/hagrid) is a public hand-gesture dataset with
photos of thousands of different people. `src/prepare_hagrid.py` maps it onto our classes:
`palm` → stop, `like` → resume, `fist` → pause, and other gestures / relaxed hands /
empty background → none. It crops a square around each hand and uses HaGRID's anonymous
`user_id` as the session id, so the train/val/test split is **by person**.

1. Download a HaGRID sample. The 30k-image, 384p sample on Hugging Face is a good size
   for a laptop (a few GB):
   ```bash
   pip install -U "huggingface_hub[cli]"
   hf download cj-mills/hagrid-sample-30k-384p --repo-type dataset --local-dir hagrid_download
   ```
   (Older `huggingface_hub` versions use `huggingface-cli download` with the same arguments.)
   You can also use any download from the official HaGRID GitHub page instead.
2. Extract every `.zip` inside `hagrid_download` (e.g. right-click → Extract All on
   Windows, `unzip` on Linux).
3. Convert:
   ```bash
   python src/prepare_hagrid.py --hagrid-dir hagrid_download --per-class 2000
   ```
   The script searches the folder recursively for HaGRID annotation `.json` files and
   images, and prints how many crops per class it saved. Re-running it replaces its own
   previous output (files starting with `hagrid-`) and leaves your own recordings alone.
   `hagrid_download/` is in `.gitignore`.

### Option B: record your own data

```bash
python src/capture.py
```

Type a session name (e.g. `rushik-desk-morning`). A preview window opens. Hold a
gesture and press the key to save the current frame (hold the key to save a burst):

| Key | Saves to |
|---|---|
| `1` | `data/raw/stop/` (open palm) |
| `2` | `data/raw/resume/` (thumbs up) |
| `3` | `data/raw/pause/` (fist) |
| `4` | `data/raw/none/` (no hand, or any other gesture) |
| `q` | quit |

The window shows per-class counts for this session and in total.

Tips:
- Record **at least 7 sessions** (more is better, e.g. 10). The split is done by
  session, so with 7 sessions you get roughly 5 train / 1 val / 1 test.
- Record **all four classes in every session**, with roughly balanced counts
  (e.g. 100–200 images per class per session).
- Change something between sessions: room, lighting, background, distance, hand,
  clothes, time of day. This is what makes the test set meaningful.
- Move the hand around (position, angle, distance) while saving.
- For `none`, include an empty scene, your face/body without a gesture, and other
  hand shapes (pointing, peace sign, waving).

## 2. Train, evaluate, export, quantize, benchmark

```bash
python src/run_all.py
```

Options: `--epochs 20 --freeze-epochs 5 --runs 500`. The first run downloads the
ImageNet MobileNetV2 weights (needs internet once). Each step can also be run on its
own, e.g. `python src/evaluate.py --model models/int8.onnx --split test`.

## 3. Live demo

```bash
python src/infer_webcam.py          # INT8 model (default)
python src/infer_webcam.py --fp32   # FP32 ONNX model, for comparison
```

Other options: `--camera 1` (another webcam), `--threads 4`, `--min-conf 0.6`
(below this confidence no command is issued). Press `q` to quit. The overlay shows
the gesture, robot command, confidence, model latency (inference only) and loop FPS
(the whole capture → preprocess → inference → display loop) separately.

## Results

All numbers below are generated by `src/update_readme.py` from the files in
`results/` (it runs at the end of `run_all.py`). Nothing is typed in by hand.

<!-- RESULTS:START -->
_No results yet. Run `python src/run_all.py` to generate them._
<!-- RESULTS:END -->

## Limitations

- **Not a certified safety system.** This is a student demo. It must not be the only
  way to stop a real robot; a physical emergency stop and proper safety engineering
  are still required.
- **Single 2D camera.** No depth; sensitive to viewpoint, occlusion, lighting and
  backgrounds that differ from the recorded ones. Only one hand/person is assumed.
- **Dataset domain gap.** HaGRID photos (people standing 0.5–4 m from a phone camera,
  hand crops) differ from a laptop webcam a hand's length away. Adding a few of your own
  `capture.py` sessions helps the live demo; the test accuracy is measured on the data's
  own test people, not on your webcam.
- **Small self-recorded dataset** (if you record your own). Few people, rooms and sessions, so the test
  accuracy says how well it works in conditions like the recordings, not in general.
- Classifies each frame independently (no temporal smoothing), so single-frame
  flicker is possible.
- Latency numbers depend on the CPU, thread count, power mode and background load;
  they are only comparable between runs on the same machine (see `results/benchmark.md`).

## Reproducibility

Random seeds are fixed (`SEED = 42` in `src/config.py`). The test set is used only for
final evaluation (after training) and in the benchmark; model selection uses the
validation set.
