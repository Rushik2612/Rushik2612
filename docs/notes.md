# EdgeHRC — notes (plain English)

These notes explain what each part of the project does and why, so you can explain
it in an interview. No result numbers are written here; those live in `results/`.

---

## The one-paragraph pitch

EdgeHRC recognises four hand gestures from a normal laptop webcam (open palm = STOP,
thumbs up = RESUME, fist = PAUSE, anything else = no command) so a person could control
a robot without touching it. Everything runs on the CPU, with no cloud. The main
question is engineering, not accuracy for its own sake: *if I compress the network from
32-bit floats to 8-bit integers (INT8 quantization), how much faster and smaller does it
get, and how much accuracy do I lose?* The benchmark answers that with measured numbers
on my own machine.

---

## Phase 1 — Data

### One shared preprocessing function (`src/preprocess.py`)
Every image, whether used for training, calibration, benchmarking or the live demo,
goes through the same function:
1. BGR → RGB (OpenCV reads BGR; the pretrained network expects RGB),
2. resize to 128×128,
3. scale pixels to 0–1,
4. subtract the ImageNet mean and divide by the ImageNet standard deviation (per
   channel), because the pretrained MobileNetV2 was trained on inputs normalised this way,
5. reorder to channels-first (3×128×128), which PyTorch and ONNX expect.

Why it matters: a classic bug in deployed ML is "training-serving skew" — the model was
trained on images prepared one way and is fed images prepared slightly differently in
production. Having one function removes that whole class of bug.

Why 128×128 and not the usual 224×224? Fewer pixels = roughly 3× less computation, and
a hand gesture filling a good part of the frame is still clearly recognisable.

### Recording (`src/capture.py`)
You type a session name, then press keys 1–4 to save frames to the class folders.
The file name contains the session id (`<session>__<class>__<index>.jpg`), which is what
makes splitting by session possible later.

### Splitting by session (`src/make_splits.py`)
Webcam frames taken a fraction of a second apart are almost identical. If frames were
split randomly, the test set would contain near-copies of training images and the test
accuracy would be falsely high (this is called *data leakage*). Instead, whole sessions
are assigned to train (~70%), validation (~15%) or test (~15%). A test session was
recorded at a different time/place, so the test accuracy estimates how well the model
works in a *new* situation. The split uses a fixed random seed so it is repeatable.

- **Train** set: the model learns from it.
- **Validation** set: used during training to pick the best epoch (model selection).
- **Test** set: touched only at the very end, to report final numbers. If you tune
  anything based on the test set, it stops being an honest estimate.

---

## Phase 2 — Training

### Why MobileNetV2?
It was designed for phones: small (a few million parameters) and cheap to run, while
still accurate. Its main trick is the depthwise separable convolution (below).

### Depthwise separable convolutions
A normal 3×3 convolution with 64 input and 64 output channels learns 64 filters, each
looking at all 64 input channels at once: 3·3·64·64 ≈ 37k weights, and the same number
of multiply-adds per pixel.

A depthwise separable convolution splits this into two cheaper steps:
1. **Depthwise**: one 3×3 filter *per channel*, each looking only at its own channel
   (spatial filtering): 3·3·64 = 576 weights.
2. **Pointwise**: a 1×1 convolution that mixes channels: 64·64 = 4,096 weights.

Total ≈ 4.7k instead of ≈ 37k, about 8× less work, with only a small accuracy cost.
MobileNetV2 also uses *inverted residual blocks*: expand channels with a 1×1 conv, do the
cheap depthwise conv in that wider space, then project back down to a narrow
"bottleneck" (with no ReLU there, a "linear bottleneck", so information is not destroyed),
with a skip connection around the block.

### Transfer learning and the two training phases (`src/train.py`)
The network starts from weights trained on ImageNet (1.2M photos, 1000 classes), so it
already knows edges, textures, skin, shapes. Only the last layer is replaced with a new
4-output layer.
- **Phase 1 (frozen backbone, first few epochs)**: only the new last layer trains. It
  starts random, so its large early gradients would otherwise flow back and damage the
  good pretrained features.
- **Phase 2 (unfrozen, lower learning rate)**: the whole network is fine-tuned gently
  (learning rate 10× smaller) to adapt the features to hands.

Loss: **cross-entropy** (standard for classification: it punishes giving low probability
to the correct class). Optimiser: **Adam** (gradient descent with per-weight adaptive
step sizes; works well without much tuning).

### Augmentation (`src/dataset.py`)
Only on training images, randomly each time an image is used: random resized crop
(hand at different positions/sizes), horizontal flip (left vs right hand), colour jitter
(different lighting/cameras), small rotation (tilted hand). This makes a small dataset
behave like a bigger, more varied one and reduces overfitting. Validation/test images
are never augmented, because we want to measure the model on real, unaltered images.

### Picking the best model
After each epoch the model is scored on the validation set; the epoch with the best
validation accuracy is saved to `models/fp32.pth`. Curves of loss/accuracy are saved to
`results/train_curves.png`. If train accuracy keeps rising while validation falls, the
model is overfitting.

### Evaluation (`src/evaluate.py`)
- **Accuracy**: fraction of images classified correctly.
- **Precision** for a class: of everything predicted as "stop", how much really was
  "stop"? (Low precision = false alarms.)
- **Recall** for a class: of all real "stop" images, how many did we catch?
  (Low recall = misses.) For a robot, recall of STOP is the most important number:
  a missed stop is worse than an unnecessary one.
- **Confusion matrix**: rows = true class, columns = predicted class; off-diagonal
  cells show exactly which gestures get mixed up (e.g. fist vs thumbs up).

Evaluation runs image-by-image (batch 1), the same way the live demo runs.

---

## Phase 3 — Optimisation

### Why ONNX? (`src/export_onnx.py`)
ONNX is an open file format for neural networks: a graph of standard operations plus
the weights. PyTorch is a training framework with lots of Python overhead; ONNX Runtime
is a lean inference engine (C++) that optimises the graph (e.g. fuses Conv + BatchNorm +
ReLU into one operation) and has fast CPU kernels, including INT8 ones. ONNX also runs
on many platforms (Windows, Linux, ARM boards, phones) without needing PyTorch installed.

The model is exported with a fixed input shape 1×3×128×128 (batch of 1, as in live use).
The export is verified: PyTorch and ONNX Runtime outputs are compared on up to 100
validation images (maximum logit difference and whether the top prediction agrees).
Result in `results/export_check.json`.

### Quantization (`src/quantize.py`)
Normally every weight and activation is a 32-bit float. **Quantization** stores them
as 8-bit integers instead, using a scale and zero-point:

    real_value ≈ scale × (int8_value − zero_point)

Benefits: 4× smaller weights (file size and memory), and integer arithmetic is faster
on CPUs (more values fit in one SIMD instruction; modern CPUs have special int8
dot-product instructions such as AVX-VNNI / AVX512-VNNI on x86 or DOT on ARM).
Cost: rounding error, which can reduce accuracy a bit.

Details of what this project does:
- **Post-training, static quantization**: no retraining. "Static" means the scales for
  activations are fixed ahead of time (from calibration), rather than computed on the
  fly for every input ("dynamic"), which is faster at runtime and is the right choice for CNNs.
- **Pre-processing (`quant_pre_process`)**: runs shape inference and graph
  optimisations (e.g. folding BatchNorm into the preceding Conv) first, so the
  quantizer sees a clean graph and fewer layers need quantizing.
- **Calibration**: about 300 training images are passed through the FP32 model, and the
  range (min/max) of values in every layer is recorded. That range decides each
  layer's scale. Calibration images must look like real inputs and must be
  preprocessed the same way; they come from the training split (never test).
- **Per-channel weights**: each output channel of a conv layer gets its own scale
  instead of one scale for the whole layer. Channels have very different weight
  ranges (especially in depthwise convs), so one shared scale would waste precision.
  Per-channel keeps accuracy much closer to FP32.
- **QDQ format**: the model keeps the original operators but inserts
  QuantizeLinear/DequantizeLinear pairs around them. ONNX Runtime recognises these
  patterns and swaps in fast integer kernels. It's the recommended, most portable format.
- **Types**: activations as unsigned 8-bit (they're mostly ≥ 0 after ReLU6), weights as
  signed 8-bit (they're centred on 0). This "U8S8" combination is the standard fast
  path on x86 CPUs.

### Benchmark (`src/benchmark.py`)
Three versions are compared: PyTorch FP32, ONNX Runtime FP32 and ONNX Runtime INT8.
Method:
- batch size 1 (the live use case),
- 50 warm-up runs first (caches, memory allocation and thread pools settle; first runs
  are always slow),
- then 500 timed runs on the same real test image,
- report the **median** (typical latency, robust to outliers) and **p95** (95% of runs
  are faster than this; shows the "bad case" jitter, which matters for real-time control),
- file size in MB, test accuracy, and speedup = baseline median / model median.
CPU name, OS, thread count and library versions are recorded, because latency numbers
are only meaningful for a specific machine and setting.

Two comparisons to talk about:
- PyTorch FP32 → ONNX FP32: gain from the runtime alone (no precision change).
- ONNX FP32 → ONNX INT8: gain from quantization alone (same runtime). This is the
  fair answer to the project's main question.

---

## Phase 4 — Deployment (`src/infer_webcam.py`)

Loop: grab a frame with OpenCV → shared `preprocess()` → ONNX Runtime INT8 model →
softmax to get probabilities → pick the top class → map to a robot command. If the top
probability is below a threshold (`--min-conf`, default 0.6) no command is issued: an
unsure model shouldn't move a robot.

Two speeds are shown separately:
- **Model latency (ms)**: time for the model call only.
- **Loop FPS**: how many complete loops per second (camera read + preprocessing +
  inference + drawing). Usually limited by the camera (often 30 FPS), not the model.
  Showing both makes it clear what the model costs versus the rest of the system.

`--fp32` switches to the FP32 ONNX model to compare live.

---

## Phase 5 — `src/run_all.py`

Runs: make splits → train → evaluate FP32 on test → export ONNX → quantize → evaluate
INT8 on test → benchmark → copy results into the README. Each step is a separate script
so it can also be rerun alone.

---

## Good interview talking points

- **Why split by session?** To avoid leakage from near-duplicate frames; the test set
  measures generalisation to a new recording.
- **Why not train on the test set or tune on it?** Then it's no longer an unbiased estimate.
- **Why INT8 and not FP16?** Most laptop CPUs have fast int8 instructions but weak or no
  FP16 compute; int8 also gives 4× (not 2×) size reduction.
- **Why might INT8 be *not* much faster on some CPUs?** Older CPUs without VNNI
  instructions gain less; the speedup also depends on thread count, and small models
  can be dominated by overhead rather than arithmetic.
- **What limits accuracy?** Small self-recorded dataset, single 2D camera, lighting and
  background changes, and frame-by-frame decisions without temporal smoothing.
- **What would I do next?** More people/rooms in the data, temporal smoothing (e.g.
  require N consistent frames before issuing a command), quantization-aware training if
  INT8 loses too much accuracy, a hand detector to crop the hand first, testing on a
  Raspberry Pi/ARM board.
- **Safety**: this is a demo, not a certified safety system; a physical emergency stop
  is always required.
