Build my complete project "EdgeHRC" end to end in this folder. First save this whole
message as CLAUDE.md so the context persists, then build everything.

PROJECT
EdgeHRC is a real-time hand gesture recognition pipeline for touchless robot control,
running fully on CPU (no cloud). Goal: show that INT8 quantization of a lightweight CNN
gives big speed and memory gains with minimal accuracy loss. Student project; don't add
features beyond this scope.

CLASSES
0 stop (open palm), 1 resume (thumbs up), 2 pause (fist), 3 none (no hand / other gesture)

STACK
Python 3.10+, torch, torchvision, onnx, onnxruntime, opencv-python, numpy, scikit-learn,
matplotlib. Laptop CPU. Must work on Windows and Linux.

FOLDER LAYOUT
data/raw/<class>/, data/splits/, src/, models/, results/, docs/notes.md, README.md,
requirements.txt

WHAT TO BUILD
1. Data: src/preprocess.py (one shared function: 128x128 RGB, ImageNet normalization,
   used everywhere). src/capture.py: asks for a session name, opens webcam with live
   preview, saves frames to data/raw/<class>/ on keys 1=stop 2=resume 3=pause 4=none,
   q=quit, shows per-class counts; session id in filenames. src/make_splits.py: split
   70/15/15 BY SESSION (never random frames), print class counts.
2. Training: src/dataset.py (train augmentations: random resized crop, flip, color
   jitter, small rotation; none for val/test). src/train.py: torchvision MobileNetV2
   ImageNet-pretrained, classifier -> 4 outputs, freeze backbone first few epochs then
   unfreeze with lower LR, Adam + cross-entropy, save best-by-val to models/fp32.pth,
   log to results/train_log.json, plot curves. src/evaluate.py: accuracy, per-class
   precision/recall, confusion matrix, works for .pth and .onnx.
3. Optimization: src/export_onnx.py: export to models/fp32.onnx with fixed 1x3x128x128
   input, verify outputs match PyTorch on 100 val images. src/quantize.py: ONNX Runtime
   quant_pre_process then quantize_static, calibration reader over ~300 train images,
   QDQ format, per-channel weights -> models/int8.onnx. src/benchmark.py: compare
   PyTorch FP32, ONNX FP32, ONNX INT8; batch 1, 50 warmup, 500+ timed runs, median and
   p95 latency, file size MB, test accuracy, speedup; record CPU, OS, thread count;
   write results/benchmark.json and results/benchmark.md.
4. Deployment: src/infer_webcam.py: OpenCV capture + shared preprocessing + ONNX
   Runtime INT8; overlay gesture, robot command (STOP/RESUME/PAUSE/no command),
   confidence, model latency ms, and loop FPS separately; flag to switch to FP32 ONNX.
5. src/run_all.py: runs splits -> train -> evaluate -> export -> quantize -> evaluate
   -> benchmark in one go.
6. README.md with setup and run instructions, results section filled from results/
   files only, and a Limitations section (not a certified safety system, single 2D
   camera, small self-recorded dataset). docs/notes.md: plain-English explanation of
   every phase and key concepts (quantization, calibration, depthwise separable
   convolutions, why ONNX) so I can explain it in interviews.

RULES
- Never invent or hardcode any metric. All numbers must come from script outputs.
- Test set only for final evaluation. Fix random seeds.
- Keep code simple, readable, commented.
- Set up git and commit when done.
- To test that everything runs before I have real data, generate a tiny dummy dataset
  of random images in a temporary folder, run the full pipeline on it end to end, fix
  any errors, then delete the dummy data and dummy outputs.

When finished, tell me exactly how to (a) record my data with capture.py and
(b) run run_all.py and then infer_webcam.py.
