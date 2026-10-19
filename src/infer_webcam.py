"""Real-time gesture -> robot command demo on the webcam, CPU only.

Uses the same preprocess() as training and ONNX Runtime with the INT8 model
(default) or the FP32 ONNX model (--fp32). The overlay shows:
  gesture + confidence, robot command, model latency (ms, inference only)
  and loop FPS (whole loop: capture + preprocess + inference + drawing).
Press q to quit.
"""
import argparse
import platform
import time

import cv2

from config import CLASSES, COMMANDS, MODELS_DIR
from preprocess import preprocess
from runtime import load_predictor, softmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", action="store_true", help="use models/fp32.onnx instead of int8")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--threads", type=int, default=None, help="ONNX Runtime CPU threads")
    ap.add_argument("--min-conf", type=float, default=0.6,
                    help="below this confidence no command is sent")
    args = ap.parse_args()

    model_path = MODELS_DIR / ("fp32.onnx" if args.fp32 else "int8.onnx")
    predict = load_predictor(model_path, threads=args.threads)
    print(f"Loaded {model_path}")

    if platform.system() == "Windows":
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam {args.camera}")

    fps = 0.0
    last = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed, stopping.")
            break

        x = preprocess(frame)[None]
        t0 = time.perf_counter()
        logits = predict(x)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        probs = softmax(logits)[0]
        idx = int(probs.argmax())
        gesture, conf = CLASSES[idx], float(probs[idx])
        command = COMMANDS[gesture] if conf >= args.min_conf else "no command"

        # Loop FPS: smoothed (exponential moving average) over whole iterations.
        now = time.perf_counter()
        inst = 1.0 / max(now - last, 1e-6)
        last = now
        fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst

        view = cv2.flip(frame, 1)  # mirrored display only
        color = (0, 0, 255) if command == "STOP" else (0, 255, 0)
        lines = [
            (f"gesture: {gesture} ({conf:.2f})", (255, 255, 255)),
            (f"command: {command}", color),
            (f"model latency: {latency_ms:.1f} ms", (255, 255, 0)),
            (f"loop FPS: {fps:.1f}", (255, 255, 0)),
            (f"model: {model_path.name}", (200, 200, 200)),
        ]
        for i, (text, c) in enumerate(lines):
            cv2.putText(view, text, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        cv2.imshow("EdgeHRC", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
