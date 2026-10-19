"""Convert the open HaGRID hand-gesture dataset into EdgeHRC's data/raw layout.

HaGRID (https://github.com/hukenovs/hagrid) has full-scene photos of people doing
gestures, with a bounding box, label and anonymous user_id for every hand.
Mapping to our classes:
    palm -> stop      like -> resume      fist -> pause
    none <- other gestures / relaxed hands ("no_gesture") / background with no hand
Each sample is a square crop around the hand (with context), so the hand fills a
similar part of the image as when you hold it up in front of a webcam.
The HaGRID user_id becomes the session id, so make_splits.py splits BY PERSON.

Usage:
    python src/prepare_hagrid.py --hagrid-dir path/to/extracted/hagrid
The folder can be any HaGRID download (full, subsample, or a Hugging Face
sample such as cj-mills/hagrid-sample-30k-384p): the script searches it
recursively for the annotation .json files and the images.
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from config import CLASSES, DATA_RAW, SEED, SEP, ensure_dirs
from preprocess import load_image

GESTURE_TO_CLASS = {"palm": "stop", "like": "resume", "fist": "pause"}
# HaGRID labels that count as "none" (hand present, but not one of our commands).
# Labels not listed here are ignored (e.g. HaGRID v2 'stop', 'grip' look too
# similar to our command gestures and would confuse the model).
NONE_LABELS = {"no_gesture", "call", "dislike", "four", "mute", "ok", "one", "peace",
               "peace_inverted", "rock", "three", "three2", "two_up", "two_up_inverted"}
PREFIX = "hagrid-"          # session ids from this script start with this
MAX_SAVE_SIDE = 256         # crops are stored at most 256x256 (model uses 128)


def load_annotations(root: Path):
    """Find every HaGRID annotation json under root.
    Returns {image_id: {"bboxes": [...], "labels": [...], "user_id": str}}."""
    anns = {}
    for jf in sorted(root.rglob("*.json")):
        try:
            data = json.loads(jf.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for image_id, a in data.items():
            if isinstance(a, dict) and "bboxes" in a and "labels" in a:
                anns[image_id] = a
    return anns


def index_images(root: Path):
    """Map image file name without extension -> path."""
    return {p.stem: p for ext in ("*.jpg", "*.jpeg", "*.png") for p in root.rglob(ext)}


def square_crop(img, box, scale):
    """Square crop centered on box=(x, y, w, h) in pixels, side = scale * max(w, h),
    shifted to stay inside the image."""
    H, W = img.shape[:2]
    x, y, w, h = box
    side = int(min(max(w, h) * scale, W, H))
    cx, cy = x + w / 2, y + h / 2
    x0 = int(np.clip(cx - side / 2, 0, W - side))
    y0 = int(np.clip(cy - side / 2, 0, H - side))
    return img[y0:y0 + side, x0:x0 + side]


def background_crop(img, boxes, side, rng):
    """Random square crop that does not overlap any hand box (None if not found)."""
    H, W = img.shape[:2]
    side = int(min(side, W, H))
    for _ in range(20):
        x0, y0 = rng.randint(0, W - side), rng.randint(0, H - side)
        if all(x0 + side <= x or x >= x0 + side or y0 + side <= y or y >= y0 + side
               for x, y, w, h in boxes):
            return img[y0:y0 + side, x0:x0 + side]
    return None


def save(crop, cls, session, idx):
    if crop.shape[0] > MAX_SAVE_SIDE:
        crop = cv2.resize(crop, (MAX_SAVE_SIDE, MAX_SAVE_SIDE), interpolation=cv2.INTER_AREA)
    path = DATA_RAW / cls / f"{session}{SEP}{cls}{SEP}{idx:05d}.jpg"
    ok, buf = cv2.imencode(".jpg", crop)
    if ok:
        buf.tofile(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hagrid-dir", required=True, help="extracted HaGRID folder")
    ap.add_argument("--per-class", type=int, default=2000,
                    help="max images per class (keeps CPU training time reasonable)")
    ap.add_argument("--scale", type=float, default=2.5,
                    help="crop side = scale x hand box size (more = more context)")
    args = ap.parse_args()

    root = Path(args.hagrid_dir)
    ensure_dirs()
    rng = random.Random(SEED)

    anns = load_annotations(root)
    images = index_images(root)
    print(f"Found {len(anns)} annotated images, {len(images)} image files.")
    if not anns:
        raise SystemExit("No HaGRID annotation .json files found under --hagrid-dir.")

    # Candidate samples per class: (image_id, box_index or None for background).
    candidates = defaultdict(list)
    background = []
    for image_id, a in anns.items():
        if image_id not in images:
            continue
        for i, label in enumerate(a["labels"]):
            if label in GESTURE_TO_CLASS:
                candidates[GESTURE_TO_CLASS[label]].append((image_id, i))
            elif label in NONE_LABELS:
                candidates["none"].append((image_id, i))
        background.append((image_id, None))

    # "none" budget: about 2/3 other hands, 1/3 background (no hand at all).
    n_bg = args.per_class // 3
    plan = {}
    for c in CLASSES:
        rng.shuffle(candidates[c])
        plan[c] = candidates[c][:args.per_class - (n_bg if c == "none" else 0)]
    rng.shuffle(background)
    plan["none"] += background[:n_bg]

    # Remove output of a previous run of this script (own recordings are kept).
    for c in CLASSES:
        (DATA_RAW / c).mkdir(parents=True, exist_ok=True)
        for p in (DATA_RAW / c).glob(f"{PREFIX}*.jpg"):
            p.unlink()

    counts, users = defaultdict(int), set()
    for c in CLASSES:
        for image_id, box_i in plan[c]:
            a = anns[image_id]
            img = load_image(images[image_id])
            H, W = img.shape[:2]
            # HaGRID boxes: [top-left x, top-left y, width, height], relative to image size.
            boxes = [(bx * W, by * H, bw * W, bh * H) for bx, by, bw, bh in a["bboxes"]]
            if box_i is None:
                hand_side = max(max(w, h) for _, _, w, h in boxes) if boxes else min(W, H) / 4
                crop = background_crop(img, boxes, hand_side * args.scale, rng)
                if crop is None:
                    continue
            else:
                crop = square_crop(img, boxes[box_i], args.scale)
            if crop.size == 0:
                continue
            user = re.sub(r"[^A-Za-z0-9]", "", str(a.get("user_id", "unknown")))[:16]
            session = PREFIX + (user or "unknown")
            save(crop, c, session, counts[c])
            counts[c] += 1
            users.add(session)

    print(f"Saved crops from {len(users)} different people (sessions):")
    for c in CLASSES:
        print(f"  {c:<7} {counts[c]}")
    print("Next: python src/run_all.py")


if __name__ == "__main__":
    main()
