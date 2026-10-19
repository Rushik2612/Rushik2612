"""Split data/raw into train/val/test (70/15/15) BY SESSION.

All frames from one recording session go to the same split. Random frame-level
splitting would put near-identical neighbouring frames in both train and test
and give a falsely high test accuracy.

Writes data/splits/{train,val,test}.csv with columns: path,label,session
(path is relative to the project root).
"""
import csv
import random
from collections import defaultdict

from config import CLASSES, DATA_RAW, ROOT, SEED, SEP, SPLITS_DIR, ensure_dirs

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def collect_images():
    """Return {session: [(relative_path, label_index), ...]}."""
    by_session = defaultdict(list)
    for label, c in enumerate(CLASSES):
        for p in sorted((DATA_RAW / c).glob("*.jpg")):
            session = p.name.split(SEP)[0]
            by_session[session].append((p.relative_to(ROOT).as_posix(), label))
    return by_session


def assign_sessions(sessions, seed=SEED):
    """Shuffle sessions with a fixed seed and give ~15% each to val and test
    (at least one session each), the rest to train."""
    if len(sessions) < 3:
        raise SystemExit(f"Need at least 3 recording sessions to split by session, "
                         f"found {len(sessions)}. Record more sessions with capture.py.")
    sessions = sorted(sessions)
    random.Random(seed).shuffle(sessions)
    n = len(sessions)
    n_val = max(1, round(n * FRACTIONS["val"]))
    n_test = max(1, round(n * FRACTIONS["test"]))
    return {
        "test": sessions[:n_test],
        "val": sessions[n_test:n_test + n_val],
        "train": sessions[n_test + n_val:],
    }


def main():
    ensure_dirs()
    by_session = collect_images()
    split_sessions = assign_sessions(list(by_session))

    total = sum(len(v) for v in by_session.values())
    print(f"Found {total} images in {len(by_session)} sessions.\n")
    header = f"{'split':<6} {'sessions':>8} {'images':>7} " + " ".join(f"{c:>7}" for c in CLASSES)
    print(header)
    print("-" * len(header))

    for split, sess_list in split_sessions.items():
        rows = [(path, label, s) for s in sess_list for path, label in by_session[s]]
        with open(SPLITS_DIR / f"{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "session"])
            w.writerows(rows)

        counts = [sum(1 for _, l, _ in rows if l == i) for i in range(len(CLASSES))]
        print(f"{split:<6} {len(sess_list):>8} {len(rows):>7} "
              + " ".join(f"{n:>7}" for n in counts)
              + f"   ({len(rows) / max(total, 1):.0%} of images)")
        for c, n in zip(CLASSES, counts):
            if n == 0:
                print(f"  WARNING: {split} has no '{c}' images.")

    print("\nSessions per split:")
    for split, sess_list in split_sessions.items():
        print(f"  {split}: {', '.join(sess_list)}")


if __name__ == "__main__":
    main()
