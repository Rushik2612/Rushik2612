"""Record gesture images from the webcam.

Usage:  python src/capture.py   (it asks for a session name)

Keys:  1 = stop (open palm)   2 = resume (thumbs up)
       3 = pause (fist)       4 = none (no hand / other gesture)
       q = quit
Each key press saves the current frame to data/raw/<class>/ as
"<session>__<class>__<index>.jpg". Holding a key down saves a burst.
"""
import platform
import re

import cv2

from config import CLASSES, DATA_RAW, SEP, ensure_dirs

KEY_TO_CLASS = {ord("1"): "stop", ord("2"): "resume", ord("3"): "pause", ord("4"): "none"}


def clean_session_name(name: str) -> str:
    """Keep letters, digits and single dashes so the name is filename-safe
    and can never contain the "__" separator."""
    name = re.sub(r"[^A-Za-z0-9]+", "-", name.strip()).strip("-")
    return name


def count_images(class_name: str, session: str = None) -> int:
    folder = DATA_RAW / class_name
    pattern = f"{session}{SEP}*.jpg" if session else "*.jpg"
    return len(list(folder.glob(pattern)))


def open_camera(index: int = 0):
    # DirectShow opens much faster on Windows; default backend elsewhere.
    if platform.system() == "Windows":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def main():
    ensure_dirs()
    for c in CLASSES:
        (DATA_RAW / c).mkdir(parents=True, exist_ok=True)

    session = ""
    while not session:
        session = clean_session_name(input("Session name (e.g. rushik-desk-day1): "))
    print(f"Session id: {session}")

    cap = open_camera(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (index 0).")

    # Start numbering after any frames already saved for this session.
    session_counts = {c: count_images(c, session) for c in CLASSES}
    total_counts = {c: count_images(c) for c in CLASSES}

    print("Keys: 1=stop 2=resume 3=pause 4=none q=quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed, stopping.")
            break

        # Show a mirrored preview (feels natural), but save the raw frame.
        view = cv2.flip(frame, 1)
        y = 25
        cv2.putText(view, f"session: {session}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        for i, c in enumerate(CLASSES):
            y += 25
            text = f"{i + 1} {c:<7} session={session_counts[c]:<4} total={total_counts[c]}"
            cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(view, "q = quit", (10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
        cv2.imshow("EdgeHRC capture", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in KEY_TO_CLASS:
            c = KEY_TO_CLASS[key]
            idx = session_counts[c]
            path = DATA_RAW / c / f"{session}{SEP}{c}{SEP}{idx:05d}.jpg"
            # imencode + tofile handles non-ASCII paths on Windows.
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                buf.tofile(str(path))
                session_counts[c] += 1
                total_counts[c] += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Saved this session:", session_counts)
    print("Total in data/raw:", total_counts)


if __name__ == "__main__":
    main()
