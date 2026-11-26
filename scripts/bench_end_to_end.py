"""End-to-end throughput of the mouse model on a recorded video: picture -> keypoints -> features -> behavior.

Each picture goes through Section 1 (the pose network, ``Eyes.find_mice``) and is pushed to the streaming scorer
(``live_scoring.LiveScorer``), which recomputes the causal features of its buffer and runs Section 2 every ``chunk``
frames. Video decoding is timed separately and is not counted in the model throughput.

    python scripts/bench_end_to_end.py [--video data/raw/calms21_video/mouse064_task1_annotator1.mp4]
                                       [--frames 600] [--device mps|cuda|cpu] [--chunk 15]

Writes one JSON line to stdout: frames, seconds, frames per second (model only and including decoding), per-stage
medians, the device and the software versions. The numbers depend on the computer; run it on yours.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etholive import model as etholive                                  # noqa: E402
from etholive import scope_model as sm                                  # noqa: E402
from etholive.live_scoring import LiveScorer                            # noqa: E402


def default_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", default="data/raw/calms21_video/mouse064_task1_annotator1.mp4")
    ap.add_argument("--model", default="models/etholive/mouse.pt")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--device", default=None)
    ap.add_argument("--chunk", type=int, default=15, help="LiveTracker uses 15 for a live camera, 5 for a recording")
    a = ap.parse_args(argv)

    import cv2
    import torch

    device = a.device or default_device()
    eyes = etholive.load_eyes(ROOT / a.model, device=device)
    behavior = sm.load_behavior(ROOT / a.model)
    cap = cv2.VideoCapture(str(ROOT / a.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    pictures, t_decode = [], time.perf_counter()
    while len(pictures) < a.frames + a.warmup:
        ok, img = cap.read()
        if not ok:
            break
        pictures.append(img)
    t_decode = time.perf_counter() - t_decode
    if len(pictures) <= a.warmup:
        raise SystemExit("video too short")

    for img in pictures[:a.warmup]:                     # first pictures: device start-up, not counted
        eyes.find_mice(img)
    eyes._memory.clear()

    scorer = LiveScorer(behavior, behavior.feature_columns, session="bench", features="basic,motion",
                        fps=30.0, chunk=a.chunk)
    t_eyes, t_score, n_rows, missed = [], [], 0, 0
    sync = (torch.mps.synchronize if device == "mps" else torch.cuda.synchronize if device == "cuda" else (lambda: None))
    start = time.perf_counter()
    for i, img in enumerate(pictures[a.warmup:]):
        t0 = time.perf_counter()
        rows = eyes.find_mice(img)
        sync()
        t1 = time.perf_counter()
        if rows is None:
            missed += 1
            continue
        rows = rows.assign(session="bench", frame=i, time_s=i / 30.0)
        n_rows += len(scorer.push(i, rows))
        t2 = time.perf_counter()
        t_eyes.append(t1 - t0)
        t_score.append(t2 - t1)
    n_rows += len(scorer.flush())
    total = time.perf_counter() - start
    n = len(pictures) - a.warmup
    per_frame_decode = t_decode / len(pictures)
    print(json.dumps({
        "video": a.video, "video_fps": fps, "frames": n, "frames_without_mice": missed, "score_rows": n_rows,
        "seconds": round(total, 3), "model_frames_per_second": round(n / total, 1),
        "frames_per_second_including_decode": round(n / (total + per_frame_decode * n), 1),
        "pose_ms_median": round(1000 * float(np.median(t_eyes)), 2),
        "scorer_ms_per_frame_median": round(1000 * float(np.median(t_score)), 2),
        "scorer_ms_per_chunk_max": round(1000 * float(np.max(t_score)), 1), "chunk": a.chunk,
        "device": device, "machine": platform.machine(), "processor": platform.processor(),
        "python": platform.python_version(), "torch": torch.__version__, "opencv": cv2.__version__,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
