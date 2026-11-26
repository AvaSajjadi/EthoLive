"""Live scoring: pose keypoints, one frame at a time -> behavior scores (new file).

The monitor wants one score row per frame (INTERFACES.md section 2). The classifier was trained on
features computed over whole sessions. This module computes the same features from a short moving
buffer of the most recent frames, so a live stream can be scored without the future.

Why a short buffer is enough: every feature uses only a trailing window (1 s by default) and a
filter that forgets quickly. Checked on one real CalMS21 validation session (7150 frames, model B):
a 6 s buffer scored in chunks of 5 frames gave the same scores as the whole-session run (largest
difference 5e-13) and took about 95 ms per chunk against a 167 ms budget. That check is repeated
by a test when the data is present locally. It is one session, not a guarantee for every setup.

Scores are produced in chunks, so a score row appears up to `chunk` frames (0.17 s at 30 fps,
chunk 5) after its frame. This is only a delay: each row still uses frames up to its own frame.

Limits:
  - Pose rows must use the CalMS21 layout: mouse "a" / "b", the 7 CalMS21 keypoint names, pixels in
    the CalMS21 image frame (1024 x 570) and a confidence "score". A camera has to be mapped to this
    frame, with the same kind of top-down view, or the scores are meaningless.
  - A frame that was never pushed (dropped) is simply absent from the output; the monitor treats
    skipped frame numbers as a gap that ends any bout.

A model with a lookahead (EthoLive fly file) scores a frame only once it has seen that many later frames: every score
comes that much later, and the last frames of a run get none.

Other animals (EthoLive fly file): each frame is filled to every animal and keypoint of the file, so a fly
that was not found counts as lost tracking. Their pixels are not the CalMS21 frame. setup="auto" measures the camera
first (flyvsfly.calibrate_live, 60 s with both flies in view): nothing is scored until then, after which the frames
still in the buffer are scored, so older frames get no score, and a run that never shows both flies for 60 s gives none.
"""

from __future__ import annotations

from collections import deque

import pandas as pd

from .classify import feature_columns, score_frames, session_features
from .flyvsfly import both_flies_seen, calibrate_live

POSE_COLUMNS = ("session", "mouse", "keypoint", "frame", "time_s", "x", "y", "score")
SCORE_ROW_KEYS = ("session", "frame", "time_s", "attack", "mount", "investigation", "tracking_ok")


class LiveScoringError(ValueError):
    """The pose rows or the model do not fit what the scorer needs."""


class LiveScorer:
    def __init__(self, model, expected_columns, *, session, features="basic,motion", fps=30.0,
                 buffer_s=6.0, chunk=5, setup=None, picture_size=None, calibrate_s=60.0):
        """setup: None (the training camera), a flyvsfly.calibrate result, or "auto" (needs picture_size: a function that
        returns the camera's (width, height), asked once the frames are there)."""
        names = [n.strip() for n in features.split(",") if n.strip()]
        if not names or not set(names) <= {"basic", "motion"} or "basic" not in names:
            raise LiveScoringError("features must be 'basic' or 'basic,motion'")
        if chunk < 1 or buffer_s * fps < chunk:
            raise LiveScoringError("the buffer must hold at least one chunk of frames")
        if setup == "auto" and picture_size is None:
            raise LiveScoringError('setup="auto" needs picture_size')
        self.model, self.expected_columns = model, list(expected_columns)
        self.session, self.fps, self.chunk = session, float(fps), int(chunk)
        self.use_motion = "motion" in names
        self.buffer_frames = int(round(buffer_s * fps))
        self.setup, self.picture_size = setup, picture_size
        self.lookahead = int(getattr(model, "lookahead", 0))
        self.calibrate_frames = int(round(calibrate_s * fps))
        self._measured = []               # setup="auto": the frames with both flies, until the camera is measured
        species = getattr(model, "species", None)
        self._grid = None
        if species is not None and species.name != "mouse":
            self._grid = pd.MultiIndex.from_product([species.animals, species.keypoints], names=["mouse", "keypoint"])
        self._buffer = deque()            # (frame, rows of that frame)
        self._pending = []                # frames not scored yet, in order
        self._last_frame = None

    def push(self, frame, pose_rows):
        """Add one frame of pose rows. Returns the score rows (dicts) that are ready, often none."""
        if self._last_frame is not None and frame <= self._last_frame:
            raise LiveScoringError(f"frame {frame} is not after frame {self._last_frame}")
        rows = pd.DataFrame(pose_rows)
        missing = [c for c in POSE_COLUMNS if c not in rows.columns]
        if missing:
            raise LiveScoringError(f"pose rows lack columns {missing}")
        if not (rows["session"] == self.session).all() or not (rows["frame"] == frame).all():
            raise LiveScoringError("pose rows must belong to this session and this frame")
        self._last_frame = frame
        self._buffer.append((frame, self._fill(rows)[list(POSE_COLUMNS)]))
        if self.setup == "auto" and both_flies_seen(self._buffer[-1][1]):
            self._measured.append(self._buffer[-1][1])
        while self._buffer[0][0] < frame - self.buffer_frames + 1:
            self._buffer.popleft()
        self._pending.append(frame)
        return self._score() if len(self._ready()) >= self.chunk else []

    def flush(self):
        """Score the frames still waiting (end of a run)."""
        rows = self._score() if self._pending else []
        self._pending = []                # frames still short of their lookahead: no score
        return rows

    def _ready(self):
        return [f for f in self._pending if f <= self._last_frame - self.lookahead]

    def _fill(self, rows):
        """Every animal and keypoint of the file, for one frame; what was not found is missing, score 0."""
        if self._grid is None:
            return rows
        out = rows.set_index(["mouse", "keypoint"]).reindex(self._grid)[["x", "y", "score"]].reset_index()
        out["score"] = out["score"].fillna(0.0)
        return out.assign(session=self.session, frame=rows["frame"].iloc[0], time_s=rows["time_s"].iloc[0])

    def _measure_camera(self):
        size = self.picture_size()
        if size is None:
            raise LiveScoringError("the source gave no picture size, so the camera cannot be measured")
        self.setup = calibrate_live(pd.concat(self._measured, ignore_index=True), size, min_frames=self.calibrate_frames)
        self._measured = []

    def _score(self):
        if self.setup == "auto":
            if len(self._measured) < self.calibrate_frames:
                self._pending = [f for f in self._pending if f >= self._buffer[0][0]]     # frames wait as far back as the buffer holds
                return []
            self._measure_camera()
        poses = pd.concat([rows for _, rows in self._buffer], ignore_index=True)
        if getattr(self.model, "methods", None):   # the scope model computes its own features (with its ScopeScore methods)
            feats = self.model.session_features(poses, setup=self.setup) if self.setup else self.model.session_features(poses)
        else:
            basic, motion = session_features(poses)
            # tracking_ok always comes from the motion table (model A has no motion features, but its rows still need it)
            extra = ([c for c in motion.columns if c not in basic.columns or c in ("session", "frame")]
                     if self.use_motion else ["session", "frame", "tracking_ok"])
            feats = basic.merge(motion[extra], on=["session", "frame"], how="inner")
        if feature_columns(feats) != self.expected_columns:
            raise LiveScoringError("the computed feature columns differ from the ones the model was trained on")
        ready = set(self._ready())
        waiting = feats[feats["frame"].isin(ready)]
        self._pending = [f for f in self._pending if f not in ready]
        if waiting.empty:
            return []
        if hasattr(self.model, "score_buffer"):   # the scope model's temporal head: it needs the frames before the waiting ones
            scored = self.model.score_buffer(feats, set(waiting["frame"]))
        else:
            scored = score_frames(self.model, waiting, [self.session])
        species = getattr(self.model, "species", None)       # EthoLive: the behaviors of its animal (flies: lunge, ...)
        behaviors = species.behaviors if species is not None else ("attack", "mount", "investigation")
        return [{"session": r["session"], "frame": int(r["frame"]), "time_s": float(r["time_s"]),
                 **{b: float(r[b]) for b in behaviors}, "tracking_ok": bool(r["tracking_ok"])}
                for r in scored.to_dict("records")]
