"""MouseScope: one model from camera frame to behavior scores (new file).

MouseScope joins the three parts that ran side by side in the hackathon build:
  1. vision:   the fine-tuned YOLO26n-pose model (pose_model.py), 7 keypoints per mouse;
  2. methods:  the ScopeScore methods, adapted to mice: tracking-quality flags (quality.py),
               causal One-Euro smoothing (filters.py) and motion features (features.py);
  3. behavior: a causal temporal network (TemporalHead, below) that replaces model B's
               per-frame logistic regression and looks at the last ~2 s of features.

Step 2 reuses the existing code on purpose, so its numbers are exactly the ones model B was
trained and validated on. Only step 3 is learned here.

Training follows the same rules as classify.py: train on manifest["train"] sessions, choose
the epoch on manifest["validation"] (frame macro F1 of attack, investigation, mount), never
read the test split. Scores keep the INTERFACES.md section 2 layout, so the monitor, the rules
and the dashboard do not change.

Causal: the network only pads on the left, so the score of frame t uses frames <= t. It sees
RECEPTIVE_FIELD frames (2.1 s at 30 fps), well inside live_scoring's 6 s buffer.

Usage (from the repo root):
    python -m etholive.mousescope train      # -> models/mousescope_head.pt + .json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .calms21 import FPS, check_manifest
from .classify import KEYS, feature_columns, load_feature_set
from .evaluate import BEHAVIORS, CLASSES, summary_metrics

MODEL_NAME = "MouseScope"          # the model's name: change it here only
MODEL_VERSION = "0.1"
DILATIONS = (1, 2, 4, 8, 16)
KERNEL = 3
RECEPTIVE_FIELD = 1 + (KERNEL - 1) * sum(DILATIONS)     # 63 frames = 2.1 s at 30 fps


class _CausalBlock(nn.Module):
    """Dilated causal convolution with a residual path. Normalization is per frame only
    (LayerNorm over channels), so no statistic mixes past and future frames."""

    def __init__(self, ch, dilation, dropout):
        super().__init__()
        self.pad = (KERNEL - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, KERNEL, dilation=dilation)
        self.mix = nn.Conv1d(ch, ch, 1)
        self.norm = nn.LayerNorm(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                   # x: batch, channels, time
        h = self.conv(nn.functional.pad(x, (self.pad, 0)))
        h = self.mix(self.drop(nn.functional.gelu(h)))
        return self.norm((x + h).transpose(1, 2)).transpose(1, 2)


class TemporalHead(nn.Module):
    """Feature sequence (batch, time, features) -> class logits (batch, time, 4)."""

    def __init__(self, n_features, width=64, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(n_features, width)
        self.blocks = nn.ModuleList(_CausalBlock(width, d, dropout) for d in DILATIONS)
        self.out = nn.Linear(width, len(CLASSES))

    def forward(self, x):
        h = self.inp(x).transpose(1, 2)
        for b in self.blocks:
            h = b(h)
        return self.out(h.transpose(1, 2))


# --- data ---------------------------------------------------------------------------------

def fit_normalizer(features, sessions, columns):
    """Median (for missing values) and mean / std after filling, from these sessions only."""
    x = features.loc[features["session"].isin(sessions), columns].astype("float32").to_numpy()
    median = np.nanmedian(x, axis=0)
    median = np.where(np.isnan(median), 0.0, median).astype("float32")
    x = np.where(np.isnan(x), median, x)
    mean, std = x.mean(axis=0), x.std(axis=0)
    return {"median": median, "mean": mean.astype("float32"),
            "std": np.where(std < 1e-6, 1.0, std).astype("float32")}


def normalize(x, norm):
    x = np.where(np.isnan(x), norm["median"], x)
    return ((x - norm["mean"]) / norm["std"]).astype("float32")


def session_arrays(features, labels, sessions, columns, norm):
    """{session: (x (T, F), y (T,) class index or None, frame table)} sorted by frame."""
    d = features[features["session"].isin(sessions)].sort_values(KEYS)
    if labels is not None:
        d = d.merge(labels[KEYS + ["label"]], on=KEYS, how="left", validate="one_to_one")
        if d["label"].isna().any():
            raise ValueError("some feature rows have no label")
    out = {}
    for sid, g in d.groupby("session", sort=True):
        x = normalize(g[columns].astype("float32").to_numpy(), norm)
        y = (pd.Categorical(g["label"], categories=CLASSES).codes.astype("int64")
             if labels is not None else None)
        out[sid] = (x, y, g[[c for c in ("session", "frame", "time_s", "tracking_ok") if c in g]])
    return out


# --- scoring ------------------------------------------------------------------------------

def predict_proba(head, x, device="cpu"):
    """Class probabilities (T, 4) for one whole session; causal, so equal to streaming."""
    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(x)[None].to(device))[0]
        return torch.softmax(logits, dim=-1).cpu().numpy()


def score_sessions(head, arrays, device="cpu"):
    """INTERFACES.md section 2 table (session, frame, time_s, tracking_ok, behaviors)."""
    parts = []
    for x, _, table in arrays.values():
        p = predict_proba(head, x, device)
        s = table.reset_index(drop=True).copy()
        for b in BEHAVIORS:
            s[b] = p[:, CLASSES.index(b)].astype(float)
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


# --- training -----------------------------------------------------------------------------

def _device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _batches(arrays, rng, window, batch):
    """Random training windows; longer sessions are picked more often."""
    items = [v for v in arrays.values() if len(v[0]) >= 2]
    sizes = np.array([len(v[0]) for v in items], float)
    pick = rng.choice(len(items), size=batch, p=sizes / sizes.sum())
    xs, ys = [], []
    for i in pick:
        x, y, _ = items[i]
        w = min(window, len(x))
        s = int(rng.integers(0, len(x) - w + 1))
        xs.append(np.pad(x[s:s + w], ((0, window - w), (0, 0))))
        ys.append(np.pad(y[s:s + w], (0, window - w), constant_values=-100))   # -100: ignored
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def train(features, labels, manifest, *, seed=0, epochs=40, steps=300, batch=32, window=256,
          lr=1e-3, width=64, dropout=0.1, patience=6, log=print):
    """Train on train sessions, keep the epoch with the best validation macro F1."""
    manifest = check_manifest(manifest)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = _device()
    columns = feature_columns(features)
    norm = fit_normalizer(features, manifest["train"], columns)
    tr = session_arrays(features, labels, manifest["train"], columns, norm)
    va = session_arrays(features, labels, manifest["validation"], columns, norm)
    counts = np.bincount(np.concatenate([v[1] for v in tr.values()]), minlength=len(CLASSES))
    weights = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))      # "balanced", as model B
    head = TemporalHead(len(columns), width, dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device),
                                  ignore_index=-100)
    best, history, waited = None, [], 0
    for epoch in range(1, epochs + 1):
        head.train()
        t0, total = time.time(), 0.0
        for _ in range(steps):
            x, y = _batches(tr, rng, window, batch)
            logits = head(x.to(device))
            loss = loss_fn(logits.reshape(-1, len(CLASSES)), y.to(device).reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        val = summary_metrics(score_sessions(head, va, device), labels)
        history.append({"epoch": epoch, "train_loss": total / steps, "val_macro_f1": val["macro_f1"]})
        log(f"  epoch {epoch:2d}: loss {total / steps:.3f}, validation macro F1 {val['macro_f1']:.3f} "
            f"({time.time() - t0:.0f} s)")
        if best is None or val["macro_f1"] > best[1]:
            best = ({k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                    val["macro_f1"], epoch)
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                break
    head.load_state_dict(best[0])
    head = head.to("cpu")
    info = {"name": MODEL_NAME, "version": MODEL_VERSION, "seed": seed, "best_epoch": best[2],
            "val_macro_f1": best[1], "history": history, "feature_columns": columns,
            "n_features": len(columns), "receptive_field_frames": RECEPTIVE_FIELD, "fps": FPS,
            "params": {"epochs": epochs, "steps": steps, "batch": batch, "window": window, "lr": lr,
                       "width": width, "dropout": dropout, "patience": patience},
            "class_weights": dict(zip(CLASSES, map(float, weights))),
            "train_sessions": len(manifest["train"]), "validation_sessions": len(manifest["validation"])}
    return head, norm, info


def save_head(path, head, norm, info):
    torch.save({"state_dict": head.state_dict(), "norm": {k: torch.from_numpy(v) for k, v in norm.items()},
                "info": info}, path)


def load_head(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    info = blob["info"]
    head = TemporalHead(info["n_features"], info["params"]["width"], info["params"]["dropout"])
    head.load_state_dict(blob["state_dict"])
    head.eval()
    return head, {k: v.numpy() for k, v in blob["norm"].items()}, info


# --- the one MouseScope file --------------------------------------------------------------
#
# models/mousescope.pt holds everything: the YOLO pose weights (as bytes) with their settings,
# the behavior head with its normalizer and feature names, and the credits. The pose part is
# read through pose_model.py (backend "mousescope"), the behavior part through
# monitor_runner.build_scorer, so the rest of the program does not change.

CREDITS = ("Pose network: Ultralytics YOLO26n-pose (AGPL-3.0), fine-tuned on the MARS hand-labeled set (Caltech).",
           "Behavior labels: CalMS21 Task 1 (Sun et al., 2021).",
           "Tracking-quality, smoothing and motion methods: ScopeScore (Sajjadi, 2026), adapted to mice.")


def pack(out_path, head_path, pose_weights, pose_settings):
    """Write the single MouseScope file from a trained head and the pose weights."""
    head = torch.load(head_path, map_location="cpu", weights_only=False)
    weights = Path(pose_weights)
    keep = {k: pose_settings[k] for k in ("imgsz", "conf", "hold") if k in pose_settings}
    blob = {"name": MODEL_NAME, "version": MODEL_VERSION, "credits": CREDITS,
            "pose": {"weights_name": weights.name, "weights": weights.read_bytes(), "settings": keep},
            "behavior": head}
    torch.save(blob, out_path)
    return out_path


def read(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("name") != MODEL_NAME:
        raise ValueError(f"{Path(path).name} is not a {MODEL_NAME} file")
    return blob


def is_mousescope(path):
    return Path(path).suffix == ".pt" and Path(path).stem.startswith(MODEL_NAME.lower())


def pose_factory(settings, root):
    """pose_model backend "mousescope": {"backend": "mousescope", "file": "models/mousescope.pt", "device": "mps"}."""
    import hashlib
    from . import pose_backends
    path = Path(settings["file"])
    path = path if path.is_absolute() else Path(root) / path
    pose = read(path)["pose"]
    digest = hashlib.sha256(pose["weights"]).hexdigest()[:12]
    cache = path.parent / ".mousescope_cache" / f"{digest}_{pose['weights_name']}"   # Ultralytics loads weights from a file
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pose["weights"])
    inner = dict(pose["settings"], weights=str(cache))
    if settings.get("device"):
        inner["device"] = settings["device"]
    return pose_backends.yolo_factory(inner, root)


class BehaviorScorer:
    """The behavior part for live_scoring.LiveScorer: scores the newest frames of a feature buffer."""

    def __init__(self, head, norm, info):
        self.head, self.norm, self.info = head, norm, info
        self.feature_columns = info["feature_columns"]

    def score_buffer(self, feats, frames):
        """feats: the buffer's feature table (one session); frames: the frames to return, all in the buffer."""
        d = feats.sort_values("frame")
        p = predict_proba(self.head, normalize(d[self.feature_columns].astype("float32").to_numpy(), self.norm))
        out = d[["session", "frame", "time_s", "tracking_ok"]].reset_index(drop=True)
        for b in BEHAVIORS:
            out[b] = p[:, CLASSES.index(b)].astype(float)
        return out[out["frame"].isin(frames)]


def load_behavior(path):
    """BehaviorScorer from models/mousescope.pt (or a bare head file)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    head_blob = blob["behavior"] if blob.get("name") == MODEL_NAME else blob
    info = head_blob["info"]
    head = TemporalHead(info["n_features"], info["params"]["width"], info["params"]["dropout"])
    head.load_state_dict(head_blob["state_dict"])
    head.eval()
    return BehaviorScorer(head, {k: v.numpy() for k, v in head_blob["norm"].items()}, info)


# --- CLI ----------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=f"{MODEL_NAME}: train the temporal behavior head")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--processed", default="data/processed")
    t.add_argument("--split", default="splits/split_manifest.json")
    t.add_argument("--models", default="models")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--epochs", type=int, default=40)
    p = sub.add_parser("pack", help="write the single models/mousescope.pt file")
    p.add_argument("--head", default="models/mousescope_head.pt")
    p.add_argument("--pose-settings", default="models/pose/pose_model_mac_yolo.json")
    p.add_argument("--out", default="models/mousescope.pt")
    a = ap.parse_args(argv)

    if a.cmd == "pack":
        settings = json.loads(Path(a.pose_settings).read_text())
        out = pack(a.out, a.head, settings["weights"], settings)
        print(f"wrote {out}: {Path(out).stat().st_size / 1e6:.1f} MB ({MODEL_NAME} {MODEL_VERSION})")
        return 0
    manifest = check_manifest(json.loads(Path(a.split).read_text()))
    feats = load_feature_set(a.processed, ["basic", "motion"])
    labels = pd.read_parquet(Path(a.processed) / "labels.parquet")
    print(f"{MODEL_NAME} {MODEL_VERSION}: {len(feature_columns(feats))} features, "
          f"receptive field {RECEPTIVE_FIELD} frames, device {_device()}")
    head, norm, info = train(feats, labels, manifest, seed=a.seed, epochs=a.epochs)
    va = session_arrays(feats, labels, manifest["validation"], info["feature_columns"], norm)
    info["validation_metrics"] = summary_metrics(score_sessions(head, va), labels)
    out = Path(a.models)
    out.mkdir(parents=True, exist_ok=True)
    save_head(out / "mousescope_head.pt", head, norm, info)
    (out / "mousescope_head.json").write_text(json.dumps(info, indent=2, default=float) + "\n")
    print(f"saved {out / 'mousescope_head.pt'} (best epoch {info['best_epoch']})")
    for k, v in info["validation_metrics"].items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
