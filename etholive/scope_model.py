"""The Live Tracker model: one model from camera frame to behavior scores, with one pack per animal.

The model has a shared core and an animal pack (species.py):

  shared core (the same for every animal)
    methods:  the ScopeScore methods: tracking-quality flags (quality.py), causal One-Euro
              smoothing (filters.py) and motion features (features.py);
    behavior: a causal temporal network (TemporalHead, below) that looks at the last ~2 s of
              features and scores the pack's behaviors;
  animal pack (one per animal: mouse now, flies next)
    the keypoints, animals, torso, skeleton and behaviors (species.Species), the fine-tuned pose
    weights, and the trained behavior head with its normalizer.

Each animal is one file, models/scope/<animal>.pt (MODEL_DIR). The mouse file is the hackathon
mouse model joined into one: YOLO26n-pose weights, the ScopeScore methods and the temporal head
that replaced model B. Its numbers are unchanged (docs/scope_model.md).

Training follows the same rules as classify.py: train on manifest["train"] sessions, choose
the epoch on manifest["validation"] (frame macro F1 of the behaviors), never read the test split.
Scores keep the INTERFACES.md section 2 layout, so the monitor, the rules and the dashboard do
not change.

Causal: the network only pads on the left, so the score of frame t uses frames <= t. It sees
RECEPTIVE_FIELD frames (2.1 s at 30 fps), well inside live_scoring's 6 s buffer.

Usage (from the repo root):
    python -m etholive.scope_model train --species mouse   # -> models/scope/mouse_head.pt + .json
    python -m etholive.scope_model pack --species mouse    # -> models/scope/mouse.pt
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

from . import species as sp
from .calms21 import check_manifest
from .classify import KEYS, feature_columns, load_feature_set
from .evaluate import summary_metrics
from .species import MOUSE

MODEL_NAME = "LiveScope"           # working name, shown to the user: change it here only
MODEL_VERSION = "0.2"
FORMAT = "live-tracker-scope/1"    # marks a model file; independent of the name
MODEL_DIR = "models/etholive"         # one file per animal: models/etholive/<animal>.pt (EthoLive, etholive.py)
DILATIONS = (1, 2, 4, 8, 16)
KERNEL = 3
RECEPTIVE_FIELD = 1 + (KERNEL - 1) * sum(DILATIONS)     # 63 frames = 2.1 s at 30 fps


def receptive_field(dilations=DILATIONS):
    return 1 + (KERNEL - 1) * sum(dilations)


def model_path(animal=sp.DEFAULT):
    """The model of one animal, without the suffix, as run configs name it: models/scope/mouse."""
    return f"{MODEL_DIR}/{sp.get(animal).name}"


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
    """Feature sequence (batch, time, features) -> class logits (batch, time, classes)."""

    def __init__(self, n_features, n_classes=len(MOUSE.classes), width=64, dropout=0.1, dilations=DILATIONS):
        super().__init__()
        self.inp = nn.Linear(n_features, width)
        self.blocks = nn.ModuleList(_CausalBlock(width, d, dropout) for d in dilations)
        self.out = nn.Linear(width, n_classes)

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


def session_arrays(features, labels, sessions, columns, norm, classes=MOUSE.classes):
    """{session: (x (T, F), y (T,) class index or None, frame table)} sorted by frame."""
    d = features[features["session"].isin(sessions)].sort_values(KEYS)
    if labels is not None:
        d = d.merge(labels[KEYS + ["label"]], on=KEYS, how="left", validate="one_to_one")
        if d["label"].isna().any():
            raise ValueError("some feature rows have no label")
    out = {}
    for sid, g in d.groupby("session", sort=True):
        x = normalize(g[columns].astype("float32").to_numpy(), norm)
        y = (pd.Categorical(g["label"], categories=list(classes)).codes.astype("int64")
             if labels is not None else None)
        if y is not None:
            y = np.where(y < 0, -100, y)                     # a label outside the classes (fly "ignore"): not trained on
        out[sid] = (x, y, g[[c for c in ("session", "frame", "time_s", "tracking_ok") if c in g]])
    return out


# --- scoring ------------------------------------------------------------------------------

def predict_proba(head, x, device="cpu"):
    """Class probabilities (T, classes) for one whole session; causal, so equal to streaming."""
    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(x)[None].to(device))[0]
        return torch.softmax(logits, dim=-1).cpu().numpy()


def _behavior_columns(table, p, species):
    for b in species.behaviors:
        table[b] = p[:, species.classes.index(b)].astype(float)
    return table


def score_sessions(head, arrays, device="cpu", species=MOUSE):
    """INTERFACES.md section 2 table (session, frame, time_s, tracking_ok, the pack's behaviors)."""
    parts = [_behavior_columns(table.reset_index(drop=True).copy(), predict_proba(head, x, device), species)
             for x, _, table in arrays.values()]
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


def train(features, labels, manifest, *, species=MOUSE, seed=0, epochs=40, steps=300, batch=32, window=256,
          lr=1e-3, width=64, dropout=0.1, patience=6, log=print):
    """Train on train sessions, keep the epoch with the best validation macro F1.

    The epoch is chosen with evaluate.summary_metrics, which scores the CalMS21 behaviors: a new
    animal pack needs its own metrics there before it can be trained."""
    if species.behaviors != MOUSE.behaviors:
        raise NotImplementedError(f"validation metrics for the {species.name} pack are not written yet")
    manifest = check_manifest(manifest)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = _device()
    classes = species.classes
    columns = feature_columns(features)
    norm = fit_normalizer(features, manifest["train"], columns)
    tr = session_arrays(features, labels, manifest["train"], columns, norm, classes)
    va = session_arrays(features, labels, manifest["validation"], columns, norm, classes)
    counts = np.bincount(np.concatenate([v[1] for v in tr.values()]), minlength=len(classes))
    weights = counts.sum() / (len(classes) * np.maximum(counts, 1))      # "balanced", as model B
    head = TemporalHead(len(columns), len(classes), width, dropout).to(device)
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
            loss = loss_fn(logits.reshape(-1, len(classes)), y.to(device).reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        val = summary_metrics(score_sessions(head, va, device, species), labels)
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
    info = {"name": MODEL_NAME, "version": MODEL_VERSION, "species": species.to_dict(), "classes": list(classes),
            "seed": seed, "best_epoch": best[2],
            "val_macro_f1": best[1], "history": history, "feature_columns": columns,
            "n_features": len(columns), "receptive_field_frames": RECEPTIVE_FIELD, "fps": species.fps,
            "params": {"epochs": epochs, "steps": steps, "batch": batch, "window": window, "lr": lr,
                       "width": width, "dropout": dropout, "patience": patience},
            "class_weights": dict(zip(classes, map(float, weights))),
            "train_sessions": len(manifest["train"]), "validation_sessions": len(manifest["validation"])}
    return head, norm, info


def save_head(path, head, norm, info):
    torch.save({"state_dict": head.state_dict(), "norm": {k: torch.from_numpy(v) for k, v in norm.items()},
                "info": info}, path)


def _head_from(blob):
    info = blob["info"]
    head = TemporalHead(info["n_features"], len(info["classes"]), info["params"]["width"], info["params"]["dropout"])
    head.load_state_dict(blob["state_dict"])
    head.eval()
    return head, {k: v.numpy() for k, v in blob["norm"].items()}, info


def load_head(path):
    return _head_from(torch.load(path, map_location="cpu", weights_only=False))


# --- the one file per animal --------------------------------------------------------------
#
# models/scope/<animal>.pt holds everything for that animal: its pack (species.Species), the pose
# weights (as bytes) with their settings, the behavior head with its normalizer and feature names,
# and the credits. The pose part is read through pose_model.py (backend "scope"), the behavior part
# through monitor_runner.build_scorer, so the rest of the program does not change.

METHODS_CREDIT = "Tracking-quality, smoothing and motion methods: ScopeScore (Sajjadi, 2026)."
POSE_CREDITS = {
    "mouse": "Pose network: Ultralytics YOLO26n-pose (AGPL-3.0), fine-tuned on the MARS hand-labeled set (Caltech).",
}
POSE_SETTINGS = {"mouse": "models/pose/pose_model_mac_yolo.json"}    # the plain YOLO settings each file is packed from


# MARS is the offline pose route (scripts/mars_pose_offline.py, TensorFlow, about 0.9 s per frame on a laptop):
# 560 MB under CC BY-NC 4.0, so the file names it and its checksums instead of carrying it.
# The CalMS21 keypoints the behavior part was trained on are MARS keypoints.
OFFLINE_POSE = {
    "mouse": {"name": "MARS top-view pose (Segalin et al., 2021)", "license": "CC BY-NC 4.0",
              "files": {"detection/MARS_top_detection_black.pb": "79f9b55a8eff1192b899e55cc0ea570c",
                        "detection/MARS_top_detection_white.pb": "4469996d5d7003c8c05d6279ed1506af",
                        "pose/MARS_top_pose.pb": "d380063a5177545721103a34fa650b56"},
              "folder": "models/mars", "script": "scripts/mars_pose_offline.py"},
}


def credits(species):
    extra = ("Offline pose and the training keypoints: MARS (Segalin et al., 2021), CC BY-NC 4.0.",
             "Model B: class-weighted logistic regression on the same features (Live Tracker hackathon build).") \
        if species.name == "mouse" else ()
    return (POSE_CREDITS.get(species.name, "Pose network: not recorded."),
            f"Behavior labels: {species.data}.", METHODS_CREDIT) + extra


def pack(out_path, head_path, pose_weights, pose_settings, methods=None, model_b=None, model_b_weight=0.0):
    """Write the single file of one animal: pose weights, ScopeScore method settings, temporal head, model B
    and the weight that mixes the two behavior parts."""
    head = torch.load(head_path, map_location="cpu", weights_only=False)
    species = sp.Species.from_dict(head["info"]["species"])
    weights = Path(pose_weights)
    keep = {k: pose_settings[k] for k in ("imgsz", "conf", "hold") if k in pose_settings}
    blob = {"format": FORMAT, "name": MODEL_NAME, "version": MODEL_VERSION,
            "species": species.to_dict(), "credits": credits(species),
            "pose": {"weights_name": weights.name, "weights": weights.read_bytes(), "settings": keep},
            "offline_pose": OFFLINE_POSE.get(species.name),
            "methods": methods, "behavior": head, "model_b": model_b,
            "fusion": {"model_b_weight": float(model_b_weight)}}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out_path)
    return out_path


def read(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or blob.get("format") != FORMAT:
        raise ValueError(f"{Path(path).name} is not a {MODEL_NAME} model file")
    return blob


def is_model_file(path):
    """True for models/scope/<animal>.pt (with or without the suffix)."""
    path = Path(path)
    return path.parent.name == Path(MODEL_DIR).name and path.with_suffix(".pt").is_file()


def _is_etholive(path):
    from . import model as etholive
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return isinstance(blob, dict) and blob.get("format") == etholive.FORMAT


def species_of(path):
    return sp.Species.from_dict(read(path)["species"])


def pose_factory(settings, root):
    """pose_model backend "scope": {"backend": "scope", "file": "models/scope/mouse.pt", "device": "mps"}."""
    import hashlib
    from . import pose_backends
    path = Path(settings["file"])
    path = path if path.is_absolute() else Path(root) / path
    if _is_etholive(path):                                       # EthoLive: its own Section 1
        from . import model as etholive
        return etholive.pose_factory(settings, root)
    pose = read(path)["pose"]
    digest = hashlib.sha256(pose["weights"]).hexdigest()[:12]
    cache = path.parent / ".cache" / f"{digest}_{pose['weights_name']}"    # Ultralytics loads weights from a file
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pose["weights"])
    inner = dict(pose["settings"], weights=str(cache))
    if settings.get("device"):
        inner["device"] = settings["device"]
    return pose_backends.yolo_factory(inner, root)


def model_b_part(joblib_path, info_path):
    """Model B (median fill, standard scaling, logistic regression) as plain numbers, so the model file
    does not depend on the scikit-learn version. Same probabilities as model_b.joblib (float64)."""
    import joblib
    info = json.loads(Path(info_path).read_text())
    return linear_start_from(joblib.load(joblib_path), info["feature_columns"], info.get("val_macro_f1"))


def linear_start_from(pipe, columns, val_macro_f1=None, classes=None):
    """The same, from a fitted pipeline in memory (the fly model B is fitted by scripts/train_fly.py).
    classes: the animal pack's class order; scikit-learn sorts classes by name, so the rows are put in this order."""
    imp, scale, lr = pipe[0], pipe[1], pipe[-1]
    if len(imp.statistics_) != len(columns) or lr.coef_.shape[1] != len(columns):
        raise ValueError("model B dropped feature columns while training; it cannot be packed")
    have = [str(c) for c in lr.classes_]
    order = [have.index(c) for c in classes] if classes is not None else list(range(len(have)))
    return {"feature_columns": list(columns), "classes": [have[i] for i in order],
            "median": imp.statistics_, "mean": scale.mean_, "scale": scale.scale_,
            "coef": lr.coef_[order], "intercept": lr.intercept_[order], "val_macro_f1": val_macro_f1}


def model_b_proba(part, table):
    x = table[part["feature_columns"]].astype("float64").to_numpy()
    z = (np.where(np.isnan(x), part["median"], x) - part["mean"]) / part["scale"]
    logit = z @ part["coef"].T + part["intercept"]
    p = np.exp(logit - logit.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def fuse(p_head, p_model_b, weight):
    """The model's probabilities: (1 - weight) * temporal head + weight * model B (weight chosen on validation)."""
    return p_head if not weight else (1.0 - weight) * p_head + weight * p_model_b


def apply_levels(p, log_levels):
    """Per-class alarm levels: each class's probability times exp(level), then renormalised (None: unchanged).
    A negative level makes a class harder to call (fly lunges in calm recordings)."""
    if log_levels is None:
        return p
    q = p * np.exp(np.asarray(log_levels, dtype=p.dtype))
    return q / q.sum(axis=1, keepdims=True)


class BehaviorScorer:
    """The behavior part for live_scoring.LiveScorer: scores the newest frames of a feature buffer.

    With a full model file it also computes its own features (session_features), with the ScopeScore
    methods the file was trained with, and mixes in model B."""

    def __init__(self, head, norm, info, methods=None, model_b=None, model_b_weight=0.0):
        self.head, self.norm, self.info = head, norm, info
        self.species = sp.Species.from_dict(info["species"])
        self.feature_columns = info["feature_columns"]
        self.methods, self.model_b, self.model_b_weight = methods, model_b, float(model_b_weight)
        self.lookahead = int(info.get("lookahead_frames", 0))   # the net's output at frame t is about frame t - lookahead

    def session_features(self, poses, setup=None):
        """The ScopeScore steps of this file's animal (scopescore_mouse.py or scopescore_fly.py).
        setup: flies only, a camera setup from flyvsfly.calibrate or calibrate_live."""
        if self.species.name.startswith("fly"):
            from . import scopescore_fly
            return scopescore_fly.session_features(poses, scopescore_fly.FlyParams(**self.methods), setup=setup)
        from . import scopescore_mouse
        return scopescore_mouse.session_features(poses, scopescore_mouse.ScopeParams(**self.methods))

    def proba(self, table):
        """Class probabilities (T, classes) for one session's feature table, sorted by frame."""
        p = predict_proba(self.head, normalize(table[self.feature_columns].astype("float32").to_numpy(), self.norm))
        if self.model_b is not None and self.model_b_weight:
            p = fuse(p, model_b_proba(self.model_b, table), self.model_b_weight)
        return apply_levels(p, getattr(self, "log_levels", None))

    def score_buffer(self, feats, frames):
        """feats: the buffer's feature table (one session); frames: the frames to return. With a lookahead, a frame's
        score comes from the output `lookahead` frames later: a frame without that later frame in the buffer is left out."""
        d = feats.sort_values("frame")
        out = _behavior_columns(d[["session", "frame", "time_s", "tracking_ok"]].reset_index(drop=True),
                                self.proba(d), self.species)
        if self.lookahead:
            later = out[["frame", *self.species.behaviors]].assign(frame=out["frame"] - self.lookahead)
            out = out[["session", "frame", "time_s", "tracking_ok"]].merge(later, on="frame", how="inner")
        return out[out["frame"].isin(frames)]


def load_behavior(path):
    """BehaviorScorer from models/scope/<animal>.pt (or a bare head file)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if _is_etholive(path):                                       # EthoLive: its own Section 2
        from . import model as etholive
        return etholive.load_behavior(path)
    if blob.get("format") != FORMAT:
        return BehaviorScorer(*_head_from(blob))
    fusion = blob.get("fusion", {})
    return BehaviorScorer(*_head_from(blob["behavior"]), methods=blob.get("methods"), model_b=blob.get("model_b"),
                          model_b_weight=fusion.get("model_b_weight", 0.0))


# --- CLI ----------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=f"{MODEL_NAME}: train an animal's behavior head, or pack its file")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--species", default=sp.DEFAULT, choices=sorted(sp.SPECIES))
    t.add_argument("--processed", default="data/processed")
    t.add_argument("--split", default="splits/split_manifest.json")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--epochs", type=int, default=40)
    p = sub.add_parser("pack", help="write the single models/scope/<animal>.pt file")
    p.add_argument("--species", default=sp.DEFAULT, choices=sorted(sp.SPECIES))
    p.add_argument("--pose-settings", help="default: the animal's entry in POSE_SETTINGS")
    a = ap.parse_args(argv)
    species = sp.get(a.species)
    out = Path(MODEL_DIR)
    head_path = out / f"{species.name}_head.pt"

    if a.cmd == "pack":
        settings = json.loads(Path(a.pose_settings or POSE_SETTINGS[species.name]).read_text())
        path = pack(out / f"{species.name}.pt", head_path, settings["weights"], settings)
        print(f"wrote {path}: {Path(path).stat().st_size / 1e6:.1f} MB ({MODEL_NAME} {MODEL_VERSION}, {species.label})")
        return 0
    manifest = check_manifest(json.loads(Path(a.split).read_text()))
    feats = load_feature_set(a.processed, ["basic", "motion"])
    labels = pd.read_parquet(Path(a.processed) / "labels.parquet")
    print(f"{MODEL_NAME} {MODEL_VERSION} ({species.label}): {len(feature_columns(feats))} features, "
          f"receptive field {RECEPTIVE_FIELD} frames, device {_device()}")
    head, norm, info = train(feats, labels, manifest, species=species, seed=a.seed, epochs=a.epochs)
    va = session_arrays(feats, labels, manifest["validation"], info["feature_columns"], norm, species.classes)
    info["validation_metrics"] = summary_metrics(score_sessions(head, va, species=species), labels)
    out.mkdir(parents=True, exist_ok=True)
    save_head(head_path, head, norm, info)
    head_path.with_suffix(".json").write_text(json.dumps(info, indent=2, default=float) + "\n")
    print(f"saved {head_path} (best epoch {info['best_epoch']})")
    for k, v in info["validation_metrics"].items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
