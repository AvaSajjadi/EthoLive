"""EthoLive: one model of our own, built from the pieces of the models this project used (new file).

  piece taken                                    from                     part of EthoLive
  backbone + neck + keypoint head (2.9 M weights) YOLO26n-pose             Section 1, Eyes
  the teaching pictures and labels               MARS hand-labeled set    Section 1, Eyes (trained on them)
  7 keypoints per mouse, black = a, white = b    MARS / CalMS21           Section 1 output
  quality flags, swap fix, smoothing, geometry   ScopeScore (9 methods)   between the sections: fixed, not learned
  102 measurements, class balancing, its fit     the hackathon Model B    Section 2, the frame branch (starts as model B)
  2 s causal memory                              new                      Section 2, the temporal branch

  Section 1 (Eyes): camera picture -> keypoints of both mice. Only the network is taken from YOLO; the picture
                    preparation, the choice of one black and one white mouse and the short-miss memory are ours,
                    so EthoLive does not run the Ultralytics predictor.
  Section 2 (Behavior): ScopeScore features -> attack / investigation / mount. logits = frame branch + memory
                    branch. The frame branch (one frame) starts with the coefficients of a simple classifier (for
                    mice the hackathon Model B) and the memory branch (the last 2.1 s) starts at zero, so before
                    training Section 2 is exactly that classifier; training adds what time shows.

The two sections are trained separately because no data set has both: the MARS set has pictures with keypoints
but no behavior labels, CalMS21 has behavior labels on keypoints but no pictures.

    python -m etholive.model check-eyes      # Section 1 against the Ultralytics predictor, real video
    python -m etholive.model train            # Section 2, train + validation only
    python -m etholive.model pack             # -> models/etholive/mouse.pt
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

from . import scope_model as sm
from . import species as sp
from .calms21 import IMAGE_SIZE, KEYPOINTS, MICE

MODEL_NAME = "EthoLive"            # the model's name, shown to the user: change it here only
MODEL_VERSION = "0.4"
FORMAT = "etholive/1"


# === Section 1: Eyes ===========================================================================

class Eyes(nn.Module):
    """Picture -> keypoints. The network is YOLO26n-pose's (fine-tuned on MARS); the rest is ours.

    Network output per anchor (5040 anchors at 384 x 640): box (4), black / white score (2), 7 x (x, y, visibility)."""

    def __init__(self, network, imgsz=(384, 640), conf=0.15, hold=3):
        super().__init__()
        self.network = network.float().eval()
        if hasattr(self.network, "fuse"):                      # each conv and its batch norm become one conv: same output,
            self.network.fuse(verbose=False)                   # about 40% faster (6.7 instead of 11.6 ms on an M-series GPU)
        self.imgsz, self.conf, self.hold = tuple(imgsz), float(conf), int(hold)
        self._memory = {}

    # picture preparation: the same letterbox the network was trained with (grey 114 border, centred)
    def prepare(self, image):
        import cv2
        h, w = image.shape[:2]
        if (w, h) != IMAGE_SIZE:
            image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            h, w = IMAGE_SIZE[1], IMAGE_SIZE[0]
        gain = min(self.imgsz[0] / h, self.imgsz[1] / w)
        nw, nh = int(round(w * gain)), int(round(h * gain))
        dw, dh = (self.imgsz[1] - nw) / 2, (self.imgsz[0] - nh) / 2
        if (nw, nh) != (w, h):
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
        image = cv2.copyMakeBorder(image, top, int(round(dh + 0.1)), left, int(round(dw + 0.1)),
                                   cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = torch.from_numpy(np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))).float() / 255.0   # BGR -> RGB
        return x, (nw / w, nh / h), left, top                 # the real scale of each axis after rounding

    def forward(self, x):
        """x (B, 3, H, W) in 0..1 -> raw network output (B, 27, anchors)."""
        out = self.network(x)
        return out[0] if isinstance(out, (list, tuple)) else out

    def pick(self, raw, gain, left, top):
        """Raw output of one picture -> one row set per mouse: the most certain anchor of each colour.
        (The anchor that NMS would keep first, so no NMS is needed for one mouse per colour.)"""
        scores = raw[4:6]                                    # (2, anchors), on the network's device
        best, cls = scores.max(dim=0)
        keep = []
        for c in range(len(MICE)):
            masked = torch.where((cls == c) & (best >= self.conf), best, torch.full_like(best, -1.0))
            keep.append(int(masked.argmax()))
        picked = raw[:, keep].cpu().numpy()                  # only the chosen anchors leave the device
        kp = picked[6:].reshape(len(KEYPOINTS), 3, len(MICE))   # keypoint, (x, y, visibility), mouse
        seen = {}
        for c, mouse in enumerate(MICE):
            if picked[4 + c, c] < self.conf or picked[4:6, c].argmax() != c:     # no anchor of this colour
                continue
            x = np.clip((kp[:, 0, c] - left) / gain[0], 0, IMAGE_SIZE[0])
            y = np.clip((kp[:, 1, c] - top) / gain[1], 0, IMAGE_SIZE[1])
            seen[mouse] = [{"mouse": mouse, "keypoint": k, "x": float(x[j]), "y": float(y[j]),
                            "score": float(min(1.0, max(0.0, kp[j, 2, c])))} for j, k in enumerate(KEYPOINTS)]
        return seen

    def held(self, seen):
        """A mouse missed for at most `hold` pictures keeps its last points (a short miss is not a loss)."""
        out = []
        for mouse in MICE:
            if mouse in seen:
                self._memory[mouse] = (seen[mouse], 0)
                out += seen[mouse]
            elif mouse in self._memory and self._memory[mouse][1] < self.hold:
                rows, n = self._memory[mouse]
                self._memory[mouse] = (rows, n + 1)
                out += rows
            else:
                self._memory.pop(mouse, None)
        return out

    @torch.no_grad()
    def find_mice(self, image):
        """BGR picture -> table mouse, keypoint, x, y, score in the 1024 x 570 CalMS21 frame, or None."""
        x, gain, left, top = self.prepare(image)
        device = next(self.network.parameters()).device
        seen = self.pick(self(x[None].to(device))[0], gain, left, top)
        rows = self.held(seen) if self.hold else [r for m in MICE for r in seen.get(m, [])]
        return pd.DataFrame(rows) if rows else None


def eyes_from_yolo(weights, imgsz=(384, 640), conf=0.15, hold=3, kind="mouse"):
    """Take the network out of a YOLO pose checkpoint (the EMA weights, as Ultralytics uses them)."""
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    return (FlyEyes if kind == "fly" else Eyes)((ck.get("ema") or ck["model"]).float(), imgsz, conf, hold)


class FlyEyes(Eyes):
    """Section 1 for flies: picture -> 5 points per fly (head, thorax, abdomen, wing tips), 2 flies.

    Differences from the mouse eyes: the picture keeps its own size (a fly chamber, not the CalMS21 frame); both flies
    are one class, so the second fly is the best detection whose box does not overlap the first (IoU < 0.5); and
    because the flies look alike, identity is kept by tracking: the detections are matched to where fly a and fly b
    were last seen (the pairing with the smaller total thorax distance; on the first picture, left to right)."""

    MAX_IOU = 0.5
    MAX_SIDE = 1280                     # a bigger picture is shrunk to this
    IGNORE_PX = 8.0                     # how near an ignored spot a detection is dropped
    SIZE_PICTURES = 90                  # pictures with both flies per measurement of the fly size
    SIZE_STEP = 0.03                    # a smaller change of the measure leaves the size alone

    def __init__(self, network, imgsz=(160, 160), conf=0.25, hold=3):
        super().__init__(network, imgsz, conf, hold)
        self._last = {}                                         # fly -> thorax (x, y) last seen
        self.frame_size = None
        self.scale = None                                       # picture -> network size, once the camera's fly size is known
        self.learning = True                                    # False: keep the size that was set
        self.ignore = []                                        # (x, y) in the picture's pixels: spots that are not flies
        self._lengths = []

    def input_size(self, h, w):
        """(height, width) for the network. A picture bigger than the training chamber starts at full size, because shrinking a
        whole arena to 160 px makes the flies too small (Boy meets boy, 640 x 480: both flies in 51% of pictures at 160), then
        goes to the size where flies are as long as in training (learn_size). The network takes any multiple of 32."""
        if max(h, w) <= max(self.imgsz):
            return self.imgsz
        s = min(self.scale or 1.0, self.MAX_SIDE / max(h, w))
        return int(np.ceil(h * s / 32) * 32), int(np.ceil(w * s / 32) * 32)

    def learn_size(self, seen):
        """A camera of unknown size: the eyes place points best with flies at the training size (median error 1-2 px against
        3-5 px at full size), so they measure the flies and rescale. This never stops: the measure shifts with the scale (from full
        size it settled in 3 steps on Boy meets boy, 0.703, true 0.673), and recordings open misleadingly, with one fly often measured
        short and false second flies, so a fixed number of rounds was used up before the second fly arrived and stayed at full size."""
        from .flyvsfly import FLIES, REFERENCE_BODY_PX
        if not self.learning or max(self.frame_size) <= max(self.imgsz) or any(f not in seen for f in FLIES):
            return
        for f in FLIES:
            p = {r["keypoint"]: (r["x"], r["y"]) for r in seen[f]}
            self._lengths.append(float(np.hypot(p["head"][0] - p["abdomen"][0], p["head"][1] - p["abdomen"][1])))
        if len(self._lengths) >= 2 * self.SIZE_PICTURES:
            new = min(1.0, REFERENCE_BODY_PX / float(np.median(self._lengths)))
            self._lengths = []
            if abs(new / (self.scale or 1.0) - 1) > self.SIZE_STEP:
                self.scale = new

    def prepare(self, image):
        import cv2
        h, w = image.shape[:2]
        self.frame_size = (w, h)
        size = self.input_size(h, w)
        gain = min(size[0] / h, size[1] / w)
        nw, nh = int(round(w * gain)), int(round(h * gain))
        dw, dh = (size[1] - nw) / 2, (size[0] - nh) / 2
        if (nw, nh) != (w, h):
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
        image = cv2.copyMakeBorder(image, top, int(round(dh + 0.1)), left, int(round(dw + 0.1)),
                                   cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = torch.from_numpy(np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))).float() / 255.0
        return x, (nw / w, nh / h), left, top

    @staticmethod
    def _iou(a, b):
        ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
        bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
        iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
        inter = iw * ih
        return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)

    def _not_ignored(self, centre, gain, left, top):
        """Not on a spot of this arena known not to be a fly (scout.look finds them: a dark mark can score like a fly). A fly's length
        cannot tell them apart here: real flies measure under half length in up to 10% of frames, mostly in fights."""
        x, y = (centre[0] - left) / gain[0], (centre[1] - top) / gain[1]
        return all(np.hypot(x - sx, y - sy) >= self.IGNORE_PX for sx, sy in self.ignore)

    def pick(self, raw, gain, left, top):
        from .flyvsfly import FLIES, KEYPOINTS as FLY_KP
        score = raw[4]
        order = torch.argsort(score, descending=True)[:50]
        cand = raw[:, order].cpu().numpy()
        found = []
        for j in range(cand.shape[1]):
            if cand[4, j] < self.conf or len(found) == 2:
                break
            if self._not_ignored(cand[:2, j], gain, left, top) and all(self._iou(cand[:4, j], cand[:4, k]) < self.MAX_IOU for k in found):
                found.append(j)
        flies = []
        for j in found:
            kp = cand[5:].reshape(len(FLY_KP), 3, -1)[:, :, j]
            w, h = self.frame_size
            x = np.clip((kp[:, 0] - left) / gain[0], 0, w)
            y = np.clip((kp[:, 1] - top) / gain[1], 0, h)
            flies.append((x, y, np.clip(kp[:, 2], 0, 1)))
        th = FLY_KP.index("thorax")
        if len(flies) == 2:
            last = [self._last.get(f) for f in FLIES]
            if None in last:
                flies.sort(key=lambda f: f[0][th])                  # first sight: left to right
            else:
                d = lambda f, p: np.hypot(f[0][th] - p[0], f[1][th] - p[1])     # noqa: E731
                if d(flies[0], last[0]) + d(flies[1], last[1]) > d(flies[1], last[0]) + d(flies[0], last[1]):
                    flies.reverse()
            names = list(FLIES)
        elif len(flies) == 1:
            seen = {f: p for f, p in self._last.items() if p is not None}
            names = [min(seen, key=lambda f: np.hypot(flies[0][0][th] - seen[f][0], flies[0][1][th] - seen[f][1]))
                     if seen else FLIES[0]]
        else:
            names = []
        out = {}
        for name, (x, y, v) in zip(names, flies):
            self._last[name] = (float(x[th]), float(y[th]))
            out[name] = [{"mouse": name, "keypoint": k, "x": float(x[i]), "y": float(y[i]), "score": float(v[i])}
                         for i, k in enumerate(FLY_KP)]
        return out

    def held(self, seen):
        from .flyvsfly import FLIES
        out = []
        for fly in FLIES:
            if fly in seen:
                self._memory[fly] = (seen[fly], 0)
                out += seen[fly]
            elif fly in self._memory and self._memory[fly][1] < self.hold:
                rows, n = self._memory[fly]
                self._memory[fly] = (rows, n + 1)
                out += rows
            else:
                self._memory.pop(fly, None)
        return out

    @torch.no_grad()
    def find_mice(self, image):
        """BGR picture -> table mouse (= fly a / b), keypoint, x, y, score in the picture's pixels, or None."""
        from .flyvsfly import FLIES
        x, gain, left, top = self.prepare(image)
        device = next(self.network.parameters()).device
        seen = self.pick(self(x[None].to(device))[0], gain, left, top)
        self.learn_size(seen)
        rows = self.held(seen) if self.hold else [r for f in FLIES for r in seen.get(f, [])]
        return pd.DataFrame(rows) if rows else None


# === Section 2: Behavior ======================================================================

class BehaviorNet(nn.Module):
    """ScopeScore features (normalized) -> class logits. Frame branch (this frame) + memory branch (causal, 2.1 s)."""

    def __init__(self, n_features, n_classes=len(sp.MOUSE.classes), width=64, dropout=0.1, dilations=sm.DILATIONS):
        super().__init__()
        self.frame = nn.Linear(n_features, n_classes)
        self.memory = sm.TemporalHead(n_features, n_classes, width, dropout, tuple(dilations))
        nn.init.zeros_(self.memory.out.weight)                # starts as the frame branch alone
        nn.init.zeros_(self.memory.out.bias)

    def forward(self, x):                                     # x: batch, time, features
        return self.frame(x) + self.memory(x)


class Team(nn.Module):
    """Several trained BehaviorNets voting: the softmax of the output is the mean of their probabilities."""

    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, x):
        p = torch.stack([torch.softmax(n(x), dim=-1) for n in self.nets]).mean(dim=0)
        return torch.log(p.clamp_min(1e-12))


def frame_branch_init(net, columns, norm, part):
    """Start the frame branch from a fitted linear classifier (`part`, see scope_model.model_b_part), rewritten for
    EthoLive's own normalization: logit = ((x - mean_c) / scale_c) . coef + b, with x = x_norm * std + mean (ours).
    The classifier's rows must be in the order of the net's classes."""
    w = np.zeros((len(part["classes"]), len(columns)))
    b = np.array(part["intercept"], float)
    index = {c: i for i, c in enumerate(columns)}
    for j, c in enumerate(part["feature_columns"]):
        i = index[c]
        k = part["coef"][:, j] / part["scale"][j]
        w[:, i] = k * norm["std"][i]
        b += k * (norm["mean"][i] - part["mean"][j])
    with torch.no_grad():
        net.frame.weight.copy_(torch.from_numpy(w).float())
        net.frame.bias.copy_(torch.from_numpy(b).float())


class _PartialLoss:
    """Class-weighted cross entropy; a partial label k (code n_classes + k) is right when the net puts its probability
    on any of its allowed classes (-log of their summed probability), weighted like "other". -100 is ignored."""

    def __init__(self, weights, allowed, other, device):
        self.w = torch.tensor(np.append(weights, [weights[other]] * len(allowed)), dtype=torch.float32, device=device)
        n = len(weights)
        self.n = n
        self.allowed = torch.zeros(len(allowed), n, dtype=torch.bool, device=device)
        for k, idx in enumerate(allowed):
            self.allowed[k, idx] = True

    def __call__(self, logits, y):
        keep = y >= 0
        logits, y = logits[keep], y[keep]
        logp = torch.log_softmax(logits, dim=-1)
        full = y < self.n
        nll = torch.empty_like(y, dtype=logp.dtype)
        nll[full] = -logp[full].gather(1, y[full, None])[:, 0]
        if (~full).any():
            mask = self.allowed[y[~full] - self.n]
            nll[~full] = -torch.logsumexp(logp[~full].masked_fill(~mask, float("-inf")), dim=-1)
        w = self.w[y]
        return (w * nll).sum() / w.sum()


def train(features, labels, manifest, start, *, species=sp.MOUSE, seed=0, epochs=40, steps=300, batch=32,
          window=256, lr=1e-3, width=64, dropout=0.1, dilations=sm.DILATIONS, patience=6, metrics=None, partial=None,
          balance="balanced", log=print):
    """Section 2 on CalMS21 train sessions; the epoch with the best validation macro F1 is kept (never test)."""
    from .calms21 import check_manifest
    from .classify import feature_columns
    from .evaluate import summary_metrics as mouse_metrics
    summary_metrics = metrics or mouse_metrics               # the animal's own scores (flies: fly_metrics.py)
    manifest = check_manifest(manifest)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = sm._device()
    classes = species.classes
    columns = feature_columns(features)
    norm = sm.fit_normalizer(features, manifest["train"], columns)
    # partial labels: {label: allowed classes}, for a data set that labels only some behaviors (fly Boy meets boy:
    # its "other" means "not lunge, not wing threat"; it may still be a tussle or a hold, which it does not label)
    partial = partial or {}
    cats = tuple(classes) + tuple(partial)
    tr = sm.session_arrays(features, labels, manifest["train"], columns, norm, cats)
    va = sm.session_arrays(features, labels, manifest["validation"], columns, norm, cats)
    ys = np.concatenate([v[1] for v in tr.values()])
    counts = np.bincount(np.where(ys >= len(classes), classes.index("other"), ys)[ys >= 0], minlength=len(classes))
    weights = counts.sum() / (len(classes) * np.maximum(counts, 1))      # class balancing, as in Model B
    if balance == "sqrt":                                                  # gentler: rare classes get less extra weight
        weights = np.sqrt(weights)
    if list(start["classes"]) != list(classes):
        raise ValueError(f"the frame branch's start has classes {start['classes']}, not the pack's {list(classes)}")
    net = BehaviorNet(len(columns), len(classes), width, dropout, dilations)
    frame_branch_init(net, columns, norm, start)
    start = summary_metrics(sm.score_sessions(net, va, "cpu", species), labels)["macro_f1"]
    log(f"  before training (frame branch alone): validation macro F1 {start:.3f}")
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = _PartialLoss(weights, [[classes.index(c) for c in allowed] for allowed in partial.values()],
                           classes.index("other"), device)
    best, history, waited = None, [], 0
    for epoch in range(1, epochs + 1):
        net.train()
        t0, total = time.time(), 0.0
        for _ in range(steps):
            x, y = sm._batches(tr, rng, window, batch)
            loss = loss_fn(net(x.to(device)).reshape(-1, len(classes)), y.to(device).reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        val = summary_metrics(sm.score_sessions(net, va, device, species), labels)
        history.append({"epoch": epoch, "train_loss": total / steps, "val_macro_f1": val["macro_f1"]})
        log(f"  epoch {epoch:2d}: loss {total / steps:.3f}, validation macro F1 {val['macro_f1']:.3f} ({time.time() - t0:.0f} s)")
        if best is None or val["macro_f1"] > best[1]:
            best = ({k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, val["macro_f1"], epoch)
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                break
    net.load_state_dict(best[0])
    net = net.to("cpu").eval()
    info = {"name": MODEL_NAME, "version": MODEL_VERSION, "species": species.to_dict(), "classes": list(classes),
            "seed": seed, "best_epoch": best[2], "val_macro_f1": best[1], "val_macro_f1_before_training": start,
            "history": history, "feature_columns": columns, "n_features": len(columns),
            "receptive_field_frames": sm.receptive_field(dilations), "fps": species.fps,
            "params": {"epochs": epochs, "steps": steps, "batch": batch, "window": window, "lr": lr,
                       "width": width, "dropout": dropout, "dilations": list(dilations), "patience": patience},
            "class_weights": dict(zip(classes, map(float, weights))), "balance": balance, "partial_labels": {k: list(v) for k, v in partial.items()},
            "train_sessions": len(manifest["train"]), "validation_sessions": len(manifest["validation"])}
    return net, norm, info


def save_behavior(path, net, norm, info):
    torch.save({"state_dict": net.state_dict(), "norm": {k: torch.from_numpy(v) for k, v in norm.items()},
                "info": info}, path)


def _net_from(info, state):
    p = info["params"]
    old = {"model_b.": "frame.", "temporal.": "memory."}             # names used before 2026-11-02
    state = {next((new + k[len(o):] for o, new in old.items() if k.startswith(o)), k): v for k, v in state.items()}
    net = BehaviorNet(info["n_features"], len(info["classes"]), p["width"], p["dropout"], p.get("dilations", sm.DILATIONS))
    net.load_state_dict(state)
    return net.eval()


def _behavior_from(blob):
    """One net, or a team (blob["team"]: the state of each member; they share the features and the normalizer)."""
    info = blob["info"]
    states = blob.get("team") or [blob["state_dict"]]
    nets = [_net_from(info, s) for s in states]
    net = nets[0] if len(nets) == 1 else Team(nets).eval()
    return net, {k: v.numpy() for k, v in blob["norm"].items()}, info


class Behavior(sm.BehaviorScorer):
    """Section 2 for live_scoring.LiveScorer: computes its ScopeScore features, then scores them."""

    def __init__(self, net, norm, info, methods, log_levels=None):
        super().__init__(net, norm, info, methods=methods)
        self.log_levels = log_levels                            # per-class alarm levels (scope_model.apply_levels)


# === the one file ===============================================================================

def team_blob(paths):
    """One Section-2 blob from several trained nets of the same design (same features, same normalizer)."""
    blobs = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    first = blobs[0]
    for b in blobs[1:]:
        if b["info"]["feature_columns"] != first["info"]["feature_columns"] or b["info"]["params"] != first["info"]["params"] \
                or any(not torch.equal(b["norm"][k], first["norm"][k]) for k in first["norm"]):
            raise ValueError("team members must share the design, the features and the normalizer")
    team = dict(first, team=[b["state_dict"] for b in blobs])
    team["info"] = dict(first["info"], team_seeds=[b["info"]["seed"] for b in blobs])
    return team


EYES_FROM = {
    "mouse": "YOLO26n-pose (Ultralytics, AGPL-3.0), fine-tuned on the MARS hand-labeled set",
    "fly": "YOLO26n-pose (Ultralytics, AGPL-3.0), fine-tuned on Fly vs. Fly Aggression and Boy meets boy frames with FlyTracker points",
}
EYES_FROM["fly_courtship"] = EYES_FROM["fly"]
CREDITS = {
    "mouse": ("Section 1 network: Ultralytics YOLO26n-pose (AGPL-3.0); trained on the MARS hand-labeled set "
              "(Segalin et al., 2021, CC BY-NC 4.0).",
              "Between the sections: ScopeScore methods (Sajjadi, 2026), adapted to mice.",
              "Section 2: frame branch started from the hackathon classifier (Model B); trained on CalMS21 Task 1 "
              "(Sun et al., 2021)."),
    "fly": ("Section 1 network: Ultralytics YOLO26n-pose (AGPL-3.0); trained on Fly vs. Fly Aggression and Boy meets boy frames, with the "
            "data set's FlyTracker points as labels (automatic, not hand-marked).",
            "Between the sections: ScopeScore methods (Sajjadi, 2026), adapted to flies.",
            "Section 2: frame branch started from a fly logistic regression; trained on Caltech Fly vs. Fly, "
            "Aggression, Boy meets boy and Courtship sets (Eyjolfsdottir et al., 2014, CC0)."),
}


CREDITS["fly_courtship"] = CREDITS["fly"]


def pack(out_path, eyes, methods, behavior_path, version=MODEL_VERSION, log_levels=None):
    """models/etholive/mouse.pt: Section 1 (the network and our settings), the ScopeScore settings, Section 2
    (behavior_path: one trained net, or a list of nets that vote as a team)."""
    behavior = (team_blob(behavior_path) if isinstance(behavior_path, (list, tuple))
                else torch.load(behavior_path, map_location="cpu", weights_only=False))
    behavior["info"].update(name=MODEL_NAME, version=version)
    species = sp.Species.from_dict(behavior["info"]["species"])
    blob = {"format": FORMAT, "name": MODEL_NAME, "version": version, "species": species.to_dict(),
            "sections": {
                "eyes": None if eyes is None else
                        {"network": eyes.network.half(), "imgsz": list(eyes.imgsz), "conf": eyes.conf, "hold": eyes.hold,
                         "kind": "fly" if isinstance(eyes, FlyEyes) else "mouse",
                         "taken_from": EYES_FROM[species.name]},
                "methods": methods,
                "log_levels": None if log_levels is None else [float(v) for v in log_levels],
                "behavior": behavior},
            "offline_pose": sm.OFFLINE_POSE.get(species.name),
            "credits": CREDITS[species.name]}
    if eyes is not None:
        eyes.network.float()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out_path)
    return out_path


def read(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or blob.get("format") != FORMAT:
        raise ValueError(f"{Path(path).name} is not a {MODEL_NAME} file")
    return blob


def is_etholive(path):
    try:
        with open(path, "rb"):
            pass
    except OSError:
        return False
    return torch.load(path, map_location="cpu", weights_only=False).get("format") == FORMAT


def load_eyes(path, device=None):
    s = read(path)["sections"]["eyes"]
    if s is None:
        raise ValueError(f"{Path(path).name} has no Section 1 (eyes) yet: score this animal from a keypoint file")
    eyes = (FlyEyes if s.get("kind") == "fly" else Eyes)(s["network"].float(), s["imgsz"], s["conf"], s["hold"])
    if device:
        eyes.network.to(device)
    return eyes


def load_behavior(path):
    blob = read(path)["sections"]
    return Behavior(*_behavior_from(blob["behavior"]), methods=blob["methods"], log_levels=blob.get("log_levels"))


def pose_factory(settings, root):
    """pose_model backend "etholive": Section 1 of the file, on the computer's device."""
    path = Path(settings["file"])
    path = path if path.is_absolute() else Path(root) / path
    eyes = load_eyes(path, settings.get("device"))
    eyes.find_mice(np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8))     # first picture: warm-up
    eyes._memory.clear()
    return eyes.find_mice


# === command line ===============================================================================

def check_eyes(video, frames=150, device="cpu", weights="models/pose/mouse_pose_best.pt"):
    """Section 1 against the Ultralytics predictor on the same pictures: largest keypoint difference in pixels."""
    import cv2
    from . import pose_backends
    yolo = pose_backends.yolo_factory({"weights": weights, "imgsz": [384, 640], "conf": 0.15, "hold": 0,
                                       "device": device}, Path("."))
    eyes = eyes_from_yolo(weights, hold=0)
    eyes.network.to(device)
    cap = cv2.VideoCapture(str(video))
    diffs, both, missing = [], 0, 0
    for _ in range(frames):
        ok, image = cap.read()
        if not ok:
            break
        a, b = yolo(image), eyes.find_mice(image)
        if a is None or b is None:
            missing += (a is None) != (b is None)
            continue
        m = a.merge(b, on=["mouse", "keypoint"], suffixes=("_y", "_s"), how="outer")
        if m.isna().any().any():
            missing += 1
            continue
        both += 1
        diffs.append(np.hypot(m["x_y"] - m["x_s"], m["y_y"] - m["y_s"]).max())
    return {"pictures_compared": both, "pictures_where_one_found_a_mouse_the_other_did_not": missing,
            "max_keypoint_difference_px": float(max(diffs)) if diffs else None,
            "median_of_max_difference_px": float(np.median(diffs)) if diffs else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=f"{MODEL_NAME}: build the model in two sections")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-eyes")
    c.add_argument("--video", default="data/raw/calms21_video/mouse064_task1_annotator1.mp4")
    c.add_argument("--frames", type=int, default=150)
    c.add_argument("--device", default="cpu")
    t = sub.add_parser("train")
    t.add_argument("--features", default="data/processed_scope/geometry_swap_smooth.parquet")
    t.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    k = sub.add_parser("pack")
    k.add_argument("--behavior", nargs="+", default=["outputs/etholive/behavior_seed0.pt"],
                   help="one trained Section 2, or several of the same design to vote as a team")
    a = ap.parse_args(argv)
    root = Path(".")
    if a.cmd == "check-eyes":
        print(json.dumps(check_eyes(a.video, a.frames, a.device), indent=2))
        return 0
    if a.cmd == "train":
        from .calms21 import check_manifest
        from .evaluate import summary_metrics
        manifest = check_manifest(json.loads((root / "splits/split_manifest.json").read_text()))
        feats = pd.read_parquet(a.features)
        labels = pd.read_parquet(root / "data/processed/labels.parquet")
        labels = labels[labels["session"].isin(set(manifest["train"]) | set(manifest["validation"]))]
        part = sm.model_b_part(root / "models/model_b.joblib", root / "models/model_b.json")
        out = root / "outputs/etholive"
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for seed in a.seeds:
            print(f"== Section 2, seed {seed}", flush=True)
            net, norm, info = train(feats, labels, manifest, part, seed=seed)
            va = sm.session_arrays(feats, labels, manifest["validation"], info["feature_columns"], norm)
            info["validation_metrics"] = summary_metrics(sm.score_sessions(net, va), labels)
            info["features"] = a.features
            save_behavior(out / f"behavior_seed{seed}.pt", net, norm, info)
            rows.append({"seed": seed, "before_training": info["val_macro_f1_before_training"], **info["validation_metrics"]})
            pd.DataFrame(rows).to_csv(out / "section2_validation.csv", index=False)
        print(pd.DataFrame(rows)[["seed", "before_training", "macro_f1", "attack_f1", "investigation_f1", "mount_f1",
                                  "attack_event_precision", "attack_event_recall"]].round(3).to_string(index=False))
        return 0
    # pack: Section 1 from the MARS-trained YOLO network, all 9 ScopeScore methods, Section 2 seed 0 (declared before training)
    import shutil
    from . import scopescore_mouse as sc
    settings = json.loads((root / sm.POSE_SETTINGS["mouse"]).read_text())
    eyes = eyes_from_yolo(root / settings["weights"], settings["imgsz"], settings["conf"], settings["hold"])
    methods = sc.ScopeParams(swap_fix=True, geometry=True, keypoint_smoothing=True).to_dict()
    (root / sm.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    if len(a.behavior) == 1:
        behavior = root / sm.MODEL_DIR / "mouse_behavior.pt"
        shutil.copy(root / a.behavior[0], behavior)
    else:
        behavior = [root / b for b in a.behavior]
    path = pack(root / sm.MODEL_DIR / "mouse.pt", eyes, methods, behavior)
    print(f"wrote {path} ({Path(path).stat().st_size / 1e6:.1f} MB, {MODEL_NAME} {MODEL_VERSION})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
