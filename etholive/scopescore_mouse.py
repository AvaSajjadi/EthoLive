"""More ScopeScore methods, adapted to mice (new file).

ScopeScore (Sajjadi, 2026, manuscript submitted; a surgical-instrument tracker whose methods are re-written here for animals) tracks a surgical tool as three points: the
tip T (the working end), the start S (the trocar end) and the center C, inside a circular field of
view. Its per-frame geometry and its smoothing carry over to a mouse seen from above: the nose is
the working end, the tail base is the trocar end, the spine is the tool's shaft and the camera
picture is the field of view. quality.py, filters.py and features.py already use three ScopeScore
methods (quality flags, causal One-Euro, motion measures). This module adds the rest that can be
measured on CalMS21 keypoints:

  ScopeScore method                                 here (mouse)
  1. tip first: the tip is found from geometry,     fix_nose_tail_swaps: the nose is the end next to
     not from the previous frame                    the ears, the tail base the end next to the hips
  2. One-Euro per endpoint, asymmetric settings     smooth_keypoints: nose = tip setting, tail base =
     (responsive tip, damped start, medium center)  start setting, the other keypoints = center setting
  3. PCA axis for S (one straight axis through      geometry: body axis, body length and width; where
     the whole shaft)                               the other mouse is in this mouse's body frame
  4. C = geodesic midpoint of the polyline          geometry: spine center at half the spine length
  5. polyline bend (Douglas-Peucker keeps real      geometry: spine straightness (chord / spine length)
     bends, drops wiggle)
  6. distance to the field-of-view edge             geometry: distance of the spine center to the
     (fov_dist) and the tip-edge state              picture edge, and an at-edge flag

Not measurable on CalMS21 (keypoints only, no video): KLT optical flow and its blend with the
detector (point_stabilizer.py, live only) and the drawing methods (Douglas-Peucker, trails).

Every method is per frame or a causal filter that forgets quickly, so live_scoring's 6 s buffer
gives the same values as a whole session.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .basic_features import _to_array, basic_features
from .calms21 import IMAGE_SIZE, KEYPOINTS, MICE, TORSO_KEYPOINTS, flag_quality
from .features import compute_features
from .filters import smooth_dataframe

KEYS = ["session", "frame"]

# ScopeScore's published One-Euro settings (methods.md, "One-Euro filter per track per endpoint")
TIP = {"min_cutoff": 0.6, "beta": 0.04}
START = {"min_cutoff": 0.1, "beta": 0.001}
CENTER = {"min_cutoff": 0.6, "beta": 0.01}


@dataclass
class ScopeParams:
    """Which ScopeScore methods run, and their settings. Chosen on validation only."""
    swap_fix: bool = True
    keypoint_smoothing: bool = False
    geometry: bool = True
    tip: dict = field(default_factory=lambda: dict(TIP))          # nose
    start: dict = field(default_factory=lambda: dict(START))      # tail base
    center: dict = field(default_factory=lambda: dict(CENTER))    # ears, neck, hips
    edge_margin_px: float = 8.0                                   # ScopeScore's --tip-edge-margin
    image_size: tuple = IMAGE_SIZE

    def to_dict(self):
        return asdict(self)


# --- 1. tip first: nose / tail-base swaps -------------------------------------------------

def _wide(poses, cols=("x", "y")):
    w = poses.set_index(KEYS + ["mouse", "keypoint"])[list(cols)].unstack("keypoint")
    return w


def swapped_frames(poses):
    """(session, frame, mouse) index -> True where the nose and the tail base look swapped.

    The rule is ScopeScore's: decide from this frame's geometry only. The nose should sit next to
    the ears and the tail base next to the hips; when the swapped reading is closer on both ends
    together, the pose model mixed them up. Needs all six points; otherwise no swap."""
    w = _wide(poses)
    x, y = w["x"], w["y"]

    def pt(k):
        return np.stack([x[k].to_numpy(float), y[k].to_numpy(float)], axis=1)

    ears = (pt("left_ear") + pt("right_ear")) / 2
    hips = (pt("left_hip") + pt("right_hip")) / 2
    nose, tail = pt("nose"), pt("tail_base")
    d = lambda a, b: np.hypot(*(a - b).T)                                     # noqa: E731
    keep = d(nose, ears) + d(tail, hips)
    swap = d(nose, hips) + d(tail, ears)
    return pd.Series(np.nan_to_num(swap, nan=np.inf) < np.nan_to_num(keep, nan=-np.inf), index=w.index)


def fix_nose_tail_swaps(poses):
    """Copy of the pose rows with nose and tail base exchanged where swapped_frames says so."""
    bad = swapped_frames(poses)
    bad = bad[bad]
    if bad.empty:
        return poses.copy()
    out = poses.copy()
    idx = pd.MultiIndex.from_frame(out[KEYS + ["mouse"]])
    hit = idx.isin(bad.index) & out["keypoint"].isin(["nose", "tail_base"]).to_numpy()
    out.loc[hit, "keypoint"] = out.loc[hit, "keypoint"].map({"nose": "tail_base", "tail_base": "nose"})
    return out


# --- 2. asymmetric One-Euro per keypoint --------------------------------------------------

def smooth_keypoints(flags, p):
    """Causal One-Euro on every keypoint track with ScopeScore's per-endpoint settings.
    Unusable points are NaN before smoothing (filters.py keeps NaN as NaN)."""
    d = flags.copy()
    usable = d["valid"].astype(bool) | d.get("imputed", False)
    d.loc[~usable, ["x", "y"]] = np.nan
    parts = []
    for name, setting in (("tip", p.tip), ("start", p.start), ("center", p.center)):
        kps = {"tip": ["nose"], "start": ["tail_base"]}.get(name)
        part = d[d["keypoint"].isin(kps)] if kps else d[~d["keypoint"].isin(["nose", "tail_base"])]
        part = smooth_dataframe(part, group_cols=("session", "mouse", "keypoint"), **setting)
        part["x"], part["y"] = part.pop("x_smooth"), part.pop("y_smooth")
        parts.append(part)
    return pd.concat(parts).loc[d.index]


# --- 3-6. tool geometry on the mouse --------------------------------------------------------

def geometry_names(mice=MICE):
    own = ["axis_len", "axis_wid", "spine_len", "spine_straight", "edge_dist", "at_edge"]
    pair = ["axis_cos"] + [f"{n}_{a}{b}" for a, b in (mice, mice[::-1]) for n in ("front", "side")] + ["spine_center_dist"]
    return [f"sc_{n}_{m}" for m in mice for n in own] + [f"sc_{n}" for n in pair]


def _pca_axis(pts):
    """pts (T, K, 2) with NaN -> unit axis (T, 2) toward the nose, length and width along / across it."""
    ok = ~np.isnan(pts).any(axis=2)                                    # (T, K)
    n = ok.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(pts, axis=1)                                 # (T, 2)
    c = np.where(ok[..., None], pts - mean[:, None], 0.0)
    cov = np.einsum("tki,tkj->tij", c, c) / np.maximum(n - 1, 1)[:, None, None]
    _, vec = np.linalg.eigh(cov)
    axis = vec[:, :, 1]                                                # largest eigenvalue
    head = pts[:, KEYPOINTS.index("nose")]                            # the tip; the ears when the nose is missing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ears = np.nanmean(pts[:, [KEYPOINTS.index("left_ear"), KEYPOINTS.index("right_ear")]], axis=1)
    head = np.where(np.isnan(head), ears, head)
    sign = np.sign(np.einsum("ti,ti->t", np.nan_to_num(head - mean), axis))
    axis = axis * np.where(sign == 0, 1.0, sign)[:, None]
    perp = np.stack([-axis[:, 1], axis[:, 0]], axis=1)
    along = np.where(ok, np.einsum("tki,ti->tk", c, axis), np.nan)
    across = np.where(ok, np.einsum("tki,ti->tk", c, perp), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        length = np.nanmax(along, axis=1) - np.nanmin(along, axis=1)
        width = np.nanmax(across, axis=1) - np.nanmin(across, axis=1)
    bad = n < 3
    axis[bad], length[bad], width[bad] = np.nan, np.nan, np.nan
    return axis, length, width


def _spine(pts):
    """Spine polyline nose -> neck -> hip midpoint -> tail base: length, straightness, geodesic center."""
    k = KEYPOINTS.index
    hips = (pts[:, k("left_hip")] + pts[:, k("right_hip")]) / 2
    line = np.stack([pts[:, k("nose")], pts[:, k("neck")], hips, pts[:, k("tail_base")]], axis=1)   # (T, 4, 2)
    seg = np.hypot(*np.moveaxis(np.diff(line, axis=1), 2, 0))                                       # (T, 3)
    length = seg.sum(axis=1)
    chord = np.hypot(*(line[:, -1] - line[:, 0]).T)
    straight = chord / np.where(length > 0, length, np.nan)
    half = length / 2
    cum = np.concatenate([np.zeros((len(seg), 1)), np.cumsum(seg, axis=1)], axis=1)                 # (T, 4)
    i = np.clip((cum[:, 1:] < half[:, None]).sum(axis=1), 0, 2)                                     # segment holding the midpoint
    r = np.arange(len(seg))
    frac = (half - cum[r, i]) / np.where(seg[r, i] > 0, seg[r, i], np.nan)
    center = line[r, i] + np.nan_to_num(frac, nan=0.0)[:, None] * (line[r, i + 1] - line[r, i])
    return length, straight, center


def geometry_features(flags, p=None):
    """The ScopeScore tool geometry for both mice, one row per session x frame (session, frame, time_s, sc_*)."""
    p = p or ScopeParams()
    index, arr = _to_array(flags, list(KEYPOINTS), list(MICE), "x", "y")             # (T, mouse, kp, xy)
    w, h = p.image_size
    cols, axes, centers, lengths = {}, [], [], []
    for j, m in enumerate(MICE):
        axis, length, width = _pca_axis(arr[:, j])
        slen, straight, center = _spine(arr[:, j])
        edge = np.minimum.reduce([center[:, 0], w - center[:, 0], center[:, 1], h - center[:, 1]])
        cols.update({f"sc_axis_len_{m}": length, f"sc_axis_wid_{m}": width, f"sc_spine_len_{m}": slen,
                     f"sc_spine_straight_{m}": straight, f"sc_edge_dist_{m}": edge,
                     f"sc_at_edge_{m}": np.where(np.isnan(edge), np.nan, (edge < p.edge_margin_px).astype(float))})
        axes.append(axis), centers.append(center), lengths.append(length)
    cols["sc_axis_cos"] = np.einsum("ti,ti->t", axes[0], axes[1])
    for me, other in ((0, 1), (1, 0)):
        rel = centers[other] - centers[me]
        perp = np.stack([-axes[me][:, 1], axes[me][:, 0]], axis=1)
        scale = np.where(lengths[me] > 0, lengths[me], np.nan)
        tag = f"{MICE[me]}{MICE[other]}"
        cols[f"sc_front_{tag}"] = np.einsum("ti,ti->t", rel, axes[me]) / scale
        cols[f"sc_side_{tag}"] = np.abs(np.einsum("ti,ti->t", rel, perp)) / scale
    cols["sc_spine_center_dist"] = np.hypot(*(centers[0] - centers[1]).T)
    out = pd.DataFrame({n: cols[n] for n in geometry_names()}).astype("float32")
    head = index.to_frame(index=False)
    head["time_s"] = flags.groupby(KEYS, sort=True)["time_s"].first().reindex(index).to_numpy()
    return pd.concat([head, out], axis=1)


# --- the whole feature step ---------------------------------------------------------------------

def session_features(poses, p=None):
    """Pose rows -> the model's feature table: basic + motion (as before) + ScopeScore geometry.

    Order of the ScopeScore steps: swaps are fixed on the raw rows (before the quality flags, so a
    swap is not flagged as a jump), the motion features keep their own body-center smoothing, and
    keypoint smoothing feeds the basic and geometry features only."""
    p = p or ScopeParams()
    if p.swap_fix:
        poses = fix_nose_tail_swaps(poses)
    flags = flag_quality(poses)
    motion = compute_features(flags, torso_keypoints=TORSO_KEYPOINTS)
    shape = smooth_keypoints(flags, p) if p.keypoint_smoothing else flags
    feats = basic_features(shape).merge(
        motion[[c for c in motion.columns if c != "time_s"]], on=KEYS, how="inner", validate="one_to_one")
    if p.geometry:
        g = geometry_features(shape, p)
        feats = feats.merge(g.drop(columns="time_s"), on=KEYS, how="inner", validate="one_to_one")
    return feats
