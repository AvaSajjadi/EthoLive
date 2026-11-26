"""ScopeScore methods for flies: EthoLive's fly pack (new file). The mouse version is scopescore_mouse.py.

A fly seen from above maps onto ScopeScore's tool the same way a mouse does, with wings in place of a jaw:

  ScopeScore (surgical tool)                       fly
  1. quality flags                                 quality.flag_tracking in the 144 x 144 chamber; the "frozen"
                                                   check is off: the wing tips are whole pixels and a resting fly
                                                   keeps them exactly, which is not a tracking failure
  2. tip first, from this frame's geometry         fix_head_abdomen: the head is the end away from the wings
  3. One-Euro per endpoint (tip / start / center)  head = tip setting, abdomen = start setting, the rest = center
  4. causal One-Euro + motion measures             features.compute_features on the thorax, with fly-sized settings
  5. PCA / straight axis of the tool               body axis head -> abdomen: length; the other fly in front / beside
  6. jaw open / closed (T2 branch)                 wing angles from the body axis: left, right, spread, asymmetry, length
  7. tool facing                                   facing: how straight each fly points at the other
  8. distance to the field-of-view edge            distance of the thorax to the chamber edge, at-edge flag

Every step is per frame or a causal filter, as for the mice, so live scoring equals whole-session scoring.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .basic_features import _to_array, basic_features
from .features import FeatureParams, compute_features
from .flyvsfly import FLIES, FRAME_SIZE, KEYPOINTS, TORSO_KEYPOINTS
from .quality import flag_tracking, interpolate_short_gaps
from .scopescore_mouse import CENTER, START, TIP
from .filters import smooth_dataframe

KEYS = ["session", "frame"]

# The mouse settings scaled by body length (a fly is about 19 px long here, a CalMS21 mouse about 100 px): not tuned.
FLY_MOTION = dict(heading_min_speed_px_s=1.0, idle_speed_px_s=2.0, close_dist_px=25.0, min_path_px=0.2)


@dataclass
class FlyParams:
    swap_fix: bool = True
    keypoint_smoothing: bool = True
    geometry: bool = True
    tip: dict = field(default_factory=lambda: dict(TIP))          # head
    start: dict = field(default_factory=lambda: dict(START))      # abdomen
    center: dict = field(default_factory=lambda: dict(CENTER))    # thorax, wing tips
    edge_margin_px: float = 2.0                                   # ScopeScore's 8 px on a 1024 px picture, scaled
    edge_cap_px: float | None = None                              # cap the edge distance (a bigger arena than training)
    frame_size: tuple = FRAME_SIZE
    motion: dict = field(default_factory=lambda: dict(FLY_MOTION))
    tip_motion: bool = False                                      # head ("tip") motion features: tried, not kept

    def to_dict(self):
        return asdict(self)


def flag_quality(poses, p):
    return interpolate_short_gaps(flag_tracking(poses, frame_size=p.frame_size, score_col="score", min_score=0.5,
                                                frozen_min_frames=10 ** 9))


# --- tip first -----------------------------------------------------------------------------

def swapped_frames(poses):
    """(session, frame, mouse) -> True where the head lies nearer the wings than the abdomen does."""
    w = poses.set_index(KEYS + ["mouse", "keypoint"])[["x", "y"]].unstack("keypoint")
    pt = lambda k: np.stack([w["x"][k].to_numpy(float), w["y"][k].to_numpy(float)], axis=1)   # noqa: E731
    wings = (pt("wing_l") + pt("wing_r")) / 2
    d_head = np.hypot(*(pt("head") - wings).T)
    d_abd = np.hypot(*(pt("abdomen") - wings).T)
    return pd.Series(np.nan_to_num(d_head, nan=np.inf) < np.nan_to_num(d_abd, nan=-np.inf), index=w.index)


def fix_head_abdomen(poses):
    bad = swapped_frames(poses)
    bad = bad[bad]
    out = poses.copy()
    if bad.empty:
        return out
    hit = pd.MultiIndex.from_frame(out[KEYS + ["mouse"]]).isin(bad.index) & out["keypoint"].isin(["head", "abdomen"]).to_numpy()
    out.loc[hit, "keypoint"] = out.loc[hit, "keypoint"].map({"head": "abdomen", "abdomen": "head"})
    return out


# --- per-point One-Euro --------------------------------------------------------------------------

def smooth_keypoints(flags, p):
    d = flags.copy()
    usable = d["valid"].astype(bool) | d.get("imputed", False)
    d.loc[~usable, ["x", "y"]] = np.nan
    parts = []
    for kps, setting in ((["head"], p.tip), (["abdomen"], p.start), (["thorax", "wing_l", "wing_r"], p.center)):
        part = smooth_dataframe(d[d["keypoint"].isin(kps)], group_cols=("session", "mouse", "keypoint"), **setting)
        part["x"], part["y"] = part.pop("x_smooth"), part.pop("y_smooth")
        parts.append(part)
    return pd.concat(parts).loc[d.index]


# --- geometry ------------------------------------------------------------------------------------

def geometry_names(flies=FLIES):
    own = ["body_len", "wing_l_ang", "wing_r_ang", "wing_spread", "wing_asym", "wing_len", "facing", "edge_dist", "at_edge"]
    pair = ["axis_cos", "head_to_other"] + [f"{n}_{a}{b}" for a, b in (flies, flies[::-1]) for n in ("front", "side")]
    return [f"fs_{n}_{f}" for f in flies for n in own] + [f"fs_{n}" for n in pair]


def _angle(u, v):
    cos = np.einsum("ti,ti->t", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.arccos(np.clip(cos, -1, 1))


def geometry_features(flags, p=None):
    p = p or FlyParams()
    index, arr = _to_array(flags, list(KEYPOINTS), list(FLIES), "x", "y")           # (T, fly, kp, xy)
    k = KEYPOINTS.index
    w, h = p.frame_size
    cols, axes, thorax, length = {}, [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j, f in enumerate(FLIES):
            head, th, abd = arr[:, j, k("head")], arr[:, j, k("thorax")], arr[:, j, k("abdomen")]
            axis = head - abd
            blen = np.linalg.norm(axis, axis=1)
            back = abd - th
            wl, wr = arr[:, j, k("wing_l")] - th, arr[:, j, k("wing_r")] - th
            al, ar = _angle(back, wl), _angle(back, wr)
            edge = np.minimum.reduce([th[:, 0], w - th[:, 0], th[:, 1], h - th[:, 1]])
            if p.edge_cap_px is not None:
                edge = np.minimum(edge, p.edge_cap_px)
            cols.update({f"fs_body_len_{f}": blen, f"fs_wing_l_ang_{f}": al, f"fs_wing_r_ang_{f}": ar,
                         f"fs_wing_spread_{f}": al + ar, f"fs_wing_asym_{f}": np.abs(al - ar),
                         f"fs_wing_len_{f}": (np.linalg.norm(wl, axis=1) + np.linalg.norm(wr, axis=1)) / 2,
                         f"fs_edge_dist_{f}": edge,
                         f"fs_at_edge_{f}": np.where(np.isnan(edge), np.nan, (edge < p.edge_margin_px).astype(float))})
            axes.append(axis / blen[:, None]), thorax.append(th), length.append(blen)
        for me, other in ((0, 1), (1, 0)):
            rel = thorax[other] - thorax[me]
            perp = np.stack([-axes[me][:, 1], axes[me][:, 0]], axis=1)
            scale = np.where(length[me] > 0, length[me], np.nan)
            tag = f"{FLIES[me]}{FLIES[other]}"
            cols[f"fs_front_{tag}"] = np.einsum("ti,ti->t", rel, axes[me]) / scale
            cols[f"fs_side_{tag}"] = np.abs(np.einsum("ti,ti->t", rel, perp)) / scale
            cols[f"fs_facing_{FLIES[me]}"] = np.cos(_angle(axes[me], rel))
        cols["fs_axis_cos"] = np.einsum("ti,ti->t", axes[0], axes[1])
        heads = [arr[:, j, k("head")] for j in range(2)]
        cols["fs_head_to_other"] = np.minimum(np.linalg.norm(heads[0] - thorax[1], axis=1),
                                              np.linalg.norm(heads[1] - thorax[0], axis=1)) / np.nanmean(length, axis=0)
    out = pd.DataFrame({n: cols[n] for n in geometry_names()}).astype("float32")
    head = index.to_frame(index=False)
    head["time_s"] = flags.groupby(KEYS, sort=True)["time_s"].first().reindex(index).to_numpy()
    return pd.concat([head, out], axis=1)


def session_features(poses, p=None, setup=None):
    """Fly pose rows -> EthoLive's fly feature table: basic + motion + ScopeScore fly geometry.

    setup: a camera setup that differs from training (flyvsfly.calibrate): the points are rescaled to the training
    fly size, the picture size follows the recording, and the edge distance is capped at half the training chamber
    (72 px; the training values never exceed it, so the trained model is unchanged)."""
    p = p or FlyParams()
    if setup is not None:
        from dataclasses import replace
        poses = poses.assign(x=poses["x"] * setup["scale"], y=poses["y"] * setup["scale"])
        p = replace(p, frame_size=tuple(setup["frame_size"]), edge_cap_px=min(FRAME_SIZE) / 2)
    if p.swap_fix:
        poses = fix_head_abdomen(poses)
    flags = flag_quality(poses, p)
    motion = compute_features(flags, params=FeatureParams(**p.motion), torso_keypoints=TORSO_KEYPOINTS)
    shape = smooth_keypoints(flags, p) if p.keypoint_smoothing else flags
    feats = basic_features(shape, torso_keypoints=TORSO_KEYPOINTS, keypoints=KEYPOINTS, mice=FLIES).merge(
        motion[[c for c in motion.columns if c != "time_s"]], on=KEYS, how="inner", validate="one_to_one")
    if p.geometry:
        feats = feats.merge(geometry_features(shape, p).drop(columns="time_s"), on=KEYS, how="inner", validate="one_to_one")
    if p.tip_motion:
        feats = feats.merge(tip_motion_features(shape), on=KEYS, how="inner", validate="one_to_one")
    return feats


def tip_motion_names(flies=FLIES):
    return [f"tm_{n}_{f}" for f in flies for n in ("head_speed", "head_accel", "head_approach", "wing_rate")]


def tip_motion_features(flags):
    """ScopeScore follows the tool tip's motion; a fly's tip is its head. Per fly: head speed and its change (px/s,
    px/s^2), how fast the head closes in on the other fly's thorax (px/s, positive = closer) and how fast the wings
    open or close (rad/s). Past and current frame only; NaN across a session start or a skipped frame."""
    index, arr = _to_array(flags, list(KEYPOINTS), list(FLIES), "x", "y")             # (T, fly, kp, xy)
    k = KEYPOINTS.index
    frames = index.get_level_values("frame").to_numpy()
    sessions = index.get_level_values("session").to_numpy()
    step = np.r_[False, (sessions[1:] == sessions[:-1]) & (np.diff(frames) == 1)]       # previous row is the frame before
    fps = 30.0

    def rate(v):
        d = np.full_like(v, np.nan)
        d[1:] = (v[1:] - v[:-1]) * fps
        d[~step] = np.nan
        return d

    cols = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j, f in enumerate(FLIES):
            head = arr[:, j, k("head")]
            vx, vy = rate(head[:, 0]), rate(head[:, 1])
            speed = np.hypot(vx, vy)
            other = arr[:, 1 - j, k("thorax")]
            gap = np.linalg.norm(head - other, axis=1)
            th = arr[:, j, k("thorax")]
            back = arr[:, j, k("abdomen")] - th
            spread = _angle(back, arr[:, j, k("wing_l")] - th) + _angle(back, arr[:, j, k("wing_r")] - th)
            cols.update({f"tm_head_speed_{f}": speed, f"tm_head_accel_{f}": rate(speed),
                         f"tm_head_approach_{f}": -rate(gap), f"tm_wing_rate_{f}": rate(spread)})
    out = index.to_frame(index=False)
    for n in tip_motion_names():
        out[n] = np.asarray(cols[n], dtype="float32")
    return out
