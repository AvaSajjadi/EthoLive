"""Caltech Fly vs. Fly, Aggression set: loader for EthoLive's fly pack (new file).

Data: Eyjolfsdottir et al. (2014), "Detecting social actions of fruit flies", ECCV; CaltechDATA record zrznw-w7386
(CC0). Aggression.zip: 10 movies of 2 male flies, 30 min each at 30 fps, one 144 x 144 px chamber per movie.
Per movie: <m>.mp4, <m>_track.mat (17 tracked values per fly and frame), <m>_feat.mat (36 derived features, not used:
EthoLive computes its own), <m>_actions.mat (expert bouts per fly). traintest.mat: movies 1-5 train, 6-10 test.

Keypoints. The tracker gives an ellipse per fly (centre, orientation, axis lengths) and the two wing tips. EthoLive
uses 5 points per fly, in the same long layout as the mice (INTERFACES.md section 1; the animal id column keeps its
name "mouse" so the shared code does not change):
    head     centre + half the major axis along the heading      heading = (cos ori, -sin ori): image y points down
    thorax   the ellipse centre                                    (checked: the wings lie behind it in 94% of frames)
    abdomen  centre - half the major axis along the heading
    wing_l, wing_r   the tracked wing tips
A missing fly (NaN) gives NaN points with score 0.

Labels: one label per frame for the pair, like CalMS21: the behavior of either fly. When both flies do something
different the first in PRIORITY wins (the short, rare actions first). "charge" (172 frames in all 10 movies) and
"unknown" are marked IGNORE: they are left out of training and of every score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FPS = 30.0
FRAME_SIZE = (144, 144)                       # width, height of one chamber
KEYPOINTS = ("head", "thorax", "abdomen", "wing_l", "wing_r")
FLIES = ("a", "b")                            # fly 1, fly 2 of the track file
TORSO_KEYPOINTS = ("thorax",)
SKELETON = (("head", "thorax"), ("thorax", "abdomen"), ("thorax", "wing_l"), ("thorax", "wing_r"))
BEHAVIORS = ("lunge", "wing_threat", "tussle", "hold")
PRIORITY = ("lunge", "tussle", "hold", "wing_threat", "charge", "unknown")
IGNORE = "ignore"
SPLIT = {"train": [1, 2, 4, 5], "validation": [3], "test": [6, 7, 8, 9, 10]}   # validation taken from the official train half
# Boy meets boy (BoyMeetsBoy_wo_movies.zip): 6 movies of about 2.3 h, 2 males; official split 1-3 train, 4-6 test.
# It labels touch, lunge, wing threat, charge and wing extension, but not tussle or hold.
BMB_SPLIT = {"train": [1, 2, 3], "test": [4, 5, 6]}
AS_OTHER = ("touch", "wing_extension")            # labelled in Boy meets boy, not aggression: counted as "other"
SPLIT_NOTE = ("Official Fly vs. Fly split: movies 1-5 train, 6-10 test. EthoLive keeps movie 3 of the train half for "
              "validation (most varied labels, few missing frames); the test movies are not read for any choice.")


BMB_FPS = 200.0                           # Boy meets boy: the raw movies say 200 fps, and lunges last 20 frames there, 4 at 30 fps


def to_fps(poses, labels, fps_from, fps_to=FPS):
    """(poses, labels) of a recording resampled to fps_to, so a second means the same to the features as in training:
    the frames nearest each tick of the new clock, renumbered. A lunge of 20 frames at 200 fps becomes 3."""
    last = int(poses["frame"].max())
    keep = np.unique(np.round(np.arange(0, last / fps_from, 1.0 / fps_to) * fps_from).astype(int))
    keep = keep[keep <= last]
    new = pd.Series(np.arange(len(keep)), index=keep)
    p = poses[poses["frame"].isin(keep)]
    p = p.assign(frame=p["frame"].map(new).astype(int), time_s=p["frame"].map(new) / fps_to)
    l = labels[labels["frame"].isin(keep)]
    return p, l.assign(frame=l["frame"].map(new).astype(int), time_s=l["frame"].map(new) / fps_to)


REFERENCE_BODY_PX = 19.45                 # median fly length (major axis) in the Aggression training movies


def calibrate(poses, first_frames=9000):
    """Camera setup of a new recording, from the poses alone (no labels): the scale that makes the median fly length
    in the first 5 minutes equal to the training flies', and the picture size the points fill, after scaling.
    Like setting a microscope's scale bar: the model was trained at about 19.5 px per fly."""
    early = poses[poses["frame"] < poses["frame"].min() + first_frames]
    w = early.pivot_table(index=["frame", "mouse"], columns="keypoint", values=["x", "y"])
    body = np.hypot(w["x"]["head"] - w["x"]["abdomen"], w["y"]["head"] - w["y"]["abdomen"])
    scale = REFERENCE_BODY_PX / float(np.nanmedian(body))
    size = (float(np.nanmax(poses["x"])) * scale * 1.02, float(np.nanmax(poses["y"])) * scale * 1.02)
    return {"scale": scale, "frame_size": size, "fly_length_px": float(np.nanmedian(body))}


def both_flies_seen(frame_rows):
    """One frame's pose rows: are the head and abdomen of every fly there, with a usable score?"""
    ends = frame_rows[frame_rows["keypoint"].isin(("head", "abdomen")) & (frame_rows["score"] >= 0.5)]
    return len(ends.dropna(subset=["x", "y"])) == 2 * len(FLIES)


def calibrate_live(poses, picture_size, min_frames=1800):
    """calibrate() for a live camera: the setup from the frames so far, or None until min_frames of them show both flies.
    Only such frames count: Boy meets boy recordings open with one fly for 20 to 100 seconds, at first often measured at
    half its length, which gave a scale that scored almost nothing. With both flies in view the length is steady, and a
    6% error in the scale changed little. A short stretch of flies does not fill the chamber, so the picture size comes
    from the camera, not from the points."""
    seen = poses[poses["score"] >= 0.5].dropna(subset=["x", "y"])
    ends = seen[seen["keypoint"].isin(("head", "abdomen"))]
    whole = ends.groupby(["frame", "mouse"])["keypoint"].nunique().eq(2).groupby("frame").sum()
    frames = whole.index[whole == len(FLIES)]
    if len(frames) < min_frames:
        return None
    seen = seen[seen["frame"].isin(frames)]
    setup = calibrate(seen, first_frames=int(seen["frame"].max() - seen["frame"].min()) + 1)
    return {**setup, "frame_size": (picture_size[0] * setup["scale"], picture_size[1] * setup["scale"])}


def _mat(path):
    from scipy.io import loadmat               # a plain data reader: .mat v5 files hold no code
    return loadmat(path)


def session_id(movie, part="aggression"):
    return f"flyvsfly/{part}/movie{movie}"


def load_movie(folder, movie, part="aggression"):
    """(poses, labels) of one movie. poses: session, mouse, keypoint, frame, time_s, x, y, score."""
    folder = Path(folder) / f"movie{movie}"
    trk = _mat(folder / f"movie{movie}_track.mat")["trk"][0, 0]
    names = [str(n[0]) for n in trk["names"][0]]
    data = trk["data"]                                             # (fly, frame, value)
    col = {n: data[:, :, names.index(n)] for n in names}
    n_frames = data.shape[1]
    hx, hy = np.cos(col["ori"]), -np.sin(col["ori"])
    half = col["major axis len"] / 2
    points = {"head": (col["pos x"] + half * hx, col["pos y"] + half * hy),
              "thorax": (col["pos x"], col["pos y"]),
              "abdomen": (col["pos x"] - half * hx, col["pos y"] - half * hy),
              "wing_l": (col["wing l x"], col["wing l y"]),
              "wing_r": (col["wing r x"], col["wing r y"])}
    sid = session_id(movie, part)
    frames = np.arange(n_frames)
    parts = []
    for f, fly in enumerate(FLIES):
        for k in KEYPOINTS:
            x, y = points[k][0][f], points[k][1][f]
            parts.append(pd.DataFrame({"session": sid, "mouse": fly, "keypoint": k, "frame": frames,
                                       "time_s": frames / FPS, "x": x, "y": y,
                                       "score": np.where(np.isfinite(x) & np.isfinite(y), 1.0, 0.0)}))
    poses = pd.concat(parts, ignore_index=True)
    return poses, load_labels(folder / f"movie{movie}_actions.mat", sid, n_frames)


def load_labels(path, sid, n_frames):
    """One label per frame: the behavior of either fly (PRIORITY when they differ), 'other' when none."""
    a = _mat(path)
    behs = [str(b[0]) for b in a["behs"][0]]
    label = np.full(n_frames, "other", dtype=object)
    rank = np.full(n_frames, len(PRIORITY))
    for j, b in enumerate(behs):
        if b in AS_OTHER:
            continue
        r = PRIORITY.index(b) if b in PRIORITY else len(PRIORITY) - 1
        for fly in range(a["bouts"].shape[0]):
            for start, end, *_ in np.atleast_2d(a["bouts"][fly, j]):
                s, e = int(start) - 1, int(end)                    # MATLAB: 1-based, end included
                if e <= s:
                    continue
                take = np.zeros(n_frames, bool)
                take[s:e] = True
                take &= r < rank
                label[take], rank[take] = b, r
    label = np.where(np.isin(label, BEHAVIORS + ("other",)), label, IGNORE)
    frames = np.arange(n_frames)
    return pd.DataFrame({"session": sid, "frame": frames, "time_s": frames / FPS, "label": label})


def load_split(folder, split):
    """Poses and labels of every movie in a split, one movie at a time."""
    out = [load_movie(folder, m) for m in SPLIT[split]]
    return pd.concat([p for p, _ in out], ignore_index=True), pd.concat([lab for _, lab in out], ignore_index=True)


def manifest():
    return {"train": [session_id(m) for m in SPLIT["train"]], "validation": [session_id(m) for m in SPLIT["validation"]],
            "test": [session_id(m) for m in SPLIT["test"]], "note": SPLIT_NOTE}


# --- scores -------------------------------------------------------------------------------------

def _bouts(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def metrics(scores, labels):
    """Frame precision / recall / F1 of each fly behavior (IGNORE frames left out), their mean (macro_f1), and lunge
    bouts: a true lunge is found when a predicted lunge frame touches it; a predicted lunge bout is right when it
    touches a true one (lunges last about 4 frames, so frame F1 alone undersells them)."""
    d = labels[["session", "frame", "label"]].merge(scores, on=["session", "frame"], how="inner", validate="one_to_one")
    d = d[d["label"] != IGNORE].sort_values(["session", "frame"])
    p = d[list(BEHAVIORS)].to_numpy(float)
    full = np.hstack([p, 1.0 - p.sum(axis=1, keepdims=True)])
    pred = np.asarray(BEHAVIORS + ("other",))[full.argmax(axis=1)]
    out, f1s = {"n_frames": int(len(d))}, []
    for b in BEHAVIORS:
        t, q = d["label"].to_numpy() == b, pred == b
        tp = int((t & q).sum())
        prec = tp / q.sum() if q.sum() else np.nan
        rec = tp / t.sum() if t.sum() else np.nan
        f1 = 2 * prec * rec / (prec + rec) if prec and rec and not np.isnan(prec + rec) else (0.0 if t.sum() or q.sum() else np.nan)
        out.update({f"{b}_precision": float(prec), f"{b}_recall": float(rec), f"{b}_f1": float(f1)})
        f1s.append(f1)
    out["macro_f1"] = float(np.nanmean(f1s))
    found = true_n = right = pred_n = 0
    for _, g in d.assign(pred=pred).groupby("session"):
        t, q = (g["label"] == "lunge").to_numpy(), (g["pred"] == "lunge").to_numpy()
        tb, qb = _bouts(t), _bouts(q)
        true_n += len(tb)
        pred_n += len(qb)
        found += sum(q[s:e].any() for s, e in tb)
        right += sum(t[s:e].any() for s, e in qb)
    out.update({"lunge_bouts": true_n, "lunge_bout_recall": found / true_n if true_n else np.nan,
                "lunge_bout_precision": right / pred_n if pred_n else np.nan})
    return out
