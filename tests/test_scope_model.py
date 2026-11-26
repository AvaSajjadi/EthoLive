"""Tests for the scope model: animal packs, the added ScopeScore methods, model B inside the file, one file per animal.

Made-up poses (SYNTHETIC session) and an untrained head: only behavior and equality matter here, not accuracy.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from etholive import scope_model as sm
from etholive import scopescore_mouse as sc
from etholive import species as sp
from etholive.calms21 import FPS, KEYPOINTS
from etholive.classify import feature_columns
from etholive.live_scoring import LiveScorer
from pose_fixtures import SESSION, features_of, make_poses, tiny_model

ALL_ON = sc.ScopeParams(swap_fix=True, geometry=True, keypoint_smoothing=True)


def straight_mouse(frames=3, swap_at=None):
    """Mouse a lies along +x (nose at the right), mouse b sits 100 px in front of it."""
    along = {"nose": 40, "left_ear": 30, "right_ear": 30, "neck": 20, "left_hip": 0, "right_hip": 0, "tail_base": -10}
    side = {"left_ear": -5, "right_ear": 5, "left_hip": -6, "right_hip": 6}
    rows = []
    for f in range(frames):
        for mouse, x0 in (("a", 300.0), ("b", 410.0)):
            for kp in KEYPOINTS:
                name = kp
                if f == swap_at and mouse == "a" and kp in ("nose", "tail_base"):
                    name = {"nose": "tail_base", "tail_base": "nose"}[kp]
                rows.append({"session": "s", "mouse": mouse, "keypoint": name, "frame": f, "time_s": f / FPS,
                             "x": x0 + along[kp], "y": 200.0 + side.get(kp, 0), "score": 0.9})
    return pd.DataFrame(rows)


def test_the_mouse_pack_round_trips_and_names_the_calms21_setup():
    assert sp.get("mouse") is sp.MOUSE and sp.MOUSE.classes == ("attack", "investigation", "mount", "other")
    assert sp.Species.from_dict(json.loads(json.dumps(sp.MOUSE.to_dict()))) == sp.MOUSE
    with pytest.raises(ValueError):
        sp.get("bee")


def test_a_swapped_nose_and_tail_base_is_found_from_one_frame_and_put_back():
    poses = straight_mouse(swap_at=1)
    bad = sc.swapped_frames(poses)
    assert bad[("s", 1, "a")] and bad.sum() == 1
    fixed = sc.fix_nose_tail_swaps(poses).set_index(["frame", "mouse", "keypoint"])["x"]
    assert fixed[(1, "a", "nose")] == 340.0 and fixed[(1, "a", "tail_base")] == 290.0


def test_the_tool_geometry_of_a_straight_mouse():
    g = sc.geometry_features(sc.flag_quality(straight_mouse())).iloc[0]
    assert g["sc_spine_straight_a"] == pytest.approx(1.0)
    assert g["sc_spine_len_a"] == pytest.approx(50.0) and g["sc_axis_len_a"] == pytest.approx(50.0, abs=1)
    assert g["sc_axis_cos"] == pytest.approx(1.0)
    assert g["sc_front_ab"] > 0 > g["sc_front_ba"]            # b is in front of a; a is behind b
    assert g["sc_side_ab"] == pytest.approx(0.0, abs=1e-6)
    assert g["sc_at_edge_a"] == 0.0


def test_with_every_added_method_off_the_features_are_the_old_ones():
    poses = make_poses(200, seed=2)
    old = features_of(poses)
    new = sc.session_features(poses, sc.ScopeParams(swap_fix=False, geometry=False))
    assert list(new.columns) == list(old.columns)
    pd.testing.assert_frame_equal(new, old, check_dtype=False)


def test_keypoint_smoothing_uses_the_published_tip_start_and_center_settings(monkeypatch):
    seen = []
    import etholive.scopescore_mouse as mod
    real = mod.smooth_dataframe
    monkeypatch.setattr(mod, "smooth_dataframe", lambda d, **kw: seen.append((sorted(set(d["keypoint"])), kw)) or real(d, **kw))
    sc.smooth_keypoints(sc.flag_quality(make_poses(20)), sc.ScopeParams())
    kps = {tuple(k): (kw["min_cutoff"], kw["beta"]) for k, kw in seen}
    assert kps[("nose",)] == (0.6, 0.04) and kps[("tail_base",)] == (0.1, 0.001)
    assert kps[("left_ear", "left_hip", "neck", "right_ear", "right_hip")] == (0.6, 0.01)


def test_model_b_as_plain_numbers_gives_the_scikit_learn_probabilities(tmp_path):
    feats = features_of(make_poses(300, seed=4))
    model, cols = tiny_model(feats)
    joblib.dump(model, tmp_path / "b.joblib")
    (tmp_path / "b.json").write_text(json.dumps({"feature_columns": cols}))
    part = sm.model_b_part(tmp_path / "b.joblib", tmp_path / "b.json")
    want = model.predict_proba(feats[cols].astype("float32").to_numpy())
    assert np.abs(sm.model_b_proba(part, feats) - want).max() < 1e-5
    p = np.full((3, 4), 0.25)
    assert sm.fuse(p, np.eye(4)[:3], 0.0) is p
    assert np.allclose(sm.fuse(p, np.eye(4)[:3], 0.5)[0], [0.625, 0.125, 0.125, 0.125])


@pytest.fixture(scope="module")
def world():
    poses = make_poses(300, seed=4)
    feats = sc.session_features(poses, ALL_ON)
    cols = feature_columns(feats)
    torch.manual_seed(0)
    head = sm.TemporalHead(len(cols)).eval()
    norm = sm.fit_normalizer(feats, [SESSION], cols)
    info = {"name": sm.MODEL_NAME, "n_features": len(cols), "feature_columns": cols, "species": sp.MOUSE.to_dict(),
            "classes": list(sp.MOUSE.classes), "params": {"width": 64, "dropout": 0.1}}
    model, bcols = tiny_model(features_of(poses))
    part = {"feature_columns": bcols, "classes": ["attack", "investigation", "mount", "other"],
            "median": model[0].statistics_, "mean": model[1].mean_, "scale": model[1].scale_,
            "coef": model[-1].coef_, "intercept": model[-1].intercept_}
    return poses, feats, cols, head, norm, info, part


def one_file(tmp_path, world):
    _, _, _, head, norm, info, part = world
    head_path = tmp_path / "head.pt"
    sm.save_head(head_path, head, norm, info)
    weights = tmp_path / "pose.pt"
    weights.write_bytes(b"not really weights")
    return sm.pack(tmp_path / sm.MODEL_DIR / "mouse.pt", head_path, weights,
                   {"imgsz": [384, 640], "conf": 0.15, "hold": 3, "device": "mps"},
                   methods=ALL_ON.to_dict(), model_b=part, model_b_weight=0.3)


def test_the_one_file_carries_every_part(tmp_path, world):
    blob = sm.read(one_file(tmp_path, world))
    assert blob["format"] == sm.FORMAT and blob["species"]["name"] == "mouse"
    assert blob["pose"]["weights"] == b"not really weights" and blob["pose"]["settings"]["conf"] == 0.15
    assert blob["methods"]["swap_fix"] and blob["methods"]["keypoint_smoothing"] and blob["fusion"]["model_b_weight"] == 0.3
    assert blob["model_b"]["coef"].shape == (4, 102) and "MARS" in blob["offline_pose"]["name"]
    text = " ".join(blob["credits"])
    assert all(w in text for w in ("YOLO26n-pose", "ScopeScore", "MARS", "Model B", "CalMS21"))


def test_live_scores_equal_whole_session_scores_with_every_method_on(tmp_path, world):
    poses, feats, *_ = world
    behavior = sm.load_behavior(one_file(tmp_path, world))
    whole = behavior.score_buffer(feats, set(feats["frame"])).set_index("frame")
    scorer = LiveScorer(behavior, behavior.feature_columns, session=SESSION, chunk=5)
    rows = []
    for f, g in poses.groupby("frame", sort=True):
        rows += scorer.push(int(f), g)
    live = pd.DataFrame(rows + scorer.flush()).set_index("frame")
    assert list(live.index) == list(range(300))
    b = ["attack", "investigation", "mount"]
    assert np.abs(live[b].to_numpy() - whole.loc[live.index, b].to_numpy()).max() < 1e-4


def test_alarm_levels_shift_a_class_and_keep_probabilities_summing_to_one():
    p = np.array([[0.4, 0.1, 0.5], [0.2, 0.2, 0.6]])
    assert sm.apply_levels(p, None) is p
    q = sm.apply_levels(p, [-np.log(4.0), 0.0, 0.0])               # class 0 four times harder to call
    assert np.allclose(q.sum(axis=1), 1.0) and q[0, 0] == pytest.approx(0.1 / 0.7)
