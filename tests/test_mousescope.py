"""Tests for MouseScope: a causal head, one file that carries both parts, and the same scores live as offline.

Made-up poses (SYNTHETIC session) and an untrained head: only causality and equality matter here, not accuracy.
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from etholive import mousescope as ms
from etholive.classify import feature_columns
from etholive.live_scoring import LiveScorer
from pose_fixtures import SESSION, features_of, make_poses


@pytest.fixture(scope="module")
def world():
    feats = features_of(make_poses(300, seed=4))
    cols = feature_columns(feats)
    torch.manual_seed(0)
    head = ms.TemporalHead(len(cols)).eval()
    norm = ms.fit_normalizer(feats, [SESSION], cols)
    info = {"name": ms.MODEL_NAME, "n_features": len(cols), "feature_columns": cols,
            "params": {"width": 64, "dropout": 0.1}}
    return feats, cols, head, norm, info


def test_the_head_never_looks_at_future_frames():
    torch.manual_seed(0)
    head = ms.TemporalHead(10).eval()
    x = torch.randn(1, 200, 10)
    later = x.clone()
    later[:, 120:] += 5.0
    assert torch.allclose(head(x)[:, :120], head(later)[:, :120])


def test_the_head_sees_exactly_its_receptive_field():
    torch.manual_seed(0)
    head = ms.TemporalHead(10).eval()
    x = torch.randn(1, 200, 10)
    early = x.clone()
    early[:, :50] = 0.0
    edge = 50 + ms.RECEPTIVE_FIELD - 1                        # the first frame whose window starts after frame 49
    assert torch.allclose(head(x)[:, edge:], head(early)[:, edge:], atol=1e-6)
    assert not torch.allclose(head(x)[:, edge - 1], head(early)[:, edge - 1])


def test_missing_values_are_filled_with_the_training_median():
    feats = pd.DataFrame({"session": ["s"] * 3, "frame": [0, 1, 2], "f": [1.0, np.nan, 3.0]})
    norm = ms.fit_normalizer(feats, ["s"], ["f"])
    assert norm["median"][0] == 2.0
    assert ms.normalize(np.array([[np.nan]], "float32"), norm)[0, 0] == pytest.approx(0.0)


def test_scores_have_the_section_2_columns(world):
    feats, cols, head, norm, _ = world
    scores = ms.score_sessions(head, ms.session_arrays(feats, None, [SESSION], cols, norm))
    assert list(scores.columns) == ["session", "frame", "time_s", "tracking_ok", "attack", "investigation", "mount"]
    assert len(scores) == 300 and scores[["attack", "investigation", "mount"]].sum(axis=1).le(1 + 1e-6).all()


def one_file(tmp_path, world):
    feats, cols, head, norm, info = world
    head_path = tmp_path / "head.pt"
    ms.save_head(head_path, head, norm, info)
    weights = tmp_path / "pose.pt"
    weights.write_bytes(b"not really weights")
    out = tmp_path / "models" / "mousescope.pt"
    out.parent.mkdir(exist_ok=True)
    return ms.pack(out, head_path, weights, {"imgsz": [384, 640], "conf": 0.15, "hold": 3, "device": "mps"})


def test_the_one_file_carries_pose_weights_settings_head_and_credits(tmp_path, world):
    blob = ms.read(one_file(tmp_path, world))
    assert blob["name"] == ms.MODEL_NAME and blob["pose"]["weights"] == b"not really weights"
    assert blob["pose"]["settings"] == {"imgsz": [384, 640], "conf": 0.15, "hold": 3}     # the device is the computer's choice
    assert any("YOLO26n-pose" in c for c in blob["credits"]) and any("ScopeScore" in c for c in blob["credits"])


def test_live_scores_equal_whole_session_scores(world):
    feats, cols, head, norm, info = world
    behavior = ms.BehaviorScorer(head, norm, info)
    poses = make_poses(300, seed=4)
    whole = behavior.score_buffer(feats, set(feats["frame"])).set_index("frame")
    scorer = LiveScorer(behavior, cols, session=SESSION, chunk=5)
    rows = []
    for f, g in poses.groupby("frame", sort=True):
        rows += scorer.push(int(f), g)
    live = pd.DataFrame(rows + scorer.flush()).set_index("frame")
    assert list(live.index) == list(range(300))
    b = ["attack", "investigation", "mount"]
    assert np.abs(live[b].to_numpy() - whole.loc[live.index, b].to_numpy()).max() < 1e-5


