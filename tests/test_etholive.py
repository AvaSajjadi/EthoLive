"""Tests for EthoLive: Section 1 (Eyes) picks and maps the mice itself, Section 2 starts as Model B (frame branch), one file.

A small fake network stands in for the YOLO network, and the poses are made up (SYNTHETIC): only the logic matters here.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from etholive import scope_model as sm
from etholive import model as sn
from etholive import scopescore_mouse as sc
from etholive.calms21 import KEYPOINTS
from etholive.classify import feature_columns
from etholive.live_scoring import LiveScorer
from pose_fixtures import SESSION, features_of, make_poses, tiny_model

ANCHORS = 10


class FakeNetwork(nn.Module):
    """Anchor 2 is a black mouse (0.9), anchor 5 a white one (0.6), anchor 7 a weaker black one (0.5)."""

    def __init__(self, white=0.6):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.white = white

    def forward(self, x):
        out = torch.zeros(x.shape[0], 27, ANCHORS)
        out[:, 4, 2], out[:, 4, 7], out[:, 5, 5] = 0.9, 0.5, self.white
        for j in range(len(KEYPOINTS)):
            out[:, 6 + 3 * j, :] = 100.0 + j          # x in network pixels
            out[:, 7 + 3 * j, :] = 14.0 + 2 * j       # y: the 14-pixel top border, then 2 px per keypoint
            out[:, 8 + 3 * j, :] = 0.8
        out[:, 6, 7] = 999.0                          # the weaker black mouse is somewhere else
        return out + self.w * 0


def test_the_picture_is_prepared_as_the_network_was_trained():
    eyes = sn.Eyes(FakeNetwork())
    x, gain, left, top = eyes.prepare(np.zeros((570, 1024, 3), np.uint8))
    assert tuple(x.shape) == (3, 384, 640) and (left, top) == (0, 14)
    assert gain == (0.625, 356 / 570)
    assert float(x[0, 0, 0]) == pytest.approx(114 / 255)          # grey border


def test_the_most_certain_mouse_of_each_colour_is_kept_and_mapped_to_the_calms21_frame():
    rows = sn.Eyes(FakeNetwork(), hold=0).find_mice(np.zeros((570, 1024, 3), np.uint8))
    assert sorted(rows["mouse"].unique()) == ["a", "b"] and len(rows) == 14
    nose_a = rows[(rows.mouse == "a") & (rows.keypoint == "nose")].iloc[0]
    assert nose_a.x == pytest.approx(100 / 0.625) and nose_a.y == pytest.approx(0.0) and nose_a.score == pytest.approx(0.8)


def test_a_mouse_below_the_confidence_is_missing_and_held_for_a_few_pictures():
    eyes = sn.Eyes(FakeNetwork(), conf=0.15, hold=2)
    picture = np.zeros((570, 1024, 3), np.uint8)
    assert set(eyes.find_mice(picture)["mouse"]) == {"a", "b"}
    eyes.network.white = 0.1                                     # the white mouse is lost
    assert [set(eyes.find_mice(picture)["mouse"]) for _ in range(3)] == [{"a", "b"}, {"a", "b"}, {"a"}]


@pytest.fixture(scope="module")
def world():
    poses = make_poses(300, seed=4)
    methods = sc.ScopeParams(swap_fix=True, geometry=True, keypoint_smoothing=True)
    feats = sc.session_features(poses, methods)
    model, bcols = tiny_model(features_of(poses))
    part = {"feature_columns": bcols, "classes": ["attack", "investigation", "mount", "other"],
            "median": model[0].statistics_, "mean": model[1].mean_, "scale": model[1].scale_,
            "coef": model[-1].coef_, "intercept": model[-1].intercept_}
    return poses, feats, methods, model, bcols, part


def test_before_training_section_2_is_exactly_the_frame_branch_start(world):
    poses, feats, _, model, bcols, part = world
    cols = feature_columns(feats)
    norm = sm.fit_normalizer(feats, [SESSION], cols)
    torch.manual_seed(0)
    net = sn.BehaviorNet(len(cols)).eval()
    sn.frame_branch_init(net, cols, norm, part)
    p = sm.predict_proba(net, sm.normalize(feats[cols].astype("float32").to_numpy(), norm))
    x = feats[bcols].astype("float32").to_numpy()
    rows = ~np.isnan(x).any(axis=1)                              # model B fills gaps with its own medians
    assert rows.sum() > 250
    assert np.abs(p[rows] - model.predict_proba(x[rows])).max() < 1e-4


def test_section_2_never_looks_at_future_frames():
    torch.manual_seed(0)
    net = sn.BehaviorNet(10).eval()
    nn.init.normal_(net.memory.out.weight)
    x = torch.randn(1, 200, 10)
    later = x.clone()
    later[:, 120:] += 5.0
    assert torch.allclose(net(x)[:, :120], net(later)[:, :120])


def one_file(tmp_path, world):
    _, feats, methods, *_ , part = world
    cols = feature_columns(feats)
    norm = sm.fit_normalizer(feats, [SESSION], cols)
    torch.manual_seed(0)
    net = sn.BehaviorNet(len(cols)).eval()
    sn.frame_branch_init(net, cols, norm, part)
    info = {"name": sn.MODEL_NAME, "n_features": len(cols), "feature_columns": cols, "species": sm.MOUSE.to_dict(),
            "classes": list(sm.MOUSE.classes), "params": {"width": 64, "dropout": 0.1}}
    sn.save_behavior(tmp_path / "behavior.pt", net, norm, info)
    return sn.pack(tmp_path / sm.MODEL_DIR / "mouse.pt", sn.Eyes(FakeNetwork()), methods.to_dict(), tmp_path / "behavior.pt")


def test_the_one_file_has_both_sections_and_the_scopescore_settings(tmp_path, world):
    blob = sn.read(one_file(tmp_path, world))
    assert blob["format"] == sn.FORMAT and set(blob["sections"]) == {"eyes", "methods", "log_levels", "behavior"} and blob["sections"]["log_levels"] is None
    assert blob["sections"]["eyes"]["conf"] == 0.15 and blob["sections"]["methods"]["swap_fix"]
    assert "MARS" in blob["offline_pose"]["name"]
    assert all(w in " ".join(blob["credits"]) for w in ("YOLO26n-pose", "MARS", "ScopeScore", "Model B", "CalMS21"))


def test_a_team_votes_with_the_mean_of_its_members_probabilities(tmp_path, world):
    _, feats, methods, *_, part = world
    cols = feature_columns(feats)
    norm = sm.fit_normalizer(feats, [SESSION], cols)
    info = {"name": sn.MODEL_NAME, "n_features": len(cols), "feature_columns": cols, "species": sm.MOUSE.to_dict(),
            "classes": list(sm.MOUSE.classes), "params": {"width": 64, "dropout": 0.1, "dilations": [1, 2, 4, 8, 16, 32]}}
    paths, nets = [], []
    for seed in range(3):
        torch.manual_seed(seed)
        net = sn.BehaviorNet(len(cols), dilations=(1, 2, 4, 8, 16, 32)).eval()
        nn.init.normal_(net.memory.out.weight, std=0.1)
        sn.save_behavior(tmp_path / f"b{seed}.pt", net, norm, dict(info, seed=seed))
        paths.append(tmp_path / f"b{seed}.pt")
        nets.append(net)
    path = sn.pack(tmp_path / "mouse.pt", sn.Eyes(FakeNetwork()), methods.to_dict(), paths)
    behavior = sn.load_behavior(path)
    assert isinstance(behavior.head, sn.Team) and behavior.info["team_seeds"] == [0, 1, 2]
    x = sm.normalize(feats[cols].astype("float32").to_numpy(), norm)
    want = np.mean([sm.predict_proba(n, x) for n in nets], axis=0)
    assert np.abs(sm.predict_proba(behavior.head, x) - want).max() < 1e-5
