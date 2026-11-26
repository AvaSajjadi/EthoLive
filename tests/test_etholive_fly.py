"""Tests for EthoLive's fly pack: Fly vs. Fly loading, the ScopeScore fly steps, fly scores, a fly model file.

Made-up flies (SYNTHETIC) except the last test, which reads the real Fly vs. Fly files when they are on this computer.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.io import savemat

from etholive import model as etholive
from etholive import flyvsfly as fv
from etholive import scope_model as sm
from etholive import scopescore_fly as sf
from etholive import species as sp
from etholive.classify import feature_columns
from etholive.live_scoring import LiveScorer, LiveScoringError

SESSION = "SYNTHETIC_fly"
DATA = Path(__file__).resolve().parents[1] / "data/raw/flyvsfly/Aggression"
MODEL_FILE = Path(__file__).resolve().parents[1] / "models/etholive/fly.pt"


def fly_rows(frame, fly, cx, cy, heading, spread=0.2, flip=False):
    d = np.array([np.cos(heading), -np.sin(heading)])
    perp = np.array([-d[1], d[0]])
    th = np.array([cx, cy])
    pts = {"head": th + 9 * d, "thorax": th, "abdomen": th - 9 * d,
           "wing_l": th - 15 * np.cos(spread) * d + 15 * np.sin(spread) * perp,
           "wing_r": th - 15 * np.cos(spread) * d - 15 * np.sin(spread) * perp}
    if flip:
        pts["head"], pts["abdomen"] = pts["abdomen"], pts["head"]
    return [{"session": SESSION, "mouse": fly, "keypoint": k, "frame": frame, "time_s": frame / fv.FPS,
             "x": float(v[0]), "y": float(v[1]), "score": 1.0} for k, v in pts.items()]


def make_flies(n, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(40, 100, size=(2, 2))
    head = rng.uniform(-np.pi, np.pi, 2)
    rows = []
    for f in range(n):
        pos = np.clip(pos + rng.normal(0, 1.0, (2, 2)), 20, 124)
        head = head + rng.normal(0, 0.1, 2)
        for j, fly in enumerate(fv.FLIES):
            rows += fly_rows(f, fly, *pos[j], head[j], spread=0.2 + 0.6 * (rng.random() < 0.1))
    return pd.DataFrame(rows)


def test_the_head_is_the_end_away_from_the_wings():
    poses = pd.DataFrame(fly_rows(0, "a", 70, 70, 0.0) + fly_rows(0, "b", 100, 70, np.pi)
                         + fly_rows(1, "a", 70, 70, 0.0, flip=True) + fly_rows(1, "b", 100, 70, np.pi))
    bad = sf.swapped_frames(poses)
    assert bad[(SESSION, 1, "a")] and bad.sum() == 1
    fixed = sf.fix_head_abdomen(poses).set_index(["frame", "mouse", "keypoint"])["x"]
    assert fixed[(1, "a", "head")] == pytest.approx(79.0)


def test_wing_angles_and_facing_of_two_flies_face_to_face():
    poses = pd.DataFrame(fly_rows(0, "a", 50, 70, 0.0, spread=0.2) + fly_rows(0, "b", 80, 70, np.pi, spread=0.8))
    g = sf.geometry_features(sf.flag_quality(poses, sf.FlyParams())).iloc[0]
    assert g["fs_wing_l_ang_a"] == pytest.approx(0.2, abs=1e-6) and g["fs_wing_spread_b"] == pytest.approx(1.6, abs=1e-6)
    assert g["fs_facing_a"] == pytest.approx(1.0) and g["fs_facing_b"] == pytest.approx(1.0)
    assert g["fs_axis_cos"] == pytest.approx(-1.0) and g["fs_body_len_a"] == pytest.approx(18.0)
    assert g["fs_front_ab"] > 0 and g["fs_side_ab"] == pytest.approx(0.0, abs=1e-6)


def test_labels_keep_the_rarer_action_and_ignore_charge_and_unknown(tmp_path):
    behs = np.empty((1, 4), dtype=object)
    behs[0, :] = ["lunge", "wing_threat", "charge", "unknown"]
    bouts = np.empty((2, 4), dtype=object)
    for i in range(2):
        for j in range(4):
            bouts[i, j] = np.zeros((0, 3), np.uint16)
    bouts[0, 1] = np.array([[1, 10, 0]], np.uint16)           # wing threat, frames 0-9
    bouts[1, 0] = np.array([[5, 6, 0]], np.uint16)            # the other fly lunges at frames 4-5
    bouts[0, 2] = np.array([[15, 16, 0]], np.uint16)          # charge
    bouts[1, 3] = np.array([[18, 18, 0]], np.uint16)          # unknown
    savemat(tmp_path / "a.mat", {"behs": behs, "bouts": bouts})
    lab = fv.load_labels(tmp_path / "a.mat", "s", 20)["label"].tolist()
    assert lab[:4] == ["wing_threat"] * 4 and lab[4:6] == ["lunge"] * 2 and lab[6:10] == ["wing_threat"] * 4
    assert lab[14:16] == [fv.IGNORE] * 2 and lab[17] == fv.IGNORE and lab[10] == "other"


def test_fly_scores_leave_out_ignored_frames_and_count_lunge_bouts():
    labels = pd.DataFrame({"session": "s", "frame": range(8),
                           "label": ["other", "lunge", "lunge", "other", fv.IGNORE, "lunge", "other", "other"]})
    scores = labels[["session", "frame"]].copy()
    for b in fv.BEHAVIORS:
        scores[b] = 0.0
    scores.loc[[1, 4, 6], "lunge"] = 0.9                      # finds bout 1, misses bout 2, one false bout at 6
    m = fv.metrics(scores, labels)
    assert m["n_frames"] == 7
    assert m["lunge_bouts"] == 2 and m["lunge_bout_recall"] == 0.5 and m["lunge_bout_precision"] == 0.5
    assert m["lunge_precision"] == 0.5 and m["lunge_recall"] == pytest.approx(1 / 3)


def test_the_fly_pack_is_registered():
    fly = sp.get("fly")
    assert fly.behaviors == fv.BEHAVIORS and fly.keypoints == fv.KEYPOINTS and fly.classes[-1] == "other"
    assert sp.Species.from_dict(fly.to_dict()) == fly


def test_a_fly_file_has_no_eyes_yet_and_scores_live_like_whole_sessions(tmp_path):
    poses = make_flies(300, seed=3)
    p = sf.FlyParams()
    feats = sf.session_features(poses, p)
    cols = feature_columns(feats)
    norm = sm.fit_normalizer(feats, [SESSION], cols)
    torch.manual_seed(0)
    net = etholive.BehaviorNet(len(cols), len(sp.FLY.classes)).eval()
    torch.nn.init.normal_(net.memory.out.weight, std=0.1)
    info = {"name": etholive.MODEL_NAME, "n_features": len(cols), "feature_columns": cols, "species": sp.FLY.to_dict(),
            "classes": list(sp.FLY.classes), "params": {"width": 64, "dropout": 0.1}}
    etholive.save_behavior(tmp_path / "b.pt", net, norm, info)
    path = etholive.pack(tmp_path / "fly.pt", None, p.to_dict(), tmp_path / "b.pt")
    with pytest.raises(ValueError, match="no Section 1"):
        etholive.load_eyes(path)
    behavior = etholive.load_behavior(path)
    assert behavior.species.name == "fly" and "Fly vs. Fly" in " ".join(etholive.read(path)["credits"])
    whole = behavior.score_buffer(feats, set(feats["frame"])).set_index("frame")
    assert list(whole.columns[-4:]) == list(fv.BEHAVIORS)
    scorer = LiveScorer(behavior, behavior.feature_columns, session=SESSION, chunk=5)
    rows = []
    for f, g in poses.groupby("frame", sort=True):
        rows += scorer.push(int(f), g)
    assert len(rows + scorer.flush()) > 0


@pytest.mark.skipif(not (DATA / "movie1/movie1_track.mat").is_file(), reason="Fly vs. Fly is not on this computer")
def test_the_real_fly_vs_fly_files_load():
    poses, labels = fv.load_movie(DATA, 1)
    assert len(labels) == 54000 and len(poses) == 54000 * 2 * len(fv.KEYPOINTS)
    assert set(labels["label"]) <= set(fv.BEHAVIORS) | {"other", fv.IGNORE}
    assert sf.swapped_frames(poses[poses.frame < 3000]).mean() < 0.05


def test_partial_labels_only_ask_for_one_of_the_allowed_classes():
    w = np.array([2.0, 1.0, 3.0, 0.5, 0.2])
    torch.manual_seed(0)
    x, y = torch.randn(40, 5), torch.randint(0, 5, (40,))
    y[0] = -100
    plain = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32), ignore_index=-100)(x, y)
    assert torch.allclose(etholive._PartialLoss(w, [], 4, "cpu")(x, y), plain)
    loss = etholive._PartialLoss(w, [[2, 3, 4]], 4, "cpu")
    rest = torch.tensor([5])
    sure_tussle = torch.tensor([[-9.0, -9.0, 9.0, -9.0, -9.0]])        # all on tussle: allowed, so almost no loss
    sure_lunge = torch.tensor([[9.0, -9.0, -9.0, -9.0, -9.0]])         # all on lunge: not allowed
    assert float(loss(sure_tussle, rest)) < 1e-6 < 10 < float(loss(sure_lunge, rest))


def test_calibration_rescales_to_the_training_fly_size():
    poses = make_flies(50, seed=1)
    big = poses.assign(x=poses["x"] * 1.5, y=poses["y"] * 1.5)
    setup = fv.calibrate(big)
    assert setup["scale"] == pytest.approx(fv.REFERENCE_BODY_PX / 27.0, rel=1e-6)   # the made-up flies are 18 px, x 1.5


class FakeFlyNetwork(torch.nn.Module):
    """3 anchors at 160 x 160: fly near x=40 (0.9), fly near x=120 (0.8), a duplicate of the first (0.85)."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.centers = [40.0, 120.0, 41.0]

    def forward(self, x):
        out = torch.zeros(x.shape[0], 20, 3)
        for j, (cx, s) in enumerate(zip(self.centers, (0.9, 0.8, 0.85))):
            out[:, 0:4, j] = torch.tensor([cx, 80.0, 20.0, 10.0])
            out[:, 4, j] = s
            for k in range(5):
                out[:, 5 + 3 * k, j], out[:, 6 + 3 * k, j], out[:, 7 + 3 * k, j] = cx + k, 80.0, 0.9
        return out + self.w * 0


def test_fly_eyes_find_two_flies_skip_the_duplicate_and_keep_who_is_who():
    eyes = etholive.FlyEyes(FakeFlyNetwork(), imgsz=(160, 160), conf=0.25, hold=0)
    picture = np.zeros((144, 144, 3), np.uint8)
    rows = eyes.find_mice(picture)
    th = rows[rows.keypoint == "thorax"].set_index("mouse")["x"]
    assert len(rows) == 10 and th["a"] < th["b"]                       # first sight: left to right; duplicate skipped
    eyes.network.centers = [120.0, 40.0, 121.0]                         # the network lists them the other way round
    th2 = eyes.find_mice(picture).query("keypoint == 'thorax'").set_index("mouse")["x"]
    assert th2["a"] == pytest.approx(th["a"]) and th2["b"] == pytest.approx(th["b"])   # identity kept by position
    assert th["a"] == pytest.approx((40 + 1) * 144 / 160, abs=1e-4)                     # 144 px picture scaled to 160, no border


def fly_behavior(tmp_path, poses):
    """A small untrained fly file, enough to run the live path."""
    p = sf.FlyParams()
    feats = sf.session_features(poses, p)
    cols = feature_columns(feats)
    torch.manual_seed(0)
    net = etholive.BehaviorNet(len(cols), len(sp.FLY.classes)).eval()
    info = {"name": etholive.MODEL_NAME, "n_features": len(cols), "feature_columns": cols, "species": sp.FLY.to_dict(),
            "classes": list(sp.FLY.classes), "params": {"width": 64, "dropout": 0.1}}
    etholive.save_behavior(tmp_path / "b.pt", net, sm.fit_normalizer(feats, [SESSION], cols), info)
    path = etholive.pack(tmp_path / "etholive/fly.pt", None, p.to_dict(), tmp_path / "b.pt")
    return etholive.load_behavior(path), path


def run_live(scorer, poses):
    rows = []
    for f, g in poses.groupby("frame", sort=True):
        rows += scorer.push(int(f), g)
    return rows + scorer.flush()


def test_a_live_camera_is_measured_before_scoring(tmp_path):
    poses = make_flies(200, seed=5)
    behavior, _ = fly_behavior(tmp_path, poses)
    big = poses.assign(x=poses["x"] * 1.5, y=poses["y"] * 1.5)
    picture = (216, 216)
    scorer = LiveScorer(behavior, behavior.feature_columns, session=SESSION, chunk=5, setup="auto",
                        picture_size=lambda: picture, calibrate_s=3.0)
    rows = run_live(scorer, big)
    assert [r["frame"] for r in rows] == list(range(200))       # the frames that waited are scored too
    assert scorer.setup["scale"] == pytest.approx(fv.REFERENCE_BODY_PX / 27.0, rel=1e-6)
    assert scorer.setup["frame_size"] == pytest.approx([s * scorer.setup["scale"] for s in picture])
    fixed = LiveScorer(behavior, behavior.feature_columns, session=SESSION, chunk=5, setup=scorer.setup)
    assert [r["lunge"] for r in rows[100:]] == pytest.approx([r["lunge"] for r in run_live(fixed, big)[100:]], abs=1e-3)


def test_only_frames_with_both_flies_measure_the_camera():
    poses = make_flies(100, seed=8)
    opening = poses[(poses["frame"] < 40) & (poses["mouse"] == "a")]                  # one fly, measured at half its length
    ends = opening["keypoint"].isin(("head", "abdomen"))
    short = opening.assign(x=np.where(ends, opening["x"] * 0.5 + 35, opening["x"]))
    mixed = pd.concat([short, poses[poses["frame"] >= 40]], ignore_index=True)
    assert fv.calibrate_live(mixed, (144, 144), min_frames=61) is None
    assert fv.calibrate_live(mixed, (144, 144), min_frames=60)["fly_length_px"] == pytest.approx(18.0, rel=1e-6)


def test_the_camera_is_not_measured_while_no_fly_is_seen(tmp_path):
    poses = make_flies(40, seed=6)
    assert fv.calibrate_live(poses, (144, 144), min_frames=41) is None
    assert fv.calibrate_live(poses.assign(score=0.0), (144, 144), min_frames=10) is None
    assert fv.calibrate_live(poses, (144, 144), min_frames=40)["frame_size"][0] > 0
    behavior, _ = fly_behavior(tmp_path, make_flies(100, seed=6))
    with pytest.raises(LiveScoringError, match="picture_size"):
        LiveScorer(behavior, behavior.feature_columns, session=SESSION, setup="auto")


def test_fly_eyes_skip_a_spot_known_not_to_be_a_fly():
    eyes = etholive.FlyEyes(FakeFlyNetwork())
    eyes.ignore = [(100.0, 50.0)]
    assert not eyes._not_ignored((100.0 + 3, 50.0), (1.0, 1.0), 0, 0) and eyes._not_ignored((120.0, 50.0), (1.0, 1.0), 0, 0)


def test_fly_eyes_keep_a_big_picture_big():
    eyes = etholive.FlyEyes(FakeFlyNetwork())
    shape = lambda h, w: tuple(eyes.prepare(np.zeros((h, w, 3), np.uint8))[0].shape)       # noqa: E731
    assert shape(144, 144) == (3, 160, 160)                                # the training chamber: as trained
    assert shape(480, 640) == (3, 480, 640)
    assert shape(2160, 3840)[1:] == (736, 1280)


def test_fly_eyes_learn_a_new_camera_from_the_fly_size():
    eyes = etholive.FlyEyes(FakeFlyNetwork())
    both = lambda length: {f: [{"keypoint": "head", "x": 0.0, "y": 0.0}, {"keypoint": "abdomen", "x": length, "y": 0.0}] for f in fv.FLIES}   # noqa: E731
    eyes.frame_size = (144, 144)
    eyes.learn_size(both(30.0))
    assert not eyes._lengths                                               # the training chamber is never rescaled
    eyes.frame_size = (640, 480)
    eyes.learn_size({"a": both(30.0)["a"]})
    assert not eyes._lengths                                               # one fly: not measured
    for _ in range(eyes.SIZE_PICTURES):
        eyes.learn_size(both(30.0))
    assert eyes.scale == pytest.approx(fv.REFERENCE_BODY_PX / 30.0)
    assert eyes.input_size(480, 640) == (320, 416)
    start = eyes.scale
    for _ in range(eyes.SIZE_PICTURES):
        eyes.learn_size(both(30.3))
    assert eyes.scale == start                                             # a small change is left alone
    eyes.scale = None
    for short, long in ((12.0, 29.0),):                                    # a misleading opening does not stop it learning
        for length in (short, long):
            for _ in range(eyes.SIZE_PICTURES):
                eyes.learn_size(both(length))
    assert eyes.scale == pytest.approx(fv.REFERENCE_BODY_PX / 29.0)
    eyes.learning = False
    eyes.learn_size(both(10.0))
    assert not eyes._lengths


def test_a_fast_recording_is_resampled_to_the_training_clock():
    poses = make_flies(201, seed=11)
    labels = pd.DataFrame({"session": SESSION, "frame": range(201), "time_s": np.arange(201) / 200.0, "label": "other"})
    labels.loc[100:119, "label"] = "lunge"                                 # 20 frames at 200 fps
    p, l = fv.to_fps(poses, labels, 200.0)
    assert sorted(p["frame"].unique()) == list(range(30)) == list(l["frame"])
    assert l["time_s"].iloc[3] == pytest.approx(0.1)
    assert (l["label"] == "lunge").sum() == 3                              # 0.1 s at 30 fps


def test_a_lookahead_gives_each_frame_the_output_from_later_and_live_equals_whole(tmp_path):
    poses = make_flies(120, seed=9)
    plain, path = fly_behavior(tmp_path, poses)
    ahead = etholive.load_behavior(path)
    ahead.lookahead = 4
    feats = plain.session_features(poses)
    every = set(feats["frame"])
    a = plain.score_buffer(feats, every).set_index("frame")
    b = ahead.score_buffer(feats, every).set_index("frame")
    assert list(b.index) == list(range(116))                                # the last 4 frames have no later output
    assert b.loc[10, "lunge"] == a.loc[14, "lunge"] and b.loc[10, "time_s"] == a.loc[10, "time_s"]
    rows = run_live(LiveScorer(ahead, ahead.feature_columns, session=SESSION, chunk=5, buffer_s=10.0), poses)
    assert [r["frame"] for r in rows] == list(range(116))
    assert [r["lunge"] for r in rows] == pytest.approx(list(b["lunge"]), abs=1e-5)


def test_a_score_waits_for_its_lookahead(tmp_path):
    poses = make_flies(40, seed=10)
    behavior, _ = fly_behavior(tmp_path, poses)
    behavior.lookahead = 6
    scorer = LiveScorer(behavior, behavior.feature_columns, session=SESSION, chunk=1)
    for f, g in poses.groupby("frame", sort=True):
        for r in scorer.push(int(f), g):
            assert r["frame"] == f - 6


@pytest.mark.skipif(not (MODEL_FILE).is_file(), reason="the fly model file is not on this computer")
def test_the_shipped_fly_file_is_0_10_with_eyes_a_team_and_a_lookahead():
    blob = etholive.read(MODEL_FILE)
    behavior = etholive.load_behavior(MODEL_FILE)
    assert blob["version"] == "0.10" and blob["sections"]["eyes"]["kind"] == "fly" and behavior.lookahead == 6
    assert isinstance(behavior.head, etholive.Team) and behavior.info["team_seeds"] == [0, 1, 2]
    assert "--arm swap" in behavior.info["features"] and "Boy meets boy" in " ".join(blob["credits"])


@pytest.mark.skipif(not (MODEL_FILE.with_name("fly_courtship.pt")).is_file(), reason="the courtship file is not on this computer")
def test_the_courtship_file_is_a_fly_file_with_its_own_behaviors():
    b = etholive.load_behavior(MODEL_FILE.with_name("fly_courtship.pt"))
    assert b.species.name == "fly_courtship" and b.lookahead == 6 and "--swap" in b.info["features"] and "copulation" in b.species.behaviors
    assert len(b.session_features(make_flies(40, seed=13)).columns) > 20          # the fly steps, not the mouse ones
