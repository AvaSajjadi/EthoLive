"""Tests for pose_backends: the YOLO backend (with a fake model, no weights) and the stub backend."""

import json

import numpy as np
import pandas as pd
import pytest

from etholive import pose_backends as pb
from etholive.calms21 import KEYPOINTS


class T:
    """Looks like a torch tensor for the two calls the backend makes."""
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class Boxes:
    def __init__(self, cls, conf):
        self.cls, self.conf = T(cls), T(conf)

    def __len__(self):
        return len(self.cls.a)


class Result:
    def __init__(self, cls, conf, xy, kp_conf=None):
        self.boxes = Boxes(cls, conf)
        self.keypoints = type("K", (), {"xy": T(xy), "conf": None if kp_conf is None else T(kp_conf)})()


class FakeModel:
    def __init__(self, result):
        self.result, self.seen = result, []

    def predict(self, image, **kw):
        self.seen.append((image.shape, kw))
        return [self.result]


def two_mice():
    xy = np.zeros((2, 7, 2))
    xy[0] = [[100 + j, 200] for j in range(7)]                         # black mouse
    xy[1] = [[700 - j, 300] for j in range(7)]                         # white mouse
    return Result([0, 1], [0.9, 0.8], xy, kp_conf=np.full((2, 7), 0.7))


def test_the_stub_backend_follows_the_clock_and_is_not_a_model(monkeypatch, tmp_path):
    rows = [{"mouse": m, "keypoint": k, "frame": f, "x": float(f), "y": 1.0, "score": 0.9}
            for f in range(100, 106) for m in "ab" for k in KEYPOINTS]
    pd.DataFrame(rows).to_parquet(tmp_path / "p.parquet", index=False)
    clock = iter([10.0, 10.0, 10.0 + 2 / 30.0, 99.0])
    monkeypatch.setattr(pb.time, "perf_counter", lambda: next(clock))
    fn = pb.stub_factory({"poses": str(tmp_path / "p.parquet"), "start_frame": 100}, tmp_path)
    img = np.zeros((4, 4, 3), np.uint8)
    assert fn(img)["x"].iloc[0] == 100.0                                # the first call is frame 100
    assert fn(img)["x"].iloc[0] == 100.0
    assert fn(img)["x"].iloc[0] == 102.0                                # two frame times later
    assert fn(img)["x"].iloc[0] == 105.0                                # past the end: the last frame


