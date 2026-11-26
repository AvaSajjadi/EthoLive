"""Pose accuracy against the MARS hand labels (scripts/accuracy_mars_handlabels.py)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mars_acc = load("accuracy_mars_handlabels")


def test_the_mars_summary_splits_touching_from_apart_and_names_the_keypoints():
    errors = np.zeros((4, 2, 7))
    errors[2:] = 12.0                                                  # the last two pictures have big errors
    touching = [False, False, True, True]
    s = mars_acc.summarize(errors, touching, with_parts=True)
    assert s["median_apart"] == 0.0 and s["median_touching"] == 12.0 and s["within_5"] == 0.5 and s["within_10"] == 0.5
    assert set(s["per_keypoint"]) == {"nose", "right_ear", "left_ear", "neck", "right_side", "left_side", "tail_base"}
    assert set(s["per_mouse"]) == {"black", "white"}


def test_the_degrade_step_keeps_the_picture_size_and_changes_it():
    img = (np.random.default_rng(0).random((570, 1024, 3)) * 255).astype(np.uint8)
    d = mars_acc.degrade(img)
    assert d.shape == img.shape and not np.array_equal(d, img)


