"""Bout rule used for alert-level evaluation (etholive.alerts)."""

import numpy as np
import pandas as pd
import pytest

from etholive.alerts import AlertRule, as_rule, run_rule
from etholive.evaluate import ATTACK_RULE_V1, monitor_event_metrics

FPS = 30.0


def table(scores, ok=None, session="s1", frames=None):
    frames = np.arange(len(scores)) if frames is None else np.asarray(frames)
    return pd.DataFrame({"session": session, "frame": frames, "time_s": frames / FPS,
                         "tracking_ok": True if ok is None else ok, "attack": scores,
                         "investigation": 0.0, "mount": 0.0})


def test_fires_once_after_min_duration():
    s = [0.0] * 5 + [0.9] * 40 + [0.0] * 5
    ev = run_rule(table(s), AlertRule("attack", min_duration_s=1.0, ask_threshold=0.6), fps=FPS)
    assert len(ev) == 1
    e = ev[0]
    assert (e["first_frame"], e["fired_frame"], e["last_frame"]) == (5, 35, 44)
    assert e["end_s"] == pytest.approx(45 / FPS)


def test_short_bout_does_not_fire():
    s = [0.9] * 29 + [0.0]
    assert run_rule(table(s), AlertRule("attack", min_duration_s=1.0, ask_threshold=0.6)) == []


def test_lost_tracking_ends_the_bout():
    s = [0.9] * 60
    ok = [True] * 20 + [False] + [True] * 39
    ev = run_rule(table(s, ok=ok), AlertRule("attack", min_duration_s=1.0, ask_threshold=0.6))
    assert [e["first_frame"] for e in ev] == [21]


def test_skipped_frame_number_ends_the_bout():
    frames = list(range(20)) + list(range(21, 61))
    ev = run_rule(table([0.9] * 60, frames=frames), AlertRule("attack", min_duration_s=1.0, ask_threshold=0.6))
    assert [e["first_frame"] for e in ev] == [21]


def test_missing_score_is_not_evidence():
    s = [0.9] * 10 + [np.nan] + [0.9] * 40
    ev = run_rule(table(s), AlertRule("attack", min_duration_s=1.0, ask_threshold=0.6))
    assert [e["first_frame"] for e in ev] == [11]


def test_threshold_is_inclusive():
    ev = run_rule(table([0.6] * 40), AlertRule("attack", min_duration_s=1.0, ask_threshold=0.6))
    assert len(ev) == 1


def test_sessions_do_not_join():
    a, b = table([0.9] * 20, session="a"), table([0.9] * 20, session="b")
    assert run_rule(pd.concat([a, b]), AlertRule("attack", min_duration_s=1.0, ask_threshold=0.6)) == []


def test_decision_uses_only_past_frames():
    base = [0.0] * 5 + [0.9] * 40 + [0.0] * 30
    changed = base[:36] + list(np.random.default_rng(0).random(len(base) - 36))
    rule = AlertRule("attack", min_duration_s=1.0, ask_threshold=0.6)
    fired = lambda s: [(e["first_frame"], e["fired_frame"]) for e in run_rule(table(s), rule)][:1]
    assert fired(base) == fired(changed) == [(5, 35)]


def test_rule_from_dict_and_object():
    r = as_rule(ATTACK_RULE_V1)
    assert (r.behavior, r.ask_threshold, r.min_duration_s) == ("attack", 0.6, 0.5)

    class Obj:
        behavior, min_duration_s, ask_threshold = "mount", 2.0, 0.4
    assert as_rule(Obj()).behavior == "mount"


def test_event_metrics_match_true_bouts():
    s = [0.0] * 10 + [0.9] * 30 + [0.0] * 20
    labels = pd.DataFrame({"session": "s1", "frame": np.arange(60),
                           "label": ["other"] * 12 + ["attack"] * 26 + ["other"] * 22})
    summary, bouts = monitor_event_metrics(table(s), labels, ATTACK_RULE_V1, fps=FPS)
    assert summary["n_events"] == 1 and summary["detected_true"] == 1
    assert summary["event_precision"] == 1.0
    assert bouts["fire_offset_s"].iloc[0] == pytest.approx((25 - 12) / FPS)
