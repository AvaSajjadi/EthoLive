"""Bout rule for alert-level evaluation.

An alert rule turns per-frame behavior scores into events, causally:

- a frame is evidence when tracking is ok, the score is finite and score >= ask_threshold;
- a bout is a run of evidence frames with consecutive frame numbers in one session;
  a non-evidence frame, a skipped frame number or the end of the session ends it;
- the bout fires an event on its first frame whose elapsed time since the bout's first
  frame is >= min_duration_s; a bout fires at most once.

This is the same rule as the streaming monitor in LiveTracker (``live_tracker.monitor``),
reduced to what evaluation needs: no journal, no provenance, no researcher answers.
``tests/test_alerts.py`` checks the rule on constructed cases; the equivalence with the
LiveTracker monitor on real validation scores is recorded in ``docs/reproducibility.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_TOL = 1e-9


@dataclass(frozen=True)
class AlertRule:
    behavior: str
    min_duration_s: float
    ask_threshold: float = 0.5
    alert_threshold: float = 0.8
    alert_mode: str = "alert"
    rule_id: str = "rule"


def as_rule(rule) -> AlertRule:
    """Accept an AlertRule, a dict with its fields, or any object with the same attributes."""
    if isinstance(rule, AlertRule):
        return rule
    get = rule.get if isinstance(rule, dict) else (lambda k, d=None: getattr(rule, k, d))
    return AlertRule(behavior=get("behavior"), min_duration_s=float(get("min_duration_s")),
                     ask_threshold=float(get("ask_threshold", 0.5)),
                     alert_threshold=float(get("alert_threshold", 0.8)),
                     alert_mode=get("alert_mode", "alert") or "alert", rule_id=get("rule_id", "rule") or "rule")


def run_rule(scores, rule, fps=30.0):
    """Events fired by ``rule`` on a score table (session, frame, tracking_ok, <behavior>, optional time_s).

    Rows must be in stream order (sorted by session, frame). Returns a list of dicts with
    session, first_frame, fired_frame, last_frame, start_s, fired_s, end_s (exclusive slot time).
    """
    rule = as_rule(rule)
    events = []
    for session, part in scores.groupby("session", sort=False):
        frame = part["frame"].to_numpy(dtype=np.int64)
        if "time_s" in part:
            t = part["time_s"].to_numpy(dtype=float)
        else:
            t = frame / fps
        ok = part["tracking_ok"].to_numpy(dtype=bool) if "tracking_ok" in part else np.ones(len(part), bool)
        s = part[rule.behavior].to_numpy(dtype=float)
        evidence = ok & np.isfinite(s) & (s >= rule.ask_threshold - _TOL)
        f0, t0 = frame[0], t[0]
        i, n = 0, len(part)
        while i < n:
            if not evidence[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and evidence[j + 1] and frame[j + 1] == frame[j] + 1:
                j += 1
            elapsed = t[i:j + 1] - t[i]
            hit = np.flatnonzero(elapsed >= rule.min_duration_s - _TOL)
            if hit.size:
                k = i + int(hit[0])
                events.append({"session": session, "first_frame": int(frame[i]), "fired_frame": int(frame[k]),
                               "last_frame": int(frame[j]), "start_s": float(t[i]), "fired_s": float(t[k]),
                               "end_s": float(t0 + (frame[j] + 1 - f0) / fps)})
            i = j + 1
    return events
