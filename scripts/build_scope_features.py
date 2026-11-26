"""Feature tables for the ScopeScore ablation, train and validation sessions only (never test).

Reads the CalMS21 Task 1 train file once and writes one table per variant of scopescore_mouse.ScopeParams
to data/processed_scope/<variant>.parquet, plus swap_rates.csv (how often the nose / tail base were swapped).

Run from the repo root:  python scripts/build_scope_features.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etholive import scopescore_mouse as sc                       # noqa: E402
from etholive.calms21 import _read_json, _sessions, check_manifest, load_calms21    # noqa: E402

VARIANTS = {
    "geometry": sc.ScopeParams(swap_fix=False, geometry=True),
    "swap": sc.ScopeParams(swap_fix=True, geometry=False),
    "geometry_swap": sc.ScopeParams(swap_fix=True, geometry=True),
    "geometry_swap_smooth": sc.ScopeParams(swap_fix=True, geometry=True, keypoint_smoothing=True),
}


def main():
    manifest = check_manifest(json.loads((ROOT / "splits/split_manifest.json").read_text()))
    keep = set(manifest["train"]) | set(manifest["validation"])
    out = ROOT / "data/processed_scope"
    out.mkdir(parents=True, exist_ok=True)
    recs = _sessions(_read_json(ROOT / "data/raw/calms21/calms21_task1_train.json"))
    sids = sorted(s for s in recs if s in keep)
    if set(sids) != keep:
        raise SystemExit("the train file does not hold every train / validation session")
    parts, rates = {v: [] for v in VARIANTS}, []
    for i, sid in enumerate(sids, 1):
        t0 = time.time()
        poses, _ = load_calms21({"g": {sid: recs[sid]}})
        sw = sc.swapped_frames(poses)
        rates.append({"session": sid, "split": "train" if sid in manifest["train"] else "validation",
                      "mouse_frames": len(sw), "swapped": int(sw.sum()), "rate": float(sw.mean())})
        for name, p in VARIANTS.items():
            f = sc.session_features(poses, p)
            num = [c for c in f.columns if c not in ("session", "frame", "time_s", "tracking_ok")]
            parts[name].append(f.astype({c: np.float32 for c in num}))
        print(f"[{i}/{len(sids)}] {sid}: {len(poses) // 14} frames, swaps {sw.mean():.2%} ({time.time() - t0:.0f} s)",
              flush=True)
        del poses
        gc.collect()
    for name, frames in parts.items():
        pd.concat(frames, ignore_index=True).to_parquet(out / f"{name}.parquet", index=False)
    pd.DataFrame(rates).to_csv(out / "swap_rates.csv", index=False)
    print("wrote", ", ".join(f"{n}.parquet" for n in parts))


if __name__ == "__main__":
    main()
