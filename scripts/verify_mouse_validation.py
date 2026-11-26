"""Score the CalMS21 Task 1 validation sessions (14) with the shipped mouse model file.

Validation only: never reads test sessions. Uses the cached ScopeScore feature table written by
scripts/build_scope_features.py (data/processed_scope/geometry_swap_smooth.parquet) and the labels written by
`python -m etholive.classify features` (data/processed/labels.parquet).

    python scripts/verify_mouse_validation.py [--model models/etholive/mouse.pt]

Expected (EthoLive 0.4, docs/etholive.md): macro F1 0.771, attack 0.741, investigation 0.838, mount 0.733.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etholive import model as etholive                                  # noqa: E402
from etholive import scope_model as sm                                  # noqa: E402
from etholive.calms21 import check_manifest                             # noqa: E402
from etholive.evaluate import summary_metrics                           # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="models/etholive/mouse.pt")
    ap.add_argument("--features", default="data/processed_scope/geometry_swap_smooth.parquet")
    a = ap.parse_args(argv)
    manifest = check_manifest(json.loads((ROOT / "splits/split_manifest.json").read_text()))
    labels = pd.read_parquet(ROOT / "data/processed/labels.parquet")
    labels = labels[labels["session"].isin(set(manifest["validation"]))]
    feats = pd.read_parquet(ROOT / a.features)
    feats = feats[feats["session"].isin(set(manifest["validation"]))]
    behavior = etholive.load_behavior(ROOT / a.model)
    va = sm.session_arrays(feats, labels, manifest["validation"], behavior.info["feature_columns"], behavior.norm)
    m = summary_metrics(sm.score_sessions(behavior.head, va), labels)
    keep = ["macro_f1", "attack_f1", "investigation_f1", "mount_f1"]
    print(json.dumps({k: round(float(m[k]), 4) for k in keep}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
