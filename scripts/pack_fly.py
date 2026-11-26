"""Pack EthoLive's fly file models/etholive/fly.pt (fly 0.10):

  Section 1  fly eyes: YOLO26n-pose, two cameras: scripts/fly_pose_dataset2.py + train_yolo_pose.py (models/pose/fly_pose2_best.pt)
  methods    the ScopeScore fly steps (scopescore_fly.FlyParams)
  Section 2  the team of 3 from scripts/train_fly_eyes2.py --arm swap --fold 3 (eye points, true clock, lookahead 6, Courtship hard negatives, flies a and b exchanged)

Run from the repo root:  python scripts/pack_fly.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etholive import model as etholive                                       # noqa: E402
from etholive import scopescore_fly as sf                           # noqa: E402

if __name__ == "__main__":
    eyes = etholive.eyes_from_yolo(ROOT / "models/pose/fly_pose2_best.pt", imgsz=(160, 160), conf=0.25, hold=3, kind="fly")
    team = [ROOT / f"outputs/etholive/fly_eyes2/swap_fold3/behavior_seed{s}.pt" for s in (0, 1, 2)]
    path = etholive.pack(ROOT / "models/etholive/fly.pt", eyes, sf.FlyParams().to_dict(), team, version="0.10")
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")
