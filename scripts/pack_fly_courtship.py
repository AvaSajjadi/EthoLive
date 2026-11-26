"""Pack models/etholive/fly_courtship.pt (0.2): the two-camera fly eyes, the ScopeScore fly steps, and the team of fold A from
scripts/train_fly_courtship.py --look 6 --swap. A separate file: the fight model is unchanged. Run from the repo root.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from etholive import model as etholive                                       # noqa: E402
from etholive import scopescore_fly as sf                           # noqa: E402

if __name__ == "__main__":
    eyes = etholive.eyes_from_yolo(ROOT / "models/pose/fly_pose2_best.pt", imgsz=(160, 160), conf=0.25, hold=3, kind="fly")
    team = [ROOT / f"outputs/etholive/fly_courtship/k6_foldA_swap/behavior_seed{s}.pt" for s in (0, 1, 2)]
    path = etholive.pack(ROOT / "models/etholive/fly_courtship.pt", eyes, sf.FlyParams().to_dict(), team, version="0.2")
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")
