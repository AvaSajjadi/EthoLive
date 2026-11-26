"""Check EthoLive's fly file through the live path, with the camera measured by LiveScorer (setup="auto"), on the whole of
one validation movie, against offline scoring of the same frames. Never reads a test movie.

  agg3  Aggression 3 from the eyes' points (data/processed_fly/eyes2/agg3.parquet), the training camera
  bmb3  Boy meets boy 3, tracker points (no video), another camera: offline uses the 5-minute flyvsfly.calibrate

Run from the repo root:  python scripts/check_fly_live.py agg3|bmb3      (about 4 and 14 minutes)
"""

import sys,json,time; from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/'src')]
import numpy as np, pandas as pd
from etholive import model as etholive, flyvsfly as fv, scope_model as sm, species as sp
from etholive.live_scoring import LiveScorer
which=sys.argv[1]
b=etholive.load_behavior(ROOT/'models/etholive/fly.pt')
keep=['macro_f1','lunge_f1','wing_threat_f1','tussle_f1','hold_f1','lunge_bout_recall','lunge_bout_precision']
if which=='agg3':
    _,lab=fv.load_movie(ROOT/'data/raw/flyvsfly/Aggression',3); sid=fv.session_id(3)
    P=pd.read_parquet(ROOT/'data/processed_fly/eyes2/agg3.parquet'); size=(144,144); ref=None
else:
    P,lab=fv.load_movie(ROOT/'data/raw/flyvsfly/BoyMeetsBoy',3,part='boymeetsboy'); sid=fv.session_id(3,'boymeetsboy')
    lab=lab.assign(label=lab['label'].replace({'rest':'other'}))
    ref=fv.calibrate(P); size=(ref['frame_size'][0]/ref['scale'],ref['frame_size'][1]/ref['scale'])
sc=LiveScorer(b,b.feature_columns,session=sid,chunk=15,fps=30,setup='auto',picture_size=lambda:size)
rows=[]; t0=time.time()
for f,g in P.groupby('frame',sort=True): rows+=sc.push(int(f),g)
rows+=sc.flush(); d=pd.DataFrame(rows); first=int(d.frame.min())
print(which,'setup',sc.setup,'rows',len(d),'first scored frame',first,f'= {first/30:.0f}s','took',round(time.time()-t0),flush=True)
lab2=lab[lab['frame']>=first] if 'frame' in lab else lab
f=b.session_features(P,setup=ref); o=b.score_buffer(f,set(f['frame'])); o=o[o.frame>=first]
out={'live auto':fv.metrics(d,lab2),'offline same frames':fv.metrics(o,lab2)}
print(pd.DataFrame({k:{c:v.get(c) for c in keep} for k,v in out.items()}).T.round(3).to_string())
