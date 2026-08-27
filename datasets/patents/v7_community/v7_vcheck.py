"""Dry-run the V matrix over all 16,000 rows before the bank run depends on it,
and record V-alone AUC + degenerate columns."""
import importlib.util, json, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
D="/lfs/skampere3/0/alexspan/norm-research/datasets/patents/v7_community/"
spec=importlib.util.spec_from_file_location("vf", D+"v_features.py")
vf=importlib.util.module_from_spec(spec); spec.loader.exec_module(vf)
pop=pd.read_csv(D+"population.csv.gz")
rows=[]
for i,r in enumerate(pop.itertuples()):
    try: rows.append(vf.vector(str(r.title), str(r.abstract), str(r.claim1)))
    except Exception as e: print("FAIL row",i,r.patent_id,repr(e)); sys.exit(1)
V=np.array(rows,dtype=float)
print("V matrix:",V.shape,"finite:",bool(np.isfinite(V).all()))
y=pop.y_fwd5.values.astype(int)
uni=[]
for j,n in enumerate(vf.V_NAMES):
    col=V[:,j]; sd=float(col.std())
    a=float(roc_auc_score(y,col)) if sd>0 else 0.5
    uni.append({"name":n,"auc":a,"sd":sd,"near_constant":bool(sd==0 or (np.bincount((col==col[0]).astype(int))[1]/len(col)>0.99 if sd>0 else True))})
uni.sort(key=lambda d:-abs(d["auc"]-.5))
print("\ntop 12 V columns by |AUC-.5|:")
for d in uni[:12]: print(f"  {d['name']:34s} auc={d['auc']:.4f} sd={d['sd']:.3g}")
deg=[d["name"] for d in uni if d["sd"]==0]
print("\ndegenerate (sd=0) columns:",deg)
np.save(D+"v_matrix_check.npy",V)
json.dump({"shape":list(V.shape),"degenerate":deg,"univariate":uni},
          open(D+"v_matrix_check.json","w"),indent=1)
print("V_CHECK_DONE")
