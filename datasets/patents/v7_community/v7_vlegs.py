"""V-only legs for V7, using the FROZEN layer-1 estimators (imported, not
reimplemented) so they are protocol-identical to the final ledger. Also the
STRUCT block and the V+STRUCT leg."""
import importlib.util, json, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0,"/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition")
import layer1_gemma_cells as L
import scaleupC_layer1 as SC
D="/lfs/skampere3/0/alexspan/norm-research/datasets/patents/v7_community/"
spec=importlib.util.spec_from_file_location("vf",D+"v_features.py")
vf=importlib.util.module_from_spec(spec); spec.loader.exec_module(vf)
pop=pd.read_csv(D+"population.csv.gz")
V=np.array([vf.vector(str(r.title),str(r.abstract),str(r.claim1)) for r in pop.itertuples()],dtype=float)
y=pop.y_fwd5.values.astype(int); groups=pop.family_group.values.astype(object)
nc=pd.to_numeric(pop.num_claims,errors="coerce").fillna(0).values.astype(float)
STRUCT=np.column_stack([nc,pop.text.str.len().values,pop.claim1.str.len().values,pop.abstract.str.len().values]).astype(float)
folds=L.outer_folds(len(y),groups,n_splits=5)
out={"n":int(len(y)),"pos_rate":float(y.mean()),"n_groups":int(len(set(groups))),"n_V":V.shape[1]}
for nm,M in [("V",V),("STRUCT",STRUCT),("V_STRUCT",np.column_stack([V,STRUCT]))]:
    auc,oof=L.linear_oof_family1(M,y,groups,folds)
    out[nm+"_lin"]=auc
    out[nm+"_lin_ci95"]=SC.group_bootstrap_auc(y,groups,oof)
    print(f"{nm+'_lin':14s} {auc:.4f}  CI {out[nm+'_lin_ci95']}")
for nm,M in [("V",V),("V_STRUCT",np.column_stack([V,STRUCT]))]:
    seeds=[L.gbm_oof_family1(M,y,groups,folds,s)["auc"] for s in L.GBM_SEEDS]
    out[nm+"_nl_seeds"]=seeds; out[nm+"_nl_mean"]=float(np.mean(seeds))
    print(f"{nm+'_nl':14s} {np.mean(seeds):.4f}  seeds {[round(s,4) for s in seeds]}")
out["V_interact"]=out["V_nl_mean"]-out["V_lin"]
json.dump(out,open(D+"v_legs.json","w"),indent=1)
print("\n",json.dumps(out,indent=1))
print("V_LEGS_DONE")
