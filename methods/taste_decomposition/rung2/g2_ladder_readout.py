#!/usr/bin/env python3
"""G2 pass-1 ladder readout: per-rung bank AUC (grouped-OOF, 24 criteria) +
mean univariate validity + per-criterion agreement with the Gemma-4-31b
reference scores. CPU."""
import numpy as np, json, glob, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
R='methods/taste_decomposition/rung2/'
z7=np.load('methods/taste_decomposition/closure/cw_community/round7_state.npz', allow_pickle=True)
names7=[str(s) for s in z7['bank_names']]
top=json.load(open(R+'g1_top24_cw.json')); crits=[t['name'] for t in top]
ref=np.column_stack([z7['VA'][:,names7.index(n)] for n in crits]).astype(float)
y=z7['y'].astype(int)
def oof_auc(X, groups):
    o=np.full(len(y),np.nan)
    for tr,te in GroupKFold(5).split(X,y,groups):
        imp=SimpleImputer(strategy='median',add_indicator=True)
        Xtr,Xte=imp.fit_transform(X[tr]),imp.transform(X[te])
        ps=[]
        for s in (0,1,2):
            m=HistGradientBoostingClassifier(max_iter=300,learning_rate=.06,max_leaf_nodes=15,random_state=s)
            m.fit(Xtr,y[tr]); ps.append(m.predict_proba(Xte)[:,1])
        o[te]=np.mean(ps,axis=0)
    return float(roc_auc_score(y,o))
out={}
for f in sorted(glob.glob(R+'g2_ladder_scores_cw_*.npz')):
    rung=f.split('_cw_')[1].replace('.npz','')
    z=np.load(f, allow_pickle=True)
    fn=[str(s) for s in z['form_names']]
    X=np.column_stack([z['X'][:,fn.index(f'{n}::a')] for n in crits])
    groups=np.array([str(g) for g in z['groups']])
    uni=float(np.mean([abs(roc_auc_score(y,np.nan_to_num(X[:,j],nan=-1))-.5)+.5 for j in range(24)]))
    agree=float(np.nanmean([spearmanr(X[m,j],ref[m,j]).statistic
        for j in range(24) for m in [np.isfinite(X[:,j])&np.isfinite(ref[:,j])] if m.sum()>200 and X[m,j].std()>0]))
    auc=oof_auc(X, groups)
    out[rung]={'bank_auc_oof':auc,'mean_univariate':uni,'agree_with_gemma4_ref':agree}
    print(f'{rung:12s} bank AUC {auc:.4f}  univ {uni:.4f}  agree(ref) {agree:.3f}', flush=True)
# reference: gemma-4-31b frozen scores, same 24 cols
groups7=np.array([str(g) for g in z7['groups']])
out['REF_gemma4_31b']={'bank_auc_oof':oof_auc(ref,groups7),
  'mean_univariate':float(np.mean([abs(roc_auc_score(y,np.nan_to_num(ref[:,j],nan=-1))-.5)+.5 for j in range(24)])),
  'agree_with_gemma4_ref':1.0}
print('REF gemma4-31b bank AUC', round(out['REF_gemma4_31b']['bank_auc_oof'],4), flush=True)
json.dump(out,open(R+'g2_ladder_readout_cw.json','w'),indent=1)
print('G2_READOUT_DONE')
