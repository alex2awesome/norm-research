#!/usr/bin/env python3
"""ADDENDUM G1 surface readout (see design doc). CPU, ~30-60 min."""
import numpy as np, json, itertools, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
R='methods/taste_decomposition/rung2/'
z=np.load(R+'g1_form_scores_cw.npz', allow_pickle=True)
z7=np.load('methods/taste_decomposition/closure/cw_community/round7_state.npz', allow_pickle=True)
names7=[str(s) for s in z7['bank_names']]
top=json.load(open(R+'g1_top24_cw.json')); crits=[t['name'] for t in top]
y=z['y'].astype(int); groups=np.array([str(g) for g in z['groups']])
fa=np.column_stack([z7['VA'][:,names7.index(n)] for n in crits]).astype(float)
fn=[str(s) for s in z['form_names']]
fb=np.column_stack([z['X'][:,fn.index(f'{n}::b')] for n in crits])
fc=np.column_stack([z['X'][:,fn.index(f'{n}::c')] for n in crits])
FORMS={'a':fa,'b':fb,'c':fc}
out={}
rel=[]; val={f:[] for f in FORMS}
for j,n in enumerate(crits):
    cors=[]
    for f1,f2 in itertools.combinations('abc',2):
        m=np.isfinite(FORMS[f1][:,j])&np.isfinite(FORMS[f2][:,j])
        cors.append(float(spearmanr(FORMS[f1][m,j],FORMS[f2][m,j]).statistic))
    rel.append(float(np.mean(cors)))
    for f in FORMS:
        col=np.nan_to_num(FORMS[f][:,j],nan=-1)
        val[f].append(float(abs(roc_auc_score(y,col)-.5)+.5))
out['reliability']={'per_criterion':dict(zip(crits,rel)),
                    'median':float(np.median(rel)),
                    'range':[float(min(rel)),float(max(rel))]}
out['univariate_validity_by_form']={f:float(np.mean(v)) for f,v in val.items()}
dis=np.minimum(np.array(val['a'])/np.sqrt(np.clip(rel,.05,1)),1)
out['disattenuated_a_mean']=float(np.mean(dis))
def oof_auc(X):
    o=np.full(len(y),np.nan)
    for tr,te in GroupKFold(5).split(X,y,groups):
        imp=SimpleImputer(strategy='median',add_indicator=True)
        Xtr,Xte=imp.fit_transform(X[tr]),imp.transform(X[te])
        ps=[]
        for s in (0,1,2):
            m=HistGradientBoostingClassifier(max_iter=300,learning_rate=.06,
                                             max_leaf_nodes=15,random_state=s)
            m.fit(Xtr,y[tr]); ps.append(m.predict_proba(Xte)[:,1])
        o[te]=np.mean(ps,axis=0)
    return float(roc_auc_score(y,o))
subsets={1:[('a',),('b',),('c',)],2:[('a','b'),('a','c'),('b','c')],3:[('a','b','c')]}
surface={}
for k in (4,8,16,24):
    for J,subs in subsets.items():
        aucs=[oof_auc(np.nanmean(np.stack([FORMS[f][:,:k] for f in sub]),axis=0))
              for sub in subs]
        surface[f'{k},{J}']=float(np.mean(aucs))
        print(f'k={k} J={J}: {surface[f"{k},{J}"]:.4f}',flush=True)
out['surface']=surface
Js=np.array([1,2,3])
for k in (8,24):
    A=np.array([surface[f'{k},{j}'] for j in Js])
    co=np.polyfit(1/Js,A,1)
    out[f'SB_extrapolation_k{k}']={'auc_by_J':A.tolist(),'A_inf':float(co[1]),
                                   'slope':float(co[0])}
g8=surface['8,3']-surface['8,1']; g24=surface['24,3']-surface['24,1']
out['cross_partial']={'J_gain_k8':float(g8),'J_gain_k24':float(g24),
                      'conjecture_confirmed':bool(g8>g24)}
out['dense_T_reference']=.792
json.dump(out,open(R+'g1_surface_cw.json','w'),indent=1)
print('G1_SURFACE_DONE',flush=True)
