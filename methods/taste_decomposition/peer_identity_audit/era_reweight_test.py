#!/usr/bin/env python3
"""Peer community residual — DISCRIMINATING TEST between the two remaining accounts
(2026-08-13, user: "what is ANY other explanation for the big taste gap?").

H-reweight (criterion nonstationarity): the criteria are fine but their weights/
directions drift by era; a pooled stack (fit on recent-heavy data) misreads 2013-2019.
  Prediction: a bank stack FIT ONLY ON PRE-2022 ROWS recovers most of the dense edge
  on pre-2022 held-out rows.
H-vocabulary/configural: pre-2022 quality speaks through channels the bank cannot
express at all (missing vocabulary, or configural combinations no marginal criterion
carries). Prediction: the within-era refit stays near the pooled bank's level.

Also: JUDGE CALIBRATION BY ERA — per-era criterion score dispersion (does the Gemma
judge score old abstracts less discriminatively?).

Frame: full revealed Layer-1 matrix (V17+A154), grouped-OOF (GroupKFold(5) by ntitle)
HistGB (frozen grid, seeds 0-2 mean) WITHIN the pre-2022 subpopulation; dense compared
on the dense-held-out ∩ pre-2022 rows. Descriptive.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

R = Path('/Users/spangher/Projects/stanford-research/norm-research')
z = np.load(R / 'datasets/peer-review/vat_3y/union_scores.npz', allow_pickle=True)
X_all, V_all = z['X'], z['V']
nt_all = [str(n) for n in z['ntitle']]
pos = {n: i for i, n in enumerate(nt_all)}

rows = [json.loads(l) for l in open(R / 'datasets/peer-review/vat_3y/revealed.jsonl')]
recs = []
for r in rows:
    try:
        yv = float(r['judgement'])
    except (TypeError, ValueError):
        continue
    if yv not in (0.0, 1.0) or r['ntitle'] not in pos:
        continue
    recs.append((r['ntitle'], int(yv), int(r['year']), r.get('split')))
print(f"revealed rows in matrix: {len(recs)}")

nt = [r[0] for r in recs]
y = np.array([r[1] for r in recs])
yr = np.array([r[2] for r in recs])
sp = np.array([r[3] for r in recs])
idx = np.array([pos[n] for n in nt])
M = np.column_stack([V_all[idx], X_all[idx]])
M = np.where(np.isnan(M), np.nanmedian(M, axis=0), M)
g = np.array(nt)

def gbm_oof(X_, y_, g_, seeds=(0, 1, 2)):
    oofs = []
    for s in seeds:
        oof = np.zeros(len(y_))
        for tr, te in GroupKFold(5).split(X_, groups=g_):
            m = HistGradientBoostingClassifier(max_leaf_nodes=31, learning_rate=.06,
                                               max_iter=400, early_stopping=True,
                                               validation_fraction=.1, n_iter_no_change=20,
                                               random_state=s)
            m.fit(X_[tr], y_[tr])
            oof[te] = m.predict_proba(X_[te])[:, 1]
        oofs.append(oof)
    return np.mean(oofs, axis=0)

out = {}
for name, mask in (("pre2022", yr <= 2021), ("2022plus", yr >= 2022), ("pooled", yr > 0)):
    o = gbm_oof(M[mask], y[mask], g[mask])
    a = float(roc_auc_score(y[mask], o))
    out[name] = {"n": int(mask.sum()), "bank_within_fit_oof": a}
    if name != "pooled":
        # pooled-fit stack read on this band (the F2-style pooled OOF restricted)
        op = gbm_oof(M, y, g)
        out[name]["bank_pooled_fit_on_band"] = float(roc_auc_score(y[mask], op[mask]))
    print(name, out[name], flush=True)

# dense comparison on held-out ∩ pre-2022 (dense preds: use the t0_rows E dense)
import importlib.util, sys
spec = importlib.util.spec_from_file_location('f2x', R / 'methods/taste_decomposition/fusion/f2_deconf.py')
f2 = importlib.util.module_from_spec(spec); sys.modules['f2x'] = f2; spec.loader.exec_module(f2)
meta, ids_E, y_E, g_E, dense_E, _ = f2.load_E('peer_revealed')
yrE = {n: yy for n, yy in zip(nt, yr)}
mE = np.array([yrE.get(str(n), 9999) <= 2021 for n in ids_E])
out["dense_heldout_pre2022"] = {"n": int(mE.sum()),
                               "dense_auc": float(roc_auc_score(y_E[mE], dense_E[mE]))}
print("dense held-out pre-2022:", out["dense_heldout_pre2022"], flush=True)

# judge calibration by era: per-criterion dispersion
disp = {}
for name, mask in (("pre2022", yr <= 2021), ("2022plus", yr >= 2022)):
    Xa = X_all[idx][mask]
    sds = np.nanstd(Xa, axis=0)
    nas = np.mean(np.isnan(X_all[idx][mask]), axis=0)
    disp[name] = {"mean_criterion_sd": float(np.nanmean(sds)),
                  "mean_na_rate": float(np.mean(nas))}
out["judge_dispersion_by_era"] = disp
print("dispersion:", json.dumps(disp), flush=True)

json.dump(out, open(R / 'methods/taste_decomposition/peer_identity_audit/era_reweight_test.json', 'w'), indent=1)
print("ERA_REWEIGHT_DONE")
