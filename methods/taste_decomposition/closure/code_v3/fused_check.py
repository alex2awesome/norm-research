#!/usr/bin/env python3
"""Design §11 check for code_v3: does the FUSED arm beat the bank?

Standing rule (design §11): the best fused arm must beat VA_nl on every cell; a final
ledger where max(fused) <= VA_nl auto-triggers an audit of that cell.

Fused arm here = grouped-OOF logistic stack of two already-out-of-sample score columns,
the bank's VA_nl OOF and the dense probability, fit by GroupKFold(repository) and read
WITHIN REPO (pooled is never a readout on this cell).  Cheap by construction: it stacks
scores, it does not refit the bank.
"""
import json, sys
sys.path.insert(0, '.'); sys.path.insert(0, '../maps_hw_si')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import cells_code as C

def stack_oof(cols, y, groups, n=5):
    X = np.column_stack(cols)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(n, len(np.unique(groups)))).split(X, groups=groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof

d = C.load(); y, g = d["y"], d["groups"]
va = np.full(len(y), np.nan)
for sp in ("eval", "test"):
    m = d["split"] == sp
    va[m] = np.mean([np.load(f"abank_rescore/code_v3_{sp}_va_nl_oof_seed{s}.npy")
                     for s in (0, 1, 2)], axis=0)
out = {"rule": "design §11 fused-must-beat-bank",
       "fused_arm": "grouped-OOF logistic stack of [VA_nl OOF, dense prob], "
                    "GroupKFold(repository), read WITHIN REPO",
       "dense_seeds_used": d["dense_seeds_have"]}
for tier, m in (("eval", d["split"] == "eval"), ("test", d["split"] == "test"),
                ("both", np.ones(len(y), bool))):
    f = np.full(len(y), np.nan)
    f[m] = stack_oof([va[m], d["dense"][m]], y[m], g[m])
    wf = C.within_repo_auc(y, f, g, m)["nwtd"]
    wv = C.within_repo_auc(y, va, g, m)["nwtd"]
    wt = C.within_repo_auc(y, d["dense"], g, m)["nwtd"]
    wd = C.within_repo_delta(y, f, va, g, m)
    out[tier] = {"fused_within": wf, "bank_VA_nl_within": wv, "T_within": wt,
                 "fused_minus_bank": wf - wv, "fused_minus_T": wf - wt,
                 "n_repos": wd["n_repos"], "fused_wins_repos": wd["a_wins_repos"],
                 "wilcoxon_p": wd.get("wilcoxon_p"),
                 "jackknife_ci95": wd.get("jackknife_ci95"),
                 "AUDIT_TRIGGER_fused_le_bank": bool(wf <= wv)}
json.dump(out, open("fused_check.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
