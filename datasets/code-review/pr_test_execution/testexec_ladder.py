#!/usr/bin/env python3
"""PR-merge ladder ON THE TEST-EXECUTED ROWS (user request 2026-08-18):
SUPERSEDED 2026-08-19 by transition_full_ladder.py (full A coverage, canonical
layer1_gemma_cells/so_votes_layer1/unified_fused_stack imports). This script's
local oof/within_repo are semantically consistent (pair-weighted, pure-group
skip) but its A join covers only 37 transition rows — see uniformity audit
notes/2026-08-19__vat-code-uniformity-audit.md.
V includes the EXECUTION features — fail-to-pass (F2P), pass-to-fail (P2F),
pass-to-pass, fail-to-fail transition counts and the baseline test count — the
things a text instrument cannot see.  Grouped-by-repo 5-fold OOF, logistic +
3-seed HistGB mean (the unified convention).

Arms: V_exec | A (bank metrics) | V_exec+A | +T (dense preds where available,
same-rows restriction reported separately).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

B = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/code-review/pr_test_execution/outputs")
tr = pd.read_csv(B / "f2p_p2f_certify_2026_07_21/transition_row_index_sk3_2026_07_21.csv")
tr = tr[tr.structurally_valid == True].copy()  # noqa: E712
# judgement is heterogeneous strings (accepted/rejected/true/1/0/unknown) — map
# explicitly, drop unmappable; verdict classes like runner_no_output/env_broken
# carry NO test information, keep only rows whose verdict implies tests actually ran
JMAP = {"accepted": 1, "rejected": 0, "true": 1, "false": 0, "1": 1, "0": 0}
tr["judgement"] = tr.judgement.astype(str).str.lower().map(JMAP)
tr = tr[tr.judgement.notna()].copy()
tr["judgement"] = tr.judgement.astype(int)
RAN = ~tr.verdict.astype(str).isin(["runner_no_output", "env_broken_no_tests",
                                    "missing_diff"])
tr = tr[RAN].copy()
tr["pr_number"] = tr.paper_id.astype(str).str.extract(r"(\d+)$").astype(float)
print(f"test-executed structurally-valid rows: {len(tr)} "
      f"({tr.repo.nunique()} repos, pos {tr.judgement.mean():.3f})")

am = pd.read_parquet(B / "pr_a_metrics_full.parquet")
acols = [c for c in am.columns if c.endswith("_score")]
am["pr_number"] = am.pr_number.astype(float)
m = tr.merge(am, on=["repo", "pr_number"], how="inner", suffixes=("", "_am"))
print(f"joined to A metrics: {len(m)} rows, {len(acols)} criteria")

EXEC = ["f2p", "p2f", "p2p", "f2f", "baseline_n"]
for c in EXEC:
    m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0)
y = m.judgement.astype(int).values
g = m.repo.astype(str).values
Vx = m[EXEC].values.astype(float)
A = m[acols].values.astype(float)


def oof(X, y, g):
    outs = []
    lo = np.zeros(len(y))
    pipe = lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                 StandardScaler(), LogisticRegression(max_iter=2000))
    for trn, te in GroupKFold(5).split(X, groups=g):
        c = pipe(); c.fit(X[trn], y[trn]); lo[te] = c.predict_proba(X[te])[:, 1]
    outs.append(lo)
    for s in (0, 1, 2):
        go = np.zeros(len(y))
        for trn, te in GroupKFold(5).split(X, groups=g):
            c = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                              HistGradientBoostingClassifier(max_leaf_nodes=31,
                                                             learning_rate=.06,
                                                             random_state=s))
            c.fit(X[trn], y[trn]); go[te] = c.predict_proba(X[te])[:, 1]
        outs.append(go)
    return np.mean(outs, axis=0)


res = {"n": int(len(m)), "n_repos": int(m.repo.nunique()),
       "pos_rate": float(y.mean()), "exec_features": EXEC}
ladder_oofs = {}
for nm, X in (("V_exec", Vx), ("A", A), ("V_exec+A", np.column_stack([Vx, A]))):
    o = oof(X, y, g)
    ladder_oofs[nm] = o
    res[nm] = {"pooled_NEVER_QUOTE": float(roc_auc_score(y, o))}
    print(f"[{nm}] pooled {res[nm]['pooled_NEVER_QUOTE']:.4f}", flush=True)

# per-exec-feature alone lines — POOLED IS COMPOSITION-DOMINATED (repo size vs
# accept rate); the quotable line is WITHIN-REPO pair-weighted, the cell's own
# discipline (and the project's MH-stratified convention).
def within_repo(pred):
    tot = wsum = 0.0
    for r in np.unique(g):
        mm = g == r
        yy = y[mm]
        if yy.min() == yy.max():
            continue
        a = roc_auc_score(yy, pred[mm])
        npair = int(yy.sum() * (len(yy) - yy.sum()))
        tot += a * npair; wsum += npair
    return float(tot / wsum) if wsum else float("nan")

res["exec_alone_pooled_NEVER_QUOTE"] = {c: float(roc_auc_score(y, m[c].values)) for c in EXEC}
res["exec_alone_within_repo"] = {c: within_repo(m[c].values.astype(float)) for c in EXEC}
print("exec within-repo:", {k: round(v, 3) for k, v in res["exec_alone_within_repo"].items()})

# T same-rows where dense preds exist
dp = pd.read_csv(B / "pr_dense_grouped_test_preds.csv")
key = None
for cand in ("paper_id", "row_id", "pr_number"):
    if cand in dp.columns:
        key = cand
        break
if key == "paper_id":
    dmap = dict(zip(dp.paper_id.astype(str), dp.prob if "prob" in dp.columns else dp.pred))
    m["dense"] = m.paper_id.astype(str).map(dmap)
    sub = m[m.dense.notna()]
    print(f"dense-covered subset: {len(sub)}")
    if len(sub) > 500:
        ys, gs = sub.judgement.astype(int).values, sub.repo.astype(str).values
        Xva = np.column_stack([sub[EXEC].values.astype(float), sub[acols].values.astype(float)])
        ova = oof(Xva, ys, gs)
        ovat = oof(np.column_stack([Xva, sub.dense.values.astype(float).reshape(-1, 1)]), ys, gs)
        res["same_rows_dense_subset"] = {
            "n": int(len(sub)),
            "V_exec+A": float(roc_auc_score(ys, ova)),
            "T_alone": float(roc_auc_score(ys, sub.dense.values.astype(float))),
            "VAT": float(roc_auc_score(ys, ovat))}
        print("same-rows:", {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in res["same_rows_dense_subset"].items()})
for nm, o in ladder_oofs.items():
    res[nm]["within_repo"] = within_repo(o)
    print(f"[{nm}] within-repo {res[nm]['within_repo']:.4f}", flush=True)
out = Path("/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition/results/pr_testexec_ladder.json")
out.write_text(json.dumps(res, indent=1))
print("PR_TESTEXEC_LADDER_DONE")


# ---- BALANCED TRANSITION ARM (user request 2026-08-18) -----------------------
# 50% rows WITH a test transition (f2p>0 or p2f>0) + 50% random no-transition
# rows, stable-seeded; same estimators; within-repo readout beside pooled.
import hashlib as _h
trans = m[(m.f2p > 0) | (m.p2f > 0)]
notr = m[(m.f2p == 0) & (m.p2f == 0)]
rng = np.random.default_rng(20260818)
n_bal = min(len(trans), len(notr))
bal = pd.concat([trans.sample(n=n_bal, random_state=7) if len(trans) > n_bal else trans,
                 notr.sample(n=n_bal, random_state=7)])
yb = bal.judgement.astype(int).values
gb = bal.repo.astype(str).values
print(f"\n[balanced] transition rows {len(trans)} | sampled {n_bal}+{n_bal} "
      f"| pos {yb.mean():.3f} | repos {bal.repo.nunique()}")
Vb = bal[EXEC].values.astype(float)
Ab = bal[acols].values.astype(float)

def within_repo_b(pred, y_, g_):
    tot = wsum = 0.0
    for r in np.unique(g_):
        mm = g_ == r
        yy = y_[mm]
        if yy.min() == yy.max():
            continue
        a = roc_auc_score(yy, pred[mm])
        npair = int(yy.sum() * (len(yy) - yy.sum()))
        tot += a * npair; wsum += npair
    return float(tot / wsum) if wsum else float("nan")

bres = {"n_transition": int(len(trans)), "n_balanced": int(2 * n_bal),
        "pos_rate": float(yb.mean())}
for nm, X in (("V_exec", Vb), ("A", Ab), ("V_exec+A", np.column_stack([Vb, Ab]))):
    o = oof(X, yb, gb)
    bres[nm] = {"pooled": float(roc_auc_score(yb, o)),
                "within_repo": within_repo_b(o, yb, gb)}
    print(f"[balanced {nm}] pooled {bres[nm]['pooled']:.4f} "
          f"within-repo {bres[nm]['within_repo']:.4f}", flush=True)
bres["has_transition_alone"] = {
    "pooled": float(roc_auc_score(yb, ((bal.f2p > 0) | (bal.p2f > 0)).astype(float))),
    "within_repo": within_repo_b(((bal.f2p > 0) | (bal.p2f > 0)).values.astype(float), yb, gb)}
bres["f2p_alone_within"] = within_repo_b(bal.f2p.values.astype(float), yb, gb)
bres["p2f_alone_within"] = within_repo_b(bal.p2f.values.astype(float), yb, gb)
print("[balanced] has-transition alone:", {k: round(v, 4) for k, v in bres["has_transition_alone"].items()},
      "| f2p w-r", round(bres["f2p_alone_within"], 4), "| p2f w-r", round(bres["p2f_alone_within"], 4))
res["balanced_transition_arm"] = bres
out.write_text(json.dumps(res, indent=1))
print("BALANCED_ARM_DONE")
