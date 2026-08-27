#!/usr/bin/env python3
"""Patents V+A harvest (phase 2, 2026-08-13): bank_v1 Gemma scores landed for the
11,988 eval+test claims. Fills the V+A column and its deconfounded companion.

Levels (raw instrument AUCs, ladder convention):
  A_nl, V+A_nl                       -- frozen Layer-1 stack, grouped OOF by app_id
  fused candidate: stack [VA_nl OOF, T]  -- §11-style; VAT column = max of variants
Deconfounded (standing rule: confounds decorrelated, never a tier):
  (c) V+A+NUIS  vs  (d) +T  -> residual w/ grouped bootstrap; §102/§103 replicate.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
TD = REPO / "methods/taste_decomposition"
D = REPO / "datasets/patents/v3_claimonly"
SC = TD / "closure/patents_claimonly"

def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[alias] = m
    spec.loader.exec_module(m); return m

F2 = _mod(TD / "fusion/f2_deconf.py", "f2m_pva")

DEP = re.compile(r"\bof claim (\d+)\b", re.I)
def content_feats(el):
    words = el.split()
    dep = DEP.search(el)
    return [1.0 if dep else 0.0, float(len(el)), float(len(words)),
            float(np.mean([len(w) for w in words])) if words else 0.0,
            float(el.count(",")), float(el.count(";")),
            float(len(re.findall(r"\bwherein\b", el, re.I))),
            float(len(re.findall(r"\d+(?:\.\d+)?", el)))]

z = np.load(SC / "patents_claimonly_r0_scores.npz", allow_pickle=True)
A_scores = z["X"]                                    # (11988, 30), NaN = judge NA
a_ids = [str(x) for x in z["row_id"]]
A_by_id = {rid: A_scores[i] for i, rid in enumerate(a_ids)}

strata = pd.read_csv(D / "harvest_strata_NEVER_AN_INPUT.csv")
rows = []
for sp in ("eval", "test"):
    d = pd.read_csv(D / f"arm_t/split/{sp}.csv")
    pt = pd.read_csv(D / f"arm_t/rm_out_seed42/preds_{sp}.csv")
    assert (d.judgement.values == pt.judgement.values).all()
    st = strata[strata.split == sp].reset_index(drop=True)
    assert (st.judgement.values == d.judgement.values).all()
    el = d.text.str.replace("CLAIM ELEMENT:\n", "", regex=False)
    rows.append(pd.DataFrame({
        "split": sp, "row_id": d.row_id, "y": d.judgement.astype(int),
        "group": d.group.astype(str), "T": pt.prob, "el": el,
        "claim_num": st.claim_num, "parent": st.parent_claim_num,
        "dep": st.is_dependent, "clen": st.char_len, "wlen": st.word_len,
        "rejection_type": st.rejection_type}))
E = pd.concat(rows, ignore_index=True)
assert set(E.row_id) <= set(A_by_id), "bank scores do not cover the ladder rows"
A = np.vstack([A_by_id[r] for r in E.row_id])
y, groups, T = E.y.values, E.group.values, E["T"].values
V = np.array([content_feats(e) for e in E.el])
NUIS = E[["claim_num", "parent", "dep", "clen", "wlen"]].astype(float).values

r_a = F2.fit_arm("clean_once", A, T, y, groups)
r_va = F2.fit_arm("clean_once", np.column_stack([V, A]), T, y, groups)
r_van = F2.fit_arm("clean_once", np.column_stack([V, A, NUIS]), T, y, groups)
prim = F2.gboot(y, r_van["_oof_VAT_nl0"], r_van["_oof_VA_nl0"], groups, n_boot=2000)

# §11-style fused: grouped-OOF logistic stack of [VA_nl OOF, T]
S = np.column_stack([r_va["_oof_VA_nl0"], T])
oof = np.zeros(len(y))
for tr, te in GroupKFold(5).split(S, groups=groups):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(S[tr], y[tr])
    oof[te] = clf.predict_proba(S[te])[:, 1]
fused = float(roc_auc_score(y, oof))

def sub(mask, s):
    return float(roc_auc_score(y[mask], s[mask]))

ev, te_ = (E.split == "eval").values, (E.split == "test").values
out = {
 "cell": "patents claim-only V+A harvest (bank_v1, 30 criteria, 4 collapsed)",
 "levels": {
   "A_nl": r_a["VA_nl_mean"], "VA_nl": r_va["VA_nl_mean"],
   "VA_nl_oof_eval": sub(ev, r_va["_oof_VA_nl0"]), "VA_nl_oof_test": sub(te_, r_va["_oof_VA_nl0"]),
   "fused_stack_VA_T": fused,
   "fused_eval": sub(ev, oof), "fused_test": sub(te_, oof),
 },
 "deconfounded": {
   "c_VA_plus_nuis": r_van["VA_nl_mean"], "d_plus_T": r_van["VAT_nl_mean"],
   "residual_d_minus_c": prim,
   "channels": "claim ordinal, parent-claim num, dependency, char/word len (decorrelated)",
 },
}
m = (E.rejection_type.astype(str).isin(["102", "103"]) | (y == 0)).values
rr = F2.fit_arm("clean_once", np.column_stack([V, A, NUIS])[m], T[m], y[m], groups[m])
out["replicate_102_103"] = {
  "n": int(m.sum()),
  "residual": F2.gboot(y[m], rr["_oof_VAT_nl0"], rr["_oof_VA_nl0"], groups[m], n_boot=2000)}

(D / "harvest_va_patents.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
print("PATENTS_VA_HARVEST_DONE", flush=True)
