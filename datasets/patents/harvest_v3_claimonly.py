#!/usr/bin/env python3
"""Patents claim-only harvest (2026-08-13): honest T, fused V3 arm, and the
DECONFOUNDED residual — ordinal/length channels decorrelated via the standard F2
machinery (nuisance block conditioning), never reported as a tier (user ruling).

Rows: eval+test (dense-held-out for both arms; app_id-grouped, T honest on all).
Arms via direction1_mirror.fit_arm (frozen Layer-1 stack, clean_once family):
  (a) V_content (8 text-derived features)
  (b) NUIS alone (claim ordinal, parent-claim num, dependency, char/word len)
  (c) V_content + NUIS
  (d) (c) + T          -> PRIMARY deconfounded residual (d)-(c), grouped bootstrap
Replicate: §102/§103-restricted T and residual (positives on prior-art grounds only).
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
TD = REPO / "methods/taste_decomposition"
D = REPO / "datasets/patents/v3_claimonly"

def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[alias] = m
    spec.loader.exec_module(m); return m

F2 = _mod(TD / "fusion/f2_deconf.py", "f2m_pat")
fit_arm, gboot = F2.fit_arm, F2.gboot

DEP = re.compile(r"\bof claim (\d+)\b", re.I)
def content_feats(el):
    words = el.split()
    dep = DEP.search(el)
    return [1.0 if dep else 0.0, float(len(el)), float(len(words)),
            float(np.mean([len(w) for w in words])) if words else 0.0,
            float(el.count(",")), float(el.count(";")),
            float(len(re.findall(r"\bwherein\b", el, re.I))),
            float(len(re.findall(r"\d+(?:\.\d+)?", el)))]

strata = pd.read_csv(D / "harvest_strata_NEVER_AN_INPUT.csv")
rows = []
for sp in ("eval", "test"):
    d = pd.read_csv(D / f"arm_t/split/{sp}.csv")
    pt = pd.read_csv(D / f"arm_t/rm_out_seed42/preds_{sp}.csv")
    pa = pd.read_csv(D / f"arm_a/rm_out_seed42/preds_{sp}.csv")
    assert (d.judgement.values == pt.judgement.values).all()
    assert (d.judgement.values == pa.judgement.values).all()
    st = strata[strata.split == sp].reset_index(drop=True)
    assert (st.judgement.values == d.judgement.values).all(), f"{sp} strata misaligned"
    el = d.text.str.replace("CLAIM ELEMENT:\n", "", regex=False)
    rows.append(pd.DataFrame({
        "split": sp, "y": d.judgement.astype(int), "group": d.group.astype(str),
        "T": pt.prob, "V3": pa.prob, "el": el,
        "claim_num": st.claim_num, "parent": st.parent_claim_num,
        "dep": st.is_dependent, "clen": st.char_len, "wlen": st.word_len,
        "rejection_type": st.rejection_type}))
E = pd.concat(rows, ignore_index=True)
y = E.y.values
groups = E.group.values
V = np.array([content_feats(e) for e in E.el])
NUIS = E[["claim_num", "parent", "dep", "clen", "wlen"]].astype(float).values
T = E["T"].values

r_v = fit_arm("clean_once", V, T, y, groups)
r_n = fit_arm("clean_once", NUIS, T, y, groups)
r_vn = fit_arm("clean_once", np.column_stack([V, NUIS]), T, y, groups)
prim = gboot(y, r_vn["_oof_VAT_nl0"], r_vn["_oof_VA_nl0"], groups, n_boot=2000)

def a(y_, s_):
    return float(roc_auc_score(y_, s_))

out = {
    "cell": "patents claim-only (examiner rejected this claim element, any ground)",
    "n_eval_test": int(len(E)), "n_groups": int(E.group.nunique()),
    "honest_T": {"eval": a(y[E.split == "eval"], T[E.split == "eval"]),
                 "test": a(y[E.split == "test"], T[E.split == "test"])},
    "V3_fused_arm": {"eval": a(y[E.split == "eval"], E.V3[E.split == "eval"]),
                     "test": a(y[E.split == "test"], E.V3[E.split == "test"])},
    "claim_num_alone": F2.alone_auc(y, E.claim_num.values.astype(float)),
    "arms": {"V_content_nl": r_v["VA_nl_mean"], "NUIS_nl": r_n["VA_nl_mean"],
             "V_plus_NUIS_nl": r_vn["VA_nl_mean"], "plus_T_nl": r_vn["VAT_nl_mean"]},
    "deconfounded_residual_d_minus_c": prim,
    "nuisance_channels": "claim ordinal, parent-claim num, dependency flag, char/word length "
                         "(decorrelated per standing rule — never a tier)",
}

m = E.rejection_type.astype(str).isin(["102", "103"]) | (y == 0)
ye, Te = y[m.values], T[m.values]
out["replicate_102_103"] = {"n": int(m.sum()), "pos_rate": float(ye.mean()),
                            "T_auc": a(ye, Te)}
rr = fit_arm("clean_once", np.column_stack([V, NUIS])[m.values], Te, ye, groups[m.values])
out["replicate_102_103"]["deconf_residual"] = gboot(
    ye, rr["_oof_VAT_nl0"], rr["_oof_VA_nl0"], groups[m.values], n_boot=2000)

(D / "harvest_v3_claimonly.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "nuisance_channels"}, indent=1))
print("PATENTS_HARVEST_DONE", flush=True)
