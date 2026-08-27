#!/usr/bin/env python3
"""code_competitions (AtCoder, "same approach as editorial") — UNPARK step 1.

User directive: "retrieve the scores we have ... go forward with new features. Don't redo
work."  So: the 139-criterion A bank is KEPT AS SCORED (never recomputed), and what gets
built is the missing V layer on the SAME 999-row population, plus V / A / V+A readouts
under the frozen Layer-1 protocol so the new numbers sit on the recorded A ledger's ruler.

POPULATION (reproduced, asserted): inner join of ac_bank_scores.parquet (1,000 rows) with
cell_ac_l1.parquet (2,495 rows) on pair_id -> n = 999, 634 canonical_pid groups,
850 positive / **149 negative** (pos rate .8509).  The absolute minority count is 149 and
is reported with every readout (mathlib lesson: never quote a rate without the count).

V LAYER: deterministic code/regex features on `candidate_code` -- no language model
anywhere in V, which is what makes it V.  Language-aware (cpp 565 / python 434).

Recorded flags carried from results/code_competitions_layer1.json:
  * no V layer had ever been built for this cell (the candidate-only exec-pass-rate layer
    was ~chance and was not used);
  * its T_dense .69 was POPULATION-MISMATCHED (computed on the full 2,495-row L1 set) and
    is NOT used here; an honest same-population T is trained separately.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

R = Path("/lfs/skampere3/0/alexspan/norm-research")
OUT = R / "methods/taste_decomposition/code_competitions"
GRID = [{"max_leaf_nodes": 15, "learning_rate": .06, "max_iter": 400},
        {"max_leaf_nodes": 31, "learning_rate": .06, "max_iter": 400}]
SEEDS = (0, 1, 2)

CPP_LIB = ["vector", "map<", "set<", "sort(", "pair<", "queue", "stack", "string",
           "priority_queue", "unordered_"]
PY_LIB = ["import ", "def ", "collections", "itertools", "heapq", "bisect", "sorted(",
          "dict(", "set(", "numpy"]


def v_features(code: str, lang: str) -> dict:
    c = str(code)
    lines = c.split("\n")
    nb = [l for l in lines if l.strip()]
    indents = [len(l) - len(l.lstrip(" ")) for l in nb if l.startswith(" ")]
    if lang == "cpp":
        cm = [l for l in nb if l.strip().startswith(("//", "/*", "*"))]
        depth, mx = 0, 0
        for ch in c:
            if ch == "{":
                depth += 1; mx = max(mx, depth)
            elif ch == "}":
                depth = max(0, depth - 1)
        nest = mx
        libs = sum(c.count(k) for k in CPP_LIB)
        nfun = len(re.findall(r"\b(?:void|int|long|double|bool|auto|string)\s+\w+\s*\(", c))
    else:
        cm = [l for l in nb if l.strip().startswith("#")]
        nest = (max(indents) // 4) if indents else 0
        libs = sum(c.count(k) for k in PY_LIB)
        nfun = len(re.findall(r"^\s*def\s+\w+", c, re.M))
    ident = re.findall(r"[A-Za-z_]\w*", c)
    return {
        "v_lang_cpp": float(lang == "cpp"),
        "v_n_chars": len(c), "v_n_lines": len(lines), "v_n_nonblank": len(nb),
        "v_blank_ratio": 1 - len(nb) / max(1, len(lines)),
        "v_mean_line_len": float(np.mean([len(l) for l in nb])) if nb else 0.0,
        "v_max_line_len": max((len(l) for l in nb), default=0),
        "v_n_comment_lines": len(cm), "v_comment_ratio": len(cm) / max(1, len(nb)),
        "v_max_nesting": nest,
        "v_n_for": len(re.findall(r"\bfor\b", c)), "v_n_while": len(re.findall(r"\bwhile\b", c)),
        "v_n_if": len(re.findall(r"\bif\b", c)), "v_n_else": len(re.findall(r"\belse\b", c)),
        "v_n_return": len(re.findall(r"\breturn\b", c)),
        "v_n_funcs": nfun, "v_lib_hits": libs,
        "v_n_includes": len(re.findall(r"^\s*(?:#include|import|from)\b", c, re.M)),
        "v_n_define": len(re.findall(r"^\s*#define\b", c, re.M)),
        "v_n_ident": len(ident), "v_n_uniq_ident": len(set(ident)),
        "v_ident_reuse": len(ident) / max(1, len(set(ident))),
        "v_digit_density": sum(ch.isdigit() for ch in c) / max(1, len(c)),
        "v_uses_int64": float(bool(re.search(r"long long|int64|10\*\*18", c))),
        "v_fast_io": float(bool(re.search(r"sync_with_stdio|scanf|sys\.stdin", c))),
        "v_has_tabs": float("\t" in c),
        "v_n_operators": sum(c.count(o) for o in ["+", "-", "*", "/", "%", "<", ">", "="]),
    }


def clean_fit(M):
    keep, meds = [], []
    for j in range(M.shape[1]):
        col = M[:, j].astype(float)
        nn = col[~np.isnan(col)]
        if len(nn) == 0:
            continue
        med = float(np.median(nn))
        cc = np.where(np.isnan(col), med, col)
        _, cnt = np.unique(cc, return_counts=True)
        if (len(cc) - cnt.max()) < 5 or cc.std() == 0:
            continue
        keep.append(j); meds.append(med)
    return np.array(keep, int), np.array(meds, float)


def prep(M):
    k, m = clean_fit(M)
    X = M[:, k].astype(float).copy()
    for i in range(X.shape[1]):
        X[np.isnan(X[:, i]), i] = m[i]
    return X


def lin_oof(X, y, g):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, groups=g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def gbm_oof(X, y, g, seed):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, groups=g):
        best, bs = None, -1
        for p in GRID:
            aucs = []
            for it, ie in GroupKFold(n_splits=3).split(X[tr], groups=g[tr]):
                m = HistGradientBoostingClassifier(**p, early_stopping=True,
                                                   validation_fraction=.1,
                                                   n_iter_no_change=20, random_state=seed)
                m.fit(X[tr][it], y[tr][it])
                aucs.append(roc_auc_score(y[tr][ie], m.predict_proba(X[tr][ie])[:, 1]))
            if np.mean(aucs) > bs:
                bs, best = np.mean(aucs), p
        m = HistGradientBoostingClassifier(**best, early_stopping=True, validation_fraction=.1,
                                           n_iter_no_change=20, random_state=seed)
        m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def main():
    b = pd.read_parquet(R / "outputs/v2_analysis/comp_fourplatform_cells/ac_bank_scores.parquet")
    c = pd.read_parquet(R / "outputs/v2_analysis/dense_ceiling/cell_ac_l1.parquet")
    d = c.merge(b, on="pair_id", how="inner", suffixes=("", "_b"))
    assert len(d) == 999 and d.canonical_pid.nunique() == 634, (len(d), d.canonical_pid.nunique())
    y = d["label"].astype(int).values
    g = d["canonical_pid"].astype(str).values
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    assert (n_pos, n_neg) == (850, 149), (n_pos, n_neg)

    a_cols = [k for k in b.columns if k.endswith("_score")]
    A = d[a_cols].astype(float).values
    V = pd.DataFrame([v_features(r.candidate_code, r.language) for r in d.itertuples()])
    V.to_parquet(OUT / "ac_v_features.parquet")

    blocks = {"V": V.values.astype(float), "A": A,
              "VA": np.column_stack([V.values.astype(float), A])}
    res = {"cell": "code_competitions (AtCoder same-approach-as-editorial)",
           "n": len(y), "n_groups": int(d.canonical_pid.nunique()),
           "pos_rate": float(y.mean()),
           "n_positive": n_pos, "n_negative_MINORITY": n_neg,
           "minority_note": "149 negatives in 999 rows -- every readout below is bounded by "
                            "this; never quote the rate without the count",
           "A_kept_from": len(a_cols), "protocol": "frozen Layer-1: StandardScaler+LR(C=1) / "
                                                   "HistGB{15,31} lr.06 400it seeds 0-2, "
                                                   "GroupKFold(5) by canonical_pid, pooled OOF AUC",
           "blocks": {}}
    for name, M in blocks.items():
        X = prep(M)
        lin = float(roc_auc_score(y, lin_oof(X, y, g)))
        nls = [float(roc_auc_score(y, gbm_oof(X, y, g, s))) for s in SEEDS]
        res["blocks"][name] = {"n_features_raw": M.shape[1], "n_features_kept": X.shape[1],
                               "lin": lin, "nl_seeds": nls, "nl_mean": float(np.mean(nls)),
                               "nl_spread": float(max(nls) - min(nls))}
        print(f"{name}: kept {X.shape[1]}/{M.shape[1]}  lin {lin:.4f}  "
              f"nl {np.mean(nls):.4f} {[round(x,4) for x in nls]}", flush=True)
    pub = {"A_lin": 0.6907027240426371, "A_nl_mean": 0.6695762600342151}
    res["reproduction_check_vs_recorded_ledger"] = {
        "recorded_A_lin": pub["A_lin"], "live_A_lin": res["blocks"]["A"]["lin"],
        "abs_diff_lin": abs(res["blocks"]["A"]["lin"] - pub["A_lin"]),
        "recorded_A_nl_mean": pub["A_nl_mean"], "live_A_nl_mean": res["blocks"]["A"]["nl_mean"],
        "abs_diff_nl": abs(res["blocks"]["A"]["nl_mean"] - pub["A_nl_mean"])}
    (OUT / "ac_v_and_readout.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["reproduction_check_vs_recorded_ledger"], indent=1))


if __name__ == "__main__":
    main()
