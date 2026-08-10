#!/usr/bin/env python3
"""Reproducibility check for the PUBLISHED code-review ladder
(V .576 pooled / A .615 pooled / V+A .632 pooled ; GroupKFold V .549 / A .596 / V+A .592).

Published protocol = scripts/pr_vat/run_vat_ladder.py:
  StandardScaler + LogisticRegression(C=0.1, max_iter=500), NaN->0,
  StratifiedKFold(5, shuffle, seed 42) for the pooled row, GroupKFold(repo) for
  the grouped row, score = MEAN OF FOLD AUCs (cross_val_score), A restricted to
  columns with >5% non-null coverage.

Surviving inputs: outputs/pr_a_metrics_full.parquet (68,083 diffs x 127 coded
aspects) + batch_runs/*/verdicts.jsonl. The V source the published run actually
used (outputs/consolidated_verdicts_ALL_final.csv) is GONE, and today's
verdicts.jsonl no longer carries smoke_rc / baseline_* / post_* columns, so V is
rebuilt from what survives (verdict category + rc + P2F/F2P gates). Exact
reproduction is therefore NOT possible; this tests whether the LEVEL replicates.
"""
import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
PTE = REPO / "datasets/code-review/pr_test_execution"

VERDICT_CATS = ["runner_no_output", "env_broken_no_tests", "no_change_all_pass",
                "no_change_still_broken", "env_broken_collection", "baseline_build_failed",
                "unknown_build_tool", "new_passing", "wall_timeout", "no_gradle",
                "fix", "regression", "new_failing", "env_broken_post_patch",
                "missing_diff", "patch_apply_failed"]


def auc_cv(X, y, groups=None, grouped=False):
    pipe = Pipeline([("s", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=500, C=0.1, solver="lbfgs"))])
    X = np.nan_to_num(X, nan=0.0)
    if grouped:
        cv = GroupKFold(n_splits=5)
        s = cross_val_score(pipe, X, y, groups=groups, cv=cv, scoring="roc_auc")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        s = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return float(s.mean()), float(s.std())


def main():
    rows = []
    for f in glob.glob(str(PTE / "batch_runs/*/verdicts.jsonl")):
        r = f.split("/")[-2]
        for l in open(f):
            l = l.strip()
            if not l:
                continue
            try:
                d = json.loads(l)
            except Exception:
                continue
            rows.append({"repo": r, "pr_number": str(d.get("pr_number")),
                         "verdict": d.get("verdict"), "rc": d.get("rc"),
                         "judgement": d.get("judgement")})
    v = pd.DataFrame(rows).drop_duplicates(["repo", "pr_number"], keep="first")
    v = v[v["judgement"].isin(["accepted", "rejected"])].copy()
    print(f"verdict rows w/ clean judgement: {len(v)}, repos {v['repo'].nunique()}")

    a = pd.read_parquet(PTE / "outputs/pr_a_metrics_full.parquet")
    a["pr_number"] = a["pr_number"].astype(str)
    m = v.merge(a, on=["repo", "pr_number"], how="inner")
    m = m.drop_duplicates(["repo", "pr_number"], keep="first")
    print(f"joined V+A: {len(m)} rows, {m['repo'].nunique()} repos "
          f"(published: 44,751 rows / 594 repos)")

    m["y"] = (m["judgement"] == "rejected").astype(int)
    m["v_rc"] = pd.to_numeric(m["rc"], errors="coerce").fillna(-1)
    m["v_p2f"] = m["verdict"].isin(["regression", "new_failing"]).astype(int)
    m["v_f2p"] = m["verdict"].isin(["fix", "new_passing"]).astype(int)
    m["v_has_signal"] = (m["v_p2f"] | m["v_f2p"]).astype(int)
    for c in VERDICT_CATS:
        m[f"v_cat_{c}"] = (m["verdict"] == c).astype(int)
    v_cols = ["v_rc", "v_p2f", "v_f2p", "v_has_signal"] + [f"v_cat_{c}" for c in VERDICT_CATS]

    a_cols = [c for c in m.columns if c.endswith("_score")]
    a_keep = [c for c in a_cols if m[c].notna().sum() > len(m) * 0.05]
    print(f"V features {len(v_cols)}, A features (>5% cov) {len(a_keep)} "
          f"(published: 8 V / ~72 A)")

    y = m["y"].values
    g = m["repo"].values
    Xv = m[v_cols].values
    Xa = m[a_keep].values
    Xva = np.hstack([Xv, Xa])
    out = {"n": int(len(m)), "n_repos": int(m["repo"].nunique()),
           "pos_rate_rejected": float(y.mean())}
    for label, grouped in (("pooled_StratifiedKFold", False), ("grouped_GroupKFold_repo", True)):
        r = {}
        for k, X in (("V", Xv), ("A", Xa), ("V+A", Xva)):
            mu, sd = auc_cv(X, y, g, grouped)
            r[k] = {"mean_of_folds": mu, "std": sd}
            print(f"  {label:26s} {k:4s}: {mu:.4f} +- {sd:.4f}")
        out[label] = r
    outp = REPO / "datasets/code-review/dense_standard_v3/abank_rescore/repro_published_ladder.json"
    outp.write_text(json.dumps(out, indent=1))
    print("wrote", outp)


if __name__ == "__main__":
    main()
