#!/usr/bin/env python3
"""Item 3 — mechanism test (NOT headline): execution-derived V features on the LC
organic cell (+ extension labels when available), python-covered subset.

Hypothesis (user): in tight solution spaces, deviation-from-canonical is
V-detectable; accepted-only filtering removed that signal. Reality: the LC pool
was never accepted-only (23% of python-ish candidates pass zero tests). Test:
do V features (pass_rate, compiled, timeouts, ...) predict the L1
editorial-similarity label, and does Bank+V beat Bank on the covered subset?

Tiers (LC frozen recipe: balanced LR / RF(500,leaf2) / rank-avg, SGKF(5) by
problem, seeds 13,21,34,55,89): Bank / Bank+V / V-only.
Also: pass_rate vs label contingency.

Output: outputs/v2_analysis/lc_cf_push/exec_diag_results.json
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from eval_extended import run_lc_recipe, read_labels, LC, PUSH  # noqa: E402

EXEC_BASE = ["compiled_f", "first_test_pass", "all_pass", "pass_rate",
             "has_runtime_err", "has_timeout", "n_tests", "log_wall_ms_median"]


def add_exec_features(df):
    out = df.copy()
    out["compiled_f"] = out["compiled"].map({True: 1.0, False: 0.0}).astype(float)
    out["first_test_pass"] = ((out["n_pass"].fillna(0) > 0) & (out["compiled_f"] == 1.0)).astype(float)
    out["all_pass"] = (out["pass_rate"] == 1.0).astype(float)
    out.loc[out["pass_rate"].isna(), "all_pass"] = np.nan
    out["has_runtime_err"] = (out["n_runtime_err"].fillna(0) > 0).astype(float)
    out.loc[out["n_runtime_err"].isna(), "has_runtime_err"] = np.nan
    out["has_timeout"] = (out["n_timeout"].fillna(0) > 0).astype(float)
    out.loc[out["n_timeout"].isna(), "has_timeout"] = np.nan
    out["log_wall_ms_median"] = np.log1p(out["wall_ms_median"].astype(float))
    return out


def main():
    # labels: original cell + any extension labels
    orig = pd.read_parquet(LC / "merged_labels.parquet")[["pair_id", "l1", "problem_id"]]
    lab_e = read_labels(str(PUSH / "lc_ext" / "results" / "shard_lcx_*_results.jsonl"))
    ext_rows = []
    for f in sorted((PUSH / "lc_ext" / "shards").glob("shard_lcx_*.jsonl")):
        for ln in open(f):
            d = json.loads(ln)
            if d["pair_id"] in lab_e:
                ext_rows.append({"pair_id": d["pair_id"], "l1": lab_e[d["pair_id"]],
                                 "problem_id": d["problem_id"],
                                 "candidate_id": str(d["candidate_id"])})
    ext = pd.DataFrame(ext_rows)
    labels = pd.concat([orig, ext], ignore_index=True) if len(ext) else orig
    print(f"labels: {len(orig)} cell + {len(ext)} ext")

    ex = pd.read_parquet(PUSH / "lc_exec_features_organic.parquet")
    ex = add_exec_features(ex)
    df = labels.merge(ex.drop(columns=["question_slug", "candidate_id"], errors="ignore"),
                      on="pair_id", how="inner")
    cov = df[df["n_tests"].notna() & (df["n_tests"] > 0)].copy()
    print(f"exec-covered labeled rows: {len(cov)} (of {len(df)} python-ish matched)")
    cov["label"] = cov["l1"].astype(int)

    # bank features
    bank = pd.read_parquet(LC / "bank_scores.parquet")
    par = pd.read_parquet(LC / "paradigm_scores.parquet")
    bp = bank.merge(par, on="candidate_id", how="left")
    eb_p = PUSH / "lc_ext" / "bank_scores.parquet"
    if eb_p.exists():
        eb = pd.read_parquet(eb_p).merge(
            pd.read_parquet(PUSH / "lc_ext" / "paradigm_scores.parquet"),
            on="candidate_id", how="left")
        bp = pd.concat([bp, eb], ignore_index=True).drop_duplicates("candidate_id")
    # candidate_id for original rows comes from merged_labels
    cid = pd.concat([
        pd.read_parquet(LC / "merged_labels.parquet")[["pair_id", "candidate_id"]],
        ext[["pair_id", "candidate_id"]] if len(ext) else
        pd.DataFrame(columns=["pair_id", "candidate_id"]),
    ], ignore_index=True)
    cid["candidate_id"] = cid["candidate_id"].astype(str)
    bp["candidate_id"] = bp["candidate_id"].astype(str)
    cov = cov.drop(columns=["candidate_id"], errors="ignore")
    cov = cov.merge(cid.drop_duplicates("pair_id"), on="pair_id", how="left")
    cov = cov.merge(bp, on="candidate_id", how="left")

    feat_bank = sorted(c for c in bp.columns if c.endswith("_score") or c.endswith("_applied"))
    y = cov["label"].values
    groups = cov["problem_id"].astype(str).values
    res = {"n_covered": int(len(cov)), "pos_rate": float(y.mean()),
           "n_neg": int((1 - y).sum()), "n_problems": int(pd.Series(groups).nunique())}

    # pass-rate vs label
    ct = pd.crosstab(cov["all_pass"], cov["label"], normalize="index")
    res["pos_rate_by_all_pass"] = {str(k): float(v) for k, v in ct[1].items()}
    res["mean_pass_rate_by_label"] = {
        str(k): float(v) for k, v in cov.groupby("label")["pass_rate"].mean().items()}
    print("pos rate by all_pass:", res["pos_rate_by_all_pass"])
    print("mean pass_rate by label:", res["mean_pass_rate_by_label"])

    Xb = cov[feat_bank].astype(float).values
    Xv = cov[EXEC_BASE].astype(float).values
    Xbv = np.hstack([Xb, Xv])
    for nm, X in [("bank_only", Xb), ("bank_plus_V", Xbv), ("V_only", Xv)]:
        r = run_lc_recipe(X, y, groups)
        res[nm] = r
        print(f"[{nm}] LR {r['lr']['mean']:.3f} RF {r['rf']['mean']:.3f} "
              f"ENS {r['ens']['mean']:.3f} (pooled {r['ens_pooled_oof']:.3f})")

    (PUSH / "exec_diag_results.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {PUSH/'exec_diag_results.json'}")


if __name__ == "__main__":
    main()
