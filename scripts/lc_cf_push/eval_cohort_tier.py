#!/usr/bin/env python3
"""Item 4 — cohort-relative tier (NOT headline; breaks candidate-only by using the
per-problem candidate cohort at prediction time).

Feature tier: candidate's deviation from the per-problem candidate centroid in
bank-feature space (leave-one-out centroid; problems with <2 labeled candidates
get NaN cohort features). Quantifies how much residual is approach-CHOICE
information (relative to what other people submitted for the same problem).

Tiers per cell (frozen per-cell recipe): Bank / Bank+Cohort / Cohort-only.
Output: outputs/v2_analysis/lc_cf_push/cohort_tier_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from eval_extended import (run_cf_recipe, run_lc_recipe, load_cf_frames,
                           load_lc_frames, PUSH)  # noqa: E402


def cohort_features(df, feat, gcol="canonical_pid"):
    """LOO centroid distances in standardized bank space."""
    X = df[feat].astype(float).values
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), med, X)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    g = df[gcol].astype(str).values
    eu = np.full(len(df), np.nan)
    cs = np.full(len(df), np.nan)
    mean_absdev = np.full(len(df), np.nan)
    cohort_n = np.zeros(len(df))
    idx = pd.Series(range(len(df))).groupby(g).groups  # wrong keying; rebuild below
    idx = {}
    for i, gg in enumerate(g):
        idx.setdefault(gg, []).append(i)
    for gg, ii in idx.items():
        ii = np.array(ii)
        cohort_n[ii] = len(ii)
        if len(ii) < 2:
            continue
        S = Z[ii].sum(0)
        for i in ii:
            cen = (S - Z[i]) / (len(ii) - 1)
            d = Z[i] - cen
            eu[i] = np.sqrt((d ** 2).mean())
            denom = (np.linalg.norm(Z[i]) * np.linalg.norm(cen) + 1e-9)
            cs[i] = float(Z[i] @ cen / denom)
            mean_absdev[i] = np.abs(d).mean()
    out = df.copy()
    out["coh_eu"] = eu
    out["coh_cos"] = cs
    out["coh_absdev"] = mean_absdev
    out["coh_n"] = cohort_n
    return out, ["coh_eu", "coh_cos", "coh_absdev", "coh_n"]


def run(name, frames_loader, runner, use_ext=True):
    orig, ext, feat = frames_loader()
    o = orig[orig["label"].notna()].copy()
    e = ext[ext["label"].notna()].copy()
    if use_ext and len(e) >= 200:
        df = pd.concat([o[["canonical_pid", "label"] + feat],
                        e[["canonical_pid", "label"] + feat]], ignore_index=True)
        which = "extended"
    else:
        df = o[["canonical_pid", "label"] + feat].copy()
        which = "original"
    df["label"] = df["label"].astype(int)
    df, coh = cohort_features(df, feat)
    multi = df["coh_n"] >= 2
    print(f"\n=== {name} ({which}): n={len(df)}, cohort>=2 coverage={multi.mean():.2f}")
    y = df["label"].values
    groups = df["canonical_pid"].astype(str).values
    res = {"population": which, "n": int(len(df)),
           "cohort_coverage_ge2": float(multi.mean())}
    for nm, cols in [("bank_only", feat), ("bank_plus_cohort", feat + coh),
                     ("cohort_only", coh)]:
        r = runner(df[cols].astype(float).values, y, groups)
        res[nm] = r
        ens = r["ens"] if isinstance(r["ens"], float) else r["ens"]["mean"]
        print(f"[{nm}] LR {r['lr']['mean']:.3f} RF {r['rf']['mean']:.3f} ENS {ens:.3f}")
    return res


def main():
    out = {
        "cf": run("CF", load_cf_frames, run_cf_recipe),
        "lc": run("LC", load_lc_frames, run_lc_recipe),
    }
    (PUSH / "cohort_tier_results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {PUSH/'cohort_tier_results.json'}")


if __name__ == "__main__":
    main()
