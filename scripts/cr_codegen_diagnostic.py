"""Why does adding 1182 codegen programs hurt RF AUC?

Diagnostic:
  1. Per-program score distributions: how many are stuck at 0.5? all-0? all-1?
  2. Per-program AUC vs label: how many actually beat 0.5?
  3. Variance per program (std of scores across 4978 datapoints).
  4. Correlation matrix among top programs - are they all measuring one thing?
  5. What text format does each program see? (sample inspection)
"""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
TASK = "code_review"
CG = REPO / "outputs/v2_analysis/cr_codegen_scores.parquet"
CODEGEN_DIR = REPO / f"runs/validity_full/v2/{TASK}/codegen_claude"
DPS_FILE = REPO / f"runs/validity_full/v2/{TASK}/datapoints.json"
ASPECTS_FILE = REPO / f"runs/validity_full/v2/{TASK}/aspects.json"


def main():
    print("Loading codegen scores...")
    df = pd.read_parquet(CG)
    y = df["y"].astype(int).values
    prog_cols = [c for c in df.columns if c not in ("datapoint_id", "y")]
    X = df[prog_cols].values
    print(f"shape: {X.shape}, label balance: y=1 frac = {y.mean():.3f}")

    # 1) Variance / degeneracy per program
    std = X.std(axis=0)
    mean = X.mean(axis=0)
    frac_default = (X == 0.5).mean(axis=0)        # fraction stuck at 0.5
    frac_zero = (X == 0.0).mean(axis=0)
    frac_one = (X == 1.0).mean(axis=0)

    print("\n=== degeneracy across 1182 programs ===")
    print(f"  fully-constant programs (std==0)         : {(std == 0).sum()}")
    print(f"  >=90% scores == 0.5 (mostly fallback)    : {(frac_default >= 0.9).sum()}")
    print(f"  >=90% scores == 0.0                      : {(frac_zero >= 0.9).sum()}")
    print(f"  >=90% scores == 1.0                      : {(frac_one >= 0.9).sum()}")
    print(f"  median std across programs               : {np.median(std):.4f}")
    print(f"  mean std across programs                 : {np.mean(std):.4f}")
    print(f"  fraction with std < 0.05                 : {(std < 0.05).mean():.1%}")
    print(f"  fraction with std < 0.10                 : {(std < 0.10).mean():.1%}")

    # 2) Per-program AUC
    print("\nComputing per-program AUC...")
    aucs = np.full(len(prog_cols), np.nan)
    for j in range(len(prog_cols)):
        if std[j] == 0:
            continue
        try:
            aucs[j] = roc_auc_score(y, X[:, j])
        except Exception:
            pass

    valid = ~np.isnan(aucs)
    # Use the "max(auc, 1-auc)" view since direction is arbitrary
    abs_auc = np.where(valid, np.maximum(aucs, 1 - aucs), np.nan)

    print(f"  programs with computable AUC          : {valid.sum()}/{len(prog_cols)}")
    print(f"  median |AUC - 0.5| (signal magnitude) : {np.nanmedian(abs_auc - 0.5):.4f}")
    print(f"  programs with |AUC - 0.5| > 0.02      : {(abs_auc > 0.52).sum()}")
    print(f"  programs with |AUC - 0.5| > 0.05      : {(abs_auc > 0.55).sum()}")
    print(f"  max |AUC - 0.5|                       : {np.nanmax(abs_auc - 0.5):.4f}")

    # 3) Top + bottom programs by signal
    print("\nTop 15 programs by |AUC - 0.5|:")
    order = np.argsort(-(abs_auc))
    for j in order[:15]:
        print(f"  {prog_cols[j]:<28} AUC={aucs[j]:.3f}  std={std[j]:.3f}  "
              f"mean={mean[j]:.3f}  frac_0.5={frac_default[j]:.2f}")

    # 4) Are the top programs all measuring same thing? (correlation)
    print("\nPairwise correlation of top 10 signal programs:")
    top10 = order[:10]
    sub = X[:, top10]
    corr = np.corrcoef(sub.T)
    names = [prog_cols[j].split("__")[0] + "/" + prog_cols[j].split("__")[1] for j in top10]
    print(" " * 15 + " ".join(f"{n[:11]:>11}" for n in names))
    for i, n in enumerate(names):
        row = " ".join(f"{corr[i, k]:>11.2f}" for k in range(10))
        print(f"  {n[:13]:<13} {row}")

    # 5) What does v2 task text actually look like? (sample inspection)
    print("\n=== sample of input text the programs see ===")
    dps = json.loads(DPS_FILE.read_text())
    sample_t = dps[0]["text"]
    print(f"  total length: {len(sample_t)} chars")
    print(f"  first 600 chars:")
    print("    " + sample_t[:600].replace("\n", "\n    "))
    # Does it contain a diff?
    has_diff = "diff --git" in sample_t or "@@" in sample_t
    print(f"\n  contains 'diff --git' or '@@' marker: {has_diff}")
    # Sample 50 datapoints
    n_has_diff = sum(1 for d in dps[:200] if "diff --git" in d.get("text", ""))
    print(f"  out of first 200 datapoints, contain diff: {n_has_diff}")

    # 6) Pull aspect names for top + bottom programs
    if ASPECTS_FILE.exists():
        aspects = json.loads(ASPECTS_FILE.read_text())
        if isinstance(aspects, dict) and "aspects" in aspects:
            aspects = aspects["aspects"]
        # Normalize: aspect ID format?
        # Programs name them a{ID}, so let's see
        print("\n=== Aspect names for top 10 ===")
        for j in order[:10]:
            aspect_id = prog_cols[j].split("__")[0]  # a359
            aspect_idx = int(aspect_id[1:])
            try:
                a = aspects[aspect_idx]
                nm = a.get("name") or a.get("text") or str(a)[:80]
                print(f"  {aspect_id} (rank{j}): {nm[:120]}")
            except (IndexError, KeyError, TypeError):
                print(f"  {aspect_id} (rank{j}): <not found>")

        # Bottom programs that are still computable
        print("\n=== Aspect names for 10 random LOW-signal programs (AUC ~ 0.5) ===")
        low_signal = [j for j in range(len(prog_cols))
                      if valid[j] and abs_auc[j] < 0.51 and std[j] > 0.05]
        np.random.seed(0)
        for j in np.random.choice(low_signal, size=min(10, len(low_signal)), replace=False):
            aspect_id = prog_cols[j].split("__")[0]
            aspect_idx = int(aspect_id[1:])
            try:
                a = aspects[aspect_idx]
                nm = a.get("name") or a.get("text") or str(a)[:80]
                print(f"  {aspect_id}: AUC={aucs[j]:.3f}  std={std[j]:.3f}  {nm[:100]}")
            except (IndexError, KeyError, TypeError):
                print(f"  {aspect_id}: <not found>")
    else:
        print(f"\n  no aspects.json at {ASPECTS_FILE}")

    # Save the per-program AUC table for follow-up
    out = pd.DataFrame({
        "program": prog_cols,
        "auc": aucs,
        "abs_auc_gap": abs_auc - 0.5,
        "std": std,
        "mean": mean,
        "frac_default": frac_default,
    })
    out_p = REPO / "outputs/v2_analysis/cr_codegen_perprogram_diagnostic.parquet"
    out.to_parquet(out_p)
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()
