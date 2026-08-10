"""Quick preview: score a sample of codegen programs against DIFF text now,
before the full 1182-program run finishes. Looks at:
  - the 17 programs that had |AUC-0.5|>0.05 on the v2 artifact (do they hold up?)
  - 50 random other programs (do new winners emerge?)
  - 30 known-degenerate programs that were stuck at 0.5 on the artifact

Reports per-program AUC on diff vs artifact for each.
"""
import importlib.util
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
TASK = "code_review"
CODEGEN_DIR = REPO / f"runs/validity_full/v2/{TASK}/codegen_claude"
DPS_FILE = REPO / f"runs/validity_full/v2/{TASK}/datapoints.json"
DENSE_FILE = REPO / "datasets/code-review/code_review_dense_4096tok.csv.gz"
ART_DIAG = REPO / "outputs/v2_analysis/cr_codegen_perprogram_diagnostic.parquet"


def load_score_fn(py_path: Path):
    spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception:
        return None
    fn = getattr(m, "score", None)
    return fn if callable(fn) else None


def main():
    print("Loading v2 + dense diff text...")
    dps = json.loads(DPS_FILE.read_text())
    dps = [d for d in dps if d.get("judgement") is not None and d.get("text")]
    v2 = pd.DataFrame([{
        "datapoint_id": d["datapoint_id"],
        "y": int(d["judgement"]),
        "title": (re.match(r"PR TITLE: ([^\n]+)", d["text"]) or [None, None])[1],
    } for d in dps]).dropna(subset=["title"])
    dense = pd.read_csv(DENSE_FILE, usecols=["text"])
    dense["title"] = dense["text"].str.extract(
        r"## PR Title\s*(.+?)(?:\n|$)", expand=False)
    j = v2.merge(dense.drop_duplicates("title", keep="first"),
                 on="title", how="left").dropna(subset=["text"])
    diff_texts = j["text"].tolist()
    y = j["y"].values.astype(int)
    print(f"  {len(diff_texts)} rows")

    # Choose programs to sample
    ad = pd.read_parquet(ART_DIAG)
    ad["abs_gap"] = (ad["auc"] - 0.5).abs()
    # The 17 known-strong on artifact
    survivors = ad.sort_values("abs_gap", ascending=False).head(17)["program"].tolist()
    # 50 random other programs
    rng = np.random.default_rng(7)
    other = ad[~ad["program"].isin(survivors)]
    sample_other = rng.choice(other["program"].values, size=50, replace=False).tolist()
    # 30 known-degenerate
    degen = ad[ad["frac_default"] >= 0.9]["program"].tolist()
    sample_degen = rng.choice(degen, size=min(30, len(degen)), replace=False).tolist()
    sample = survivors + sample_other + list(sample_degen)
    print(f"  sampling {len(sample)} programs ({len(survivors)} survivors + "
          f"{len(sample_other)} random + {len(sample_degen)} degen)")

    # Load each program file
    results = []
    for prog_name in sample:
        aspect_id, ver_flavor = prog_name.split("__")
        py_file = CODEGEN_DIR / f"{aspect_id}_{ver_flavor}.py"
        if not py_file.exists():
            continue
        fn = load_score_fn(py_file)
        if fn is None:
            continue
        scores = np.full(len(diff_texts), 0.5, dtype=np.float32)
        n_err = 0
        for i, t in enumerate(diff_texts):
            try:
                v = fn(t)
                if isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v)):
                    scores[i] = float(max(0.0, min(1.0, v)))
                else:
                    n_err += 1
            except Exception:
                n_err += 1
        std = float(scores.std())
        frac_def = float((scores == 0.5).mean())
        try:
            auc_diff = roc_auc_score(y, scores)
        except Exception:
            auc_diff = np.nan
        results.append({
            "program": prog_name,
            "bucket": ("survivor" if prog_name in survivors else
                       "degen" if prog_name in sample_degen else "random"),
            "auc_diff": auc_diff,
            "abs_gap_diff": abs(auc_diff - 0.5) if auc_diff == auc_diff else np.nan,
            "std_diff": std,
            "frac_default_diff": frac_def,
            "auc_artifact": ad.loc[ad["program"] == prog_name, "auc"].iloc[0],
            "abs_gap_artifact": ad.loc[ad["program"] == prog_name, "abs_gap"].iloc[0],
            "frac_default_artifact":
                ad.loc[ad["program"] == prog_name, "frac_default"].iloc[0],
        })
    res = pd.DataFrame(results)
    out = REPO / "outputs/v2_analysis/cr_codegen_diff_preview.parquet"
    res.to_parquet(out)

    print("\n=== Bucket summaries ===")
    for b in ["survivor", "random", "degen"]:
        sub = res[res["bucket"] == b]
        if not len(sub):
            continue
        print(f"\n{b.upper()}  (n={len(sub)})")
        print(f"  median |AUC-0.5| on diff      = {sub['abs_gap_diff'].median():.4f}")
        print(f"  median |AUC-0.5| on artifact  = {sub['abs_gap_artifact'].median():.4f}")
        print(f"  median std on diff            = {sub['std_diff'].median():.4f}")
        print(f"  median frac_default on diff   = {sub['frac_default_diff'].median():.3f}")
        print(f"  programs |AUC-0.5|>0.05 (diff) = "
              f"{(sub['abs_gap_diff'] > 0.05).sum()}/{len(sub)}")
        print(f"  programs |AUC-0.5|>0.10 (diff) = "
              f"{(sub['abs_gap_diff'] > 0.10).sum()}/{len(sub)}")

    print("\n=== Survivors: did they hold up on diff? ===")
    aspects = json.loads((REPO / f"runs/validity_full/v2/{TASK}/aspects.json").read_text())
    s = res[res["bucket"] == "survivor"].sort_values("abs_gap_diff", ascending=False)
    print(f"  {'program':<28} {'aucART':>7} {'aucDIFF':>8} {'gapART':>7} {'gapDIFF':>8} aspect")
    for _, r in s.iterrows():
        idx = int(r["program"].split("__")[0][1:])
        nm = aspects[idx].get("name") if idx < len(aspects) else "?"
        print(f"  {r['program']:<28} {r['auc_artifact']:.3f}   "
              f"{r['auc_diff']:.3f}   {r['abs_gap_artifact']:+.3f}  "
              f"{r['abs_gap_diff']:+.3f}   {nm}")

    print("\n=== New strong programs from random/degen sample (|AUC-0.5|>0.05 on diff) ===")
    new = res[(res["bucket"] != "survivor") & (res["abs_gap_diff"] > 0.05)]\
        .sort_values("abs_gap_diff", ascending=False)
    if len(new) == 0:
        print("  none — no random/degen-bucket programs cross 0.05 on diff input")
    else:
        for _, r in new.iterrows():
            idx = int(r["program"].split("__")[0][1:])
            nm = aspects[idx].get("name") if idx < len(aspects) else "?"
            print(f"  {r['program']:<28} bucket={r['bucket']:<8} "
                  f"aucART={r['auc_artifact']:.3f} -> aucDIFF={r['auc_diff']:.3f}  {nm}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
