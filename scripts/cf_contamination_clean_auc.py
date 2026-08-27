"""Recompute CF bank-feature AUC restricted to the contamination-clean subset.

Reads per-editorial contamination flags from cf_contamination_probe_per_editorial.parquet,
joins to cf_bank_scores + cf_pairs + cf label shards, then trains grouped 5-fold
LR + RF on all-vs-clean-vs-contam subsets.
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


def grouped_auc(X, y, g, n_splits=5):
    if len(np.unique(y)) < 2 or len(np.unique(g)) < 2:
        return None
    n_splits = min(n_splits, len(np.unique(g)))
    gkf = GroupKFold(n_splits=n_splits)
    lr_aucs, rf_aucs = [], []
    for tr, te in gkf.split(X, y, g):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
        rf = RandomForestClassifier(n_estimators=200, random_state=0,
                                    class_weight="balanced", n_jobs=4).fit(X[tr], y[tr])
        lr_aucs.append(roc_auc_score(y[te], lr.predict_proba(X[te])[:, 1]))
        rf_aucs.append(roc_auc_score(y[te], rf.predict_proba(X[te])[:, 1]))
    return {
        "lr_auc_mean": float(np.mean(lr_aucs)) if lr_aucs else None,
        "lr_auc_std": float(np.std(lr_aucs)) if lr_aucs else None,
        "rf_auc_mean": float(np.mean(rf_aucs)) if rf_aucs else None,
        "rf_auc_std": float(np.std(rf_aucs)) if rf_aucs else None,
        "n_pairs": int(len(X)),
        "n_groups": int(len(np.unique(g))),
        "pos_rate": float(np.mean(y)),
        "n_folds_used": len(lr_aucs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-editorial", default="/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/cf_contamination_probe_per_editorial.parquet")
    ap.add_argument("--probe-json", default="/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/cf_contamination_probe.json")
    ap.add_argument("--probe-md", default="/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/cf_contamination_probe.md")
    ap.add_argument("--bank-scores", default="/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/comp_fourplatform_cells/cf_bank_scores.parquet")
    ap.add_argument("--pairs", default="/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/comp_fourplatform_cells/cf_pairs.parquet")
    ap.add_argument("--labels-dir", default="/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/comp_fourplatform_label_shards/results")
    args = ap.parse_args()

    per_ed = pd.read_parquet(args.per_editorial)
    print("per-editorial rows:", len(per_ed))
    print("contaminated frac:", per_ed.contaminated.mean())

    clean_pids  = set(per_ed.loc[~per_ed.contaminated, "canonical_pid"].unique())
    contam_pids = set(per_ed.loc[ per_ed.contaminated, "canonical_pid"].unique())
    covered_pids = clean_pids | contam_pids
    print(f"probe coverage: {len(covered_pids)} pids ({len(clean_pids)} clean, {len(contam_pids)} contam)")

    scores = pd.read_parquet(args.bank_scores)
    pairs  = pd.read_parquet(args.pairs)
    print("scores:", scores.shape, "pairs:", pairs.shape)

    # collect labels from shards
    rows = []
    for fn in sorted(os.listdir(args.labels_dir)):
        if not (fn.startswith("shard_cf_") and fn.endswith(".jsonl")): continue
        with open(os.path.join(args.labels_dir, fn)) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    lab = pd.DataFrame(rows)
    print("label rows:", len(lab))
    # keep only label columns we need
    lab = lab[["pair_id","claude_label"]].rename(columns={"claude_label": "label_y"})
    lab = lab[lab.label_y.isin([0, 1])]
    print("valid 0/1 labels:", len(lab))

    feature_cols = [c for c in scores.columns if c.startswith("a") and (c.endswith("_score") or c.endswith("_applied"))]
    print("feature cols:", len(feature_cols))

    # merge: scores (drop claude_label/label_status to avoid collisions) + labels + pairs(canonical_pid)
    sc = scores.drop(columns=[c for c in ["claude_label","label_status","platform","language"] if c in scores.columns])
    merged = sc.merge(lab, on="pair_id", how="inner")
    merged = merged.merge(pairs[["pair_id","canonical_pid"]], on="pair_id", how="left")
    print("merged rows:", len(merged))

    # restrict to pairs whose canonical_pid is covered by probe
    merged_probe = merged[merged.canonical_pid.isin(covered_pids)].copy()
    merged_probe["clean_flag"]  = merged_probe.canonical_pid.isin(clean_pids)
    merged_probe["contam_flag"] = merged_probe.canonical_pid.isin(contam_pids)
    print("probe-covered labelled pairs:", len(merged_probe),
          "(clean:", int(merged_probe.clean_flag.sum()),
          "contam:", int(merged_probe.contam_flag.sum()), ")")

    X_full = merged[feature_cols].fillna(0.0).to_numpy()
    y_full = merged["label_y"].astype(int).to_numpy()
    g_full = merged["canonical_pid"].fillna("UNK").to_numpy()

    X_probe = merged_probe[feature_cols].fillna(0.0).to_numpy()
    y_probe = merged_probe["label_y"].astype(int).to_numpy()
    g_probe = merged_probe["canonical_pid"].fillna("UNK").to_numpy()

    results = {}
    print("computing full-cell AUC ...")
    results["full_eval_cell"] = grouped_auc(X_full, y_full, g_full)
    print("  ", results["full_eval_cell"])

    print("computing probe-covered AUC ...")
    results["probe_covered_all"] = grouped_auc(X_probe, y_probe, g_probe)
    print("  ", results["probe_covered_all"])

    mask = merged_probe.clean_flag.to_numpy()
    print(f"computing clean-subset AUC (n={mask.sum()}) ...")
    results["clean_subset"] = grouped_auc(X_probe[mask], y_probe[mask], g_probe[mask]) if mask.sum() else None
    print("  ", results["clean_subset"])

    mask = merged_probe.contam_flag.to_numpy()
    print(f"computing contam-subset AUC (n={mask.sum()}) ...")
    results["contam_subset"] = grouped_auc(X_probe[mask], y_probe[mask], g_probe[mask]) if mask.sum() else None
    print("  ", results["contam_subset"])

    # update probe JSON in place
    with open(args.probe_json) as f:
        payload = json.load(f)
    payload["clean_subset_auc"] = results
    payload["clean_subset_n_pids"] = len(clean_pids)
    payload["contam_subset_n_pids"] = len(contam_pids)
    with open(args.probe_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print("updated", args.probe_json)

    # update MD: append AUC block
    with open(args.probe_md) as f:
        md = f.read()
    md = md.split("## Clean-subset AUC")[0]  # drop any prior partial block
    md += "\n## Bank-feature AUC: full eval cell vs clean vs contaminated subset\n\n"
    md += "```json\n" + json.dumps(results, indent=2) + "\n```\n"
    with open(args.probe_md, "w") as f:
        f.write(md)
    print("updated", args.probe_md)


if __name__ == "__main__":
    main()
