"""Combined lasso v2: harder look at code + judge.

The v1 combined lasso (AUC 0.596) underperformed code-alone (AUC 0.620),
likely because LogisticRegressionCV picked a single C across 1448 features
and over-shrank.

This script tries:
  1. Combined L1 with WIDER Cs grid (20 values, more lenient than default 10).
  2. Combined ElasticNet (L1+L2 blend) — keeps correlated code/judge pairs.
  3. Combined Ridge (L2 only) — no feature selection, just regularization;
     answers "if we don't force sparsity, does adding judge HELP?".
  4. Stacked / "best of both" — code-OOF-prob AND judge-OOF-prob as the only
     two features in a final logistic. Asks: are judge predictions a useful
     second opinion if we don't try to fit individual judge features?

Reuses the same feature-building logic as validity_lasso.py.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, "scripts")
from validity_lasso import load_codes, load_judge, to_feature_matrix, mean


def main():
    base = Path("runs/validity_full/full_v1")
    datapoints = json.loads((base / "datapoints.json").read_text())
    labels = np.array([d["judgement"] for d in datapoints])
    dp_ids = [d["datapoint_id"] for d in datapoints]

    r1_metrics = json.loads((base / "r1_metrics.json").read_text())
    r2_aspects = json.loads((base / "r2_aspects.json").read_text())
    r1_ids = [m["metric_id"] for m in r1_metrics]
    r2_ids = [a["aspect_id"] for a in r2_aspects]
    r1_names = {m["metric_id"]: m["name"] for m in r1_metrics}
    r2_names = {a["aspect_id"]: a["name"] for a in r2_aspects}

    print("loading scores...")
    qwen_r1_code = load_codes(base / "codegen_exec_results_qwen_all.jsonl")
    qwen_r2_code = load_codes(base / "codegen_exec_results_qwen_r2.jsonl",
                               mid_field="aspect_id")
    r1_judge = load_judge(base, "judge_r1_responses",
                           "judge_r1_manifest.json", "metric_id")
    r2_judge = load_judge(base, "judge_responses_llama",
                           "judge_manifest.json", "aspect_id")

    print("building matrices...")
    X_r1_code = to_feature_matrix(qwen_r1_code, dp_ids, r1_ids)
    X_r2_code = to_feature_matrix(qwen_r2_code, dp_ids, r2_ids)
    X_r1_judge = to_feature_matrix(r1_judge, dp_ids, r1_ids)
    X_r2_judge = to_feature_matrix(r2_judge, dp_ids, r2_ids)

    X_code = np.hstack([X_r1_code, X_r2_code])
    X_judge = np.hstack([X_r1_judge, X_r2_judge])
    X_all = np.hstack([X_code, X_judge])

    names_code = ([f"code_r1_{r1_names[m]}" for m in r1_ids] +
                  [f"code_r2_{r2_names[a]}" for a in r2_ids])
    names_judge = ([f"judge_r1_{r1_names[m]}" for m in r1_ids] +
                   [f"judge_r2_{r2_names[a]}" for a in r2_ids])
    names_all = names_code + names_judge

    print(f"  X_code: {X_code.shape}, X_judge: {X_judge.shape}, "
          f"X_all: {X_all.shape}, y mean: {labels.mean():.3f}")

    from sklearn.linear_model import (LogisticRegressionCV, LogisticRegression,
                                       SGDClassifier)
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    def fit_l1(X, names, label, Cs):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = LogisticRegressionCV(
            penalty="l1", solver="saga", Cs=Cs, cv=cv,
            scoring="roc_auc", max_iter=10000, n_jobs=-1, random_state=0,
        )
        clf.fit(Xs, labels)
        oof = cross_val_predict(
            LogisticRegression(penalty="l1", solver="saga",
                                C=clf.C_[0], max_iter=10000),
            Xs, labels, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        auc = roc_auc_score(labels, oof)
        coefs = clf.coef_[0]
        nz = [(names[j], coefs[j]) for j in range(len(coefs))
              if abs(coefs[j]) > 1e-6]
        nz.sort(key=lambda x: -abs(x[1]))
        return {"C": float(clf.C_[0]), "n_selected": len(nz),
                "auc": float(auc), "top": nz[:20], "oof": oof}

    def fit_elasticnet(X, names, label, Cs, l1_ratios):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = LogisticRegressionCV(
            penalty="elasticnet", solver="saga", Cs=Cs, l1_ratios=l1_ratios,
            cv=cv, scoring="roc_auc", max_iter=10000, n_jobs=-1, random_state=0,
        )
        clf.fit(Xs, labels)
        oof = cross_val_predict(
            LogisticRegression(penalty="elasticnet", solver="saga",
                                C=clf.C_[0], l1_ratio=clf.l1_ratio_[0],
                                max_iter=10000),
            Xs, labels, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        auc = roc_auc_score(labels, oof)
        coefs = clf.coef_[0]
        nz = [(names[j], coefs[j]) for j in range(len(coefs))
              if abs(coefs[j]) > 1e-6]
        nz.sort(key=lambda x: -abs(x[1]))
        return {"C": float(clf.C_[0]), "l1_ratio": float(clf.l1_ratio_[0]),
                "n_selected": len(nz), "auc": float(auc), "top": nz[:20],
                "oof": oof}

    def fit_l2(X, names, label, Cs):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = LogisticRegressionCV(
            penalty="l2", solver="lbfgs", Cs=Cs, cv=cv,
            scoring="roc_auc", max_iter=10000, n_jobs=-1, random_state=0,
        )
        clf.fit(Xs, labels)
        oof = cross_val_predict(
            LogisticRegression(penalty="l2", solver="lbfgs",
                                C=clf.C_[0], max_iter=10000),
            Xs, labels, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        auc = roc_auc_score(labels, oof)
        coefs = clf.coef_[0]
        # L2 doesn't zero out; report top-20 by |coef|
        top_idx = np.argsort(-np.abs(coefs))[:20]
        top = [(names[j], coefs[j]) for j in top_idx]
        return {"C": float(clf.C_[0]), "n_selected": int(len(coefs)),
                "auc": float(auc), "top": top, "oof": oof}

    Cs_wide = np.logspace(-3, 2, 20).tolist()
    Cs_default = 10

    results = {}

    print("\n--- (a) CODE-only L1 (baseline) ---")
    r = fit_l1(X_code, names_code, "code_l1", Cs_default)
    print(f"  C={r['C']:.4g}  selected={r['n_selected']}  AUC={r['auc']:.3f}")
    results["code_l1"] = r

    print("\n--- (b) JUDGE-only L1 (baseline) ---")
    r = fit_l1(X_judge, names_judge, "judge_l1", Cs_default)
    print(f"  C={r['C']:.4g}  selected={r['n_selected']}  AUC={r['auc']:.3f}")
    results["judge_l1"] = r

    print("\n--- (c) COMBINED L1 with WIDER Cs grid (20 values) ---")
    r = fit_l1(X_all, names_all, "combined_l1_wide", Cs_wide)
    print(f"  C={r['C']:.4g}  selected={r['n_selected']}  AUC={r['auc']:.3f}")
    n_code = sum(1 for n, c in r["top"] if n.startswith("code_"))
    n_judge = sum(1 for n, c in r["top"] if n.startswith("judge_"))
    print(f"  top-20: {n_code} code, {n_judge} judge")
    for n, c in r["top"][:15]:
        print(f"    {c:+.3f}  {n[:80]}")
    results["combined_l1_wide"] = r

    print("\n--- (d) COMBINED ElasticNet (L1+L2) ---")
    r = fit_elasticnet(X_all, names_all, "combined_elastic",
                       Cs_wide, [0.1, 0.3, 0.5, 0.7, 0.9])
    print(f"  C={r['C']:.4g}  l1_ratio={r['l1_ratio']:.2f}  "
          f"selected={r['n_selected']}  AUC={r['auc']:.3f}")
    n_code = sum(1 for n, c in r["top"] if n.startswith("code_"))
    n_judge = sum(1 for n, c in r["top"] if n.startswith("judge_"))
    print(f"  top-20: {n_code} code, {n_judge} judge")
    for n, c in r["top"][:15]:
        print(f"    {c:+.3f}  {n[:80]}")
    results["combined_elastic"] = r

    print("\n--- (e) COMBINED Ridge (L2 only, no sparsity) ---")
    r = fit_l2(X_all, names_all, "combined_l2", Cs_wide)
    print(f"  C={r['C']:.4g}  AUC={r['auc']:.3f}")
    n_code = sum(1 for n, c in r["top"] if n.startswith("code_"))
    n_judge = sum(1 for n, c in r["top"] if n.startswith("judge_"))
    print(f"  top-20 by |coef|: {n_code} code, {n_judge} judge")
    for n, c in r["top"][:15]:
        print(f"    {c:+.3f}  {n[:80]}")
    results["combined_l2"] = r

    print("\n--- (f) STACKED: just [code-OOF-prob, judge-OOF-prob] as features ---")
    code_oof = results["code_l1"]["oof"].reshape(-1, 1)
    judge_oof = results["judge_l1"]["oof"].reshape(-1, 1)
    Xs_stack = np.hstack([code_oof, judge_oof])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xs_stack)
    clf = LogisticRegressionCV(
        penalty="l2", solver="lbfgs", Cs=Cs_wide, cv=cv,
        scoring="roc_auc", max_iter=10000, n_jobs=-1, random_state=0,
    )
    clf.fit(Xs, labels)
    oof = cross_val_predict(
        LogisticRegression(penalty="l2", solver="lbfgs",
                            C=clf.C_[0], max_iter=10000),
        Xs, labels, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    auc = roc_auc_score(labels, oof)
    print(f"  AUC={auc:.3f}")
    print(f"  coefs: code_oof={clf.coef_[0][0]:+.3f}  judge_oof={clf.coef_[0][1]:+.3f}")
    results["stacked_oofs"] = {"auc": float(auc),
                                "coef_code": float(clf.coef_[0][0]),
                                "coef_judge": float(clf.coef_[0][1])}

    print("\n\n=== SUMMARY ===")
    print(f"{'Model':<35} {'AUC':>6} {'#sel':>6} {'C':>10}")
    for k, r in results.items():
        c = f"{r.get('C', 0):.3g}" if r.get('C') else "-"
        n = r.get('n_selected', '-')
        print(f"  {k:<33} {r['auc']:>6.3f} {str(n):>6} {c:>10}")

    out = {k: {kk: vv for kk, vv in v.items() if kk != "oof"}
           for k, v in results.items()}
    (base / "lasso_combined_v2.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote runs/validity_full/full_v1/lasso_combined_v2.json")


if __name__ == "__main__":
    main()
