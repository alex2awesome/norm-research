"""Lasso analysis: how well do code metrics predict accept/reject?
   How much does the judge add beyond code?

Builds feature matrices from per-(metric, datapoint) scores averaged across
5 paraphrases. Runs:
  1. Lasso on code-only features (R1 Qwen + R2 Qwen direct)
  2. Lasso on judge-only features (R1 + R2 judge)
  3. Lasso on combined (code + judge) — which features dominate?
  4. Residualization: regress label on code-predicted-probability, then
     fit judge on residual → AUC lift from judge given code

5-fold cross-validation throughout; AUC reported.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
import re

import numpy as np


def mean(xs): return statistics.mean(xs) if xs else 0.0


def parse_score(p):
    raw = p.read_text().strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    if m: raw = m.group(1).strip()
    try: return json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        return json.loads(raw[s:e + 1])


def load_codes(path, mid_field="metric_id"):
    d = defaultdict(lambda: defaultdict(dict))
    if not Path(path).exists(): return d
    for line in open(path):
        r = json.loads(line)
        if r.get("score") is None: continue
        d[r[mid_field]][r["paraphrase_idx"]][r["datapoint_id"]] = r["score"]
    return d


def load_judge(base, sub_dir, manifest_file, mid_field):
    d = defaultdict(lambda: defaultdict(dict))
    manifest = json.loads((base / manifest_file).read_text())
    rdir = base / sub_dir
    for entry in manifest:
        rp = rdir / f"{entry['key']}.json"
        if not rp.exists(): continue
        try:
            obj = parse_score(rp)
            for sc in obj.get("scores", []):
                dp_id = sc.get("id"); s = sc.get("score")
                if dp_id is None or s is None: continue
                d[entry[mid_field]][entry["paraphrase_idx"]][dp_id] = s / 10.0
        except Exception:
            continue
    return d


def to_feature_matrix(scores_per_metric, dp_ids, metric_ids):
    """Each (dp, metric) cell = mean across paraphrases. Missing → 0.5 (neutral)."""
    X = np.full((len(dp_ids), len(metric_ids)), 0.5, dtype=np.float32)
    for j, mid in enumerate(metric_ids):
        per_para = scores_per_metric.get(mid, {})
        for i, dp in enumerate(dp_ids):
            vs = [per_para[pi].get(dp) for pi in per_para if dp in per_para[pi]]
            vs = [v for v in vs if v is not None]
            if vs:
                X[i, j] = float(mean(vs))
    return X


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

    # Build feature matrices
    print("building matrices...")
    X_r1_code = to_feature_matrix(qwen_r1_code, dp_ids, r1_ids)
    X_r2_code = to_feature_matrix(qwen_r2_code, dp_ids, r2_ids)
    X_r1_judge = to_feature_matrix(r1_judge, dp_ids, r1_ids)
    X_r2_judge = to_feature_matrix(r2_judge, dp_ids, r2_ids)

    X_code = np.hstack([X_r1_code, X_r2_code])
    X_judge = np.hstack([X_r1_judge, X_r2_judge])
    X_all = np.hstack([X_code, X_judge])

    feature_names_code = (
        [f"code_r1_{r1_names[m]}" for m in r1_ids] +
        [f"code_r2_{r2_names[a]}" for a in r2_ids]
    )
    feature_names_judge = (
        [f"judge_r1_{r1_names[m]}" for m in r1_ids] +
        [f"judge_r2_{r2_names[a]}" for a in r2_ids]
    )
    feature_names_all = feature_names_code + feature_names_judge

    print(f"  X_code:  {X_code.shape}")
    print(f"  X_judge: {X_judge.shape}")
    print(f"  X_all:   {X_all.shape}")
    print(f"  y mean:  {labels.mean():.3f}")

    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    def lasso_cv(X, y, names, label, n_splits=5):
        """5-fold CV: get out-of-fold predictions, AUC, and selected features."""
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        # Logistic with L1 (lasso-equivalent). Use saga solver.
        clf = LogisticRegressionCV(
            penalty="l1", solver="saga", Cs=10, cv=cv,
            scoring="roc_auc", max_iter=10000, n_jobs=-1, random_state=0,
        )
        clf.fit(Xs, y)
        # Cross-validated AUC for honest estimate
        oof = cross_val_predict(
            LogisticRegression(penalty="l1", solver="saga",
                                C=clf.C_[0], max_iter=10000),
            Xs, y, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        auc = roc_auc_score(y, oof)
        # Selected features (non-zero coefficients in full fit)
        coefs = clf.coef_[0]
        nonzero = [(names[j], coefs[j]) for j in range(len(coefs)) if abs(coefs[j]) > 1e-6]
        nonzero.sort(key=lambda x: -abs(x[1]))
        return {
            "label": label,
            "C": float(clf.C_[0]),
            "n_features_selected": len(nonzero),
            "oof_auc": float(auc),
            "top_features": nonzero[:20],
            "oof_preds": oof,
        }

    print("\n--- CODE-ONLY LASSO ---")
    r_code = lasso_cv(X_code, labels, feature_names_code, "code")
    print(f"  selected {r_code['n_features_selected']}/{X_code.shape[1]} features")
    print(f"  5-fold AUC: {r_code['oof_auc']:.3f}")
    print(f"  Top 15 features:")
    for n, c in r_code["top_features"][:15]:
        print(f"    {c:+.3f}  {n[:80]}")

    print("\n--- JUDGE-ONLY LASSO ---")
    r_judge = lasso_cv(X_judge, labels, feature_names_judge, "judge")
    print(f"  selected {r_judge['n_features_selected']}/{X_judge.shape[1]} features")
    print(f"  5-fold AUC: {r_judge['oof_auc']:.3f}")
    print(f"  Top 15 features:")
    for n, c in r_judge["top_features"][:15]:
        print(f"    {c:+.3f}  {n[:80]}")

    print("\n--- COMBINED LASSO ---")
    r_both = lasso_cv(X_all, labels, feature_names_all, "both")
    print(f"  selected {r_both['n_features_selected']}/{X_all.shape[1]} features")
    print(f"  5-fold AUC: {r_both['oof_auc']:.3f}")
    n_code_sel = sum(1 for n, c in r_both["top_features"] if n.startswith("code_"))
    n_judge_sel = sum(1 for n, c in r_both["top_features"] if n.startswith("judge_"))
    print(f"  In top 20: {n_code_sel} code features, {n_judge_sel} judge features")
    print(f"  Top 15 features:")
    for n, c in r_both["top_features"][:15]:
        print(f"    {c:+.3f}  {n[:80]}")

    # Residualize: judge features on code OOF prediction
    print("\n--- RESIDUAL JUDGE (judge → label, controlling for code OOF prob) ---")
    # Add code OOF prob as a single "summary feature" that captures code's predictive power
    # Then ask: do judge features explain the residual?
    X_judge_plus_code_prob = np.hstack([X_judge, r_code["oof_preds"].reshape(-1, 1)])
    names_jpcp = feature_names_judge + ["__code_oof_prob__"]
    r_resid = lasso_cv(X_judge_plus_code_prob, labels, names_jpcp,
                       "judge|code_prob")
    print(f"  selected {r_resid['n_features_selected']} features")
    print(f"  5-fold AUC: {r_resid['oof_auc']:.3f}")
    # How much extra AUC does judge add?
    print(f"  Code-only AUC:  {r_code['oof_auc']:.3f}")
    print(f"  + Judge AUC:    {r_resid['oof_auc']:.3f}  "
          f"(lift: {r_resid['oof_auc']-r_code['oof_auc']:+.3f})")
    print(f"  Top 10 features in residual model:")
    for n, c in r_resid["top_features"][:10]:
        marker = "★" if n == "__code_oof_prob__" else " "
        print(f"   {marker}{c:+.3f}  {n[:80]}")

    # Save full results
    out = {}
    for k, r in [("code", r_code), ("judge", r_judge), ("both", r_both),
                  ("judge_given_code", r_resid)]:
        out[k] = {
            "C": r["C"],
            "n_features_selected": r["n_features_selected"],
            "oof_auc": r["oof_auc"],
            "top_30_features": r["top_features"][:30],
        }
    (base / "lasso_results.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {base}/lasso_results.json")


if __name__ == "__main__":
    main()
