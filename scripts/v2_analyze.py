"""v2 analysis: per-aspect stats + lasso AUC for code-only / judge-only /
combined / stacked.

Features = per (aspect, datapoint) mean score across paraphrases / variants.
- Code: 3 variants per aspect → mean (also keep per-variant for richer feature set)
- Judge: 3 paraphrases per aspect → mean

Reports:
  runs/validity_full/full_v2/analysis_per_aspect.json   per-aspect stats
  runs/validity_full/full_v2/analysis_summary.md        human-readable
  runs/validity_full/full_v2/lasso_results.json         lasso AUCs
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


def mean(xs): return statistics.mean(xs) if xs else 0.0
def std(xs):  return statistics.pstdev(xs) if len(xs) >= 2 else 0.0
def pearson(xs, ys):
    if len(xs) < 2: return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx = sum((x-mx)**2 for x in xs)**.5
    dy = sum((y-my)**2 for y in ys)**.5
    return num / max(dx*dy, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/validity_full/full_v2")
    args = ap.parse_args()

    base = Path(args.run_dir)
    datapoints = json.loads((base / "datapoints.json").read_text())
    labels = {d["datapoint_id"]: d["judgement"] for d in datapoints}
    dp_ids = [d["datapoint_id"] for d in datapoints]
    label_arr = np.array([labels[dp] for dp in dp_ids])

    aspects = json.loads(
        (Path("runs/validity_full/full_v1") / "r2_aspects.json").read_text())
    aspect_by_id = {a["aspect_id"]: a for a in aspects}

    # ===== Load code scores =====
    print("loading code scores...")
    # aspect_id -> variant -> dp -> score
    code = defaultdict(lambda: defaultdict(dict))
    code_path = base / "codegen_exec_results.jsonl"
    if code_path.exists():
        for line in code_path.open():
            r = json.loads(line)
            if r["score"] is None: continue
            code[r["aspect_id"]][r["variant"]][r["datapoint_id"]] = r["score"]
    print(f"  code: {len(code)} aspects with scores")

    # ===== Load judge scores =====
    print("loading judge scores...")
    # aspect_id -> paraphrase_idx -> dp -> score (only applicable ones)
    judge = defaultdict(lambda: defaultdict(dict))
    judge_path = base / "judge_scores.jsonl"
    if judge_path.exists():
        for line in judge_path.open():
            r = json.loads(line)
            if r["score"] is None: continue
            judge[r["aspect_id"]][r["paraphrase_idx"]][r["datapoint_id"]] = r["score"]
    print(f"  judge: {len(judge)} aspects with scores")

    # ===== Per-aspect stats =====
    per_aspect = []
    for aid, asp in aspect_by_id.items():
        row = {"aspect_id": aid, "name": asp["name"]}

        # Code per variant
        code_variants = code.get(aid, {})
        for variant, dp_scores in code_variants.items():
            xs = [dp_scores[d] for d in dp_ids if d in dp_scores]
            ys = [labels[d] for d in dp_ids if d in dp_scores]
            row[f"code_{variant}_n"] = len(xs)
            row[f"code_{variant}_mean"] = round(mean(xs), 3)
            row[f"code_{variant}_std"] = round(std(xs), 3)
            row[f"code_{variant}_label_rho"] = round(pearson(xs, ys), 3)
        # Code variant stability: per dp std across variants
        if len(code_variants) >= 2:
            dp_to_variant_scores = defaultdict(list)
            for vname, dp_scores in code_variants.items():
                for d, s in dp_scores.items():
                    dp_to_variant_scores[d].append(s)
            stds = [std(v) for v in dp_to_variant_scores.values() if len(v) >= 2]
            row["code_variant_sigma"] = round(mean(stds), 3)
            # Mean across variants per dp = aggregated code score
            mean_xs = [mean(v) for d, v in dp_to_variant_scores.items()
                       if d in labels]
            mean_dps = [d for d in dp_to_variant_scores if d in labels]
            ys = [labels[d] for d in mean_dps]
            row["code_mean_label_rho"] = round(pearson(mean_xs, ys), 3)
            row["code_n_dp"] = len(mean_dps)

        # Judge per paraphrase
        judge_paras = judge.get(aid, {})
        for p_idx, dp_scores in judge_paras.items():
            xs = [dp_scores[d] for d in dp_ids if d in dp_scores]
            ys = [labels[d] for d in dp_ids if d in dp_scores]
            row[f"judge_p{p_idx}_n"] = len(xs)
            row[f"judge_p{p_idx}_mean"] = round(mean(xs), 3)
            row[f"judge_p{p_idx}_label_rho"] = round(pearson(xs, ys), 3)
        if len(judge_paras) >= 2:
            dp_to_para_scores = defaultdict(list)
            for p_idx, dp_scores in judge_paras.items():
                for d, s in dp_scores.items():
                    dp_to_para_scores[d].append(s)
            stds = [std(v) for v in dp_to_para_scores.values() if len(v) >= 2]
            row["judge_para_sigma"] = round(mean(stds), 3)
            mean_xs = [mean(v) for d, v in dp_to_para_scores.items()
                       if d in labels]
            mean_dps = [d for d in dp_to_para_scores if d in labels]
            ys = [labels[d] for d in mean_dps]
            row["judge_mean_label_rho"] = round(pearson(mean_xs, ys), 3)
            row["judge_n_dp"] = len(mean_dps)

        # Convergent: code mean vs judge mean
        if "code_n_dp" in row and "judge_n_dp" in row:
            code_per_dp = {d: mean(v)
                            for d, v in dp_to_variant_scores.items()}
            # Re-collect judge dp scores
            judge_per_dp = {}
            for p_idx, dp_scores in judge_paras.items():
                for d, s in dp_scores.items():
                    judge_per_dp.setdefault(d, []).append(s)
            judge_per_dp = {d: mean(v) for d, v in judge_per_dp.items()}
            common = sorted(set(code_per_dp) & set(judge_per_dp))
            if len(common) >= 5:
                cx = [code_per_dp[d] for d in common]
                jx = [judge_per_dp[d] for d in common]
                row["convergent_rho"] = round(pearson(cx, jx), 3)
                row["n_common"] = len(common)

        per_aspect.append(row)

    (base / "analysis_per_aspect.json").write_text(json.dumps(per_aspect, indent=1))
    print(f"  wrote analysis_per_aspect.json ({len(per_aspect)} aspects)")

    # ===== Lasso AUC =====
    print("\nbuilding lasso feature matrices...")
    aspect_ids_with_code = [r["aspect_id"] for r in per_aspect
                             if "code_n_dp" in r]
    aspect_ids_with_judge = [r["aspect_id"] for r in per_aspect
                              if "judge_n_dp" in r]

    def to_X(scores_dict, ids):
        X = np.full((len(dp_ids), len(ids)), 0.5, dtype=np.float32)
        for j, aid in enumerate(ids):
            per_obj = scores_dict.get(aid, {})
            for i, d in enumerate(dp_ids):
                vs = [per_obj[k].get(d) for k in per_obj if d in per_obj[k]]
                vs = [v for v in vs if v is not None]
                if vs:
                    X[i, j] = float(mean(vs))
        return X

    X_code = to_X(code, aspect_ids_with_code)
    X_judge = to_X(judge, aspect_ids_with_judge)
    print(f"  X_code: {X_code.shape}  X_judge: {X_judge.shape}")

    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    def lasso_cv(X, y, names, label):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        clf = LogisticRegressionCV(
            penalty="l1", solver="saga", Cs=15, cv=cv,
            scoring="roc_auc", max_iter=10000, n_jobs=-1, random_state=0)
        clf.fit(Xs, y)
        oof = cross_val_predict(
            LogisticRegression(penalty="l1", solver="saga",
                                C=clf.C_[0], max_iter=10000),
            Xs, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        auc = roc_auc_score(y, oof)
        coefs = clf.coef_[0]
        nonzero = [(names[j], float(coefs[j])) for j in range(len(coefs))
                   if abs(coefs[j]) > 1e-6]
        nonzero.sort(key=lambda x: -abs(x[1]))
        return {"label": label, "C": float(clf.C_[0]),
                "n_selected": len(nonzero), "oof_auc": float(auc),
                "top_features": nonzero[:25], "oof_preds": oof}

    names_code = [f"code_{aid}_{aspect_by_id[aid]['name']}"
                  for aid in aspect_ids_with_code]
    names_judge = [f"judge_{aid}_{aspect_by_id[aid]['name']}"
                   for aid in aspect_ids_with_judge]

    print("\nLASSO: code-only")
    r_code = lasso_cv(X_code, label_arr, names_code, "code")
    print(f"  AUC: {r_code['oof_auc']:.3f}  ({r_code['n_selected']} feats selected)")

    print("\nLASSO: judge-only")
    r_judge = lasso_cv(X_judge, label_arr, names_judge, "judge")
    print(f"  AUC: {r_judge['oof_auc']:.3f}  ({r_judge['n_selected']} feats selected)")

    print("\nLASSO: combined")
    X_all = np.hstack([X_code, X_judge])
    names_all = names_code + names_judge
    r_both = lasso_cv(X_all, label_arr, names_all, "combined")
    print(f"  AUC: {r_both['oof_auc']:.3f}  ({r_both['n_selected']} feats selected)")
    n_c = sum(1 for n, c in r_both["top_features"] if n.startswith("code_"))
    n_j = sum(1 for n, c in r_both["top_features"] if n.startswith("judge_"))
    print(f"  top-25: {n_c} code / {n_j} judge")

    print("\nLASSO: stacked (code-OOF + judge-OOF only)")
    X_stack = np.column_stack([r_code["oof_preds"], r_judge["oof_preds"]])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    Xs = StandardScaler().fit_transform(X_stack)
    clf = LogisticRegressionCV(penalty="l2", solver="lbfgs",
                                Cs=15, cv=cv, scoring="roc_auc",
                                max_iter=10000, n_jobs=-1)
    clf.fit(Xs, label_arr)
    oof = cross_val_predict(
        LogisticRegression(penalty="l2", C=clf.C_[0], max_iter=10000),
        Xs, label_arr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    auc_stack = roc_auc_score(label_arr, oof)
    print(f"  AUC: {auc_stack:.3f}  coefs: code={clf.coef_[0][0]:+.3f} "
          f"judge={clf.coef_[0][1]:+.3f}")

    out = {}
    for k, r in [("code", r_code), ("judge", r_judge), ("combined", r_both)]:
        out[k] = {kk: vv for kk, vv in r.items() if kk != "oof_preds"}
    out["stacked"] = {"auc": float(auc_stack),
                       "coef_code": float(clf.coef_[0][0]),
                       "coef_judge": float(clf.coef_[0][1])}
    (base / "lasso_results.json").write_text(json.dumps(out, indent=1))

    # ===== Summary markdown =====
    lines = ["# v2 validity results", "",
             f"n_datapoints: {len(dp_ids)}",
             f"label balance: {int(sum(label_arr))} accept / "
             f"{len(label_arr)-int(sum(label_arr))} reject",
             f"aspects with code: {len(aspect_ids_with_code)}",
             f"aspects with judge: {len(aspect_ids_with_judge)}",
             "",
             "## Lasso AUC",
             "",
             "| Model | AUC | # features selected |",
             "|---|---|---|",
             f"| Code only | {r_code['oof_auc']:.3f} | {r_code['n_selected']} |",
             f"| Judge only | {r_judge['oof_auc']:.3f} | {r_judge['n_selected']} |",
             f"| Combined | {r_both['oof_auc']:.3f} | {r_both['n_selected']} |",
             f"| Stacked (code-OOF + judge-OOF) | {auc_stack:.3f} | 2 |",
             "",
             "## Top code features (combined model)",
             "",
             "| Feature | Coef |",
             "|---|---|"]
    for n, c in [(n, c) for n, c in r_both["top_features"]
                 if n.startswith("code_")][:15]:
        lines.append(f"| {n[:70]} | {c:+.3f} |")
    lines += ["", "## Top judge features (combined model)", "",
              "| Feature | Coef |", "|---|---|"]
    for n, c in [(n, c) for n, c in r_both["top_features"]
                 if n.startswith("judge_")][:15]:
        lines.append(f"| {n[:70]} | {c:+.3f} |")
    (base / "analysis_summary.md").write_text("\n".join(lines))
    print(f"\nwrote {base/'analysis_summary.md'}")


if __name__ == "__main__":
    main()
