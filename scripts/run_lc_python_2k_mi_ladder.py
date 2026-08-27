"""MI ladder on the Claude-labeled 2K LC Python set: bank vs bank+a478-a482.

Loads:
  - Existing bank scores (lc_python_metric_scores.parquet)
  - New runnability scores (lc_python_runnability_scores.parquet)
  - Claude labels (lc_python_2k_claude_labels.parquet) -- label is 0/1
    "is this candidate similar in approach to the editorial?"

Evaluations (GroupKFold by question_slug, k=5):
  - bank_only       (RF, LR)
  - bank_plus_new   (RF, LR)
  - new_only        (RF, LR)
  - tfidf_baseline  (LR, char_wb 3-5)

Reports per-decile pass rate, per-metric importance for a478-a482, and
the per-decile correlation between max_sim and a480.

Output: outputs/v2_analysis/lc_python_2k_runnability_mi_report.json
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["HOME"] = "/lfs/skampere3/0/alexspan"
REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
BASE = REPO / "outputs/v2_analysis"

BANK_SCORES = BASE / "lc_python_metric_scores.parquet"
RUNN_SCORES = BASE / "lc_python_runnability_scores.parquet"
CLAUDE_LABELS = BASE / "lc_python_2k_claude_labels.parquet"
SAMPLE_2K = BASE / "lc_python_2k_sample.parquet"
CAND_CORPUS = BASE / "lc_candidate_corpus.parquet"

OUT_REPORT = BASE / "lc_python_2k_runnability_mi_report.json"

NEW_METRIC_IDS = ["a478", "a479", "a480", "a481", "a482"]
N_FOLDS = 5
SEED = 42

sys.path.insert(0, str(REPO))


# ------- AST-only a478/a479 ------- #

_STDLIB = set(getattr(sys, "stdlib_module_names", ()) or ())
_STDLIB.update({
    "abc", "argparse", "array", "ast", "asyncio", "base64", "binascii",
    "bisect", "builtins", "calendar", "cmath", "collections", "concurrent",
    "contextlib", "copy", "csv", "ctypes", "datetime", "decimal", "dis",
    "enum", "errno", "fnmatch", "fractions", "functools", "gc", "getopt",
    "glob", "gzip", "hashlib", "heapq", "hmac", "html", "http", "importlib",
    "inspect", "io", "ipaddress", "itertools", "json", "logging", "math",
    "multiprocessing", "numbers", "operator", "os", "pathlib", "pickle",
    "pprint", "queue", "random", "re", "shutil", "signal", "socket",
    "sqlite3", "statistics", "string", "struct", "subprocess", "sys",
    "tempfile", "textwrap", "threading", "time", "timeit", "token",
    "tokenize", "trace", "traceback", "types", "typing", "unicodedata",
    "unittest", "urllib", "uuid", "warnings", "weakref", "xml", "zipfile",
    "zlib", "__future__",
})
_THIRD_PARTY = {
    "numpy", "np", "scipy", "pandas", "pd", "sortedcontainers",
    "more_itertools", "blist", "networkx", "sympy", "matplotlib",
    "pytest", "tqdm", "regex", "z3", "pulp", "cvxpy", "intervaltree",
    "bitarray", "torch", "tensorflow", "keras", "sklearn",
}


def score_a478(code: str):
    if not isinstance(code, str) or not code.strip():
        return None
    try:
        ast.parse(code)
        return 1.0
    except SyntaxError:
        return 0.0
    except Exception:
        return None


def score_a479(code: str):
    if not isinstance(code, str) or not code.strip():
        return None
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    total = 0
    ok = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                total += 1
                if top in _STDLIB or top in _THIRD_PARTY:
                    ok += 1
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            top = node.module.split(".", 1)[0]
            total += 1
            if top in _STDLIB or top in _THIRD_PARTY:
                ok += 1
    if total == 0:
        return None
    return ok / total


# ------- CV helpers ------- #

def _safe_auc(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def _eval_cv(X, y, groups, model_kind, seed=SEED, n_folds=N_FOLDS):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    aucs = []
    gkf = GroupKFold(n_splits=n_folds)
    for tr, te in gkf.split(X, y, groups):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        imp = SimpleImputer(strategy="median")
        Xtr_i = imp.fit_transform(Xtr)
        Xte_i = imp.transform(Xte)
        if model_kind == "rf":
            m = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                       n_jobs=-1, class_weight="balanced",
                                       random_state=seed)
            m.fit(Xtr_i, ytr)
            p = m.predict_proba(Xte_i)[:, 1]
        else:
            sc = StandardScaler(with_mean=False)
            Xtr_s = sc.fit_transform(Xtr_i)
            Xte_s = sc.transform(Xte_i)
            m = LogisticRegression(C=1.0, max_iter=2000,
                                   class_weight="balanced",
                                   random_state=seed)
            m.fit(Xtr_s, ytr)
            p = m.predict_proba(Xte_s)[:, 1]
        aucs.append(_safe_auc(yte, p))
    aucs = [a for a in aucs if not math.isnan(a)]
    if not aucs:
        return float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.std(aucs))


def _rf_importance(X, y, groups, feature_names, seed=SEED, n_folds=N_FOLDS):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_folds)
    imps = np.zeros(X.shape[1])
    n_used = 0
    for tr, _ in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        # Fill NaN with column median (or 0 if all NaN); preserves shape.
        Xtr = X[tr].copy()
        for j in range(Xtr.shape[1]):
            col = Xtr[:, j]
            mask = np.isnan(col)
            if mask.any():
                vals = col[~mask]
                fill = float(np.median(vals)) if vals.size else 0.0
                Xtr[mask, j] = fill
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                   n_jobs=-1, class_weight="balanced",
                                   random_state=seed)
        m.fit(Xtr, y[tr])
        imps += m.feature_importances_
        n_used += 1
    if n_used == 0:
        return {}
    imps /= n_used
    return dict(zip(feature_names, imps.tolist()))


def _tfidf_lr(code_arr, y, grp):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    aucs = []
    gkf = GroupKFold(n_splits=N_FOLDS)
    for tr, te in gkf.split(code_arr, y, grp):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                              min_df=5, max_features=50000)
        Xt = vec.fit_transform(code_arr[tr])
        Xv = vec.transform(code_arr[te])
        m = LogisticRegression(C=1.0, max_iter=2000,
                               class_weight="balanced", random_state=SEED)
        m.fit(Xt, y[tr])
        p = m.predict_proba(Xv)[:, 1]
        aucs.append(_safe_auc(y[te], p))
    aucs = [a for a in aucs if not math.isnan(a)]
    if not aucs:
        return float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.std(aucs))


# ------- main ------- #

def main():
    print("[1] loading inputs...", flush=True)
    bank = pd.read_parquet(BANK_SCORES)
    runn = pd.read_parquet(RUNN_SCORES)
    claude = pd.read_parquet(CLAUDE_LABELS)
    samp = pd.read_parquet(SAMPLE_2K)
    cands = pd.read_parquet(CAND_CORPUS)
    print(f"  bank rows={len(bank)}  runn rows={len(runn)}  claude rows={len(claude)}  samp rows={len(samp)}",
          flush=True)

    # Merge sample with claude labels
    df = samp.merge(claude, on="pair_id", how="inner")
    df = df.merge(cands[["candidate_id", "code"]].rename(columns={"code": "raw_code"}),
                  on="candidate_id", how="inner")
    print(f"  joined sample+labels+corpus: {len(df)}", flush=True)

    # Inline a478, a479
    print("[2] computing a478 / a479 inline...", flush=True)
    df["a478_score"] = df["raw_code"].map(score_a478)
    df["a479_score"] = df["raw_code"].map(score_a479)
    df["a478_applied"] = df["a478_score"].notna().astype(int)
    df["a479_applied"] = df["a479_score"].notna().astype(int)
    print(f"  a478 mean={df['a478_score'].mean():.3f}  applied={df['a478_applied'].mean():.3f}", flush=True)
    print(f"  a479 mean={df['a479_score'].mean():.3f}  applied={df['a479_applied'].mean():.3f}", flush=True)

    # Merge runnability (a480, a481, a482)
    runn_cols = ["candidate_id", "a480", "a481", "a482"]
    df = df.merge(runn[runn_cols], on="candidate_id", how="left")
    df["a480_score"] = df["a480"].astype(float)
    df["a481_score"] = df["a481"].astype(float)
    df["a482_score"] = df["a482"].astype(float)
    df["a480_applied"] = df["a480_score"].notna().astype(int)
    df["a481_applied"] = df["a481_score"].notna().astype(int)
    df["a482_applied"] = df["a482_score"].notna().astype(int)
    print(f"  a480 coverage: {df['a480_applied'].sum()}/{len(df)}  pass rate (among applied): {df.loc[df.a480_applied==1, 'a480_score'].mean():.3f}",
          flush=True)
    print(f"  a481 coverage: {df['a481_applied'].sum()}/{len(df)}", flush=True)

    # Merge bank scores
    df = df.merge(bank, on="candidate_id", how="left")
    print(f"  final df shape: {df.shape}", flush=True)

    # === Per-decile pass rate ===
    print("[3] per-decile pass rate...", flush=True)
    df["decile"] = pd.qcut(df["max_sim"], 10, labels=False, duplicates="drop")
    decile_table = df.groupby("decile").agg(
        n=("candidate_id", "size"),
        a480_passrate=("a480_score", "mean"),
        max_sim_mean=("max_sim", "mean"),
        claude_label_mean=("label", "mean"),
        a478_mean=("a478_score", "mean"),
        a479_mean=("a479_score", "mean"),
    ).reset_index()
    print(decile_table.to_string(), flush=True)

    # === Build feature sets ===
    bank_score_cols = sorted([c for c in bank.columns if c.endswith("_score")])
    bank_applied_cols = [c[:-len("_score")] + "_applied" for c in bank_score_cols]
    bank_score_cols = [c for c in bank_score_cols if c[:-len("_score")] not in NEW_METRIC_IDS]
    bank_applied_cols = [c[:-len("_score")] + "_applied" for c in bank_score_cols]
    new_score_cols = [f"{n}_score" for n in NEW_METRIC_IDS]
    new_applied_cols = [f"{n}_applied" for n in NEW_METRIC_IDS]

    y = df["label"].astype(int).values
    groups = df["question_slug"].astype(str).values
    code = df["raw_code"].astype(str).fillna("").values

    print(f"[4] feature counts: bank={len(bank_score_cols)} new={len(new_score_cols)}",
          flush=True)
    print(f"   label pos={int(y.sum())}/{len(y)}  pos_rate={y.mean():.3f}", flush=True)

    feature_sets = {
        "bank_only": bank_score_cols + bank_applied_cols,
        "bank_plus_new": bank_score_cols + bank_applied_cols + new_score_cols + new_applied_cols,
        "new_only": new_score_cols + new_applied_cols,
    }

    results = {
        "n_total": int(len(df)),
        "pos_rate": float(y.mean()),
        "a480_pass_rate_overall": float(df["a480_score"].mean()),
        "a480_coverage": int(df["a480_applied"].sum()),
        "per_decile": decile_table.to_dict(orient="records"),
    }

    for fname, cols in feature_sets.items():
        cols_avail = [c for c in cols if c in df.columns]
        X = df[cols_avail].astype(float).values
        print(f"[5] {fname}: X.shape={X.shape}", flush=True)
        rf_m, rf_s = _eval_cv(X, y, groups, "rf")
        lr_m, lr_s = _eval_cv(X, y, groups, "lr")
        results[f"{fname}_RF"] = [rf_m, rf_s]
        results[f"{fname}_LR"] = [lr_m, lr_s]
        print(f"   RF={rf_m:.4f}+-{rf_s:.4f}  LR={lr_m:.4f}+-{lr_s:.4f}", flush=True)

    print("[6] tfidf baseline...", flush=True)
    tf_m, tf_s = _tfidf_lr(code, y, groups)
    results["tfidf_LR"] = [tf_m, tf_s]
    print(f"   tfidf_LR={tf_m:.4f}+-{tf_s:.4f}", flush=True)

    # === Per-metric importance ===
    print("[7] computing per-metric importance (RF on bank+new)...", flush=True)
    cols = [c for c in feature_sets["bank_plus_new"] if c in df.columns]
    Xfull = df[cols].astype(float).values
    imps = _rf_importance(Xfull, y, groups, cols)
    new_imps = {k: v for k, v in imps.items()
                if any(k.startswith(f"{nid}_") for nid in NEW_METRIC_IDS)}
    top30 = sorted(imps.items(), key=lambda kv: -kv[1])[:30]
    results["importances"] = {
        "top_30_bank_plus_new": [list(t) for t in top30],
        "new_metrics_only": dict(sorted(new_imps.items(), key=lambda kv: -kv[1])),
    }

    OUT_REPORT.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[done] wrote {OUT_REPORT}\n", flush=True)
    print(json.dumps({
        "bank_only_RF": results["bank_only_RF"],
        "bank_plus_new_RF": results["bank_plus_new_RF"],
        "new_only_RF": results["new_only_RF"],
        "bank_only_LR": results["bank_only_LR"],
        "bank_plus_new_LR": results["bank_plus_new_LR"],
        "new_only_LR": results["new_only_LR"],
        "tfidf_LR": results["tfidf_LR"],
        "a480_pass_rate_overall": results["a480_pass_rate_overall"],
        "new_metric_importances": results["importances"]["new_metrics_only"],
    }, indent=2))


if __name__ == "__main__":
    main()
