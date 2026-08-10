"""
MI ladder against Claude's binary label on the 2K Python LC sample.

8 cells (features x model):
- Bank only   x {RF, LR}
- Bank + G1-G6 (a472-a477) x {RF, LR}
- G1-G6 only x {RF, LR}
- TF-IDF char_wb 3-5gram (max 20K) x {RF, LR}

GroupKFold by question_slug, 5 folds. Report AUC mean +/- std.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE = Path("/Users/spangher/Projects/stanford-research/norm-research/outputs/v2_analysis")
WORK = BASE / "lc_python_2k_work"

# -------- Load labels from shards --------
def load_labels():
    rows = []
    for p in sorted(WORK.glob("labels_*.json")):
        d = json.load(open(p))
        for r in d:
            if "pair_id" in r and "label" in r:
                rows.append({"pair_id": r["pair_id"], "label": int(r["label"]),
                             "reason": r.get("brief_reason", "")})
    return pd.DataFrame(rows)

labels = load_labels()
print(f"Loaded {len(labels)} labels")
print(f"Label distribution overall:\n{labels['label'].value_counts(normalize=True)}")

samp = pd.read_parquet(BASE / "lc_python_2k_sample.parquet")
scores = pd.read_parquet(BASE / "lc_python_metric_scores.parquet")
metric_names = json.load(open(BASE / "lc_metric_id_to_name.json"))

df = samp.merge(labels, on="pair_id", how="inner")
df = df.merge(scores, on="candidate_id", how="left")
print(f"Joined: {len(df)} rows")

print(f"\nLabel by decile:")
print(df.groupby("decile")["label"].agg(["mean", "count"]))

# -------- Feature sets --------
all_score_cols = [c for c in df.columns if c.endswith("_score")]
g_ids = [472, 473, 474, 475, 476, 477]
g_cols = [f"a{i}_score" for i in g_ids if f"a{i}_score" in df.columns]
print(f"\nTotal score cols: {len(all_score_cols)}, G1-G6 present: {g_cols}")
bank_cols = [c for c in all_score_cols if c not in g_cols]
print(f"Bank cols (excluding G1-G6): {len(bank_cols)}")

# Fill NaN scores with 0 (means metric not applied for that candidate -> neutral)
X_bank = df[bank_cols].fillna(0).values
X_g = df[g_cols].fillna(0).values
X_bank_plus_g = df[bank_cols + g_cols].fillna(0).values

# TF-IDF on candidate code
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000, lowercase=True)
X_tfidf_full = vec.fit_transform(df["code"].astype(str).tolist())
print(f"TF-IDF shape: {X_tfidf_full.shape}")

y = df["label"].values
groups = df["question_slug"].values

# -------- CV runner --------
gkf = GroupKFold(n_splits=5)
def cv_auc(X, model_factory, name):
    aucs = []
    for fold, (tr, va) in enumerate(gkf.split(X if not hasattr(X, "toarray") else np.zeros(X.shape[0]), y, groups)):
        if hasattr(X, "tocsr"):
            Xtr, Xva = X[tr], X[va]
        else:
            Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        if len(np.unique(yva)) < 2:
            continue
        mdl = model_factory()
        mdl.fit(Xtr, ytr)
        if hasattr(mdl, "predict_proba"):
            p = mdl.predict_proba(Xva)[:, 1]
        else:
            p = mdl.decision_function(Xva)
        aucs.append(roc_auc_score(yva, p))
    return float(np.mean(aucs)), float(np.std(aucs)), aucs

def rf_factory():
    return RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                                  n_jobs=-1, random_state=42, class_weight="balanced")

def lr_factory_dense():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced", random_state=42)),
    ])

def lr_factory_sparse():
    return LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced", random_state=42, solver="liblinear")

def rf_factory_sparse():
    # RF doesn't support sparse natively for tree splits well; we densify per fold (20K x ~1600 rows fits)
    return RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                                  n_jobs=-1, random_state=42, class_weight="balanced")

# RF on sparse TF-IDF: sklearn supports sparse input for RF (since ~0.24) so we'll pass through.
results = {}
print("\n--- Running MI ladder ---")

for name, X, mdl_rf, mdl_lr in [
    ("bank_only",       X_bank,         rf_factory, lr_factory_dense),
    ("bank_plus_g",     X_bank_plus_g,  rf_factory, lr_factory_dense),
    ("g_only",          X_g,            rf_factory, lr_factory_dense),
]:
    m, s, all_ = cv_auc(X, mdl_rf, f"{name}_RF")
    results[f"{name}_RF"] = (m, s, all_)
    print(f"  {name}_RF: AUC = {m:.4f} +/- {s:.4f}")
    m, s, all_ = cv_auc(X, mdl_lr, f"{name}_LR")
    results[f"{name}_LR"] = (m, s, all_)
    print(f"  {name}_LR: AUC = {m:.4f} +/- {s:.4f}")

# TF-IDF — refit per fold to avoid leakage
def cv_auc_tfidf(model_factory, dense_required=False):
    aucs = []
    for fold, (tr, va) in enumerate(gkf.split(np.zeros(len(y)), y, groups)):
        vec_f = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                max_features=20000, lowercase=True)
        Xtr = vec_f.fit_transform(df.iloc[tr]["code"].astype(str).tolist())
        Xva = vec_f.transform(df.iloc[va]["code"].astype(str).tolist())
        ytr, yva = y[tr], y[va]
        if len(np.unique(yva)) < 2:
            continue
        mdl = model_factory()
        if dense_required:
            mdl.fit(Xtr.toarray(), ytr)
            p = mdl.predict_proba(Xva.toarray())[:, 1]
        else:
            mdl.fit(Xtr, ytr)
            p = mdl.predict_proba(Xva)[:, 1] if hasattr(mdl, "predict_proba") else mdl.decision_function(Xva)
        aucs.append(roc_auc_score(yva, p))
    return float(np.mean(aucs)), float(np.std(aucs)), aucs

m, s, all_ = cv_auc_tfidf(rf_factory_sparse, dense_required=False)
results["tfidf_RF"] = (m, s, all_)
print(f"  tfidf_RF: AUC = {m:.4f} +/- {s:.4f}")
m, s, all_ = cv_auc_tfidf(lr_factory_sparse, dense_required=False)
results["tfidf_LR"] = (m, s, all_)
print(f"  tfidf_LR: AUC = {m:.4f} +/- {s:.4f}")

# -------- Save report --------
report = {
    "n_total": int(len(df)),
    "n_unique_slugs": int(df["question_slug"].nunique()),
    "label_distribution": df["label"].value_counts(normalize=True).round(4).to_dict(),
    "label_by_decile": df.groupby("decile")["label"].agg(["mean", "count"]).round(4).to_dict(),
    "auc": {k: {"mean": round(v[0], 4), "std": round(v[1], 4),
                "folds": [round(x, 4) for x in v[2]]} for k, v in results.items()},
    "g_cols_used": g_cols,
    "n_bank_features": len(bank_cols),
    "n_tfidf_features": X_tfidf_full.shape[1],
}
out_path = BASE / "lc_python_2k_mi_ladder_report.json"
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nReport written: {out_path}")

# Also save merged labels parquet
labels_out = BASE / "lc_python_2k_labels.parquet"
df[["pair_id", "candidate_id", "question_slug", "max_sim", "decile", "label", "reason"]].to_parquet(labels_out, index=False)
print(f"Labels parquet: {labels_out}")
