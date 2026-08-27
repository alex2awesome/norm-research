#!/usr/bin/env python
"""Confound / spurious-feature / leakage audit for math_se_v3_3_propensity_balanced.

Adapted from ../v3_leakage_audit/audit.py. Trains on the provided `train`
split only; all reported held-out numbers are on the provided `test` split.
Nothing is ever refit on test.

v3.3 additions vs the v3 audit:
  - AUC of the build-time `propensity` column itself on test (should be ~0.5
    by construction of the decile x year balancing).
  - decile-by-class balance table.

Outputs (all in this folder):
  - results.json                  : all AUC numbers
  - top_features_answer_word.csv  : top +/-40 LR coefficients (answer-only word model)
  - top_features_question_word.csv: top +/-40 LR coefficients (question-only word model)
  - single_feature_aucs.csv       : structural confound probe table
  - year_balance.csv              : class balance by answer_year
  - decile_balance.csv            : class balance by propensity decile
  - position_balance.csv          : class balance by answer_position
"""
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "math_se_v3_3_propensity_balanced.csv.gz"
RNG = 0

print("loading...", flush=True)
df = pd.read_csv(DATA)

# ---- parse question / answer portions --------------------------------------
MARKER = "\n\nAnswer: "
parts = df["text"].str.split(MARKER, n=1, regex=False)
df["q_text"] = parts.str[0].str.removeprefix("Question: ")
df["a_text"] = parts.str[1]
assert df["a_text"].notna().all()

tr = df[df["split"] == "train"]
te = df[df["split"] == "test"]
ytr, yte = tr["judgement"].values, te["judgement"].values
print(f"total={len(df)} train={len(tr)} test={len(te)}  "
      f"train pos rate={ytr.mean():.4f} test pos rate={yte.mean():.4f}", flush=True)

# split hygiene
overlap = set(tr["question_id"]) & set(te["question_id"])
print(f"train/test question_id overlap: {len(overlap)}", flush=True)

results = {"n_total": int(len(df)), "n_train": int(len(tr)), "n_test": int(len(te)),
           "n_unique_questions": int(df["question_id"].nunique()),
           "train_test_question_overlap": int(len(overlap)),
           "pos_rate_train": round(float(ytr.mean()), 4),
           "pos_rate_test": round(float(yte.mean()), 4)}

# ---- text models -------------------------------------------------------------
def fit_eval(name, col, vec):
    Xtr = vec.fit_transform(tr[col])
    Xte = vec.transform(te[col])
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=RNG)
    lr.fit(Xtr, ytr)
    auc_tr = roc_auc_score(ytr, lr.decision_function(Xtr))
    auc_te = roc_auc_score(yte, lr.decision_function(Xte))
    results[name] = {"train_auc": round(auc_tr, 4), "test_auc": round(auc_te, 4),
                     "n_features": Xtr.shape[1]}
    print(f"{name}: train AUC={auc_tr:.4f}  test AUC={auc_te:.4f}  "
          f"({Xtr.shape[1]} features)", flush=True)
    return vec, lr

def word_vec():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200_000,
                           sublinear_tf=True, strip_accents="unicode")

def char_vec():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                           max_features=200_000, sublinear_tf=True)

vec_aw, lr_aw = fit_eval("answer_word_1_2", "a_text", word_vec())
vec_ac, lr_ac = fit_eval("answer_char_3_5", "a_text", char_vec())
vec_fw, lr_fw = fit_eval("fulltext_word_1_2", "text", word_vec())
vec_qw, lr_qw = fit_eval("question_word_1_2", "q_text", word_vec())

# ---- top +/- 40 features ------------------------------------------------------
def top_features(vec, lr, fname, k=40):
    names = np.array(vec.get_feature_names_out())
    coefs = lr.coef_[0]
    order = np.argsort(coefs)
    rows = []
    for idx in order[::-1][:k]:
        rows.append({"direction": "positive", "feature": names[idx],
                     "coef": round(float(coefs[idx]), 4)})
    for idx in order[:k]:
        rows.append({"direction": "negative", "feature": names[idx],
                     "coef": round(float(coefs[idx]), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / fname, index=False)
    return out

top_aw = top_features(vec_aw, lr_aw, "top_features_answer_word.csv")
top_qw = top_features(vec_qw, lr_qw, "top_features_question_word.csv")
print("top answer-word features saved", flush=True)

# ---- structural single-feature probes (test split, no fitting needed: AUC is
#      rank-based; we report the AUC of the raw feature and its orientation) ---
def n_display_math(s):
    return s.count("$$") // 2 + len(re.findall(r"\\begin\{(align|equation|gather|eqnarray|cases|array)", s)) + s.count("\\[")

def feats(frame):
    a = frame["a_text"]
    return pd.DataFrame({
        "answer_len_chars": a.str.len(),
        "answer_len_tokens": a.str.split().str.len(),
        "n_latex_blocks": a.str.count(r"\$") // 2,
        "n_display_math": a.apply(n_display_math),
        "has_numbered_list": a.str.contains(r"(?m)^\s*\d+[\.\)]\s", regex=True).astype(int),
        "n_paragraphs": a.str.count("\n\n") + 1,
        "answer_year": frame["answer_year"],
        "answer_age_gap_days": frame["answer_age_gap_days"],
        "n_answers_on_question": frame["n_answers_on_question"],
        "answer_position": frame["answer_position"],
        "propensity": frame["propensity"],
    })

F_te = feats(te)
probe_rows = []
for c in F_te.columns:
    auc = roc_auc_score(yte, F_te[c].values)
    probe_rows.append({"feature": c, "raw_auc": round(auc, 4),
                       "abs_auc": round(max(auc, 1 - auc), 4),
                       "higher_predicts": "positive" if auc >= 0.5 else "negative"})
probes = pd.DataFrame(probe_rows).sort_values("abs_auc", ascending=False)
probes.to_csv(HERE / "single_feature_aucs.csv", index=False)
print(probes.to_string(index=False), flush=True)

# propensity column itself (should be ~0.5 by construction)
auc_prop = roc_auc_score(yte, te["propensity"].values)
results["propensity_column_test_auc"] = round(float(auc_prop), 4)
print(f"propensity column test AUC={auc_prop:.4f} (expect ~0.5)", flush=True)

# combined structural model: how much do ALL structural features buy together?
from sklearn.preprocessing import StandardScaler
STRUCT_COLS = [c for c in F_te.columns if c != "propensity"]
F_tr = feats(tr)
sc = StandardScaler().fit(F_tr[STRUCT_COLS])
lr_s = LogisticRegression(max_iter=2000, random_state=RNG).fit(
    sc.transform(F_tr[STRUCT_COLS]), ytr)
auc_struct = roc_auc_score(yte, lr_s.decision_function(sc.transform(F_te[STRUCT_COLS])))
results["structural_all_features_LR"] = {"test_auc": round(auc_struct, 4)}
print(f"all-structural LR test AUC={auc_struct:.4f}", flush=True)

# ---- year drift ---------------------------------------------------------------
yb = (df.groupby("answer_year")["judgement"]
        .agg(n="size", pos_rate="mean").reset_index())
yb["pos_rate"] = yb["pos_rate"].round(4)
yb.to_csv(HERE / "year_balance.csv", index=False)
print(yb.to_string(index=False), flush=True)
results["year_drift"] = {
    "year_auc_test": float(probes.loc[probes.feature == "answer_year", "raw_auc"].iloc[0]),
    "max_abs_dev_from_0.5": round(float((yb.loc[yb.n >= 200, "pos_rate"] - 0.5).abs().max()), 4),
}

# ---- decile balance (the balancing axis of the v3.3 build) ---------------------
db = (df.groupby("decile")["judgement"]
        .agg(n="size", pos_rate="mean").reset_index())
db["pos_rate"] = db["pos_rate"].round(4)
db.to_csv(HERE / "decile_balance.csv", index=False)
print(db.to_string(index=False), flush=True)
results["decile_balance_max_abs_dev"] = round(float((db["pos_rate"] - 0.5).abs().max()), 4)

# ---- position residual: within answer_position == 1 ----------------------------
p1_all = df[df["answer_position"] == 1]
p1_te = te[te["answer_position"] == 1]
auc_len_p1 = roc_auc_score(p1_te["judgement"], p1_te["a_text"].str.len())
results["position1_residual"] = {
    "n_all": int(len(p1_all)), "pos_rate_all": round(float(p1_all["judgement"].mean()), 4),
    "n_test": int(len(p1_te)), "pos_rate_test": round(float(p1_te["judgement"].mean()), 4),
    "len_chars_auc_test_within_pos1": round(float(auc_len_p1), 4),
}
# per-position class balance for the report
pos_bal = (df.groupby("answer_position")["judgement"]
             .agg(n="size", pos_rate="mean").reset_index().head(10))
pos_bal["pos_rate"] = pos_bal["pos_rate"].round(4)
pos_bal.to_csv(HERE / "position_balance.csv", index=False)
print(pos_bal.to_string(index=False), flush=True)
print(json.dumps(results["position1_residual"], indent=2), flush=True)

with open(HERE / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print("done")
