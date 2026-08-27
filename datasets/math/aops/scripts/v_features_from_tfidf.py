"""New deterministic V features inspired by the top TF-IDF features of the
same-approach-as-editorial label, plus the ex-ante ladder re-run.
All features are regex/lexicon counters over the post's own text (ex-ante)."""
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = Path(__file__).resolve().parents[1]

d = pd.read_parquet(BASE / "approach_verdicts_joined.parquet").drop_duplicates("post_id")
d = d[d.same_approach.notna() & d.body_noquote.notna()].copy()
d["y"] = d.same_approach.astype(int)
# DATA CLEANING (user, 2026-06-12): drop wiki transcriptions/adaptations —
# sim_word>=0.5 posts are 94-96% y=1 by construction (they copy the editorial)
_sim = (pd.read_parquet(BASE / "editorial_similarity.parquet")
        .drop_duplicates("post_id")[["post_id", "sim_word"]]
        .rename(columns={"sim_word": "_simw"}))
d = d.merge(_sim, on="post_id", how="left")
n0 = len(d)
d = d[d._simw.fillna(0) < 0.5].copy()
print(f"cleaning: dropped {n0 - len(d)} transcription-band posts (sim>=0.5); n={len(d)}, base rate {d.y.mean():.3f}")
oof = pd.read_parquet(BASE / "editorial_register_oof.parquet")[["post_id", "p_ed"]].drop_duplicates("post_id")
d = d.merge(oof, on="post_id", how="left")
txt = d.body_noquote.str.lower()
ntok = txt.str.split().str.len().clip(lower=1)

LEX = {
    # complete-solution presentation conventions
    "v_boxed": r"\\boxed",
    "v_hide_block": r"\[hide",
    "v_answer_stmt": r"\banswer (?:is|of)\b|\bour answer\b|\bthe answer\b",
    # deductive scaffolding
    "v_deductive": r"\b(?:thus|therefore|hence|implies|it follows)\b",
    # meta-discussion / non-solution register
    "v_meta_doubt": r"\b(?:wrong|incorrect|mistake|why|how do|help|stuck|hint|"
                    r"tried|guessed|confused|understand|not sure|seems)\b",
    "v_first_person": r"\b(?:i|my|me|im|i'm|i've|i think)\b",
    # heavy machinery (routes editorials rarely take)
    "v_heavy_machinery": r"\b(?:inversion|inversive|projective|pascal|desargues|"
                         r"generating function|determinant|vmatrix|"
                         r"roots? of unity filter|complex bash|"
                         r"barycentric|trig bash|coordinate bash|"
                         r"brute force|python|wolfram|calculator)\b",
    # mainstream elementary techniques
    "v_standard_tech": r"\b(?:right triangle|pythagorean|pythag|similar triangles?|"
                       r"casework|cases?|counting|ways|total|substitut|"
                       r"factor|symmetry|telescop)\b",
    # proof-problem framing
    "v_proof_framing": r"\b(?:prove|proof|q\.?e\.?d|wts|we want to show)\b|\\blacksquare",
}
for name, pat in LEX.items():
    d[name] = txt.str.count(pat) / ntok
d["v_numeral_density"] = txt.str.count(r"(?<![\w.])\d{1,4}(?![\w.])") / ntok
d["v_question_marks"] = txt.str.count(r"\?") / ntok
NEW_V = list(LEX) + ["v_numeral_density", "v_question_marks"]

# spot-check: 2 example snippets per feature (validate before trusting)
print("=== SPOT-CHECK (top-scoring snippet per feature) ===")
for f in NEW_V:
    top = d.loc[d[f].idxmax()]
    snip = re.sub(r"\s+", " ", top.body_noquote)[:110]
    print(f"{f:22s} max={top[f]:.3f}  y={top.y}  | {snip}")

print("\n=== single-feature AUCs (pooled, direction-free) ===")
for f in NEW_V:
    auc = roc_auc_score(d.y, d[f])
    print(f"{f:22s} AUC = {max(auc, 1 - auc):.3f}  ({'+' if auc >= .5 else '-'})")

# ladder
d2 = d.dropna(subset=["p_ed"]).reset_index(drop=True)
d2["log_len"] = np.log1p(d2.body_noquote.str.len())
d2["n_display_math"] = d2.body_noquote.str.count(r"\\\[|\\begin\{align")
d2["latex_density"] = d2.body_noquote.str.count(r"\$") / (d2.body_noquote.str.len() + 1)
d2["is_correct_f"] = d2.is_correct.fillna(False).astype(int)
d2["has_answer_f"] = d2.has_answer.fillna(False).astype(int)

OLD_V = ["log_len", "n_display_math", "latex_density", "is_correct_f", "has_answer_f"]
y, g = d2.y.values, d2.problem.values
gkf = GroupKFold(5)

def cv(cols, name, extra=None):
    X = StandardScaler().fit_transform(d2[cols])
    if extra is not None:
        X = np.column_stack([X, extra])
    p = cross_val_predict(LogisticRegression(max_iter=3000), X, y,
                          cv=gkf, groups=g, method="predict_proba")[:, 1]
    print(f"{name:46s} AUC = {roc_auc_score(y, p):.3f}")

print(f"\n=== EX-ANTE ladder with new V features (n={len(d2)}) ===")
cv(OLD_V, "V-old (struct + answer)")
cv(NEW_V, "V-new (TF-IDF-inspired lexicons)")
cv(OLD_V + NEW_V, "V-old + V-new")
cv(OLD_V + NEW_V + ["p_ed"], "V-all + register")

# TF-IDF reference + does V-new survive alongside it
oofp = np.zeros(len(d2))
for tr, te in gkf.split(d2, y, g):
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Xtr = vec.fit_transform(d2.body_noquote.iloc[tr])
    Xte = vec.transform(d2.body_noquote.iloc[te])
    clf = LogisticRegression(max_iter=3000).fit(Xtr, y[tr])
    oofp[te] = clf.predict_proba(Xte)[:, 1]
print(f"{'post-only TF-IDF (reference)':46s} AUC = {roc_auc_score(y, oofp):.3f}")
cv(OLD_V + NEW_V + ["p_ed"], "everything + TF-IDF", extra=oofp)

out = BASE / "v_features_same_approach.parquet"
d2[["post_id", "problem", "y"] + OLD_V + NEW_V + ["p_ed"]].to_parquet(out)
print(f"\n[saved] {out}")
