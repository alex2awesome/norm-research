"""Code-parallel framing for AoPS: y = LLM-judged 'forum solution is
substantially the same approach as an editorial (wiki) solution' (same_approach),
then climb the V/A ladder: how much of that judgment is recoverable from
deterministic features (V) and from the learned register score (A-cheap)?
Group-split by problem throughout."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1]

df = pd.read_parquet(BASE / "approach_verdicts_joined.parquet").drop_duplicates("post_id")
df = df[df.same_approach.notna()].copy()
df["y"] = df.same_approach.astype(int)

es = pd.read_parquet(BASE / "editorial_similarity.parquet").drop_duplicates("post_id")
es["log_len"] = np.log1p(es.body_noquote.str.len())
es["n_display_math"] = es.body_noquote.str.count(r"\\\[|\\begin\{align")
es["latex_density"] = es.body_noquote.str.count(r"\$") / (es.body_noquote.str.len() + 1)
oof = pd.read_parquet(BASE / "editorial_register_oof.parquet")[["post_id", "p_ed"]].drop_duplicates("post_id")

d = (df[["post_id", "problem", "y", "is_correct", "has_answer", "n_wiki_shown"]]
     .merge(es[["post_id", "sim_word", "sim_char", "match_sim", "log_len",
                "n_display_math", "latex_density"]], on="post_id", how="left")
     .merge(oof, on="post_id", how="left"))
d["is_correct_f"] = d.is_correct.fillna(False).astype(int)
d["has_answer_f"] = d.has_answer.fillna(False).astype(int)
print(f"n={len(d)}, base rate y=1: {d.y.mean():.3f}, problems: {d.problem.nunique()}")

groups = d.problem
gkf = GroupKFold(n_splits=5)

def cv_auc(cols, name):
    sub = d.dropna(subset=cols)
    X, y, g = sub[cols].values, sub.y.values, sub.problem.values
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(clf, X, y, cv=gkf, groups=g, method="predict_proba")[:, 1]
    print(f"{name:42s} AUC = {roc_auc_score(y, p):.3f}  (n={len(sub)})")
    return roc_auc_score(y, p)

print("\n=== single features (group-CV LR) ===")
for c in ["sim_word", "sim_char", "match_sim", "log_len", "n_display_math",
          "latex_density", "is_correct_f", "has_answer_f", "p_ed"]:
    cv_auc([c], c)

print("\n=== ladder (group-CV LR) ===")
V_LEX = ["sim_word", "sim_char", "match_sim"]
V_STRUCT = ["log_len", "n_display_math", "latex_density"]
V_ANS = ["is_correct_f", "has_answer_f"]
cv_auc(V_STRUCT, "V-struct (length/math density)")
cv_auc(V_ANS, "V-answer (extraction + correctness)")
cv_auc(V_LEX, "V-lexical (similarity to wiki)")
cv_auc(V_LEX + V_STRUCT + V_ANS, "V-all")
cv_auc(["p_ed"], "A-cheap (register score alone)")
cv_auc(V_LEX + V_STRUCT + V_ANS + ["p_ed"], "V-all + register")

print("\n=== robustness: same ladder, y restricted to confident rows ===")
dc = d.copy()
dfj = df[["post_id", "salvaged", "approach_label"]]
dc = dc.merge(dfj, on="post_id", how="left")
mask = (~dc.salvaged.fillna(False).astype(bool)) & \
       (dc.approach_label.str.lower().fillna("") != "not a solution")
d = dc[mask].reset_index(drop=True)
print(f"n={len(d)}, base rate: {d.y.mean():.3f}")
cv_auc(V_LEX, "V-lexical")
cv_auc(V_LEX + V_STRUCT + V_ANS, "V-all")
cv_auc(V_LEX + V_STRUCT + V_ANS + ["p_ed"], "V-all + register")
