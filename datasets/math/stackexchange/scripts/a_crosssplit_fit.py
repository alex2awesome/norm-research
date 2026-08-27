"""Address the methodological question: the A-combiner LR should be trained on
the train split and evaluated on eval/test (not CV'd within a single split).
Show that the A-layer AUC is robust to which split fits the 15-param combiner —
i.e. we do NOT need the full 80K train split scored; ~2.7K rows fit it fine."""
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

A = [f"a{n:02d}" for n in range(1, 15)]
df = pd.DataFrame([json.loads(l) for l in open("a_metric_verdicts_all.jsonl")
                   if "judge_failed" not in l]).drop_duplicates("answer_id")
sp = {s: df[df.split == s] for s in ["train", "eval", "test"]}
for s, d in sp.items():
    print(f"{s}: n={len(d)} base={d.judgement.mean():.3f}")


def fit_eval(tr, te):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
    clf.fit(tr[A].values, tr.judgement.values)
    return roc_auc_score(te.judgement.values, clf.predict_proba(te[A].values)[:, 1])


print("\n=== A-combiner LR: FIT ON TRAIN split, evaluate held-out ===")
print(f"  fit train(2672) -> eval : AUC={fit_eval(sp['train'], sp['eval']):.3f}")
print(f"  fit train(2672) -> test : AUC={fit_eval(sp['train'], sp['test']):.3f}")
print("=== cross-split fits (both fully scored) ===")
print(f"  fit eval -> test        : AUC={fit_eval(sp['eval'], sp['test']):.3f}")
print(f"  fit test -> eval        : AUC={fit_eval(sp['test'], sp['eval']):.3f}")
print("=== within-split 5-fold CV (what was originally reported) ===")
for s in ["eval", "test"]:
    d = sp[s]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
    p = cross_val_predict(clf, d[A].values, d.judgement.values,
                          cv=GroupKFold(5), groups=d.question_id.values,
                          method="predict_proba")[:, 1]
    print(f"  CV within {s:4s}        : AUC={roc_auc_score(d.judgement.values, p):.3f}")
print("=== per-metric needs NO training (raw score vs label) ===")
for s in ["eval", "test"]:
    print(f"  {s}: a07 elegance AUC={roc_auc_score(sp[s].judgement, sp[s].a07):.3f}")
