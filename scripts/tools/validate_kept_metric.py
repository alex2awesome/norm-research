"""Final honest read for the ONE kept metric (peer-review, label_contrast arm):
fit bank vs bank+metric on pooled discover+guard, evaluate ONCE on the untouched test split."""
import sys, json; sys.path.insert(0, "methods")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.io_metrics import (
    REPO_ROOT, load_rubric_metrics_from_dir, make_design, make_vllm_judge_scorer,
    materialize, three_way_split, MetricSpec)
from metrics_tree_infilling.run import DATASET_CONFIGS

dcfg = DATASET_CONFIGS["peer-review"]
led = json.load(open("outputs/ctree/arm_comparison/peer-review-cv/label_contrast/global_infill_ledger.json"))
kept = [l for l in led["ledgers"] if l["status"] == "kept"][0]
print(f"validating: {kept['name']}", flush=True)

cfg = InfillConfig(
    random_seed=0, proposer_backend="anthropic", proposer_model="glm-5.2",
    materialize_backend="anthropic", materialize_model="glm-5.2",
    llm_concurrency=2, max_text_tokens=700, verbose=False,
    min_auc_gain=0.01, min_bits_gain=0.005, acceptance_eval="cv",
    viability_min_applicability=0.10, viability_min_std=0.05,
    id_column=dcfg["id"], text_column=dcfg["text"], label_column=dcfg["label"],
    output_dir="outputs/ctree/arm_comparison/peer-review-cv",
    cache_dir="outputs/ctree/B_tree/judge_cache", curated_z_only=True,
    include_text_length_in_z=False)

# EXACT same data path as run_arm_comparison
df = pd.read_csv(REPO_ROOT / dcfg["split"], low_memory=False).dropna(subset=[dcfg["text"], dcfg["label"]])
df[dcfg["label"]] = pd.to_numeric(df[dcfg["label"]], errors="coerce")
df = df.dropna(subset=[dcfg["label"]]); df[dcfg["label"]] = df[dcfg["label"]].astype(int)
df = df.sample(min(400 + 200, len(df)), random_state=7).reset_index(drop=True)
df_d, df_g, df_t = three_way_split(df, cfg)
print(f"d/g/t = {len(df_d)}/{len(df_g)}/{len(df_t)}", flush=True)

bank = load_rubric_metrics_from_dir("datasets/peer-review/medoid-bank")[:40]
judge = make_vllm_judge_scorer(cfg)
probe = df_d.sample(min(60, len(df_d)), random_state=1)[dcfg["text"]].astype(str).tolist()
lv, apl = judge(bank, probe)
viable = [bank[j] for j in range(len(bank))
          if apl[:, j].mean() > 0.10 and np.std(lv[apl[:, j], j]) > 0.05]
new = MetricSpec(metric_id="kept_lc", name=kept["name"], description=kept["description"],
                 kind="judge", guidance=kept["rubric"])

df_tr = pd.concat([df_d, df_g]).reset_index(drop=True)
sm_tr = materialize(viable + [new], df_tr, cfg, judge)      # d+g cached; new metric fresh
sm_te = materialize(viable + [new], df_t, cfg, judge)       # test fresh
y_tr = df_tr[dcfg["label"]].to_numpy(); y_te = df_t[dcfg["label"]].to_numpy()

X_tr, fn, _, spec = make_design(sm_tr, df_tr, cfg)
X_te, _, _, _ = make_design(sm_te, df_t, cfg, spec)
new_cols = [j for j, f in enumerate(fn) if kept["name"][:20] in f]
base_cols = [j for j in range(X_tr.shape[1]) if j not in new_cols]

def ev(cols):
    lr = LogisticRegression(max_iter=2000).fit(X_tr[:, cols], y_tr)
    p = np.clip(lr.predict_proba(X_te[:, cols])[:, 1], 1e-9, 1-1e-9)
    auc = roc_auc_score(y_te, p)
    q = np.clip(y_tr.mean(), 1e-9, 1-1e-9)
    bits = np.mean(y_te*np.log2(p)+(1-y_te)*np.log2(1-p)) - np.mean(y_te*np.log2(q)+(1-y_te)*np.log2(1-q))
    return auc, bits

a0, b0 = ev(base_cols); a1, b1 = ev(list(range(X_tr.shape[1])))
print(f"\nTEST (n={len(y_te)}, untouched):", flush=True)
print(f"  bank          AUC={a0:.4f} bits={b0:.4f}", flush=True)
print(f"  bank+metric   AUC={a1:.4f} bits={b1:.4f}", flush=True)
print(f"  DELTA         AUC={a1-a0:+.4f} bits={b1-b0:+.4f}", flush=True)
