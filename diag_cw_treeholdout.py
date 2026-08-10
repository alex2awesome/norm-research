"""Does the curated-z tree GENERALIZE? Fit tree on 500-item discover sample (cached),
evaluate bank-GLM per-leaf vs global on a FRESH 500-item holdout (needs new judge calls
for the holdout — glm-5.2, ~26 rubrics x 500 items, concurrency 2).
Report: global AUC vs tree-routed per-leaf AUC on holdout + per-leaf label rates."""
import sys; sys.path.insert(0, "methods")
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.io_metrics import (
    REPO_ROOT, load_rubric_metrics, make_design, make_vllm_judge_scorer, materialize)
from metrics_tree_infilling.depth_dial import embed
from metrics_tree_infilling.mob.glmtree import GapTree
from metrics_tree_infilling.run import DATASET_CONFIGS

dcfg = DATASET_CONFIGS["creative-writing"]
df_full = pd.read_csv(REPO_ROOT / dcfg["split"], low_memory=False).dropna(subset=[dcfg["text"], dcfg["label"]])
df_full[dcfg["label"]] = pd.to_numeric(df_full[dcfg["label"]], errors="coerce")
df_full = df_full.dropna(subset=[dcfg["label"]])
n_full = len(df_full)
df_full["_orig_pos"] = np.arange(n_full)
disc = df_full.sample(500, random_state=7).reset_index(drop=True)
rest = df_full.drop(df_full.sample(500, random_state=7).index)
hold = rest.sample(500, random_state=23).reset_index(drop=True)
for d in (disc, hold):
    d[dcfg["label"]] = d[dcfg["label"]].astype(int)
    d["source_half"] = np.where(d["_orig_pos"] < n_full // 2, "A", "B")

cfg = InfillConfig(
    n_permutations=999, min_node_size=30, max_depth=4, random_seed=0,
    proposer_backend="anthropic", proposer_model="glm-5.2",
    materialize_backend="anthropic", materialize_model="glm-5.2",
    llm_concurrency=2, max_text_tokens=700, verbose=False,
    id_column=dcfg["id"], text_column=dcfg["text"], label_column=dcfg["label"],
    output_dir="outputs/ctree/B_tree", cache_dir="outputs/ctree/B_tree/judge_cache",
    viability_min_applicability=0.1, viability_min_std=0.05,
    include_text_length_in_z=False, extra_z_columns=["source_half", "text_cluster"])

rubrics = load_rubric_metrics("creative-writing", limit=40)
judge = make_vllm_judge_scorer(cfg)
probe = disc.sample(60, random_state=1)[dcfg["text"]].astype(str).tolist()
lv, ap = judge(rubrics, probe)
viable = [rubrics[j] for j in range(len(rubrics)) if ap[:, j].mean() > 0.1 and np.std(lv[ap[:, j], j]) > 0.05]
sm_d = materialize(viable, disc, cfg, judge)
print("discover materialized (cache)", flush=True)
sm_h = materialize(viable, hold, cfg, judge)
print("holdout materialized (fresh)", flush=True)

# shared text-cluster model: fit on discover, apply to both
E_d = embed(disc[dcfg["text"]].astype(str).str[:1500].tolist(), "all-MiniLM-L6-v2")
km = KMeans(6, n_init=8, random_state=0).fit(E_d)
disc["text_cluster"] = km.labels_.astype(str)
E_h = embed(hold[dcfg["text"]].astype(str).str[:1500].tolist(), "all-MiniLM-L6-v2")
hold["text_cluster"] = km.predict(E_h).astype(str)

y_d = disc[dcfg["label"]].to_numpy(); y_h = hold[dcfg["label"]].to_numpy()
X_d, fn, Zf_d, spec = make_design(sm_d, disc, cfg)
X_h, _, Zf_h, _ = make_design(sm_h, hold, cfg, spec)   # SAME DesignSpec => same columns
Z_d = {k: v for k, v in Zf_d.items() if k in ("source_half", "text_cluster")}

tree = GapTree(cfg).fit(X_d, y_d, Z_d, fn)
print(f"tree nodes={len(tree.all_nodes())} terminals={len(tree.terminal_nodes())}", flush=True)

def route_leaf(tree, zrow):
    node = tree.root
    while node.split is not None:
        node = node.left if node.split.goes_left(zrow[node.split.variable]) else node.right
    return node

def leaf_of(df_):
    ids = []
    for i in range(len(df_)):
        zrow = {k: df_.iloc[i][k] for k in ("source_half", "text_cluster")}
        ids.append(id(route_leaf(tree, zrow)))
    return np.array(ids)

leaves_d, leaves_h = leaf_of(disc), leaf_of(hold)

# global bank GLM (fit discover, eval holdout)
g = LogisticRegression(max_iter=2000).fit(X_d, y_d)
auc_global = roc_auc_score(y_h, g.predict_proba(X_h)[:, 1])

# tree-routed: per-leaf GLM fit on discover members, predict holdout members
p_h = np.zeros(len(hold))
for lid in np.unique(leaves_h):
    tr = leaves_d == lid; te = leaves_h == lid
    if tr.sum() >= 20 and len(np.unique(y_d[tr])) == 2:
        m = LogisticRegression(max_iter=2000).fit(X_d[tr], y_d[tr])
        p_h[te] = m.predict_proba(X_h[te])[:, 1]
    else:
        p_h[te] = g.predict_proba(X_h[te])[:, 1]
    print(f"  leaf {lid%9973}: n_disc={tr.sum()} n_hold={te.sum()} "
          f"rate_d={y_d[tr].mean() if tr.sum() else float('nan'):.2f} "
          f"rate_h={y_h[te].mean() if te.sum() else float('nan'):.2f}", flush=True)
auc_tree = roc_auc_score(y_h, p_h)
print(f"\nHOLDOUT AUC  global-GLM={auc_global:.4f}   tree-routed per-leaf={auc_tree:.4f}   "
      f"delta={auc_tree-auc_global:+.4f}", flush=True)

# CONTROL: is the gain moderation or just main effects of the axes?
# global GLM with bank + one-hot(source_half, text_cluster) as features
Fd = pd.get_dummies(disc[["source_half", "text_cluster"]]).astype(float)
Fh = pd.get_dummies(hold[["source_half", "text_cluster"]]).astype(float)
Fh = Fh.reindex(columns=Fd.columns, fill_value=0.0)
Xa_d = np.column_stack([X_d, Fd.to_numpy()])
Xa_h = np.column_stack([X_h, Fh.to_numpy()])
ga = LogisticRegression(max_iter=2000).fit(Xa_d, y_d)
auc_aug = roc_auc_score(y_h, ga.predict_proba(Xa_h)[:, 1])
# axes ONLY (no bank): how much do source+cluster carry alone?
gx = LogisticRegression(max_iter=2000).fit(Fd.to_numpy(), y_d)
auc_axes = roc_auc_score(y_h, gx.predict_proba(Fh.to_numpy())[:, 1])
print(f"CONTROL AUCs  axes-only={auc_axes:.4f}   global+axes(additive)={auc_aug:.4f}   "
      f"tree-routed={auc_tree:.4f}   moderation-specific delta={auc_tree-auc_aug:+.4f}", flush=True)
