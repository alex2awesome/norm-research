"""Reproduce the curated-z CW tree (cache-hit, deterministic) and dump its full shape:
nodes with sizes/label-rates/splits, plus the 26 viable metric names used as X regressors."""
import sys; sys.path.insert(0, "methods")
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
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
df = df_full.sample(min(500, n_full), random_state=7).reset_index(drop=True)
df[dcfg["label"]] = df[dcfg["label"]].astype(int)
df["source_half"] = np.where(df["_orig_pos"] < n_full // 2, "A", "B")

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
probe = df.sample(min(60, len(df)), random_state=1)[dcfg["text"]].astype(str).tolist()
lv, ap = judge(rubrics, probe)
viable = [rubrics[j] for j in range(len(rubrics)) if ap[:, j].mean() > 0.1 and np.std(lv[ap[:, j], j]) > 0.05]
print(f"== X regressors: {len(viable)} viable rubric metrics (of {len(rubrics)} probed, "
      f"pool={len(load_rubric_metrics('creative-writing'))}) ==", flush=True)
for m in viable:
    print(f"  - {m.name}", flush=True)

sm = materialize(viable, df, cfg, judge)
y = df[dcfg["label"]].to_numpy()
E = embed(df[dcfg["text"]].astype(str).str[:1500].tolist(), "all-MiniLM-L6-v2")
km = KMeans(6, n_init=8, random_state=0).fit(E)
df["text_cluster"] = km.labels_.astype(str)
sizes = df.groupby("text_cluster")[dcfg["label"]].agg(["size", "mean"])
print(f"\ntext_cluster sizes/label-rates:\n{sizes}", flush=True)

X, fn, Zfull, spec = make_design(sm, df, cfg)
Z = {k: v for k, v in Zfull.items() if k in ("source_half", "text_cluster")}
tree = GapTree(cfg).fit(X, y, Z, fn)

def dump(node, indent=""):
    n = len(node.indices)
    rate = y[node.indices].mean()
    if node.split is None:
        print(f"{indent}LEAF  n={n} rate={rate:.2f}", flush=True)
    else:
        best = min(node.fluct, key=lambda r: r.adj_pvalue) if node.fluct else None
        p = f" adj_p={best.adj_pvalue:.4f}" if best else ""
        print(f"{indent}NODE  n={n} rate={rate:.2f} SPLIT {node.split.describe()}{p}", flush=True)
        dump(node.left, indent + "  |-L ")
        dump(node.right, indent + "  |-R ")

print("\n== TREE ==", flush=True)
dump(tree.root)

# top standardized root coefficients: which metrics carry the within-node GLM
lrbeta = tree.root.beta
if lrbeta is not None and len(lrbeta) == X.shape[1] + 1:
    coefs = lrbeta[1:] * X.std(axis=0)
    top = np.argsort(-np.abs(coefs))[:8]
    print("\ntop |standardized| root-GLM coefficients (metric -> label, whole sample):", flush=True)
    for t in top:
        print(f"  {coefs[t]:+.3f}  {fn[t]}", flush=True)
