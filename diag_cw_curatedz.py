"""Curated-z positive control: same data/bank, but z = ONLY hypothesized moderator axes
(source_half + text_cluster + length). m drops 48 -> ~3, Bonferroni bar 0.05/3=0.017,
so source_half's p~0.003 should now pass IF the machinery is right. This is the
partykit-intended design: z = few hypothesized moderators, not the whole bank."""
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
sm = materialize(viable, df, cfg, judge)
y = df[dcfg["label"]].to_numpy()

E = embed(df[dcfg["text"]].astype(str).str[:1500].tolist(), "all-MiniLM-L6-v2")
df["text_cluster"] = KMeans(6, n_init=8, random_state=0).fit_predict(E).astype(str)

X, fn, Zfull, spec = make_design(sm, df, cfg)
# CURATED z: only the hypothesized moderator axes; X (the GLM regressors) unchanged
Z = {k: v for k, v in Zfull.items() if k in ("source_half", "text_cluster")}
print(f"curated z axes: {list(Z.keys())} (m={len(Z)}); X keeps {X.shape[1]} bank columns", flush=True)
tree = GapTree(cfg).fit(X, y, Z, fn)
sp = tree.root.split
print(f"root_split = {sp.describe() if sp else 'STUMP'}", flush=True)
if tree.root.fluct:
    for rr in sorted(tree.root.fluct, key=lambda t: t.adj_pvalue):
        print(f"   {rr.variable[:32]:32} stat={rr.statistic:.1f} p={rr.pvalue:.4f} adj_p={rr.adj_pvalue:.4f}", flush=True)
for nd in tree.all_nodes():
    if nd.split is not None:
        print(f"   depth={nd.depth} n={len(nd.indices)} SPLIT {nd.split.describe()}", flush=True)
print(f"nodes={len(tree.all_nodes())} terminals={len(tree.terminal_nodes())}", flush=True)
