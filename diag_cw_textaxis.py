"""Does the MOB tree find structure once it has a TEXT-CONTENT axis in z?
Cache-reuse (free): materialize the 26 viable rubrics from B_tree cache, cluster texts by
embedding, add the cluster id as a categorical z covariate, and compare the tree WITH vs
WITHOUT that axis. If it splits on text_cluster => moderation is real AND the tree was just
missing the axis (the fix = put text-content/genre into z)."""
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
df = pd.read_csv(REPO_ROOT / dcfg["split"], low_memory=False).dropna(subset=[dcfg["text"], dcfg["label"]])
df[dcfg["label"]] = pd.to_numeric(df[dcfg["label"]], errors="coerce"); df = df.dropna(subset=[dcfg["label"]])
df = df.sample(min(500, len(df)), random_state=7).reset_index(drop=True)
df[dcfg["label"]] = df[dcfg["label"]].astype(int)

cfg = InfillConfig(
    n_permutations=999, min_node_size=30, max_depth=4, random_seed=0,
    proposer_backend="anthropic", proposer_model="glm-5.2",
    materialize_backend="anthropic", materialize_model="glm-5.2",
    llm_concurrency=2, max_text_tokens=700, verbose=False,
    id_column=dcfg["id"], text_column=dcfg["text"], label_column=dcfg["label"],
    output_dir="outputs/ctree/B_tree", cache_dir="outputs/ctree/B_tree/judge_cache",
    viability_min_applicability=0.1, viability_min_std=0.05,
    include_text_length_in_z=False)

rubrics = load_rubric_metrics("creative-writing", limit=40)
judge = make_vllm_judge_scorer(cfg)
probe = df.sample(min(60, len(df)), random_state=1)[dcfg["text"]].astype(str).tolist()
lv, ap = judge(rubrics, probe)
viable = [rubrics[j] for j in range(len(rubrics)) if ap[:, j].mean() > 0.1 and np.std(lv[ap[:, j], j]) > 0.05]
print(f"viable {len(viable)}/{len(rubrics)}", flush=True)
sm = materialize(viable, df, cfg, judge)
y = df[dcfg["label"]].to_numpy()

# text-content axis: embedding clusters
E = embed(df[dcfg["text"]].astype(str).str[:1500].tolist(), "all-MiniLM-L6-v2")
df["text_cluster"] = KMeans(6, n_init=8, random_state=0).fit_predict(E).astype(str)
print(f"text-cluster sizes: {dict(df['text_cluster'].value_counts())}", flush=True)


def fit_tree(extra_z, label):
    c = InfillConfig(**{**cfg.__dict__, "extra_z_columns": extra_z})
    X, fn, Z, spec = make_design(sm, df, c)
    tree = GapTree(c).fit(X, y, Z, fn)
    sp = tree.root.split
    print(f"\n[{label}] nodes={len(tree.all_nodes())} "
          f"root_split={sp.describe() if sp else 'STUMP'}", flush=True)
    if tree.root.fluct:
        top = sorted(tree.root.fluct, key=lambda r: r.adj_pvalue)[:4]
        for r in top:
            print(f"    {r.variable[:30]:30} stat={r.statistic:.1f} adj_p={r.adj_pvalue:.4f}", flush=True)
    return tree


fit_tree([], "z = rubrics only (current)")
fit_tree(["text_cluster"], "z = rubrics + text_cluster (content axis)")
