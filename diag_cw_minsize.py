"""min_node_size sweep (cache-reuse, free): is min_node_size=30 blocking the borderline splits?
The root shows adj_p~0.047 on NA indicators but stumps — likely because the NA child is <30
items. Try smaller thresholds and report whether a real tree grows, and the NA group sizes."""
import sys; sys.path.insert(0, "methods")
import numpy as np, pandas as pd
from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.io_metrics import (
    REPO_ROOT, load_rubric_metrics, make_design, make_vllm_judge_scorer, materialize)
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
    viability_min_applicability=0.1, viability_min_std=0.05, include_text_length_in_z=False)
rubrics = load_rubric_metrics("creative-writing", limit=40)
judge = make_vllm_judge_scorer(cfg)
probe = df.sample(min(60, len(df)), random_state=1)[dcfg["text"]].astype(str).tolist()
lv, ap = judge(rubrics, probe)
viable = [rubrics[j] for j in range(len(rubrics)) if ap[:, j].mean() > 0.1 and np.std(lv[ap[:, j], j]) > 0.05]
sm = materialize(viable, df, cfg, judge)
y = df[dcfg["label"]].to_numpy()

# NA group sizes (why splits fail min_node_size)
print("NA indicator group sizes (n where rubric is N/A):", flush=True)
for j in range(sm.levels.shape[1]):
    n_na = int((~sm.applicable[:, j]).sum())
    if n_na > 0 and n_na < 60:
        print(f"   {sm.metric_names[j][:34]:34} NA={n_na} app={500-n_na}", flush=True)

for mns in (30, 20, 15, 10):
    c = InfillConfig(**{**cfg.__dict__, "min_node_size": mns})
    X, fn, Z, spec = make_design(sm, df, c)
    tree = GapTree(c).fit(X, y, Z, fn)
    splits = [n.split.describe() for n in tree.all_nodes() if n.split is not None]
    print(f"\nmin_node_size={mns}: nodes={len(tree.all_nodes())} terminals={len(tree.terminal_nodes())} "
          f"splits={len(splits)}", flush=True)
    for s in splits[:6]:
        print(f"    {s}", flush=True)
