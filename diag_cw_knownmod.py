"""Known-moderator positive control on REAL CW data (cache-reuse, free).
litbench-to-train.csv.gz is a concat of two halves with label rates ~0.47 vs ~0.15 — a real,
known intercept-moderator the tree has never been shown. Add `source_half` to z:
  - if MOB splits on it => the machinery detects real moderation on real CW data, so the
    stump on rubric-z's is a substantive negative ("no moderation in rubric space"), and the
    missing-axis hypothesis gets direct support (a true moderator existed outside z);
  - if MOB misses a 0.32 base-rate break => genuine power/design problem.
"""
import sys; sys.path.insert(0, "methods")
import numpy as np, pandas as pd
from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.io_metrics import (
    REPO_ROOT, load_rubric_metrics, make_design, make_vllm_judge_scorer, materialize)
from metrics_tree_infilling.mob.glmtree import GapTree
from metrics_tree_infilling.run import DATASET_CONFIGS

dcfg = DATASET_CONFIGS["creative-writing"]
df_full = pd.read_csv(REPO_ROOT / dcfg["split"], low_memory=False).dropna(subset=[dcfg["text"], dcfg["label"]])
df_full[dcfg["label"]] = pd.to_numeric(df_full[dcfg["label"]], errors="coerce")
df_full = df_full.dropna(subset=[dcfg["label"]])
n_full = len(df_full)
# half boundary from the ORIGINAL row order (concat structure), before sampling
df_full["_orig_pos"] = np.arange(n_full)
df = df_full.sample(min(500, n_full), random_state=7).reset_index(drop=True)
df[dcfg["label"]] = df[dcfg["label"]].astype(int)
df["source_half"] = np.where(df["_orig_pos"] < n_full // 2, "A", "B")

r = df.groupby("source_half")[dcfg["label"]].agg(["mean", "size"])
print(f"label rate by half (sample of {len(df)}):\n{r}", flush=True)

cfg = InfillConfig(
    n_permutations=999, min_node_size=30, max_depth=4, random_seed=0,
    proposer_backend="anthropic", proposer_model="glm-5.2",
    materialize_backend="anthropic", materialize_model="glm-5.2",
    llm_concurrency=2, max_text_tokens=700, verbose=False,
    id_column=dcfg["id"], text_column=dcfg["text"], label_column=dcfg["label"],
    output_dir="outputs/ctree/B_tree", cache_dir="outputs/ctree/B_tree/judge_cache",
    viability_min_applicability=0.1, viability_min_std=0.05,
    include_text_length_in_z=False, extra_z_columns=["source_half"])

rubrics = load_rubric_metrics("creative-writing", limit=40)
judge = make_vllm_judge_scorer(cfg)
probe = df.sample(min(60, len(df)), random_state=1)[dcfg["text"]].astype(str).tolist()
lv, ap = judge(rubrics, probe)
viable = [rubrics[j] for j in range(len(rubrics)) if ap[:, j].mean() > 0.1 and np.std(lv[ap[:, j], j]) > 0.05]
print(f"viable {len(viable)}/{len(rubrics)} (cache-driven)", flush=True)
sm = materialize(viable, df, cfg, judge)
y = df[dcfg["label"]].to_numpy()

X, fn, Z, spec = make_design(sm, df, cfg)
tree = GapTree(cfg).fit(X, y, Z, fn)
sp = tree.root.split
print(f"\nroot_split = {sp.describe() if sp else 'STUMP'}", flush=True)
if tree.root.fluct:
    for rr in sorted(tree.root.fluct, key=lambda t: t.adj_pvalue)[:6]:
        print(f"   {rr.variable[:32]:32} stat={rr.statistic:.1f} p={rr.pvalue:.4f} adj_p={rr.adj_pvalue:.4f}", flush=True)
splits = [n.split.describe() for n in tree.all_nodes() if n.split is not None]
print(f"nodes={len(tree.all_nodes())} terminals={len(tree.terminal_nodes())} splits={splits[:8]}", flush=True)

# follow-up: where did source_half rank, and why did adj_p=0.048<0.05 not split?
rows = sorted(tree.root.fluct, key=lambda t: t.adj_pvalue)
for rank, rr in enumerate(rows):
    if rr.variable == "source_half":
        print(f"\nsource_half rank={rank+1}/{len(rows)} stat={rr.statistic:.1f} "
              f"p={rr.pvalue:.4f} adj_p={rr.adj_pvalue:.4f}", flush=True)
print(f"total z candidates m={len(rows)}; p-floor={1/(1+cfg.n_permutations):.4f} "
      f"-> min adj_p={len(rows)/(1+cfg.n_permutations):.4f}", flush=True)
# does the bank ABSORB the axis? label~half raw vs residualized on metric levels
from sklearn.linear_model import LogisticRegression
half = (df["source_half"] == "A").astype(int).to_numpy()
lr = LogisticRegression(max_iter=2000).fit(X, y)
resid = y - lr.predict_proba(X)[:, 1]
import numpy as _np
print(f"raw label gap A-B: {y[half==1].mean()-y[half==0].mean():+.3f}", flush=True)
print(f"residual gap A-B (after bank GLM): {resid[half==1].mean()-resid[half==0].mean():+.3f}", flush=True)
