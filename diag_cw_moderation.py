"""Moderation test (cache-reuse, ~free): do CW rubric->label relationships differ across
text subpopulations? Reuses the B_tree judge cache (same 40 rubrics, same 500-item sample,
same max_chars => identical prompt hashes => cache hits). If per-cluster coefficient vectors
diverge, MOB-relevant heterogeneity EXISTS (the tree is underpowered/missing the axis)."""
import sys; sys.path.insert(0, "methods")
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.io_metrics import (
    REPO_ROOT, load_rubric_metrics, make_design, make_vllm_judge_scorer, materialize)
from metrics_tree_infilling.depth_dial import embed
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
    viability_min_applicability=0.1, viability_min_std=0.05)

rubrics = load_rubric_metrics("creative-writing", limit=40)
judge = make_vllm_judge_scorer(cfg)
probe = df.sample(min(60, len(df)), random_state=1)[dcfg["text"]].astype(str).tolist()
lv, ap = judge(rubrics, probe)
viable = [rubrics[j] for j in range(len(rubrics)) if ap[:, j].mean() > 0.1 and np.std(lv[ap[:, j], j]) > 0.05]
print(f"viable {len(viable)}/{len(rubrics)} (cache-driven)", flush=True)

sm = materialize(viable, df, cfg, judge)
y = df[dcfg["label"]].to_numpy()
Xd, fn, _, spec = make_design(sm, df, cfg)
keep = np.where(Xd.std(axis=0) > 1e-6)[0]
print(f"non-degenerate cols {len(keep)}/{Xd.shape[1]}", flush=True)

texts = df[dcfg["text"]].astype(str).str[:1500].tolist()
E = embed(texts, "all-MiniLM-L6-v2")
K = 8
km = KMeans(K, n_init=8, random_state=0).fit(E)
print(f"global label~levels AUC baseline...", flush=True)
coefs, rates = [], []
for k in range(K):
    idx = np.where(km.labels_ == k)[0]
    rates.append(float(y[idx].mean()) if len(idx) else float("nan"))
    if len(idx) < 25 or len(np.unique(y[idx])) < 2:
        coefs.append(None); continue
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xd[idx][:, keep], y[idx])
    coefs.append(lr.coef_[0])
    top = np.argsort(-np.abs(lr.coef_[0]))[:4]
    tops = [(str(fn[keep[t]])[:26], round(float(lr.coef_[0][t]), 2)) for t in top]
    print(f"  cluster {k} n={len(idx)} rate={y[idx].mean():.2f} top={tops}", flush=True)
print(f"label-rate by cluster {[round(r,2) for r in rates]}", flush=True)

valid = [c for c in coefs if c is not None]
if len(valid) >= 2:
    M = np.array(valid)
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True).clip(min=1e-9)
    cos = Mn @ Mn.T
    off = cos[~np.eye(len(valid), dtype=bool)]
    print(f"\nMODERATION SIGNAL — cross-cluster coef-vector cosine: mean {off.mean():.2f} "
          f"min {off.min():.2f}  (LOW => rubrics predict differently per subpopulation => "
          "MOB-relevant structure exists; HIGH => additive/stable)", flush=True)
