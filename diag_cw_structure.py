"""Offline diagnostic (no GLM): rubric hierarchy + CW text subpopulations.
Scoped to the rubrics the ctree actually uses (first 40) AND a larger sample of the full space."""
import sys; sys.path.insert(0, "methods")
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from metrics_tree_infilling.io_metrics import load_rubric_metrics, REPO_ROOT
from metrics_tree_infilling.depth_dial import embed
from metrics_tree_infilling.run import DATASET_CONFIGS


def hierarchy(rubrics, label):
    descs = [f"{r.name}. {r.description}" for r in rubrics]
    E = embed(descs, "all-MiniLM-L6-v2")
    D = 1.0 - E @ E.T; np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    print(f"\n== {label}: {len(rubrics)} rubrics ==", flush=True)
    for t in (0.45, 0.55, 0.65):
        lab = fcluster(Z, t=t, criterion="distance")
        sizes = sorted(np.bincount(lab), reverse=True)
        n_multi = int(sum(1 for s in sizes if s > 1))
        print(f"  cosine-dist<{t}: {len(set(lab))} clusters ({n_multi} multi-member), top sizes {sizes[:8]}", flush=True)
    lab = fcluster(Z, t=0.55, criterion="distance")
    for c in sorted(set(lab)):
        members = [rubrics[i].name[:34] for i in range(len(rubrics)) if lab[i] == c]
        if len(members) > 1:
            print(f"    C{c}({len(members)}): {members[:5]}", flush=True)


allrub = load_rubric_metrics("creative-writing", limit=300)
hierarchy(allrub[:40], "A1. the 40 rubrics the ctree used")
hierarchy(allrub, "A2. 300-rubric sample of the full space (73,702 total)")

print("\n== B. CW text subpopulations ==", flush=True)
dcfg = DATASET_CONFIGS["creative-writing"]
df = pd.read_csv(REPO_ROOT / dcfg["split"], low_memory=False).dropna(subset=[dcfg["text"], dcfg["label"]])
df[dcfg["label"]] = pd.to_numeric(df[dcfg["label"]], errors="coerce"); df = df.dropna(subset=[dcfg["label"]])
samp = df.sample(min(1000, len(df)), random_state=7)
texts = samp[dcfg["text"]].astype(str).str[:1500].tolist()
y = samp[dcfg["label"]].to_numpy().astype(int)
print(f"{len(texts)} texts, base label-rate {y.mean():.2f}", flush=True)
Et = embed(texts, "all-MiniLM-L6-v2")
for K in (5, 10):
    km = KMeans(K, n_init=8, random_state=0).fit(Et)
    rates = [float(y[km.labels_ == k].mean()) for k in range(K)]
    sizes = [int((km.labels_ == k).sum()) for k in range(K)]
    print(f"  K={K}: label-rate by cluster {[round(r,2) for r in rates]} sizes {sizes} "
          f"(spread {max(rates)-min(rates):.2f})", flush=True)
