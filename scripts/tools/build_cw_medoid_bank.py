"""Coverage-selected CW rubric bank: embed all 73,702 deduped rubrics (MiniLM, CPU-fine),
KMeans k=40, take the medoid of each cluster -> a 40-rubric bank that SPANS the pool instead
of reading the first taxonomy's head. Output: datasets/creative-writing/medoid-bank/bank.json
(load with load_rubric_metrics_from_dir)."""
import sys, json; sys.path.insert(0, "methods")
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from metrics_tree_infilling.io_metrics import REPO_ROOT, load_rubric_metrics
from metrics_tree_infilling.depth_dial import embed

K = 40
specs = load_rubric_metrics("creative-writing")
print(f"pool: {len(specs)} rubrics", flush=True)
texts = [f"{s.name}. {s.description}"[:400] for s in specs]
E = embed(texts, "all-MiniLM-L6-v2")
E = E / np.linalg.norm(E, axis=1, keepdims=True).clip(min=1e-9)
km = KMeans(K, n_init=4, random_state=0, verbose=0).fit(E)
medoids = []
for k in range(K):
    idx = np.where(km.labels_ == k)[0]
    d = ((E[idx] - km.cluster_centers_[k] / np.linalg.norm(km.cluster_centers_[k]).clip(min=1e-9)) ** 2).sum(1)
    medoids.append((int(idx[np.argmin(d)]), len(idx)))

out = Path(REPO_ROOT / "datasets/creative-writing/medoid-bank")
out.mkdir(parents=True, exist_ok=True)
doc = {"extracted": {"rubrics_metrics": [
    {"name": specs[i].name, "description": specs[i].description,
     "guidance": specs[i].guidance, "cluster_size": n}
    for i, n in medoids]}}
with open(out / "bank.json", "w") as f:
    json.dump(doc, f, indent=2)
print(f"wrote {K} medoids -> {out/'bank.json'}", flush=True)
for i, n in sorted(medoids, key=lambda t: -t[1])[:12]:
    print(f"  [{n:6d} rubrics] {specs[i].name[:70]}", flush=True)
