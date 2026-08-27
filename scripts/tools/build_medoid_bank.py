"""Coverage-selected rubric bank for any task: embed the full pool, KMeans k=40, medoids.
Usage: python scripts/tools/build_medoid_bank.py <task-dir-name>"""
import sys, json; sys.path.insert(0, "methods")
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from metrics_tree_infilling.io_metrics import REPO_ROOT, load_rubric_metrics
from metrics_tree_infilling.depth_dial import embed

task = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 40
specs = load_rubric_metrics(task)
print(f"[{task}] pool: {len(specs)} rubrics", flush=True)
texts = [f"{s.name}. {s.description}"[:400] for s in specs]
E = embed(texts, "all-MiniLM-L6-v2")
E = E / np.linalg.norm(E, axis=1, keepdims=True).clip(min=1e-9)
km = KMeans(K, n_init=4, random_state=0).fit(E)
medoids = []
for k in range(K):
    idx = np.where(km.labels_ == k)[0]
    c = km.cluster_centers_[k] / np.linalg.norm(km.cluster_centers_[k]).clip(min=1e-9)
    medoids.append((int(idx[np.argmin(((E[idx] - c) ** 2).sum(1))]), len(idx)))
out = Path(REPO_ROOT / f"datasets/{task}/medoid-bank")
out.mkdir(parents=True, exist_ok=True)
doc = {"extracted": {"rubrics_metrics": [
    {"name": specs[i].name, "description": specs[i].description,
     "guidance": specs[i].guidance, "cluster_size": n} for i, n in medoids]}}
json.dump(doc, open(out / "bank.json", "w"), indent=2)
print(f"wrote {K} medoids -> {out/'bank.json'}", flush=True)
