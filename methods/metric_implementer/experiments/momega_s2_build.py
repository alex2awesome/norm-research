"""S2 build: encode-pass manifests (candidates x probes) + corpus-pass manifests
(candidates x corpus). Form ids: C0..C5 -> 700..705. Runs on sk3."""
import json
from pathlib import Path

BANKS = Path("/lfs/skampere3/0/alexspan/outputs/ecert_slice_v1")
MD = Path("/lfs/skampere3/0/alexspan/mention_auc")
cands = json.load(open(BANKS / "momega_candidates.json"))
per = {}
for r in cands:
    for j in range(6):
        per.setdefault(r["task"], []).append(
            {"metric_id": r["metric"], "form_idx": 700 + j,
             "rubric": r["candidates"][f"C{j}"][:1800]})
for task, rows in per.items():
    json.dump(rows, open(MD / f"mo_{task}_manifest.json", "w"), indent=0)
    print(task, len(rows), "rubrics")
