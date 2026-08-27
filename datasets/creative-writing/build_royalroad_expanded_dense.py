#!/usr/bin/env python3
"""dense_standard layout for the EXPANDED RoyalRoad cell (n=1,742).
Splits are the per-fiction stable hash already stored on the population, so no
row moves across a boundary as the population grows."""
import json, os
from pathlib import Path
import pandas as pd
REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
CELL = REPO / "datasets/creative-writing/royalroad_stubs"
d = pd.read_csv(CELL / "va_expanded/population.csv.gz")
out = CELL / "dense_expanded"; (out / "split").mkdir(parents=True, exist_ok=True)
cols = ["text", "judgement", "group", "row_id"]
d[cols].to_csv(out / "data.csv", index=False)
for sp in ("train", "eval", "test"):
    d.loc[d.split == sp, cols].to_csv(out / f"split/{sp}.csv", index=False)
c = d.split.value_counts().to_dict(); pos = d.groupby("split").judgement.sum().astype(int).to_dict()
man = {"cell": "cw_royalroad_verdict_expanded", "n": int(len(d)),
       "n_pos": int(d.judgement.sum()), "split_row_counts": {k: int(v) for k, v in c.items()},
       "split_pos_counts_ABSOLUTE": {k: int(v) for k, v in pos.items()},
       "split_fractions": {k: round(v / len(d), 4) for k, v in c.items()},
       "recipe": "Llama-3.1-8B LoRA r16/a32 lr5e-5 batch16 len1024 2ep gradckpt "
                 "select-on-eval + --class_weight_auto, seeds 42/1/2"}
(out / "manifest.json").write_text(json.dumps(man, indent=2))
print(json.dumps(man, indent=1))
