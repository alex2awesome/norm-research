#!/usr/bin/env python3
"""Label-blind TRAIN-SPLIT exemplars for the V7 A-bank proposer pass.
y is NEVER attached. Deterministic sha256 order, 4 disjoint batches."""
import hashlib, json, sys
import pandas as pd
OUTD = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/v7_community/"
pop = pd.read_csv(OUTD + "population.csv.gz")
tr = pop[pop.split == "train"].copy()
tr["k"] = [hashlib.sha256(("v7-mine|" + str(p)).encode()).hexdigest() for p in tr.patent_id]
tr = tr.sort_values("k")
NB, PER = 4, 14
for b in range(NB):
    rows = tr.iloc[b * PER:(b + 1) * PER]
    recs = [{"exemplar": i + 1, "title": r.title,
             "abstract": r.abstract[:1400], "claim_1": r.claim1[:2200]}
            for i, r in enumerate(rows.itertuples())]
    with open(OUTD + f"exemplars_batch{b}.json", "w") as f:
        json.dump(recs, f, indent=1)
    print(f"batch{b}: {len(recs)} exemplars")
print("n_train", len(tr))
