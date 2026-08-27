#!/usr/bin/env python3
"""V7 split repair: replace the imported claim-fell bucketer with a plain
STABLE HASH on the family group.

WHY THE IMPORTED BUCKETER FAILED HERE (worth recording -- it will bite any other
near-singleton-group cell): `stable_hash_bucket_map` scores a candidate bucket as
   size_term + lam * sum_b (pos_rate_b - overall_rate)^2 ,  lam = 2.5
Adding a single row to a nearly-empty bucket swings that bucket's pos rate to 0
or 1 (penalty ~lam*0.25), while adding it to the large bucket barely moves the
rate. With 15,973 groups of median size 1 the pos-rate term therefore dominates
the size term at every step and the greedy pours everything into one bucket:
train 15,972 / eval 14 / test 14. The term is load-bearing on claim-fell (few
large app groups, corr(group size, y) = +.30); here corr(group size, y) = +.013,
so it buys nothing and costs the split. Plain stable hashing is both the build
spec ("stable-hash grouped splits") and the standing no-seeded-shuffle rule.
"""
import hashlib, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

OUTD = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/v7_community/"
pop = pd.read_csv(OUTD + "population.csv.gz")
man = json.load(open(OUTD + "population_manifest.json"))

def bucket(g):
    h = int(hashlib.sha1(("v7-split|" + str(g)).encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else ("eval" if h < 90 else "test")

pop["split"] = pop.family_group.map(bucket)
sizes = pop.split.value_counts().to_dict()
rates = pop.groupby("split").y_fwd5.mean().to_dict()
print("split sizes:", sizes)
print("split pos rates:", {k: round(v, 4) for k, v in rates.items()})

# no family may straddle a split
straddle = pop.groupby("family_group").split.nunique()
assert int(straddle.max()) == 1, "family straddles splits"
print("families straddling splits:", int((straddle > 1).sum()))

# cohort coverage across splits
print("cohorts in each split:", pop.groupby("split").cohort.nunique().to_dict())
y = pop.y_fwd5.values.astype(int)
print("split-identity alone AUC:",
      round(float(roc_auc_score(y, pd.factorize(pop.split)[0].astype(float))), 4))

pop.to_csv(OUTD + "population.csv.gz", index=False, compression="gzip")
man["split_sizes"] = sizes
man["split_pos_rates"] = rates
man["splitter"] = ("stable sha1 hash of family_group into 80/10/10 "
                   "('v7-split|' + group). REPLACES the imported "
                   "stable_hash_bucket_map, which collapses to a single bucket on "
                   "near-singleton groups -- see v7_fix_splits.py docstring.")
man["n_families_straddling_splits"] = 0
json.dump(man, open(OUTD + "population_manifest.json", "w"), indent=2, default=str)
print("V7_SPLIT_FIX_DONE")
