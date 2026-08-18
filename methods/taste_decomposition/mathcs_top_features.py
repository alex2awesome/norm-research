#!/usr/bin/env python3
"""Top V and A features for the six math/CS cells (2 domains x 3 y-types), for the
notebook's cross-channel comparison.  Per cell: alone-AUC of every A criterion
(NA median-imputed, the readout_va_gemma univariate convention) and every V
feature, top-10 each.  The mathse_multiy and so_votes shard sets carry BOTH the
verdict and community y's on one matrix; the bounty cells have their own shards.
CPU, sk3.  Output: results/mathcs_top_features.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

NR = Path("/lfs/skampere3/0/alexspan/norm-research")
RESULTS = NR / "methods/taste_decomposition/results"
sys.path.insert(0, str(NR / "methods/taste_decomposition"))
import scaleupC_layer1 as SC  # noqa: E402


def load_bank(tag, outdir):
    """ALWAYS through the framework loader — manual shard concatenation does NOT
    match the meta y order (first run: every alone-AUC pinned at ~.50)."""
    meta, A, V, groups, shard, ids = SC.load_scaleupC_bank(tag, out=NR / outdir)
    return A, V, list(meta["a_names"]), list(meta["v_names"]), meta


def alone(col, y):
    v = col.astype(float).copy()
    fin = np.isfinite(v)
    if fin.sum() < 50 or np.nanstd(v) == 0:
        return None
    v[~fin] = float(np.nanmedian(v))
    if np.std(v) == 0:
        return None
    return float(roc_auc_score(y, v))


def tops(M, names, y, k=10):
    rows = []
    for j in range(M.shape[1]):
        a = alone(M[:, j], y)
        if a is not None:
            rows.append({"name": names[j], "alone_auc": round(a, 4),
                         "dev": round(abs(a - .5), 4)})
    rows.sort(key=lambda r: -r["dev"])
    return rows[:k]


out = {}

# ---- mathse_multiy shards: accepted + vote y's on one matrix -----------------
A, V, an, vn, m = load_bank("mathse_multiy", "outputs/va_gemma_banks_scaleupC")
ys = m["ys"]
y_acc = np.array(ys["accepted_verdict"], dtype=float)
y_vote = np.array(ys["vote_score"], dtype=float)
out["mathse_accepted (VERDICT)"] = {"A": tops(A, an, y_acc.astype(int)),
                                    "V": tops(V, vn, y_acc.astype(int))}
kv = np.isfinite(y_vote)
out["mathse_vote (COMMUNITY)"] = {"A": tops(A[kv], an, y_vote[kv].astype(int)),
                                  "V": tops(V[kv], vn, y_vote[kv].astype(int))}

# ---- so_votes shards: accepted + vote --------------------------------------
A, V, an, vn, m = load_bank("so_votes", "outputs/va_gemma_banks_so_votes")
ys = m["ys"]
y_acc = np.array(ys["accepted_verdict"], dtype=float)
y_vote = np.array(ys["vote_score"], dtype=float)
out["so_accepted (VERDICT)"] = {"A": tops(A, an, y_acc.astype(int)),
                                "V": tops(V, vn, y_acc.astype(int))}
kv = np.isfinite(y_vote)
out["so_votes (COMMUNITY)"] = {"A": tops(A[kv], an, y_vote[kv].astype(int)),
                               "V": tops(V[kv], vn, y_vote[kv].astype(int))}

# ---- bounty cells -----------------------------------------------------------
for cell, tag in (("mathse_bounty (CURATED)", "mathse_bounty"),
                  ("so_bounty (CURATED)", "so_bounty")):
    A, V, an, vn, m = load_bank(tag, f"outputs/va_gemma_banks_{tag}")
    y = np.array(m["ys"]["bounty"], dtype=float).astype(int)
    out[cell] = {"A": tops(A, an, y), "V": tops(V, vn, y)}

(RESULTS / "mathcs_top_features.json").write_text(json.dumps(out, indent=1))
for cell, d in out.items():
    print(f"== {cell}")
    for r in d["A"][:3]:
        print(f"   A {r['name'][:52]:54s} {r['alone_auc']:.3f}")
    for r in d["V"][:2]:
        print(f"   V {r['name'][:52]:54s} {r['alone_auc']:.3f}")
print("MATHCS_TOPS_DONE")
