#!/usr/bin/env python3
"""Recovery MI for the patents claim-comparison (disclosure) metric: I(M; Y) in bits.

Puts the disclosure/anticipation metric on the same recovery axis as the rest of the project
(C(R(Ω)) = I(M_ω; label)) instead of only AUC. M = claim disclosure score (softmin of element
scores, the pipeline's own aggregation); Y = label_fell. Reports I(M;Y), H(Y), the normalized
fraction I/H(Y), and AUC for reference, on retrieved_only and mixture conditions.

NOTE: this is a READOUT of an already-built, label-free metric (reconstruction-only preserved) —
the metric was never optimized against Y. Run ON sk3 (CPU):
  python3 scripts/patents_recovery_mi.py
"""
import json, sys
import numpy as np
sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research/scripts")
from score_gold_mixture import claim_scores_with_T
from sklearn.metrics import roc_auc_score

TB = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/processed/gold_mixture_testbed_v1"
FILES = {"semantic_delta": f"{TB}/units_semdelta_elscores.json"}


def entropy(p):
    p = np.asarray(p, float); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mi_binned(y, m, bins=10):
    """I(M;Y) with M quantile-binned (robust to scale), Y binary. Bits."""
    y = np.asarray(y, int); m = np.asarray(m, float)
    # quantile bin edges (dedup)
    edges = np.unique(np.quantile(m, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    b = np.clip(np.digitize(m, edges[1:-1]), 0, len(edges) - 2)
    n = len(y)
    Hy = entropy(np.bincount(y) / n)
    Hy_given_m = 0.0
    for bi in np.unique(b):
        mask = b == bi
        w = mask.mean()
        Hy_given_m += w * entropy(np.bincount(y[mask], minlength=2) / mask.sum())
    return max(0.0, Hy - Hy_given_m), Hy


def main():
    for name, path in FILES.items():
        d = json.load(open(path))
        print(f"\n=== {name} ===", flush=True)
        for cond in ("retrieved_only", "mixture"):
            if cond not in d:
                continue
            rows = d[cond]
            per_claim = {k: (v[0], v[1], tuple(v[2]), v[3]) for k, v in rows.items() if v[1]}
            cs, _ = claim_scores_with_T(per_claim)
            y = np.array([per_claim[k][0] for k in cs])
            m = np.array([cs[k] for k in cs])
            if len(set(y.tolist())) < 2:
                continue
            mi, Hy = mi_binned(y, m)
            auc = roc_auc_score(y, m)
            print(f"  {cond:16s} n={len(y):5d}  AUC={auc:.3f}  I(M;Y)={mi:.4f} bits  "
                  f"H(Y)={Hy:.3f}  recovery={mi/Hy:.1%} of H(Y)", flush=True)
        # reference: what MI would a perfect (AUC=1) and a strong (AUC=0.75) metric give at this base rate?
    print("\nref: at balanced base rate H(Y)=1.0 bit; AUC 0.60 metrics typically carry ~1-3% of H(Y);"
          " AUC 0.75 ~ 8-12%; a 'high-signal promptable' task shows tens of % recoverable.", flush=True)
    print("CAVEAT (Codex audit): pooled over claims with no app-clustered resampling — apps with "
          "many claims dominate and errors cluster by app. Treat MI as a point estimate, not a "
          "precise population value.", flush=True)
    print("RECOVERY_MI_DONE", flush=True)


if __name__ == "__main__":
    main()
