"""Per-metric test-retest reliability + noise ceilings (PRECONDITION for the 5.1 slope).

Audit F3: subjective metrics plausibly have lower judge reliability, mechanically attenuating
rho on the articulation side - an upward slope could be pure measurement artifact. So every
metric gets a reliability estimate BEFORE the headline is read, and every reported rho gets an
attenuation-corrected twin.

Reliability here = mean pairwise Spearman between independent target-judgment vectors of the
same metric (across reps and/or prompt forms), Spearman-Brown-corrected to the length actually
averaged at analysis time.

CPU-only; consumes the same grid npz layout as everything else.

Usage:
  python -m methods.tacit_channels.channels.peer_review.reliability_ceilings \
      --scores-root <root> --job <target job> --domain peer-review \
      --out outputs/tacit_channels/peer_review/reliability.json
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
from collections import defaultdict

import numpy as np

from methods.tacit_channels.channels.common import spearman


def load_replicate_vectors(root: str, job: str, domain: str):
    """{(cell_id, arm_id): [vector per (rep, form) replicate]} - replicates NOT averaged."""
    vecs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, job, f"grid_{domain}_*_rep*.npz"))):
        d = np.load(path, allow_pickle=True)
        scores = np.asarray(d["scores"])
        for i, s in enumerate(d["meta"]):
            m = json.loads(s)
            vecs[(m["cell_id"], m["arm_id"])].append(scores[i])
    return vecs


def reliability(vectors: list[np.ndarray]) -> dict:
    """Mean pairwise Spearman across replicates + Spearman-Brown to the averaged length."""
    if len(vectors) < 2:
        return {"n_replicates": len(vectors), "r_single": None, "r_mean_of_k": None}
    pairs = [spearman(a, b) for a, b in itertools.combinations(vectors, 2)]
    pairs = [p for p in pairs if not np.isnan(p)]
    if not pairs:
        return {"n_replicates": len(vectors), "r_single": None, "r_mean_of_k": None}
    r1 = float(np.mean(pairs))
    k = len(vectors)
    r_k = (k * r1) / (1 + (k - 1) * r1) if r1 > -1 / (k - 1) else None
    return {"n_replicates": k, "r_single": round(r1, 4),
            "r_mean_of_k": round(r_k, 4) if r_k is not None else None}


def attenuation_corrected(rho: float, rel_x: float, rel_y: float) -> float | None:
    """rho_true = rho_observed / sqrt(rel_x * rel_y); None outside valid range."""
    if rel_x is None or rel_y is None or rel_x <= 0 or rel_y <= 0:
        return None
    denom = (rel_x * rel_y) ** 0.5
    return round(rho / denom, 4) if denom > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-root", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--arm", default="name",
                    help="which arm's replicates to assess (default: name)")
    ap.add_argument("--min-reliability", type=float, default=0.5,
                    help="flag metrics below this as slope-uninterpretable")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vecs = load_replicate_vectors(args.scores_root, args.job, args.domain)
    report, flagged = {}, []
    for (cell_id, arm_id), vectors in sorted(vecs.items()):
        if arm_id != args.arm:
            continue
        rel = reliability(vectors)
        report[cell_id] = rel
        if rel["r_single"] is not None and rel["r_single"] < args.min_reliability:
            flagged.append(cell_id)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"job": args.job, "domain": args.domain, "arm": args.arm,
               "min_reliability": args.min_reliability,
               "ceilings": report, "flagged_low_reliability": flagged},
              open(args.out, "w"), indent=2)
    n2 = sum(1 for r in report.values() if r["n_replicates"] >= 2)
    print(f"{len(report)} metrics; {n2} with >=2 replicates; "
          f"{len(flagged)} flagged below r={args.min_reliability}")
    if n2 < len(report):
        print("WARNING: metrics with <2 replicates have NO ceiling - the slope is not "
              "interpretable for them until a second rep/form pass runs (audit F3).")


if __name__ == "__main__":
    main()
