"""Surface the families in the tau DECISION ZONE for manual same/different review.

Clusters the canonical forms at the loosest tau (0.90, complete linkage), then
for every multi-member family computes the MIN pairwise cosine similarity among
its members -- i.e. the complete-linkage threshold at which it formed. Families
are printed sorted by that min-sim.

Reading guide:
  - A family whose min-sim is, say, 0.95 only forms once tau <= 0.95. If its
    members are genuinely the same concept, then tau=0.97 SPLIT them -> a false
    negative at 0.97.
  - A family whose min-sim is 0.91 and whose members are NOT all the same
    concept is a false positive at tau<=0.91.
So one read of the 0.90-0.97 band answers both questions at once.

Usage: python scripts/examine_tau_bands.py --tasks creative-writing,code-review
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
OUT = ROOT / "outputs" / "analyses"
EMB = ROOT / "notebooks" / "_explore_cache" / "bge"
FORMS = OUT / "canon_real_forms.jsonl"


def load_task(task):
    rows = [json.loads(l) for l in FORMS.open() if json.loads(l)["task"] == task]
    rows.sort(key=lambda r: r["idx"])
    emb = np.load(EMB / f"emb_bge_canon_{task}.npy").astype(np.float64)
    return rows, emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="creative-writing,code-review")
    ap.add_argument("--lo", type=float, default=0.895, help="show families min-sim >= lo")
    ap.add_argument("--hi", type=float, default=0.975, help="... and < hi (decision zone)")
    ap.add_argument("--max-fam", type=int, default=28)
    args = ap.parse_args()

    for task in args.tasks.split(","):
        rows, emb = load_task(task)
        lab = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="complete",
            distance_threshold=1.0 - 0.90).fit_predict(emb)
        fam = defaultdict(list)
        for i, c in enumerate(lab):
            fam[c].append(i)

        zone = []  # (min_sim, member_indices)
        for members in fam.values():
            if len(members) < 2:
                continue
            E = emb[members]
            E = E / np.linalg.norm(E, axis=1, keepdims=True)
            sims = E @ E.T
            np.fill_diagonal(sims, 1.0)
            min_sim = float(sims.min())
            if args.lo <= min_sim < args.hi:
                zone.append((min_sim, members))
        zone.sort()

        print(f"\n{'='*90}\n{task}: {len(zone)} families in the decision zone "
              f"(min-sim {args.lo}-{args.hi}), loosest first\n{'='*90}")
        for min_sim, members in zone[:args.max_fam]:
            print(f"\n  [min-sim {min_sim:.3f}]  {len(members)} leaves:")
            for i in members[:10]:
                print(f"    - {rows[i]['canonical'][:108]}")
            if len(members) > 10:
                print(f"    ... +{len(members)-10} more")


if __name__ == "__main__":
    main()
