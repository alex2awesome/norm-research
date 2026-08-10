"""Spot-check bs=40 vs bs=200 subagent R1 on peer-review.

Diff in disagreement direction:
  - bs40-only merged  -> bs200 split (is bs200 right to split or being timid?)
  - bs200-only merged -> bs40 split  (is bs200 catching something bs40 missed?)
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path


def load_r1(path: Path):
    d = json.loads(path.read_text())
    fams = d.get("families", [])
    mp = {}
    for fi, f in enumerate(fams):
        for m in f.get("cluster_ids") or f.get("members") or []:
            s = str(m).strip().lstrip("C")
            try: mp[int(s)] = fi
            except ValueError: continue
    return mp, fams


def pairs_from(mp, fams):
    by_fam = defaultdict(list)
    for c, fi in mp.items():
        by_fam[fi].append(c)
    pairs = set()
    for cids in by_fam.values():
        if len(cids) >= 2:
            for a, b in combinations(sorted(cids), 2):
                pairs.add((a, b))
    return pairs


bs40_map, bs40_fams = load_r1(Path(
    "outputs/analyses/structural_metrics/r1_v4a_subagent/"
    "r1_families_peer-review.json"))
bs200_map, bs200_fams = load_r1(Path(
    "outputs/analyses/structural_metrics/r1_v4a_subagent_bs200/"
    "r1_families_peer-review.json"))
reps = json.loads(Path("/tmp/r1_subagent/peer-review/clusters_repr.json"
                       ).read_text())

bs40_pairs = pairs_from(bs40_map, bs40_fams)
bs200_pairs = pairs_from(bs200_map, bs200_fams)

both = bs40_pairs & bs200_pairs
only40 = bs40_pairs - bs200_pairs   # bs40 merged, bs200 split
only200 = bs200_pairs - bs40_pairs  # bs200 merged, bs40 split

print(f"pair counts: bs40={len(bs40_pairs)} bs200={len(bs200_pairs)} "
      f"both={len(both)} only_bs40={len(only40)} only_bs200={len(only200)}")

rng = random.Random(0)


def show(title, pairs, n=6):
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")
    sample = rng.sample(pairs, min(n, len(pairs)))
    for i, (a, b) in enumerate(sample, 1):
        print(f"\n[{i}] C{a}  vs  C{b}")
        print(f"  rep_a: {reps[str(a)][:180]}")
        print(f"  rep_b: {reps[str(b)][:180]}")
        f40a = bs40_fams[bs40_map[a]]
        f40b = bs40_fams[bs40_map[b]]
        f200a = bs200_fams[bs200_map[a]]
        f200b = bs200_fams[bs200_map[b]]
        print(f"  bs40:  a→ {f40a['name'][:55]}  |  b→ {f40b['name'][:55]}")
        print(f"  bs200: a→ {f200a['name'][:55]}  |  b→ {f200b['name'][:55]}")


show("ONLY BS40 MERGED (bs200 split — too timid? or right?)", list(only40))
show("ONLY BS200 MERGED (bs40 split — bs200 caught a missed merge?)",
     list(only200))
show("BOTH AGREE (merged)", list(both), n=4)
