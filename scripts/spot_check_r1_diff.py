"""Spot-check Claude-subagent R1 vs sk3-Llama R1 on peer-review.

Strategy:
  1. Build per-cluster family lookup for each R1 version.
  2. For every pair (c_i, c_j) of clusters, tag:
       - "agree-merge"  = both put them in the same family
       - "agree-split"  = different family in both
       - "claude-merged-only"  = same in subagent, different in sk3
       - "llama-merged-only"   = same in sk3, different in subagent
  3. Restrict to pairs where AT LEAST ONE put them in a multi-member family
     (to keep things diagnostic; singleton-singleton pairs are trivial).
  4. Sample N pairs from each disagreement category and print rep texts
     side-by-side so the user can eyeball who's right.

This is the most diagnostic comparison: it surfaces exactly the merge
decisions where the two pipelines diverge.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path


def load_r1(path: Path) -> dict[int, int]:
    """Return cluster_id -> family_id mapping."""
    d = json.loads(path.read_text())
    fams = d.get("families", [])
    mp: dict[int, int] = {}
    for fi, f in enumerate(fams):
        members = f.get("cluster_ids") or f.get("members") or []
        for m in members:
            s = str(m).strip()
            if s.startswith("C"):
                s = s[1:]
            try:
                mp[int(s)] = fi
            except ValueError:
                continue
    return mp, fams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer-review")
    ap.add_argument("--n-per-category", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    llama_path = Path("outputs/analyses/structural_metrics/r1_v4a/"
                      f"r1_families_{args.task}.json")
    claude_path = Path("outputs/analyses/structural_metrics/r1_v4a_subagent/"
                       f"r1_families_{args.task}.json")
    reps = json.loads(Path(f"/tmp/r1_subagent/{args.task}/"
                           "clusters_repr.json").read_text())

    llama_map, llama_fams = load_r1(llama_path)
    claude_map, claude_fams = load_r1(claude_path)

    common = set(llama_map) & set(claude_map)
    print(f"common clusters: {len(common)}")

    # Group clusters by family for both
    llama_by_fam: dict[int, list[int]] = defaultdict(list)
    for c, fi in llama_map.items():
        llama_by_fam[fi].append(c)
    claude_by_fam: dict[int, list[int]] = defaultdict(list)
    for c, fi in claude_map.items():
        claude_by_fam[fi].append(c)

    # Only consider pairs from multi-member families on either side
    llama_pairs: set[tuple[int, int]] = set()
    for cids in llama_by_fam.values():
        if len(cids) >= 2:
            for a, b in combinations(sorted(cids), 2):
                llama_pairs.add((a, b))
    claude_pairs: set[tuple[int, int]] = set()
    for cids in claude_by_fam.values():
        if len(cids) >= 2:
            for a, b in combinations(sorted(cids), 2):
                claude_pairs.add((a, b))

    both_merged = llama_pairs & claude_pairs
    llama_only = llama_pairs - claude_pairs   # llama merged, claude split
    claude_only = claude_pairs - llama_pairs  # claude merged, llama split

    print(f"merged-pair counts:")
    print(f"  llama: {len(llama_pairs)}")
    print(f"  claude: {len(claude_pairs)}")
    print(f"  both agree (merged):    {len(both_merged)}")
    print(f"  only llama merged:      {len(llama_only)}")
    print(f"  only claude merged:     {len(claude_only)}")

    def print_section(title: str, pairs: list[tuple[int, int]]):
        print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")
        sample = rng.sample(pairs, min(args.n_per_category, len(pairs)))
        for i, (a, b) in enumerate(sample, 1):
            la = llama_fams[llama_map[a]]
            lb = llama_fams[llama_map[b]]
            ca = claude_fams[claude_map[a]]
            cb = claude_fams[claude_map[b]]
            print(f"\n[{i}] C{a}  vs  C{b}")
            print(f"  rep_a: {reps[str(a)][:200]}")
            print(f"  rep_b: {reps[str(b)][:200]}")
            print(f"  LLAMA:  a→ {la['name'][:60]}  |  b→ {lb['name'][:60]}")
            print(f"  CLAUDE: a→ {ca['name'][:60]}  |  b→ {cb['name'][:60]}")

    print_section("BOTH MERGED  (sanity-check agreement)", list(both_merged))
    print_section("LLAMA-ONLY MERGED  (Claude split — is Claude right?)",
                  list(llama_only))
    print_section("CLAUDE-ONLY MERGED  (Llama split — is Claude right?)",
                  list(claude_only))


if __name__ == "__main__":
    main()
