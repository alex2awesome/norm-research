"""Inspect the 56 hard FPs Fork 3 produced.

For each v6 score=0 (unrelated) pair where both clusters ended up in the same
Fork 3 family, find:
  - the merge edge (family pair Claude said YES on) that brought them together
  - the two cluster rep texts
  - the v6 judge's reasoning

Helps decide: are those 56 actually fine merges (v6 was wrong) or real Claude
errors?
"""
from __future__ import annotations

import json
import random
from pathlib import Path

# Load Fork 3 result + base R1 + family meta
fork3 = json.loads(Path("outputs/analyses/structural_metrics/"
                         "r1_v4a_lora_fork3_merge/"
                         "r1_families_peer-review.json").read_text())
base_r1 = json.loads(Path("outputs/analyses/structural_metrics/"
                          "r1_v4a_subagent_lora_bs400/"
                          "r1_families_peer-review.json").read_text())
fam_meta = json.loads(Path("/tmp/r1_fork3/peer-review/family_meta.json").read_text())
reps = json.loads(Path("/tmp/r1_subagent/peer-review/clusters_repr.json").read_text())

# cluster -> merged-family-id  (using cluster_ids list per fork3 family)
cluster_to_merged_fam = {}
for mi, mf in enumerate(fork3["families"]):
    for c in mf["cluster_ids"]:
        cluster_to_merged_fam[int(c)] = mi

# Load v6 verdicts for peer-review
v6_pairs = [json.loads(l) for l in
            open("outputs/analyses/structural_metrics/validation/"
                 "peer-review_v6_verdicts.jsonl")]

# Map key -> cluster_id
key2cluster = {k: int(v) for k, v in json.loads(
    open("outputs/analyses/structural_metrics/clusters_peer-review.json"
         ).read()).items()}

# Find hard FPs: v6 score=0 + both clusters in same Fork 3 family
hard_fps = []
for p in v6_pairs:
    if p["score"] != 0: continue
    ka, kb = p["key_a"], p["key_b"]
    if ka not in key2cluster or kb not in key2cluster: continue
    ca, cb = key2cluster[ka], key2cluster[kb]
    if ca == cb: continue
    if ca not in cluster_to_merged_fam or cb not in cluster_to_merged_fam:
        continue
    if cluster_to_merged_fam[ca] != cluster_to_merged_fam[cb]: continue
    # Find the SOURCE base families
    source_a = None; source_b = None
    for fi, bf in enumerate(base_r1["families"]):
        bcids = [int(str(c).lstrip("C")) for c in
                 (bf.get("cluster_ids") or bf.get("members") or [])
                 if str(c).lstrip("C").isdigit()]
        if ca in bcids: source_a = fi
        if cb in bcids: source_b = fi
    hard_fps.append({
        "score": p["score"],
        "ca": ca, "cb": cb,
        "rep_a": reps[str(ca)][:200], "rep_b": reps[str(cb)][:200],
        "canonical_a": p.get("canonical_a", "")[:200],
        "canonical_b": p.get("canonical_b", "")[:200],
        "judge_reasoning": p.get("judge_reasoning", "")[:400],
        "source_fa": source_a, "source_fb": source_b,
        "same_source": source_a == source_b,
    })

print(f"Total hard FPs (Fork 3 merged a v6 score=0 pair): {len(hard_fps)}")
already_in_base = sum(1 for h in hard_fps if h["same_source"])
new_from_fork3 = len(hard_fps) - already_in_base
print(f"  already in same base family (NOT from Fork 3 merge): {already_in_base}")
print(f"  NEW from Fork 3 merge: {new_from_fork3}")

# Show 10 random NEW ones
new_only = [h for h in hard_fps if not h["same_source"]]
sample = random.Random(0).sample(new_only, min(10, len(new_only)))
for i, h in enumerate(sample, 1):
    print(f"\n[{i}] CLUSTERS C{h['ca']} vs C{h['cb']}  (v6 score=0)")
    print(f"  rep_a: {h['rep_a']}")
    print(f"  rep_b: {h['rep_b']}")
    fa = base_r1["families"][h["source_fa"]]
    fb = base_r1["families"][h["source_fb"]]
    print(f"  base family A: {fa.get('name', '')[:80]}")
    print(f"  base family B: {fb.get('name', '')[:80]}")
    print(f"  v6 reasoning: {h['judge_reasoning']}")
