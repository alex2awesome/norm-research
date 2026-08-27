#!/usr/bin/env python
"""Stage 2: cross-tab the three family verdicts from family_verdict_join_v1_raw.json.

Outputs (per task ANALYSIS spec):
  (a) family-robust plateau set: metrics FLAT-or-FALLING in ALL THREE families
      -> top 30 by combined flatness (mean of the 3 slopes, ties by max-slope), with
         task + name.
  (b) metrics RISING in exactly one family (the dialect set): counts by owning family,
      broken down by task.
  (c) Qwen2.5-vs-Qwen3 agreement rate (hive-mind probe): pairwise disagreement rates
      between all 3 family pairs, both on the full 3-class verdict and on the binary
      RISING-vs-not axis.

Writes family_verdict_join_v1.json (full) + family_verdict_join_v1.md (summary).
"""
import json
from collections import Counter, defaultdict

OM = "/lfs/skampere3/0/alexspan/outputs/osl_multi"
FAMS = ["llama", "qwen25", "qwen3"]

d = json.load(open(f"{OM}/family_verdict_join_v1_raw.json"))
meta, rows = d["meta"], d["rows"]

# ---- keep only rows where all 3 families produced a real verdict (not INSUFFICIENT) ----
usable = [r for r in rows if all(r["families"][f]["verdict"] != "INSUFFICIENT" for f in FAMS)]
dropped = len(rows) - len(usable)
print(f"usable rows (all 3 families have >=3 rungs): {len(usable)} / {len(rows)} "
      f"({dropped} dropped as INSUFFICIENT in >=1 family)")

for r in usable:
    r["verdict3"] = {f: r["families"][f]["verdict"] for f in FAMS}
    r["slope3"] = {f: r["families"][f]["slope"] for f in FAMS}

# ---- (a) family-robust plateau set ----
plateau = [r for r in usable if all(r["verdict3"][f] in ("FLAT", "FALLING") for f in FAMS)]
for r in plateau:
    r["combined_flatness"] = sum(r["slope3"][f] for f in FAMS) / 3.0
    r["max_slope"] = max(r["slope3"][f] for f in FAMS)
plateau_sorted = sorted(plateau, key=lambda r: (r["combined_flatness"], r["max_slope"]))
top30 = plateau_sorted[:30]
print(f"\n(a) family-robust plateau set (FLAT-or-FALLING in ALL 3 families): "
      f"{len(plateau)} / {len(usable)} metrics "
      f"({100*len(plateau)/len(usable):.1f}%)")
by_task_plateau = Counter(r["task"] for r in plateau)
print("    by task:", dict(by_task_plateau))

# ---- (b) rising-in-exactly-one-family (dialect set) ----
def rising_owner(r):
    owners = [f for f in FAMS if r["verdict3"][f] == "RISING"]
    return owners[0] if len(owners) == 1 else None

dialect = [(r, rising_owner(r)) for r in usable]
dialect = [(r, o) for r, o in dialect if o is not None]
print(f"\n(b) rising-in-exactly-one-family (dialect set): {len(dialect)} / {len(usable)} "
      f"({100*len(dialect)/len(usable):.1f}%)")
owner_counts = Counter(o for _, o in dialect)
print("    owned by:", dict(owner_counts))
owner_task = defaultdict(lambda: Counter())
for r, o in dialect:
    owner_task[o][r["task"]] += 1
for o in FAMS:
    print(f"    {o}: {dict(owner_task[o])}")

# ---- (c) qwen2.5 vs qwen3 hive-mind probe ----
pairs = [("llama", "qwen25"), ("llama", "qwen3"), ("qwen25", "qwen3")]
disagree3 = {}
disagree_bin = {}
for a, b in pairs:
    n = len(usable)
    d3 = sum(1 for r in usable if r["verdict3"][a] != r["verdict3"][b])
    dbin = sum(1 for r in usable
               if (r["verdict3"][a] == "RISING") != (r["verdict3"][b] == "RISING"))
    disagree3[f"{a}_vs_{b}"] = round(d3 / n, 4)
    disagree_bin[f"{a}_vs_{b}"] = round(dbin / n, 4)
print("\n(c) hive-mind probe -- pairwise disagreement rate (3-class verdict):")
for k, v in disagree3.items():
    print(f"    {k}: {v}")
print("    pairwise disagreement rate (binary RISING-vs-not):")
for k, v in disagree_bin.items():
    print(f"    {k}: {v}")
hive_mind_verdict = ("Qwen3 disagrees with Qwen2.5 (same lab, different generation) about as "
                      "often as Llama disagrees with either Qwen family -> family-dialect is "
                      "NOT purely a same-lab artifact"
                      if disagree3["qwen25_vs_qwen3"] >= 0.7 * min(disagree3["llama_vs_qwen25"],
                                                                    disagree3["llama_vs_qwen3"])
                      else "Qwen3 agrees with Qwen2.5 much more than Llama agrees with either "
                           "-> some of the family-dialect signal IS a same-lab artifact")
print(f"    -> {hive_mind_verdict}")

out = dict(
    meta=dict(**meta, n_rows_total=len(rows), n_usable=len(usable), n_dropped_insufficient=dropped),
    plateau_set=dict(n=len(plateau), pct_of_usable=round(100 * len(plateau) / len(usable), 2),
                      by_task=dict(by_task_plateau),
                      top30=[dict(task=r["task"], name=r["name"],
                                  combined_flatness=round(r["combined_flatness"], 5),
                                  slopes=r["slope3"], verdicts=r["verdict3"])
                             for r in top30]),
    dialect_set=dict(n=len(dialect), pct_of_usable=round(100 * len(dialect) / len(usable), 2),
                      owner_counts=dict(owner_counts),
                      owner_by_task={o: dict(owner_task[o]) for o in FAMS}),
    hive_mind_probe=dict(pairwise_disagreement_3class=disagree3,
                         pairwise_disagreement_binary_rising=disagree_bin,
                         verdict=hive_mind_verdict),
    full_rows=[dict(task=r["task"], name=r["name"], n_frontier_items=r["n_frontier_items"],
                    verdict3=r["verdict3"], slope3=r["slope3"],
                    top_minus_mid=({f: r["families"][f].get("top_minus_mid") for f in FAMS}))
               for r in usable],
)
json.dump(out, open(f"{OM}/family_verdict_join_v1.json", "w"), indent=1)
print(f"\n-> wrote {OM}/family_verdict_join_v1.json")
