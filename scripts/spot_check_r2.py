"""Extensive R2 spot-check across all 11 tasks.

For each task, samples and inspects:
  - 5 LARGEST aspects (verify they're coherent themes, not umbrellas)
  - 5 random multi-member aspects (size 3-15)
  - 5 singletons (could they have joined existing aspects?)
  - Cross-batch duplicate-name detection (same concept in multiple batches)

Outputs a human-readable report per task.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

TASKS = ["peer-review", "news-homepages", "grant-funding", "notice-and-comment",
         "legal-outcome-prediction", "patents", "creative-writing",
         "math-stackexchange", "humor", "press-releases", "code-review"]


def load_task(task):
    r2 = json.loads(Path(f"outputs/analyses/structural_metrics/r2_v1_subagent/"
                         f"r2_aspects_{task}.json").read_text())
    fam_meta = json.loads(Path(f"/tmp/r2_subagent/{task}/family_meta.json"
                               ).read_text())
    meta_by_fi = {m["fi"]: m for m in fam_meta}
    return r2, meta_by_fi


def show_aspect(a, meta_by_fi, show_n=5):
    print(f"\n  [{a['n_families']}] {a['name']}")
    print(f"        {a['description'][:140]}")
    members = a['family_ids'][:show_n]
    for fi in members:
        m = meta_by_fi.get(fi, {"name": "?"})
        print(f"          - F{fi}: {m['name'][:80]}")
    if a['n_families'] > show_n:
        print(f"          ... +{a['n_families'] - show_n} more members")


def cross_batch_dup_check(r2, top_k=20):
    """Find aspect names that recur in multiple batches (sign of cross-batch
    fragmentation that R2.5 would catch)."""
    name_to_batches = defaultdict(set)
    name_to_aspects = defaultdict(list)
    for a in r2["aspects"]:
        # normalize aspect name: lowercase, strip punctuation
        norm = re.sub(r"[^a-z0-9 ]", "", a["name"].lower()).strip()
        name_to_batches[norm].add(a["source_batch"])
        name_to_aspects[norm].append(a)
    multi_batch = [(norm, batches, name_to_aspects[norm])
                   for norm, batches in name_to_batches.items()
                   if len(batches) >= 2]
    # Order by total family count across copies
    multi_batch.sort(key=lambda x: -sum(a["n_families"] for a in x[2]))
    return multi_batch[:top_k]


def main():
    rng = random.Random(0)
    out_path = Path("notes/2026-05-24__r2-spot-check-report.md")
    out_path.parent.mkdir(exist_ok=True)
    lines = []
    lines.append("# R2 spot-check report\n")
    lines.append("Auto-generated. Each task: 5 largest aspects, 5 random mid-size aspects,\n"
                 "5 singletons (potential mis-classifications), and cross-batch duplicate-name\n"
                 "candidates (signal that an R2.5 merge pass would help).\n")

    summary_rows = []

    for task in TASKS:
        try:
            r2, meta = load_task(task)
        except Exception as e:
            print(f"SKIP {task}: {e}")
            continue
        aspects = r2["aspects"]
        sizes = [a["n_families"] for a in aspects]
        n_singletons = sum(1 for s in sizes if s == 1)

        lines.append(f"\n\n## {task}\n")
        lines.append(f"- R1 families: {r2['n_r1_families']}")
        lines.append(f"- R2 aspects: {r2['n_r2_aspects']} (compression {r2['compression']:.2f}×)")
        lines.append(f"- max aspect size: {max(sizes)}")
        lines.append(f"- singletons: {n_singletons} ({n_singletons/len(aspects)*100:.0f}%)\n")

        # 5 largest
        lines.append(f"### Top 5 largest aspects")
        biggest = sorted(aspects, key=lambda a: -a["n_families"])[:5]
        for a in biggest:
            lines.append(f"\n**[{a['n_families']}] {a['name']}**  ")
            lines.append(f"_{a['description']}_")
            lines.append(f"")
            for fi in a["family_ids"][:6]:
                m = meta.get(fi, {"name": "?"})
                lines.append(f"  - F{fi}: {m['name']}")
            if a["n_families"] > 6:
                lines.append(f"  - ... +{a['n_families']-6} more members")

        # 5 random mid-size (3-15)
        mid = [a for a in aspects if 3 <= a["n_families"] <= 15]
        if mid:
            lines.append(f"\n### 5 random mid-size aspects (size 3-15)")
            for a in rng.sample(mid, min(5, len(mid))):
                lines.append(f"\n**[{a['n_families']}] {a['name']}**  ")
                lines.append(f"_{a['description'][:200]}_")
                lines.append(f"")
                for fi in a["family_ids"]:
                    m = meta.get(fi, {"name": "?"})
                    lines.append(f"  - F{fi}: {m['name']}")

        # 5 random singletons (check for orphans)
        sing = [a for a in aspects if a["n_families"] == 1]
        if sing:
            lines.append(f"\n### 5 random singletons (could these join other aspects?)")
            for a in rng.sample(sing, min(5, len(sing))):
                fi = a["family_ids"][0]
                m = meta.get(fi, {"name": "?", "description": ""})
                lines.append(f"\n- **{a['name']}**: F{fi} ({m['name']})")

        # Cross-batch duplicates
        dups = cross_batch_dup_check(r2, top_k=15)
        n_dup = len(dups)
        if dups:
            lines.append(f"\n### Cross-batch duplicate aspect names ({n_dup} found)")
            lines.append(f"_These are concept that the LLM independently produced in multiple"
                         f" batches. They would be merged by an R2.5 pass._\n")
            for norm, batches, aspect_list in dups[:10]:
                total_fams = sum(a["n_families"] for a in aspect_list)
                lines.append(f"- **{aspect_list[0]['name']}** — "
                             f"{len(batches)} batches, "
                             f"total {total_fams} R1 families")

        summary_rows.append({
            "task": task,
            "r1": r2["n_r1_families"],
            "r2": r2["n_r2_aspects"],
            "compression": r2["compression"],
            "max_size": max(sizes),
            "singletons": n_singletons,
            "n_cross_batch_dups": n_dup,
        })

    # Summary table at top
    summary_lines = ["\n## Cross-task summary\n",
                     "| Task | R1 fams | R2 aspects | Compression | Max size | Singletons | "
                     "Cross-batch dup names |",
                     "|---|---|---|---|---|---|---|"]
    for r in summary_rows:
        summary_lines.append(
            f"| {r['task']} | {r['r1']} | {r['r2']} | {r['compression']:.2f}× | "
            f"{r['max_size']} | {r['singletons']} | {r['n_cross_batch_dups']} |")
    # Insert summary after the title
    final = lines[:5] + summary_lines + lines[5:]
    out_path.write_text("\n".join(final))
    print(f"\nwrote {out_path}")
    print(f"\nSummary:")
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
