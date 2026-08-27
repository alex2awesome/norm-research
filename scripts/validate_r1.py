"""Manual validation of R1 families across tasks.

For each --task in the comma list (default = a diverse 4-task sample):
  - 5 largest families: representative + sample members (eyeball coherence).
  - 5 random mid-size (3-7 member) families.
  - 3 duplicate-name cases (different families sharing the same name -- show
    descriptions so we can confirm they encode different sub-rules, not the
    same rule fragmented).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
FORMS = ROOT / "outputs" / "analyses" / "canon_all_real_forms.jsonl"
METR = ROOT / "outputs" / "analyses" / "structural_metrics"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="code-review,creative-writing,humor,notice-and-comment")
    ap.add_argument("--r1-dir", default=str(METR),
                    help="Directory with r1_families_<task>.json")
    args = ap.parse_args()
    tasks = args.tasks.split(",")
    r1_dir = Path(args.r1_dir)
    rng = random.Random(0)

    forms_by_task = defaultdict(list)
    for line in FORMS.open():
        r = json.loads(line)
        forms_by_task[r["task"]].append(r)

    for task in tasks:
        fp = r1_dir / f"r1_families_{task}.json"
        cp = METR / f"clusters_{task}.json"
        if not fp.exists() or not cp.exists():
            print(f"{task}: missing files -- skipped")
            continue
        d = json.loads(fp.read_text())
        cl = json.loads(cp.read_text())
        members = defaultdict(list)
        for r in forms_by_task[task]:
            members[cl[r["key"]]].append(r["canonical"] or "")
        reps = {c: Counter(m).most_common(1)[0][0] for c, m in members.items()}

        fams = d["families"]
        sizes = [len(f["cluster_ids"]) for f in fams]
        n_multi = sum(1 for s in sizes if s > 1)
        names = [f["name"].lower().strip() for f in fams]
        nc = Counter(names)
        dups = {k: v for k, v in nc.items() if v > 1}

        print(f"\n{'=' * 78}")
        print(f"{task}: {len(fams)} R1 families "
              f"({n_multi} multi, {sizes.count(1)} singletons), "
              f"max {max(sizes)}, {len(dups)} duplicated names")
        print(f"{'=' * 78}")

        print("\n-- TOP-5 LARGEST FAMILIES --")
        for f in sorted(fams, key=lambda x: -len(x["cluster_ids"]))[:5]:
            print(f"\n  [{len(f['cluster_ids'])}] {f['name']}")
            print(f"      {f.get('description', '')[:130]}")
            for cid in f["cluster_ids"][:7]:
                print(f"      - {reps.get(cid, '?')[:96]}")
            if len(f["cluster_ids"]) > 7:
                print(f"      ... +{len(f['cluster_ids']) - 7} more")

        print("\n-- 5 RANDOM MID-SIZE FAMILIES (3-7 members) --")
        mid = [f for f in fams if 3 <= len(f["cluster_ids"]) <= 7]
        for f in rng.sample(mid, min(5, len(mid))):
            print(f"\n  [{len(f['cluster_ids'])}] {f['name']}")
            print(f"      {f.get('description', '')[:130]}")
            for cid in f["cluster_ids"]:
                print(f"      - {reps.get(cid, '?')[:96]}")

        print("\n-- 3 DUPLICATE-NAME CASES (do the descriptions differ?) --")
        dup_names_sorted = sorted(dups.items(), key=lambda x: -x[1])[:3]
        for dn, cnt in dup_names_sorted:
            print(f"\n  \"{dn}\" appears {cnt}x:")
            copies = [f for f in fams if f["name"].lower().strip() == dn]
            for i, f in enumerate(copies):
                print(f"    #{i+1} [{len(f['cluster_ids'])}] {f.get('description', '')[:140]}")
                for cid in f["cluster_ids"][:3]:
                    print(f"       - {reps.get(cid, '?')[:88]}")


if __name__ == "__main__":
    main()
