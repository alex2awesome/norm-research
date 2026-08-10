"""Apply meta-merge subagent verdicts: union-find merged families into a new R1.

Reads:
  /tmp/r1_meta_merge/<task>/responses/batch_*.json   (groups of F<n> ids per batch)
  /tmp/r1_meta_merge/<task>/family_meta.json         (fi -> original family)
  outputs/analyses/structural_metrics/<base_r1_dir>/r1_families_<task>.json

Writes:
  outputs/analyses/structural_metrics/<out_dir>/r1_families_<task>.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_response(p: Path) -> list[list[int]]:
    """Parse one batch response into list of F-id groups."""
    raw = p.read_text().strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    if m:
        raw = m.group(1).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        obj = json.loads(raw[start:end + 1])
    groups = []
    for g in obj.get("groups", []):
        members = []
        for m in g.get("members", []):
            s = str(m).strip().lstrip("F")
            try: members.append(int(s))
            except ValueError: continue
        if members:
            groups.append({
                "members": members,
                "name": g.get("name", ""),
                "description": g.get("description", ""),
            })
    return groups


def union_find(parents, a, b):
    ra, rb = a, b
    while parents[ra] != ra: ra = parents[ra]
    while parents[rb] != rb: rb = parents[rb]
    if ra != rb:
        parents[ra] = rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer-review")
    ap.add_argument("--base-r1-dir", default="r1_v4a_subagent_lora_bs400")
    ap.add_argument("--work-dir", default="/tmp/r1_meta_merge")
    ap.add_argument("--out-dir", default="r1_v4a_lora_meta_merge")
    args = ap.parse_args()

    work = Path(args.work_dir) / args.task
    out_dir = Path("outputs/analyses/structural_metrics") / args.out_dir
    out_dir.mkdir(exist_ok=True, parents=True)

    fam_meta = json.loads((work / "family_meta.json").read_text())
    n = max(m["fi"] for m in fam_meta) + 1
    parents = list(range(n))

    # Apply union-find from all batches
    n_groups = n_merge_edges = 0
    for bp in sorted((work / "responses").glob("batch_*.json")):
        groups = parse_response(bp)
        for g in groups:
            n_groups += 1
            members = g["members"]
            for i in range(1, len(members)):
                union_find(parents, members[0], members[i])
                n_merge_edges += 1

    # Group families by root
    by_root = {}
    new_name_desc = {}  # root -> chosen name/desc (from largest group)
    for bp in sorted((work / "responses").glob("batch_*.json")):
        groups = parse_response(bp)
        for g in groups:
            if len(g["members"]) >= 2:
                # representative root for this group
                rt = g["members"][0]
                while parents[rt] != rt: rt = parents[rt]
                # Use the meta-merge-given name/desc if not already chosen, or
                # if this group is larger
                prev = new_name_desc.get(rt)
                if prev is None or len(g["members"]) > prev["n"]:
                    new_name_desc[rt] = {"name": g["name"],
                                         "description": g["description"],
                                         "n": len(g["members"])}

    base_r1 = json.loads((Path("outputs/analyses/structural_metrics")
                          / args.base_r1_dir
                          / f"r1_families_{args.task}.json").read_text())
    base_fams = base_r1["families"]
    by_fi = {m["fi"]: m for m in fam_meta}

    # Construct merged families
    merged = {}
    for fi, m in by_fi.items():
        root = fi
        while parents[root] != root: root = parents[root]
        if root not in merged:
            base_f = base_fams[root]
            chosen = new_name_desc.get(root)
            merged[root] = {
                "name": chosen["name"] if chosen else base_f.get("name", ""),
                "description": (chosen["description"] if chosen
                                else base_f.get("description", "")),
                "cluster_ids": [],
                "source_family_ids": [],
            }
        merged[root]["cluster_ids"].extend(m["cluster_ids"])
        merged[root]["source_family_ids"].append(fi)

    out_fams = list(merged.values())

    out_path = out_dir / f"r1_families_{args.task}.json"
    out_path.write_text(json.dumps({
        "task": args.task,
        "method": "meta_merge",
        "base": args.base_r1_dir,
        "n_groups_emitted": n_groups,
        "n_merge_edges": n_merge_edges,
        "n_base_families": len(base_fams),
        "n_merged_families": len(out_fams),
        "families": out_fams,
    }, indent=1))

    sizes = [len(f["cluster_ids"]) for f in out_fams]
    print(f"base: {len(base_fams)} families")
    print(f"after meta-merge: {len(out_fams)} families "
          f"(merged {len(base_fams) - len(out_fams)} families away)")
    print(f"size dist: max={max(sizes)}, mean={sum(sizes)/len(sizes):.1f}, "
          f"singletons={sum(1 for s in sizes if s == 1)}, "
          f"multi={sum(1 for s in sizes if s >= 2)}, "
          f">=30={sum(1 for s in sizes if s >= 30)}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
