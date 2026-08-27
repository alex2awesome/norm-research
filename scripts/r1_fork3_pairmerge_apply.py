"""Apply Fork 3 pairwise merge verdicts via union-find.

Reads:
  /tmp/r1_fork3/<task>/batches.jsonl           (per-batch pair record: fa, fb, cos)
  /tmp/r1_fork3/<task>/responses/batch_*.json  (verdicts list)
  /tmp/r1_fork3/<task>/family_meta.json
  outputs/analyses/structural_metrics/<base>/r1_families_<task>.json

Writes:
  outputs/analyses/structural_metrics/<out_dir>/r1_families_<task>.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_verdicts(p: Path) -> list[dict]:
    raw = p.read_text().strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    if m: raw = m.group(1).strip()
    try: obj = json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        obj = json.loads(raw[s:e + 1])
    return obj.get("verdicts", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer-review")
    ap.add_argument("--base-r1-dir", default="r1_v4a_subagent_lora_bs400")
    ap.add_argument("--work-dir", default="/tmp/r1_fork3")
    ap.add_argument("--out-dir", default="r1_v4a_lora_fork3_merge")
    args = ap.parse_args()

    work = Path(args.work_dir) / args.task
    out_dir = Path("outputs/analyses/structural_metrics") / args.out_dir
    out_dir.mkdir(exist_ok=True, parents=True)

    base_r1 = json.loads((Path("outputs/analyses/structural_metrics")
                          / args.base_r1_dir
                          / f"r1_families_{args.task}.json").read_text())
    base_fams = base_r1["families"]
    n = len(base_fams)
    parents = list(range(n))
    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parents[ra] = rb

    # Load batches.jsonl into ordered pair list (matches subagent prompts)
    batch_pairs = []  # list of lists; each inner = pairs in that batch in order
    for line in (work / "batches.jsonl").open():
        rec = json.loads(line)
        batch_pairs.append((rec["batch_idx"], rec["pairs"]))

    # Aggregate verdicts (ordered desc by cos because that's how we batched)
    yes_edges = []
    no_edges = []
    for bi, pairs in batch_pairs:
        rp = work / "responses" / f"batch_{bi}.json"
        if not rp.exists():
            print(f"missing batch_{bi}.json")
            continue
        verdicts = parse_verdicts(rp)
        v_by_idx = {v.get("pair_idx"): v for v in verdicts}
        for k, p in enumerate(pairs, start=1):
            v = v_by_idx.get(k)
            if v is None: continue
            decision = str(v.get("merge", "")).strip().upper()
            if decision == "YES":
                yes_edges.append((p["fa"], p["fb"], p["cos"]))
            else:
                no_edges.append((p["fa"], p["fb"], p["cos"]))

    print(f"YES: {len(yes_edges)}   NO: {len(no_edges)}   "
          f"total: {len(yes_edges) + len(no_edges)}")

    # Apply YES edges (cos desc -- already in that order)
    for a, b, c in yes_edges:
        union(a, b)

    # Aggregate families by root
    merged = {}
    for fi, f in enumerate(base_fams):
        r = find(fi)
        if r not in merged:
            merged[r] = {
                "name": base_fams[r].get("name", ""),
                "description": base_fams[r].get("description", ""),
                "cluster_ids": [],
                "source_family_ids": [],
            }
        cids = [int(str(c).lstrip("C")) for c in
                (f.get("cluster_ids") or f.get("members") or [])
                if str(c).lstrip("C").isdigit()]
        merged[r]["cluster_ids"].extend(cids)
        merged[r]["source_family_ids"].append(fi)

    out_fams = list(merged.values())
    sizes = [len(f["cluster_ids"]) for f in out_fams]
    print(f"base: {n} families")
    print(f"after pairwise merge: {len(out_fams)} families "
          f"(merged {n - len(out_fams)} families away)")
    print(f"size dist: max={max(sizes)}, mean={sum(sizes)/len(sizes):.1f}, "
          f"singletons={sum(1 for s in sizes if s == 1)}, "
          f"multi={sum(1 for s in sizes if s >= 2)}, "
          f">=30={sum(1 for s in sizes if s >= 30)}, "
          f">=50={sum(1 for s in sizes if s >= 50)}")

    out_path = out_dir / f"r1_families_{args.task}.json"
    out_path.write_text(json.dumps({
        "task": args.task,
        "method": "fork3_pairwise_merge",
        "base": args.base_r1_dir,
        "n_yes_edges": len(yes_edges),
        "n_no_edges": len(no_edges),
        "n_base_families": n,
        "n_merged_families": len(out_fams),
        "families": out_fams,
    }, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
