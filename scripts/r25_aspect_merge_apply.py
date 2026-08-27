"""Apply R2.5 cross-batch merge verdicts via union-find.

Reads:
  /tmp/r2_5/<task>/batches.jsonl                (per-batch pair records)
  /tmp/r2_5/<task>/responses/batch_*.json       (subagent YES/NO verdicts)
  outputs/analyses/structural_metrics/r2_v1_subagent/r2_aspects_<task>.json

Writes:
  outputs/analyses/structural_metrics/r2_v2_merged/r2_aspects_<task>.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_verdicts(p):
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
    ap.add_argument("--work-dir", default="/tmp/r2_5")
    ap.add_argument("--r2-dir", default="r2_v1_subagent")
    ap.add_argument("--out-dir", default="r2_v2_merged")
    args = ap.parse_args()

    work = Path(args.work_dir) / args.task
    base_r2 = json.loads((Path("outputs/analyses/structural_metrics")
                          / args.r2_dir
                          / f"r2_aspects_{args.task}.json").read_text())
    aspects = base_r2["aspects"]
    n = len(aspects)
    parents = list(range(n))
    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parents[ra] = rb

    yes = no = 0
    for line in (work / "batches.jsonl").open():
        rec = json.loads(line)
        bi = rec["batch_idx"]
        rp = work / "responses" / f"batch_{bi}.json"
        if not rp.exists():
            continue
        try:
            verdicts = parse_verdicts(rp)
        except Exception as e:
            print(f"  parse fail batch_{bi}: {e}")
            continue
        v_by_idx = {v.get("pair_idx"): v for v in verdicts}
        for k, pair in enumerate(rec["pairs"], 1):
            v = v_by_idx.get(k)
            if v is None: continue
            decision = str(v.get("merge", "")).strip().upper()
            if decision == "YES":
                union(pair["ai_a"], pair["ai_b"])
                yes += 1
            else:
                no += 1

    # Group by root
    merged_by_root = {}
    for ai in range(n):
        r = find(ai)
        if r not in merged_by_root:
            merged_by_root[r] = {
                "name": aspects[r]["name"],
                "description": aspects[r]["description"],
                "n_families": 0,
                "family_ids": [],
                "source_aspect_ids": [],
                "source_batches": set(),
            }
        merged_by_root[r]["n_families"] += aspects[ai]["n_families"]
        merged_by_root[r]["family_ids"].extend(aspects[ai]["family_ids"])
        merged_by_root[r]["source_aspect_ids"].append(aspects[ai]["aspect_id"])
        merged_by_root[r]["source_batches"].add(aspects[ai]["source_batch"])

    out_aspects = []
    for r, m in merged_by_root.items():
        m["source_batches"] = sorted(m["source_batches"])
        out_aspects.append(m)

    out_dir = Path("outputs/analyses/structural_metrics") / args.out_dir
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f"r2_aspects_{args.task}.json"
    out_path.write_text(json.dumps({
        "task": args.task,
        "method": "r2_5_pairwise_merge",
        "n_yes": yes,
        "n_no": no,
        "n_base_aspects": n,
        "n_merged_aspects": len(out_aspects),
        "aspects": out_aspects,
    }, indent=1))
    print(f"{args.task}: {n} -> {len(out_aspects)} aspects "
          f"(YES={yes}, NO={no}, merged {n - len(out_aspects)} away)")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
