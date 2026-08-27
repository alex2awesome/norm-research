"""Aggregate R2 batch responses into per-task r2_aspects_<task>.json.

Note: each batch sees only ~400 R1 families. The same concept can appear as
different aspects across different batches (cross-batch fragmentation, just
like R1 had). This script just concatenates; cross-batch R2 merging is a
separate problem (R2.5 / Fork3-for-R2).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse(p):
    raw = p.read_text().strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    if m: raw = m.group(1).strip()
    try: return json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        return json.loads(raw[s:e + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--work-dir", default="/tmp/r2_subagent")
    ap.add_argument("--out-dir",
                    default="outputs/analyses/structural_metrics/r2_v1_subagent")
    args = ap.parse_args()

    work = Path(args.work_dir) / args.task
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    fam_meta = json.loads((work / "family_meta.json").read_text())
    meta_by_fi = {m["fi"]: m for m in fam_meta}

    all_aspects = []
    seen_fis = set()
    duplicate_fis = set()
    n_unknown = 0
    for rp in sorted((work / "responses").glob("batch_*.json"),
                     key=lambda p: int(p.stem.split("_")[1])):
        try:
            obj = parse(rp)
        except Exception as e:
            print(f"parse fail {rp.name}: {e}")
            continue
        bi = int(rp.stem.split("_")[1])
        for j, a in enumerate(obj.get("aspects", [])):
            mems = []
            for m in a.get("members", []):
                s = str(m).strip().lstrip("F")
                try: fi = int(s)
                except ValueError: n_unknown += 1; continue
                if fi in seen_fis: duplicate_fis.add(fi); continue
                seen_fis.add(fi)
                mems.append(fi)
            all_aspects.append({
                "aspect_id": f"b{bi}_a{j}",
                "name": a.get("name", ""),
                "description": a.get("description", ""),
                "n_families": len(mems),
                "family_ids": mems,
                "source_batch": bi,
            })

    missing_fis = sorted(set(meta_by_fi.keys()) - seen_fis)
    sizes = [a["n_families"] for a in all_aspects]
    print(f"task: {args.task}")
    print(f"  R1 families expected: {len(fam_meta)}")
    print(f"  R1 families covered:  {len(seen_fis)}  missing: {len(missing_fis)}")
    print(f"  duplicate family ids: {len(duplicate_fis)}")
    print(f"  R2 aspects (all batches): {len(all_aspects)}")
    print(f"  compression: {len(seen_fis)/max(len(all_aspects),1):.2f}x")
    print(f"  size dist: max={max(sizes)}, mean={sum(sizes)/len(sizes):.1f}, "
          f"singletons={sum(1 for s in sizes if s == 1)}, "
          f">=20={sum(1 for s in sizes if s >= 20)}")
    out_path = Path(args.out_dir) / f"r2_aspects_{args.task}.json"
    out_path.write_text(json.dumps({
        "task": args.task,
        "n_r1_families": len(fam_meta),
        "n_r2_aspects": len(all_aspects),
        "compression": len(seen_fis)/max(len(all_aspects),1),
        "n_missing_r1_fams": len(missing_fis),
        "n_duplicate_r1_fams": len(duplicate_fis),
        "missing_sample": missing_fis[:20],
        "aspects": all_aspects,
    }, indent=1))


if __name__ == "__main__":
    main()
