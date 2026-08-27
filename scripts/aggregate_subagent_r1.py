"""Aggregate per-batch subagent R1 outputs into a single r1_families file.

Reads:
  /tmp/r1_subagent/<task>/responses/batch_*.json
  /tmp/r1_subagent/<task>/batches.jsonl              (for expected cluster_ids)
  /tmp/r1_subagent/<task>/clusters_repr.json         (for rep text per cluster)

Writes:
  outputs/analyses/structural_metrics/r1_v4a_subagent/r1_families_<task>.json
  outputs/analyses/structural_metrics/r1_v4a_subagent/r1_metrics_<task>.json

Validates: every cluster id present in batches.jsonl ends up in exactly one
family. Reports missing / duplicate / out-of-set ids.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def parse_one(path: Path) -> dict:
    """Parse one batch response; salvage if the model wrapped it in prose."""
    raw = path.read_text().strip()
    # Strip markdown fences if present
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    if m:
        raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Take the largest {...} blob
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--work-dir", default="/tmp/r1_subagent")
    ap.add_argument("--output-dir",
                    default="outputs/analyses/structural_metrics/r1_v4a_subagent")
    args = ap.parse_args()

    work = Path(args.work_dir) / args.task
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Expected cluster ids per batch
    expected = {}
    for line in (work / "batches.jsonl").open():
        rec = json.loads(line)
        expected[rec["batch_idx"]] = set(rec["cluster_ids"])
    all_expected = set().union(*expected.values())
    print(f"task={args.task} batches={len(expected)} "
          f"expected_clusters={len(all_expected)}")

    reps = json.loads((work / "clusters_repr.json").read_text())

    # Parse each batch
    all_families = []
    seen_cids: Counter = Counter()
    per_batch_summary = []

    for bi in sorted(expected.keys()):
        rp = work / "responses" / f"batch_{bi}.json"
        if not rp.exists():
            print(f"  MISSING batch_{bi}.json", file=sys.stderr)
            continue
        try:
            obj = parse_one(rp)
        except Exception as e:
            print(f"  PARSE FAIL batch_{bi}: {e}", file=sys.stderr)
            continue
        fams = obj.get("families", [])
        b_cids: set[int] = set()
        for f in fams:
            cids = []
            for m in f.get("members", []):
                # Members are "C1234"; strip prefix
                s = str(m).strip()
                if s.startswith("C"):
                    s = s[1:]
                try:
                    c = int(s)
                except ValueError:
                    continue
                cids.append(c)
                seen_cids[c] += 1
                b_cids.add(c)
            all_families.append({
                "batch_idx": bi,
                "family_id": f"b{bi}_{len(all_families)}",
                "name": f.get("name", ""),
                "description": f.get("description", ""),
                "cluster_ids": cids,
                "n_clusters": len(cids),
            })
        exp = expected[bi]
        missing = exp - b_cids
        extra = b_cids - exp
        per_batch_summary.append({
            "batch_idx": bi,
            "n_families": len(fams),
            "n_cids_seen": len(b_cids),
            "n_cids_expected": len(exp),
            "missing": sorted(missing)[:10],
            "extra": sorted(extra)[:10],
        })

    # Global coverage
    seen_all = set(seen_cids.keys())
    missing_all = all_expected - seen_all
    duplicates = {c: n for c, n in seen_cids.items() if n > 1}
    extra_all = seen_all - all_expected

    print(f"\nCoverage: seen={len(seen_all)}/{len(all_expected)}  "
          f"missing={len(missing_all)}  duplicates={len(duplicates)}  "
          f"out_of_set={len(extra_all)}")
    print(f"Families: total={len(all_families)}  "
          f"max_size={max((f['n_clusters'] for f in all_families), default=0)}")

    sizes = Counter(f["n_clusters"] for f in all_families)
    singletons = sum(1 for f in all_families if f["n_clusters"] == 1)
    dup_in_fam = sum(1 for f in all_families if f["n_clusters"] >= 2)
    big = sum(1 for f in all_families if f["n_clusters"] >= 20)
    huge = sum(1 for f in all_families if f["n_clusters"] >= 30)
    print(f"Singletons: {singletons}  multi: {dup_in_fam}  ≥20: {big}  ≥30: {huge}")

    out_data = {
        "task": args.task,
        "method": "v4a_subagent",
        "n_families": len(all_families),
        "n_clusters_expected": len(all_expected),
        "n_clusters_seen": len(seen_all),
        "coverage": len(seen_all) / max(len(all_expected), 1),
        "n_missing": len(missing_all),
        "n_duplicates": len(duplicates),
        "n_extra": len(extra_all),
        "families": all_families,
    }
    out_path = out_dir / f"r1_families_{args.task}.json"
    out_path.write_text(json.dumps(out_data, indent=1))
    print(f"\nwrote {out_path}")

    metrics = {
        "task": args.task,
        "method": "v4a_subagent",
        "n_clusters": len(all_expected),
        "n_families": len(all_families),
        "max_family_size": max((f['n_clusters'] for f in all_families),
                               default=0),
        "n_families_ge20": big,
        "n_families_ge30": huge,
        "n_singletons": singletons,
        "n_multi": dup_in_fam,
        "coverage": len(seen_all) / max(len(all_expected), 1),
        "n_missing": len(missing_all),
        "n_duplicates": len(duplicates),
        "n_extra": len(extra_all),
        "missing_sample": sorted(missing_all)[:20],
        "duplicate_sample": dict(list(duplicates.items())[:20]),
        "per_batch": per_batch_summary,
    }
    (out_dir / f"r1_metrics_{args.task}.json").write_text(
        json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
