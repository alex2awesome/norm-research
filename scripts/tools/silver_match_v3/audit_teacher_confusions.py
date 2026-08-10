#!/usr/bin/env python3
"""Audit exact high-volume teacher agreement against independent human labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .common import read_jsonl, sha256_file


def safe_rate(n: int, d: int) -> float | None:
    return n / d if d else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--human-panel", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path, proposal_path = Path(args.manifest), Path(args.proposals)
    human_paths = [Path(path) for path in args.human_panel]
    manifest = json.loads(manifest_path.read_text())
    proposals = list(read_jsonl(proposal_path))
    humans = [row for path in human_paths for row in read_jsonl(path)]
    tasks = sorted({row["task"] for row in humans})
    report = {"tasks": {}}
    for task in tasks:
        proposal_by_uid = {
            str(row["norm_uid"]): row
            for row in proposals
            if row.get("task") == task and row.get("decision") == "MATCH"
        }
        bank_path = Path(manifest["banks"][task]["path"])
        bank = json.loads(bank_path.read_text())["metrics"]
        names = {str(row["metric_id"]): str(row["name"]) for row in bank}
        task_report = {"proposal_matches": len(proposal_by_uid), "splits": {}}
        all_pairs: Counter[tuple[str, str]] = Counter()
        for split in ("train", "dev"):
            rows = [
                row
                for row in humans
                if row.get("task") == task
                and row.get("split") == split
                and row.get("decision") == "MATCH"
            ]
            overlap = [(row, proposal_by_uid[str(row["norm_uid"])]) for row in rows if str(row["norm_uid"]) in proposal_by_uid]
            agreements = sum(a.get("metric_id") == b.get("metric_id") for a, b in overlap)
            high = [(a, b) for a, b in overlap if b.get("confidence") == "high"]
            high_agreements = sum(a.get("metric_id") == b.get("metric_id") for a, b in high)
            pairs = Counter(
                (str(a["metric_id"]), str(b["metric_id"]))
                for a, b in overlap
                if a.get("metric_id") != b.get("metric_id")
            )
            all_pairs.update(pairs)
            task_report["splits"][split] = {
                "human_matches": len(rows),
                "proposal_overlap": len(overlap),
                "exact_agreement": agreements,
                "exact_agreement_rate": safe_rate(agreements, len(overlap)),
                "high_confidence_overlap": len(high),
                "high_confidence_exact_agreement": high_agreements,
                "high_confidence_exact_agreement_rate": safe_rate(high_agreements, len(high)),
            }
        task_report["top_confusions"] = [
            {
                "human_metric_id": human_id,
                "human_metric_name": names.get(human_id),
                "proposal_metric_id": proposal_id,
                "proposal_metric_name": names.get(proposal_id),
                "count": count,
            }
            for (human_id, proposal_id), count in all_pairs.most_common(50)
        ]
        report["tasks"][task] = task_report
    report["input_hashes"] = {
        "manifest": sha256_file(manifest_path),
        "proposals": sha256_file(proposal_path),
        "human_panels": {str(path): sha256_file(path) for path in human_paths},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
