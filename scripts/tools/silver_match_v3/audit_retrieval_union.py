#!/usr/bin/env python3
"""Report paired top-k capture and unions for already frozen retrievers.

The audit never chooses a model or fusion weight.  Every system is evaluated
with a dev-selected fusion report, and every labeled item must be present for
every system so union gains cannot be inflated by unpaired coverage.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import read_jsonl, sha256_file, write_jsonl
from .materialize_frozen_fusion import rerank_candidates
from .optimize_retrieval_fusion import COMPONENTS


def keyed(values: Sequence[str], flag: str) -> dict[str, list[Path]]:
    output: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        if "=" not in value:
            raise ValueError(f"{flag} must use NAME=PATH")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"invalid {flag}: {value}")
        output[name].append(Path(path).resolve())
    return dict(output)


def load_frozen_fusion(path: Path) -> tuple[dict[str, float], float, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("selection_split") != "dev":
        raise ValueError(f"fusion was not selected on dev: {path}")
    raw = (report.get("selected") or {}).get("component_weights")
    if not isinstance(raw, dict) or set(raw) != set(COMPONENTS):
        raise ValueError(f"invalid fusion weights: {path}")
    weights = {key: float(raw[key]) for key in COMPONENTS}
    constant = float(report.get("rank_constant", 0))
    if constant <= 0:
        raise ValueError(f"invalid rank constant: {path}")
    return weights, constant, report


def validate_candidate_provenance(
    candidate_path: Path, fusion_path: Path, fusion: Mapping[str, Any]
) -> str:
    actual = sha256_file(candidate_path)
    expected = (fusion.get("candidate_inputs") or {}).get(str(candidate_path))
    if expected is not None:
        if expected != actual:
            raise ValueError(f"candidate hash mismatch: {candidate_path}")
        return "direct_dev_fusion_input"
    meta_path = candidate_path.with_suffix(candidate_path.suffix + ".meta.json")
    if not meta_path.exists():
        raise ValueError(f"unlinked candidate lacks metadata: {candidate_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("fusion_weights_sha256") != sha256_file(fusion_path):
        raise ValueError(f"candidate was not materialized with frozen fusion: {candidate_path}")
    if meta.get("output_sha256") != actual:
        raise ValueError(f"candidate metadata output hash mismatch: {candidate_path}")
    return "postselection_frozen_fusion_materialization"


def load_unique_candidates(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            uid = str(row["norm_uid"])
            if uid in output:
                raise ValueError(f"duplicate candidate UID: {uid}")
            output[uid] = row
    return output


def summarize_capture(
    labels: Sequence[Mapping[str, Any]], captures: Mapping[str, set[str]]
) -> dict[str, Any]:
    names = sorted(captures)
    all_uids = {str(row["norm_uid"]) for row in labels}
    report: dict[str, Any] = {
        "n_match": len(labels),
        "systems": {
            name: {
                "captured": len(captures[name]),
                "recall": len(captures[name]) / len(labels) if labels else None,
            }
            for name in names
        },
        "unique_rescues": {
            name: len(captures[name] - set().union(*(captures[x] for x in names if x != name)))
            for name in names
        },
        "unions": {},
    }
    for size in range(2, len(names) + 1):
        for group in itertools.combinations(names, size):
            captured = set().union(*(captures[name] for name in group))
            report["unions"]["+".join(group)] = {
                "captured": len(captured),
                "recall": len(captured) / len(labels) if labels else None,
            }
    union_all = set().union(*(captures[name] for name in names))
    report["missed_by_all"] = len(all_uids - union_all)
    return report


def audit(
    labels: Sequence[Mapping[str, Any]],
    systems: Mapping[str, Mapping[str, dict[str, Any]]],
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    captures: dict[str, set[str]] = {name: set() for name in systems}
    items = []
    for label in labels:
        uid, gold = str(label["norm_uid"]), str(label["metric_id"])
        hit = {}
        ranks = {}
        for name, system in systems.items():
            row = system["candidates"].get(uid)
            if row is None:
                raise KeyError(f"paired item absent from {name}: {uid}")
            ordered = rerank_candidates(
                row.get("candidates") or [],
                system["weights"],
                float(system["rank_constant"]),
                top_k,
            )
            ids = [str(value["metric_id"]) for value in ordered]
            found = gold in ids
            hit[name] = found
            ranks[name] = ids.index(gold) + 1 if found else None
            if found:
                captures[name].add(uid)
        items.append(
            {
                "norm_uid": uid,
                "corpus": label.get("corpus"),
                "metric_id": gold,
                "captured": hit,
                "rank_within_top_k": ranks,
            }
        )
    by_corpus = {}
    for corpus in sorted({str(row.get("corpus") or "") for row in labels}):
        subset = [row for row in labels if str(row.get("corpus") or "") == corpus]
        subset_uids = {str(row["norm_uid"]) for row in subset}
        by_corpus[corpus] = summarize_capture(
            subset,
            {name: values & subset_uids for name, values in captures.items()},
        )
    return {"all": summarize_capture(labels, captures), "by_corpus": by_corpus}, items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--system-candidate", action="append", default=[], required=True)
    parser.add_argument("--system-fusion", action="append", default=[], required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", required=True)
    parser.add_argument("--items-output")
    args = parser.parse_args()

    candidate_paths = keyed(args.system_candidate, "--system-candidate")
    fusion_values = keyed(args.system_fusion, "--system-fusion")
    if set(candidate_paths) != set(fusion_values) or any(
        len(paths) != 1 for paths in fusion_values.values()
    ):
        raise ValueError("every system needs candidates and exactly one fusion report")
    labels_path = Path(args.labels).resolve()
    labels = [
        row
        for row in read_jsonl(labels_path)
        if row.get("split") == args.split and row.get("decision") == "MATCH"
    ]
    if not labels:
        raise ValueError("no MATCH labels for requested split")
    systems = {}
    provenance = {}
    for name in sorted(candidate_paths):
        fusion_path = fusion_values[name][0]
        weights, constant, fusion = load_frozen_fusion(fusion_path)
        roles = {
            str(path): validate_candidate_provenance(path, fusion_path, fusion)
            for path in candidate_paths[name]
        }
        systems[name] = {
            "weights": weights,
            "rank_constant": constant,
            "candidates": load_unique_candidates(candidate_paths[name]),
        }
        provenance[name] = {
            "candidate_inputs": {
                str(path): sha256_file(path) for path in candidate_paths[name]
            },
            "candidate_roles": roles,
            "fusion_report": str(fusion_path),
            "fusion_report_sha256": sha256_file(fusion_path),
            "component_weights": weights,
            "rank_constant": constant,
        }
    metrics, items = audit(labels, systems, args.top_k)
    report = {
        "role": "paired_capture_reporting_only_no_selection",
        "selection_performed": False,
        "split": args.split,
        "top_k": args.top_k,
        "labels": str(labels_path),
        "labels_sha256": sha256_file(labels_path),
        "systems": provenance,
        "metrics": metrics,
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.items_output:
        items_output = Path(args.items_output).resolve()
        if items_output.exists():
            raise FileExistsError(f"refusing to overwrite item audit: {items_output}")
        write_jsonl(items_output, items)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
