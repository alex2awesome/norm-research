#!/usr/bin/env python3
"""Partition any frozen task identity panel before labels or predictions exist."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl


def _key(seed: int, namespace: str, row: dict[str, object]) -> tuple[str, str]:
    group = str(row["source_group"])
    value = hashlib.sha256(f"{seed}\x1f{namespace}\x1f{group}".encode()).hexdigest()
    return value, group


def _largest_remainder(sizes: dict[str, int], total: int) -> dict[str, int]:
    population = sum(sizes.values())
    if total < 0 or total > population:
        raise ValueError("partition total is outside panel capacity")
    exact = {key: total * value / population for key, value in sizes.items()}
    result = {key: int(value) for key, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(sizes, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--panel-freeze", required=True)
    parser.add_argument("--identities", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--train-count", required=True, type=int)
    parser.add_argument("--optimize-count", required=True, type=int)
    parser.add_argument("--exclude-panel", action="append", default=[])
    parser.add_argument("--stratify-by-corpus", action="store_true")
    args = parser.parse_args()

    freeze_path = Path(args.panel_freeze).resolve()
    identities_path = Path(args.identities).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    rows = list(read_jsonl(identities_path))
    if (
        freeze.get("schema_version") != "silver-match-v3-clean-gepa-panel-freeze-v1"
        or freeze.get("status") != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or freeze.get("task") != args.task
        or freeze.get("required_upstream_split") != "train"
        or freeze.get("outputs", {}).get("identities", {}).get("sha256")
        != sha256_file(identities_path)
        or len(rows) != int(freeze.get("selected_count", -1))
    ):
        raise ValueError("identity panel is not a linked upstream-train freeze for this task")
    if args.train_count + args.optimize_count != len(rows):
        raise ValueError("partition counts must exhaust the frozen panel")
    uids = [str(row.get("norm_uid") or "") for row in rows]
    groups = [str(row.get("source_group") or "") for row in rows]
    if "" in uids or "" in groups or len(uids) != len(set(uids)) or len(groups) != len(set(groups)):
        raise ValueError("frozen identities are empty or duplicated")

    excluded_uids: set[str] = set()
    excluded_groups: set[str] = set()
    exclusion_refs: list[dict[str, object]] = []
    for raw in args.exclude_panel:
        path = Path(raw).resolve()
        excluded = list(read_jsonl(path))
        excluded_uids.update(str(row["norm_uid"]) for row in excluded)
        excluded_groups.update(str(row["source_group"]) for row in excluded)
        exclusion_refs.append(
            {"path": str(path), "sha256": sha256_file(path), "count": len(excluded)}
        )
    if set(uids) & excluded_uids or set(groups) & excluded_groups:
        raise ValueError("distillation identities overlap an excluded panel")

    if args.stratify_by_corpus:
        by_corpus: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            by_corpus[str(row["corpus"])].append(row)
        sizes = {key: len(value) for key, value in sorted(by_corpus.items())}
        train_quota = _largest_remainder(sizes, args.train_count)
        train_uids: set[str] = set()
        for corpus, values in sorted(by_corpus.items()):
            ordered = sorted(values, key=lambda row: _key(args.seed, corpus, row))
            train_uids.update(str(row["norm_uid"]) for row in ordered[: train_quota[corpus]])
    else:
        ordered = sorted(rows, key=lambda row: _key(args.seed, "global", row))
        train_uids = {str(row["norm_uid"]) for row in ordered[: args.train_count]}

    assignments: list[dict[str, object]] = []
    for row in rows:
        role = "train" if str(row["norm_uid"]) in train_uids else "optimize"
        assignments.append(
            {
                "schema_version": "silver-match-v3-frozen-identity-partition-v1",
                "task": args.task,
                "norm_uid": row["norm_uid"],
                "corpus": row["corpus"],
                "source_group": row["source_group"],
                "upstream_split": row["upstream_split"],
                "remediation_role": role,
                "labels_predictions_metric_ids_reasons_mi_or_outcomes_used": False,
            }
        )
    assignments.sort(key=lambda row: (str(row["remediation_role"]), str(row["norm_uid"])))
    role_counts = Counter(str(row["remediation_role"]) for row in assignments)
    if role_counts != Counter({"train": args.train_count, "optimize": args.optimize_count}):
        raise AssertionError("partition role counts drifted")

    output_root.mkdir(parents=True, exist_ok=False)
    output = output_root / "assignments.jsonl"
    write_jsonl(output, assignments)
    report = {
        "schema_version": "silver-match-v3-frozen-identity-partition-freeze-v1",
        "status": "FROZEN_BEFORE_ANY_DISTILLATION_LABELS_OR_PREDICTIONS",
        "task": args.task,
        "seed": args.seed,
        "stratified_by_corpus": args.stratify_by_corpus,
        "role_counts": dict(sorted(role_counts.items())),
        "role_by_corpus": {
            role: dict(
                sorted(
                    Counter(
                        str(row["corpus"])
                        for row in assignments
                        if row["remediation_role"] == role
                    ).items()
                )
            )
            for role in ("train", "optimize")
        },
        "cross_role_uid_or_source_group_overlap": 0,
        "inputs": {
            "panel_freeze": {"path": str(freeze_path), "sha256": sha256_file(freeze_path)},
            "identities": {"path": str(identities_path), "sha256": sha256_file(identities_path)},
            "exclusions": exclusion_refs,
        },
        "output": {"path": str(output), "sha256": sha256_file(output), "count": len(assignments)},
        "content_contract": {
            "identity_and_source_group_fields_only": True,
            "labels_predictions_metric_ids_reasons_mi_or_outcomes_used": False,
            "seed_search_or_performance_tuning_used": False,
        },
    }
    report_path = output_root / "FREEZE.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "freeze_sha256": sha256_file(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
