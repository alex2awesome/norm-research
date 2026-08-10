#!/usr/bin/env python3
"""Freeze identity-only capacity and deterministic seeds for fresh task panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file
from .make_calibration import split_group_for


def _resolve(path: str, anchor: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (anchor.parent / value).resolve()


def _seed(digest: str, namespace: str) -> int:
    return int(hashlib.sha256(f"{digest}\x1f{namespace}".encode()).hexdigest()[:8], 16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--upstream-role-reference", required=True)
    parser.add_argument("--upstream-role-field", default="split")
    parser.add_argument("--exclusion-union", required=True)
    parser.add_argument("--required-upstream-split", default="train")
    parser.add_argument("--select-count", type=int, required=True)
    parser.add_argument("--select-min-per-corpus", type=int, required=True)
    parser.add_argument("--distill-count", type=int, required=True)
    parser.add_argument("--distill-min-per-corpus", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if min(
        args.select_count,
        args.distill_count,
        args.select_min_per_corpus,
        args.distill_min_per_corpus,
    ) < 1:
        parser.error("panel counts and per-corpus minima must be positive")

    manifest_path = Path(args.manifest).resolve()
    roles_path = Path(args.upstream_role_reference).resolve()
    exclusions_path = Path(args.exclusion_union).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    norms: dict[str, dict[str, Any]] = {}
    task_corpora: set[str] = set()
    for corpus, meta in sorted((manifest.get("corpora") or {}).items()):
        if meta.get("task") != args.task:
            continue
        task_corpora.add(str(corpus))
        for row in read_jsonl(_resolve(str(meta["path"]), manifest_path)):
            uid = str(row.get("norm_uid") or "")
            if not uid or uid in norms:
                raise ValueError(f"missing or duplicate canonical UID: {uid!r}")
            if row.get("task") != args.task or row.get("corpus") != corpus:
                raise ValueError(f"canonical task/corpus mismatch: {uid}")
            norms[uid] = row
    if not norms:
        raise ValueError(f"manifest has no canonical norms for {args.task}")

    role_by_uid: dict[str, str] = {}
    role_by_group: dict[str, str] = {}
    role_counts: Counter[str] = Counter()
    for row in read_jsonl(roles_path):
        uid = str(row.get("norm_uid") or "")
        if uid not in norms or uid in role_by_uid:
            raise ValueError(f"unknown or duplicate role UID: {uid!r}")
        role = str(row.get(args.upstream_role_field) or "")
        group = split_group_for(norms[uid])
        supplied = str(row.get("source_group") or "")
        if supplied and supplied != group:
            raise ValueError(f"role source-group mismatch: {uid}")
        prior = role_by_group.setdefault(group, role)
        if prior != role:
            raise ValueError(f"upstream role splits source group: {group}")
        role_by_uid[uid] = role
        role_counts[role] += 1
    if set(role_by_uid) != set(norms):
        raise ValueError("upstream role reference is not exactly canonical-task complete")

    excluded_uids: set[str] = set()
    excluded_groups: set[str] = set()
    for row in read_jsonl(exclusions_path):
        uid = str(row.get("norm_uid") or "")
        if uid not in norms or uid in excluded_uids:
            raise ValueError(f"unknown or duplicate exclusion UID: {uid!r}")
        group = split_group_for(norms[uid])
        supplied = str(row.get("source_group") or "")
        if supplied and supplied != group:
            raise ValueError(f"exclusion source-group mismatch: {uid}")
        excluded_uids.add(uid)
        excluded_groups.add(group)

    by_group: dict[str, list[str]] = defaultdict(list)
    group_corpus: dict[str, str] = {}
    for uid, row in norms.items():
        group = split_group_for(row)
        if role_by_uid[uid] != args.required_upstream_split:
            continue
        if uid in excluded_uids or group in excluded_groups:
            continue
        corpus = str(row["corpus"])
        prior_corpus = group_corpus.setdefault(group, corpus)
        if prior_corpus != corpus:
            raise ValueError(f"source group spans corpora: {group}")
        by_group[group].append(uid)

    groups_by_corpus: dict[str, list[str]] = {corpus: [] for corpus in task_corpora}
    uids_by_corpus: Counter[str] = Counter()
    for group, uids in by_group.items():
        corpus = group_corpus[group]
        groups_by_corpus[corpus].append(group)
        uids_by_corpus[corpus] += len(uids)

    digest = hashlib.sha256()
    for group in sorted(by_group):
        corpus = group_corpus[group]
        for uid in sorted(by_group[group]):
            digest.update(f"{corpus}\t{group}\t{uid}\n".encode())
    capacity_digest = digest.hexdigest()
    group_counts = {key: len(value) for key, value in sorted(groups_by_corpus.items())}
    total_groups = sum(group_counts.values())
    task_corpus_count = len(task_corpora)
    select_feasible = (
        total_groups >= args.select_count
        and args.select_count >= task_corpus_count * args.select_min_per_corpus
        and all(value >= args.select_min_per_corpus for value in group_counts.values())
    )
    distill_feasible = (
        total_groups >= args.distill_count
        and args.distill_count >= task_corpus_count * args.distill_min_per_corpus
        and all(value >= args.distill_min_per_corpus for value in group_counts.values())
    )
    joint_feasible = (
        total_groups >= args.select_count + args.distill_count
        and all(
            value >= args.select_min_per_corpus + args.distill_min_per_corpus
            for value in group_counts.values()
        )
    )
    passed = select_feasible and distill_feasible and joint_feasible
    report = {
        "schema_version": "silver-match-v3-fresh-panel-capacity-freeze-v1",
        "status": "PASS_FROZEN_BEFORE_PANEL_SELECTION" if passed else "FAIL_INSUFFICIENT_CAPACITY",
        "task": args.task,
        "required_upstream_split": args.required_upstream_split,
        "canonical_uids": len(norms),
        "task_corpora": sorted(task_corpora),
        "upstream_role_counts": dict(sorted(role_counts.items())),
        "exclusion_union": {
            "uids": len(excluded_uids),
            "source_groups": len(excluded_groups),
        },
        "eligible_capacity": {
            "uids": sum(uids_by_corpus.values()),
            "source_groups": total_groups,
            "uids_by_corpus": dict(sorted(uids_by_corpus.items())),
            "source_groups_by_corpus": group_counts,
            "identity_capacity_sha256": capacity_digest,
        },
        "predeclared_panels": {
            "select": {
                "count": args.select_count,
                "min_per_corpus": args.select_min_per_corpus,
                "feasible_before_selection": select_feasible,
                "selection_seed": _seed(capacity_digest, "select"),
            },
            "distill": {
                "count": args.distill_count,
                "min_per_corpus": args.distill_min_per_corpus,
                "feasible_before_selection": distill_feasible,
                "selection_seed": _seed(capacity_digest, "distill"),
            },
            "joint_source_group_disjointness_feasible": joint_feasible,
        },
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "upstream_role_reference": {
                "path": str(roles_path),
                "sha256": sha256_file(roles_path),
                "field": args.upstream_role_field,
                "canonical_task_complete": True,
            },
            "exclusion_union": {
                "path": str(exclusions_path),
                "sha256": sha256_file(exclusions_path),
            },
        },
        "content_contract": {
            "fields_used": ["norm_uid", "task", "corpus", "source_group", args.upstream_role_field],
            "truth_labels_read": False,
            "metric_ids_read": False,
            "model_predictions_read": False,
            "mi_or_outcomes_read": False,
            "seed_search_or_performance_tuning_used": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "output_sha256": sha256_file(output)}, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
