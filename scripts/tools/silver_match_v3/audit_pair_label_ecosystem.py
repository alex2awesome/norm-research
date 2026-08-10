#!/usr/bin/env python3
"""Audit whether rubric-pair verdicts are safe retriever supervision.

The v6 verdict file labels relationships between *bank-source rubrics*.  This
audit quantifies how much survives an unambiguous mapping into the current
metric bank and whether train-discovered ``related but different`` edges have
enough held-out support to be used as explicit hard negatives.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .build_relations import load_hierarchy_task
from .common import read_jsonl, sha256_file
from .config import DEFAULT_HIERARCHY_ROOT


REQUIRED_FIELDS = {
    "task",
    "split",
    "key_a",
    "key_b",
    "canonical_a",
    "canonical_b",
    "score",
}
PROVENANCE_FIELDS_NEEDED_FOR_NORM_METRIC_TEACHERS = {
    "norm_uid",
    "norm",
    "context",
    "metric_id",
    "bank_source_sha256",
    "judge_model",
    "judge_prompt_sha256",
}


def heldout_related_graph(
    aggregates: Mapping[str, Mapping[tuple[int, int], Counter[int]]],
    min_train_related: int,
) -> dict[str, Any]:
    selected = {
        edge
        for edge, counts in aggregates.get("train", {}).items()
        if counts[1] >= min_train_related
        and counts[0] == 0
        and counts[2] == 0
    }
    eval_counts: Counter[int] = Counter()
    eval_edges = 0
    for edge in selected:
        counts = aggregates.get("eval", {}).get(edge)
        if counts:
            eval_edges += 1
            eval_counts.update(counts)
    return {
        "selection_rule": (
            f"unique-anchor train metric-pair has >={min_train_related} RELATED labels "
            "and zero SAME/UNRELATED labels"
        ),
        "selected_train_edges": len(selected),
        "selected_metric_pairs": [
            [f"a{left}", f"a{right}"] for left, right in sorted(selected)
        ],
        "edges_with_any_eval_label": eval_edges,
        "eval_label_counts": {
            "unrelated_0": eval_counts[0],
            "related_1": eval_counts[1],
            "same_2": eval_counts[2],
        },
    }


def audit(
    pair_path: Path,
    hierarchy_root: Path,
    tasks: list[str],
    min_train_related: int,
) -> dict[str, Any]:
    hierarchies = {
        task: load_hierarchy_task(task, hierarchy_root) for task in tasks
    }
    observed_fields: set[str] = set()
    task_stats = {task: Counter() for task in tasks}
    graph = {
        task: {
            "train": defaultdict(Counter),
            "eval": defaultdict(Counter),
        }
        for task in tasks
    }
    total = 0
    for row in read_jsonl(pair_path):
        total += 1
        observed_fields.update(row)
        task = str(row.get("task") or "")
        if task not in hierarchies:
            continue
        stats = task_stats[task]
        stats["rows_seen"] += 1
        split, score = row.get("split"), row.get("score")
        stats[f"split_{split}"] += 1
        if score not in (0, 1, 2):
            stats["invalid_score"] += 1
            continue
        stats[f"score_{score}"] += 1
        memberships = hierarchies[task]["key_memberships"]
        left = memberships.get(str(row.get("key_a") or ""), set())
        right = memberships.get(str(row.get("key_b") or ""), set())
        if not left or not right:
            stats["unmapped_to_current_bank"] += 1
            continue
        if len(left) != 1 or len(right) != 1:
            stats["ambiguous_anchor_membership"] += 1
            continue
        a, b = next(iter(left)), next(iter(right))
        if a == b:
            stats["same_current_metric"] += 1
            continue
        stats["unique_anchor_cross_metric"] += 1
        if split in ("train", "eval"):
            graph[task][split][tuple(sorted((a, b)))][int(score)] += 1

    per_task = {}
    for task in tasks:
        relation = heldout_related_graph(graph[task], min_train_related)
        stats = dict(sorted(task_stats[task].items()))
        seen = stats.get("rows_seen", 0)
        per_task[task] = {
            "row_audit": stats,
            "unmapped_rate": (
                stats.get("unmapped_to_current_bank", 0) / seen if seen else None
            ),
            "multiply_assigned_current_leaf_keys": hierarchies[task][
                "multiply_assigned_leaf_keys"
            ],
            "heldout_related_graph": relation,
        }
    missing_provenance = sorted(
        PROVENANCE_FIELDS_NEEDED_FOR_NORM_METRIC_TEACHERS - observed_fields
    )
    return {
        "schema_version": "silver-match-v3-pair-ecosystem-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pair_labels": str(pair_path),
        "pair_labels_sha256": sha256_file(pair_path),
        "total_rows": total,
        "observed_fields": sorted(observed_fields),
        "required_pair_fields_present": sorted(REQUIRED_FIELDS & observed_fields),
        "missing_norm_metric_teacher_provenance": missing_provenance,
        "tasks": per_task,
        "adjudication": {
            "norm_to_metric_gradient_supervision": "REJECT",
            "explicit_hard_negative_graph": "REJECT",
            "safe_uses": [
                "conservative current-bank equivalence audit",
                "broad family diagnostics without exact-ID credit",
            ],
            "reasons": [
                "Rows label rubric-to-rubric relationships, not extracted norm-to-metric relevance.",
                "The artifact lacks norm UID/text/context, current bank hash, and judge model/prompt provenance.",
                "Most task rows do not map into the current bank; some source keys map to multiple metrics.",
                "Train-derived RELATED edges have extremely sparse held-out coverage, and peer review receives held-out SAME labels on nominally different edges.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-labels",
        default="/lfs/skampere3/0/alexspan/norm_embed/all_verdicts.jsonl",
    )
    parser.add_argument("--hierarchy-root", default=str(DEFAULT_HIERARCHY_ROOT))
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["peer-review", "legal-outcome-prediction", "notice-and-comment"],
    )
    parser.add_argument("--min-train-related", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite pair-label audit: {output}")
    report = audit(
        Path(args.pair_labels).resolve(),
        Path(args.hierarchy_root).resolve(),
        list(args.tasks),
        args.min_train_related,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
