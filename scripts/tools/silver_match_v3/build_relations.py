#!/usr/bin/env python3
"""Build conservative metric-equivalence and R3-family relations.

The current hierarchy exposes two different semantic relations and they must
not be conflated:

* ``r3_expanded.merged_groups`` are proposed SAME-CONSTRUCT merges.
* ``r3_expanded.grandparents`` are SUBSUMPTION families whose children remain
  distinct constructs.

An R3 merge is credited as metric equivalence only when every metric pair in
the group is independently supported by the norm_embed 0/1/2 pair labels.
The defaults are intentionally strict: at least three raw-rubric SAME labels,
at least 90% SAME among all decisive/related labels, and no UNRELATED label.
Groups lacking evidence remain useful as broad family relations but are not
silently promoted to equivalence.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import SCHEMA_VERSION
from .common import normalize_space, read_jsonl, sha256_file, stable_uid
from .config import DEFAULT_HIERARCHY_ROOT, DEFAULT_OUTPUT_ROOT, TASK_TO_HIERARCHY


RELATION_SCHEMA_VERSION = "silver-match-relations-v1"
DEFAULT_PAIR_LABELS = Path("/lfs/skampere3/0/alexspan/norm_embed/all_verdicts.jsonl")


@dataclass(frozen=True)
class Thresholds:
    min_same_pairs: int = 3
    min_same_rate: float = 0.90
    max_unrelated_pairs: int = 0

    def validate(self) -> None:
        if self.min_same_pairs < 1:
            raise ValueError("min_same_pairs must be positive")
        if not 0.0 <= self.min_same_rate <= 1.0:
            raise ValueError("min_same_rate must be in [0, 1]")
        if self.max_unrelated_pairs < 0:
            raise ValueError("max_unrelated_pairs must be nonnegative")


def metric_id(index: int) -> str:
    return f"a{index}"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _valid_indices(values: Iterable[Any], n_metrics: int, source: str) -> tuple[int, ...]:
    indices = tuple(sorted({value for value in values if isinstance(value, int)}))
    bad = [value for value in indices if value < 0 or value >= n_metrics]
    if bad:
        raise ValueError(f"out-of-range R2 metric IDs in {source}: {bad[:5]}")
    return indices


def load_hierarchy_task(
    task: str, hierarchy_root: Path
) -> dict[str, Any]:
    r2_path = hierarchy_root / TASK_TO_HIERARCHY[task]
    r3_path = hierarchy_root / f"{task}_general_r3_expanded.json"
    if not r2_path.exists():
        raise FileNotFoundError(r2_path)
    if not r3_path.exists():
        raise FileNotFoundError(r3_path)
    r2, r3 = _load_json(r2_path), _load_json(r3_path)
    groups = r2.get("merged_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"no merged_groups in {r2_path}")
    n_metrics = len(groups)

    key_memberships: dict[str, set[int]] = defaultdict(set)
    for index, group in enumerate(groups):
        for leaf in group.get("all_leaves") or []:
            if not isinstance(leaf, dict):
                continue
            key = normalize_space(leaf.get("key"))
            if key:
                key_memberships[key].add(index)

    merge_groups = []
    for r3_index, group in enumerate(r3.get("merged_groups") or []):
        indices = _valid_indices(
            group.get("source_r2_cluster_ids") or [],
            n_metrics,
            f"{r3_path}:merged_groups[{r3_index}]",
        )
        if len(indices) < 2:
            continue
        merge_groups.append(
            {
                "r3_index": r3_index,
                "name": normalize_space(group.get("merged_name")),
                "description": normalize_space(group.get("merged_description")),
                "indices": indices,
            }
        )

    grandparents = []
    for r3_index, group in enumerate(r3.get("grandparents") or []):
        indices = _valid_indices(
            (
                child.get("r2_cluster_id")
                for child in group.get("children") or []
                if isinstance(child, dict)
            ),
            n_metrics,
            f"{r3_path}:grandparents[{r3_index}]",
        )
        if len(indices) < 2:
            continue
        grandparents.append(
            {
                "r3_index": r3_index,
                "name": normalize_space(group.get("grandparent_name")),
                "description": normalize_space(group.get("grandparent_description")),
                "indices": indices,
            }
        )

    return {
        "task": task,
        "r2_path": r2_path,
        "r3_path": r3_path,
        "n_metrics": n_metrics,
        "key_memberships": key_memberships,
        "merge_groups": merge_groups,
        "grandparents": grandparents,
        "leaf_key_count": len(key_memberships),
        "multiply_assigned_leaf_keys": sum(
            len(indices) > 1 for indices in key_memberships.values()
        ),
    }


def collect_pair_evidence(
    pair_labels_path: Path,
    tasks: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, dict[tuple[int, int], dict[tuple[str, str], set[int]]]],
    dict[str, dict[str, int]],
]:
    """Map raw-rubric labels to every current R2 membership, without key overwrite.

    A raw leaf can occur in more than one current R2 metric.  Expanding its set
    of memberships preserves that provenance.  Duplicate labels for the same
    raw pair are retained as a score set; downstream summarization takes the
    minimum score, so disagreement can only make equivalence harder to earn.
    """
    evidence: dict[
        str, dict[tuple[int, int], dict[tuple[str, str], set[int]]]
    ] = {task: defaultdict(lambda: defaultdict(set)) for task in tasks}
    stats: dict[str, Counter[str]] = {task: Counter() for task in tasks}

    for row in read_jsonl(pair_labels_path):
        task = normalize_space(row.get("task"))
        if task not in tasks:
            continue
        task_stats = stats[task]
        task_stats["rows_seen"] += 1
        score = row.get("score")
        if score not in (0, 1, 2):
            task_stats["rows_invalid_score"] += 1
            continue
        key_a = normalize_space(row.get("key_a"))
        key_b = normalize_space(row.get("key_b"))
        if not key_a or not key_b or key_a == key_b:
            task_stats["rows_invalid_keys"] += 1
            continue
        memberships = tasks[task]["key_memberships"]
        left, right = memberships.get(key_a, set()), memberships.get(key_b, set())
        if not left or not right:
            task_stats["rows_unmapped"] += 1
            continue
        raw_pair = tuple(sorted((key_a, key_b)))
        added = False
        for a in left:
            for b in right:
                if a == b:
                    continue
                metric_pair = tuple(sorted((a, b)))
                evidence[task][metric_pair][raw_pair].add(int(score))
                added = True
        if added:
            task_stats["rows_cross_metric"] += 1
        else:
            task_stats["rows_same_metric_only"] += 1
    return evidence, {task: dict(counter) for task, counter in stats.items()}


def summarize_evidence(
    raw_pair_scores: dict[tuple[str, str], set[int]] | None,
    thresholds: Thresholds,
) -> dict[str, Any]:
    scores = [min(values) for values in (raw_pair_scores or {}).values() if values]
    counts = Counter(scores)
    total = sum(counts.values())
    same_rate = counts[2] / total if total else 0.0
    qualified = (
        counts[2] >= thresholds.min_same_pairs
        and same_rate >= thresholds.min_same_rate
        and counts[0] <= thresholds.max_unrelated_pairs
    )
    reasons = []
    if counts[2] < thresholds.min_same_pairs:
        reasons.append("insufficient_same_support")
    if same_rate < thresholds.min_same_rate:
        reasons.append("same_rate_below_threshold")
    if counts[0] > thresholds.max_unrelated_pairs:
        reasons.append("unrelated_contradiction")
    return {
        "same": counts[2],
        "related": counts[1],
        "unrelated": counts[0],
        "total_unique_raw_pairs": total,
        "same_rate": same_rate,
        "conflicting_raw_pairs": sum(
            len(values) > 1 for values in (raw_pair_scores or {}).values()
        ),
        "qualified": qualified,
        "reasons": reasons,
    }


def _relation_id(task: str, kind: str, name: str, ids: list[str]) -> str:
    return stable_uid(task, kind, name, *ids)[:20]


def build_task_relations(
    hierarchy: dict[str, Any],
    evidence: dict[tuple[int, int], dict[tuple[str, str], set[int]]],
    pair_stats: dict[str, int],
    thresholds: Thresholds,
) -> dict[str, Any]:
    task = hierarchy["task"]
    accepted_equivalence = []
    merge_audit = []
    occupied: set[int] = set()

    for group in hierarchy["merge_groups"]:
        pair_audits = []
        for pair in itertools.combinations(group["indices"], 2):
            summary = summarize_evidence(evidence.get(tuple(sorted(pair))), thresholds)
            pair_audits.append(
                {
                    "metric_ids": [metric_id(pair[0]), metric_id(pair[1])],
                    **summary,
                }
            )
        accepted = bool(pair_audits) and all(row["qualified"] for row in pair_audits)
        ids = [metric_id(index) for index in group["indices"]]
        audit = {
            "r3_index": group["r3_index"],
            "r3_name": group["name"],
            "metric_ids": ids,
            "accepted_as_equivalent": accepted,
            "pair_evidence": pair_audits,
        }
        merge_audit.append(audit)
        if not accepted:
            continue
        overlap = occupied.intersection(group["indices"])
        if overlap:
            raise ValueError(
                f"accepted R3 equivalence groups overlap for {task}: {sorted(overlap)}"
            )
        occupied.update(group["indices"])
        accepted_equivalence.append(
            {
                "equivalence_id": _relation_id(task, "equivalence", group["name"], ids),
                "metric_ids": ids,
                "r3_index": group["r3_index"],
                "r3_name": group["name"],
                "relation": "same_construct",
                "provenance": "r3_merged_group_and_pair_labels",
                "pair_evidence": pair_audits,
            }
        )

    families = []
    for kind, relation, groups in (
        ("r3_merge", "same_construct_candidate", hierarchy["merge_groups"]),
        ("r3_grandparent", "subsumption", hierarchy["grandparents"]),
    ):
        for group in groups:
            ids = [metric_id(index) for index in group["indices"]]
            families.append(
                {
                    "family_id": _relation_id(task, kind, group["name"], ids),
                    "metric_ids": ids,
                    "r3_index": group["r3_index"],
                    "r3_name": group["name"],
                    "relation": relation,
                    "provenance": kind,
                }
            )

    metric_relations = {
        metric_id(index): {
            "equivalent_metric_ids": [metric_id(index)],
            "family_ids": [],
            "family_metric_ids": [metric_id(index)],
        }
        for index in range(hierarchy["n_metrics"])
    }
    for group in accepted_equivalence:
        for mid in group["metric_ids"]:
            metric_relations[mid]["equivalent_metric_ids"] = list(group["metric_ids"])
    for family in families:
        for mid in family["metric_ids"]:
            metric_relations[mid]["family_ids"].append(family["family_id"])
            metric_relations[mid]["family_metric_ids"] = sorted(
                set(metric_relations[mid]["family_metric_ids"]).union(family["metric_ids"]),
                key=lambda value: int(value[1:]),
            )
    for value in metric_relations.values():
        value["family_ids"].sort()

    nontrivial_family_metrics = sum(
        len(value["family_metric_ids"]) > 1 for value in metric_relations.values()
    )
    return {
        "metric_count": hierarchy["n_metrics"],
        "bank_source_path": str(hierarchy["r2_path"]),
        "bank_source_sha256": sha256_file(hierarchy["r2_path"]),
        "r3_source_path": str(hierarchy["r3_path"]),
        "r3_source_sha256": sha256_file(hierarchy["r3_path"]),
        "equivalence_groups": accepted_equivalence,
        "families": families,
        "metric_relations": metric_relations,
        "r3_merge_audit": merge_audit,
        "audit": {
            "leaf_key_count": hierarchy["leaf_key_count"],
            "multiply_assigned_leaf_keys": hierarchy["multiply_assigned_leaf_keys"],
            "pair_labels": pair_stats,
            "r3_merge_candidates": len(hierarchy["merge_groups"]),
            "accepted_equivalence_groups": len(accepted_equivalence),
            "metrics_with_nontrivial_equivalence": len(occupied),
            "r3_merge_families": len(hierarchy["merge_groups"]),
            "r3_subsumption_families": len(hierarchy["grandparents"]),
            "metrics_with_nontrivial_family": nontrivial_family_metrics,
        },
    }


def build_relations(
    hierarchy_root: Path,
    pair_labels_path: Path,
    tasks: list[str],
    thresholds: Thresholds,
) -> dict[str, Any]:
    thresholds.validate()
    unknown = sorted(set(tasks).difference(TASK_TO_HIERARCHY))
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    hierarchy = {
        task: load_hierarchy_task(task, hierarchy_root) for task in sorted(set(tasks))
    }
    evidence, pair_stats = collect_pair_evidence(pair_labels_path, hierarchy)
    return {
        "schema_version": SCHEMA_VERSION,
        "relation_schema_version": RELATION_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pair_labels_path": str(pair_labels_path),
        "pair_labels_sha256": sha256_file(pair_labels_path),
        "thresholds": {
            "min_same_pairs": thresholds.min_same_pairs,
            "min_same_rate": thresholds.min_same_rate,
            "max_unrelated_pairs": thresholds.max_unrelated_pairs,
            "group_rule": "every_metric_pair_must_qualify",
            "duplicate_raw_pair_rule": "minimum_score_wins",
        },
        "tasks": {
            task: build_task_relations(
                hierarchy[task], evidence[task], pair_stats[task], thresholds
            )
            for task in sorted(hierarchy)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchy-root", default=str(DEFAULT_HIERARCHY_ROOT))
    parser.add_argument("--pair-labels", default=str(DEFAULT_PAIR_LABELS))
    parser.add_argument(
        "--tasks", nargs="+", default=sorted(TASK_TO_HIERARCHY)
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT_ROOT / "relations.json")
    )
    parser.add_argument("--min-same-pairs", type=int, default=3)
    parser.add_argument("--min-same-rate", type=float, default=0.90)
    parser.add_argument("--max-unrelated-pairs", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_relations(
        Path(args.hierarchy_root),
        Path(args.pair_labels),
        args.tasks,
        Thresholds(
            min_same_pairs=args.min_same_pairs,
            min_same_rate=args.min_same_rate,
            max_unrelated_pairs=args.max_unrelated_pairs,
        ),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        task: value["audit"] for task, value in result["tasks"].items()
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
