#!/usr/bin/env python3
"""Audit complementary retriever captures on independently labeled matches.

This is capture/recapture as a diversity diagnostic, not an independence
assumption.  The report measures each lane, their top-k union, marginal unique
captures, pairwise overlap, and an exact upper confidence bound on empirical
union misses.  Exhaustive rescue remains the stronger production guarantee.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .audit_false_abstentions import clopper_pearson_upper
from .build_abstention_rescue import LANES
from .common import normalize_space, read_jsonl, sha256_file


CAPTURE_LANES = ("rank", *LANES[:-1])
RANK_CAPTURE_DEPTHS = (1, 5, 10, 20, 50, 100, 150, 200, 285)
COMPONENT_CAPTURE_DEPTHS = (1, 2, 5, 10, 20, 30, 40, 50)


def _unique(
    paths: Iterable[Path],
    kind: str,
    *,
    only_uids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            uid = normalize_space(row.get("norm_uid"))
            if not uid:
                raise ValueError(f"{kind} row missing norm_uid in {path}")
            if only_uids is not None and uid not in only_uids:
                continue
            if uid in output:
                raise ValueError(f"duplicate {kind} norm_uid: {uid}")
            output[uid] = row
    return output


def lane_topk(candidates: list[Mapping[str, Any]], lane: str, k: int) -> set[str]:
    if k <= 0:
        raise ValueError("k must be positive")

    def key(row: Mapping[str, Any]):
        try:
            rank = int(row.get(lane))
        except (TypeError, ValueError):
            rank = 10**9
        try:
            fallback = int(row.get("rank"))
        except (TypeError, ValueError):
            fallback = 10**9
        return rank, fallback, normalize_space(row.get("metric_id"))

    ordered = sorted(candidates, key=key)
    return {normalize_space(row.get("metric_id")) for row in ordered[:k]}


def _component_min_rank(row: Mapping[str, Any]) -> int | None:
    """Return the best rank across preserved component and direct lane ranks."""

    values: list[int] = []
    nested = row.get("component_lane_ranks")
    if isinstance(nested, Mapping):
        for ranks in nested.values():
            if not isinstance(ranks, Mapping):
                continue
            for raw in ranks.values():
                try:
                    rank = int(raw)
                except (TypeError, ValueError):
                    continue
                if rank > 0:
                    values.append(rank)
    lane_ranks = row.get("lane_ranks")
    if isinstance(lane_ranks, Mapping):
        for raw in lane_ranks.values():
            try:
                rank = int(raw)
            except (TypeError, ValueError):
                continue
            if rank > 0:
                values.append(rank)
    return min(values) if values else None


def _summarize(
    records: list[dict[str, Any]],
    *,
    lanes: tuple[str, ...],
    alpha: float,
    target: float,
) -> dict[str, Any]:
    n = len(records)
    lane_hits = Counter()
    unique_hits = Counter()
    patterns = Counter()
    union_misses = 0
    pairwise_jaccard: dict[str, list[float]] = defaultdict(list)
    pairwise_capture = Counter()
    union_candidate_counts: list[int] = []
    ranks_by_system: dict[str, list[int | None]] = defaultdict(list)
    depths_by_system: dict[str, set[int]] = defaultdict(set)
    component_ranks_by_system: dict[str, list[int | None]] = defaultdict(list)
    component_counts_by_system: dict[str, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        gold = record["metric_id"]
        sets = record["sets"]
        union_candidates = set().union(*sets.values()) if sets else set()
        union_candidate_counts.append(len(union_candidates))
        for system, rank in (record.get("rank_by_system") or {}).items():
            ranks_by_system[system].append(rank)
        for system, depth in (record.get("candidate_depth_by_system") or {}).items():
            depths_by_system[system].add(int(depth))
        for system, rank in (record.get("component_rank_by_system") or {}).items():
            component_ranks_by_system[system].append(rank)
        for system, counts in (
            record.get("component_candidate_count_by_system") or {}
        ).items():
            for depth, count in counts.items():
                component_counts_by_system[system][int(depth)].append(int(count))
        captured = {lane for lane, ids in sets.items() if gold in ids}
        for lane in captured:
            lane_hits[lane] += 1
        if len(captured) == 1:
            unique_hits[next(iter(captured))] += 1
        patterns["+".join(lane for lane in lanes if lane in captured) or "NONE"] += 1
        if not captured:
            union_misses += 1
        for i, left in enumerate(lanes):
            for right in lanes[i + 1 :]:
                union = sets[left] | sets[right]
                pair = f"{left}|{right}"
                pairwise_jaccard[pair].append(len(sets[left] & sets[right]) / len(union) if union else 0.0)
                pairwise_capture[f"{pair}:both"] += int(gold in sets[left] and gold in sets[right])
                pairwise_capture[f"{pair}:left_only"] += int(gold in sets[left] and gold not in sets[right])
                pairwise_capture[f"{pair}:right_only"] += int(gold not in sets[left] and gold in sets[right])
                pairwise_capture[f"{pair}:neither"] += int(gold not in sets[left] and gold not in sets[right])
    upper = clopper_pearson_upper(union_misses, n, alpha=alpha)
    ordered_union_counts = sorted(union_candidate_counts)

    def percentile(fraction: float) -> int | None:
        if not ordered_union_counts:
            return None
        index = round((len(ordered_union_counts) - 1) * fraction)
        return ordered_union_counts[index]

    rank_capture: dict[str, Any] = {}
    for system, ranks in sorted(ranks_by_system.items()):
        depths = depths_by_system[system]
        if len(depths) != 1:
            raise ValueError(
                f"candidate depth differs within capture system {system}: {sorted(depths)}"
            )
        maximum_depth = next(iter(depths))
        observed = sorted(rank for rank in ranks if rank is not None)

        def rank_percentile(fraction: float) -> int | None:
            if not observed:
                return None
            index = round((len(observed) - 1) * fraction)
            return observed[index]

        capture_at_depth = {
            str(depth): {
                "count": sum(rank is not None and rank <= depth for rank in ranks),
                "rate": (
                    sum(rank is not None and rank <= depth for rank in ranks) / len(ranks)
                    if ranks
                    else None
                ),
            }
            for depth in RANK_CAPTURE_DEPTHS
            if depth <= maximum_depth
        }
        if str(maximum_depth) not in capture_at_depth:
            capture_at_depth[str(maximum_depth)] = {
                "count": len(observed),
                "rate": len(observed) / len(ranks) if ranks else None,
            }
        rank_capture[system] = {
            "candidate_depth": maximum_depth,
            "gold_present_count": len(observed),
            "gold_absent_count": len(ranks) - len(observed),
            "gold_rank_quantiles_when_present": {
                "p50": rank_percentile(0.5),
                "p90": rank_percentile(0.9),
                "p95": rank_percentile(0.95),
                "p99": rank_percentile(0.99),
                "max": max(observed) if observed else None,
            },
            "capture_at_depth": dict(
                sorted(capture_at_depth.items(), key=lambda item: int(item[0]))
            ),
        }

    component_capture: dict[str, Any] = {}
    for system, ranks in sorted(component_ranks_by_system.items()):
        depth_rows = component_counts_by_system[system]
        if not depth_rows:
            continue
        curve = {}
        for depth in sorted(depth_rows):
            counts = sorted(depth_rows[depth])
            if len(counts) != len(ranks):
                raise ValueError(
                    f"component candidate-count coverage differs: {system}/{depth}"
                )

            def count_percentile(fraction: float) -> int:
                index = round((len(counts) - 1) * fraction)
                return counts[index]

            captured = sum(rank is not None and rank <= depth for rank in ranks)
            curve[str(depth)] = {
                "gold_capture_count": captured,
                "gold_capture_rate": captured / len(ranks) if ranks else None,
                "unique_candidate_count": {
                    "min": min(counts),
                    "p50": count_percentile(0.5),
                    "p90": count_percentile(0.9),
                    "max": max(counts),
                    "mean": sum(counts) / len(counts),
                },
            }
        component_capture[system] = {
            "definition": "union of every preserved component/direct lane at rank <= depth",
            "curve": curve,
        }

    return {
        "gold_matches": n,
        "capture_count_by_lane": {lane: lane_hits[lane] for lane in lanes},
        "capture_rate_by_lane": {lane: lane_hits[lane] / n if n else None for lane in lanes},
        "unique_marginal_capture_by_lane": {lane: unique_hits[lane] for lane in lanes},
        "union_capture_count": n - union_misses,
        "union_capture_rate": (n - union_misses) / n if n else None,
        "union_miss_count": union_misses,
        "union_miss_rate": union_misses / n if n else None,
        "union_miss_upper_bound": upper,
        "confidence_level_one_sided": 1.0 - alpha,
        "target_upper_bound": target,
        "under_target_supported": upper is not None and upper < target,
        "unique_candidate_union_size": {
            "min": min(ordered_union_counts) if ordered_union_counts else None,
            "p50": percentile(0.5),
            "p90": percentile(0.9),
            "max": max(ordered_union_counts) if ordered_union_counts else None,
            "mean": (
                sum(ordered_union_counts) / len(ordered_union_counts)
                if ordered_union_counts
                else None
            ),
        },
        "candidate_rank_capture_by_system": rank_capture,
        "component_union_capture_by_system": component_capture,
        "capture_patterns": dict(sorted(patterns.items())),
        "pairwise_mean_slate_jaccard": {
            pair: sum(values) / len(values) for pair, values in sorted(pairwise_jaccard.items())
        },
        "pairwise_gold_capture_counts": dict(sorted(pairwise_capture.items())),
    }


def audit_candidate_capture(
    label_paths: list[Path],
    candidate_paths: list[Path],
    *,
    k: int = 50,
    alpha: float = 0.05,
    target: float = 0.05,
    allow_prefix_missing_gold: bool = False,
) -> dict[str, Any]:
    labels = _unique(label_paths, "label")
    if not candidate_paths:
        raise ValueError("at least one candidate system is required")
    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for index, path in enumerate(candidate_paths):
        stem = path.stem.lower()
        if "adapter" in stem:
            name = "adapter"
        elif "nemotron" in stem and "base" in stem:
            name = "nemotron_base"
        elif "bge" in stem:
            name = "bge"
        else:
            name = f"system{index}"
        grouped_paths[name].append(path)
    systems: list[tuple[str, tuple[Path, ...], dict[str, dict[str, Any]]]] = [
        (
            name,
            tuple(paths),
            _unique(paths, f"candidate:{name}", only_uids=set(labels)),
        )
        for name, paths in sorted(grouped_paths.items())
    ]
    lanes = (
        CAPTURE_LANES
        if len(systems) == 1
        else tuple(f"{name}:{lane}" for name, _, _ in systems for lane in CAPTURE_LANES)
    )
    records = []
    missing = []
    for uid, label in labels.items():
        if label.get("decision") != "MATCH":
            continue
        system_rows = [(name, rows.get(uid)) for name, _, rows in systems]
        if any(row is None for _, row in system_rows):
            missing.append(uid)
            continue
        metric_id = normalize_space(label.get("metric_id"))
        sets = {}
        rank_by_system: dict[str, int | None] = {}
        candidate_depth_by_system: dict[str, int] = {}
        component_rank_by_system: dict[str, int | None] = {}
        component_candidate_count_by_system: dict[str, dict[int, int]] = {}
        for name, candidate in system_rows:
            assert candidate is not None
            if normalize_space(label.get("task")) != normalize_space(candidate.get("task")):
                raise ValueError(f"task mismatch for {uid} in {name}")
            rows = list(candidate.get("candidates") or [])
            candidate_depth_by_system[name] = len(rows)
            bank_ids = {normalize_space(row.get("metric_id")) for row in rows}
            if metric_id not in bank_ids and not allow_prefix_missing_gold:
                raise ValueError(
                    f"gold metric absent from candidate bank for {uid} in {name}: {metric_id}"
                )
            rank_by_system[name] = next(
                (
                    position
                    for position, row in enumerate(rows, 1)
                    if normalize_space(row.get("metric_id")) == metric_id
                ),
                None,
            )
            component_ranks = [_component_min_rank(row) for row in rows]
            if any(rank is not None for rank in component_ranks):
                component_rank_by_system[name] = next(
                    (
                        rank
                        for row, rank in zip(rows, component_ranks, strict=True)
                        if normalize_space(row.get("metric_id")) == metric_id
                    ),
                    None,
                )
                component_candidate_count_by_system[name] = {
                    depth: sum(
                        rank is not None and rank <= depth
                        for rank in component_ranks
                    )
                    for depth in COMPONENT_CAPTURE_DEPTHS
                }
            for lane in CAPTURE_LANES:
                key = lane if len(systems) == 1 else f"{name}:{lane}"
                sets[key] = lane_topk(rows, lane, k)
        records.append(
            {
                "norm_uid": uid,
                "task": label["task"],
                "corpus": label["corpus"],
                "split": label.get("split"),
                "metric_id": metric_id,
                "sets": sets,
                "rank_by_system": rank_by_system,
                "candidate_depth_by_system": candidate_depth_by_system,
                "component_rank_by_system": component_rank_by_system,
                "component_candidate_count_by_system": (
                    component_candidate_count_by_system
                ),
            }
        )
    if missing:
        raise ValueError(f"candidates miss {len(missing)} gold MATCH UIDs; first={sorted(missing)[:3]}")
    if not records:
        raise ValueError("no joined gold MATCH rows")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[f"task:{record['task']}"] .append(record)
        groups[f"corpus:{record['corpus']}"] .append(record)
        if record.get("split"):
            groups[f"task_split:{record['task']}:{record['split']}"] .append(record)
    return {
        "schema_version": "silver-match-v3-candidate-capture-v1",
        "k": k,
        "lanes": list(lanes),
        "label_inputs": {str(path): sha256_file(path) for path in label_paths},
        "candidate_inputs": {str(path): sha256_file(path) for path in candidate_paths},
        "allow_prefix_missing_gold": allow_prefix_missing_gold,
        "overall": _summarize(records, lanes=lanes, alpha=alpha, target=target),
        "groups": {
            group: _summarize(rows, lanes=lanes, alpha=alpha, target=target)
            for group, rows in sorted(groups.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", action="append", required=True)
    parser.add_argument("--candidates", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target", type=float, default=0.05)
    parser.add_argument(
        "--allow-prefix-missing-gold",
        action="store_true",
        help="Treat a gold metric absent from an audited prefix lane as a retrieval miss.",
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit_candidate_capture(
        [Path(path).resolve() for path in args.labels],
        [Path(path).resolve() for path in args.candidates],
        k=args.k,
        alpha=args.alpha,
        target=args.target,
        allow_prefix_missing_gold=args.allow_prefix_missing_gold,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "gold_matches": report["overall"]["gold_matches"],
        "union_capture_rate": report["overall"]["union_capture_rate"],
        "union_miss_upper_bound": report["overall"]["union_miss_upper_bound"],
        "under_target_supported": report["overall"]["under_target_supported"],
        "output": str(output),
        "output_sha256": sha256_file(output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
