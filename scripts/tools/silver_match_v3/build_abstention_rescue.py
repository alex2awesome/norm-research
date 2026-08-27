#!/usr/bin/env python3
"""Build diverse, complementary 50-metric trials for provisional abstentions.

Each input candidate row must contain a full frozen task bank with component
ranks.  Metrics already shown to the primary adjudicator are excluded, and
successive trials use different retrieval lanes.  The union of the primary
slate and all rescue trials is asserted to equal the bank exactly.  Thus any
remaining false abstention is an adjudication error, not a retrieval omission.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import normalize_space, read_jsonl, sha256_file, write_jsonl


DEFAULT_ELIGIBLE = {
    "MATCH_FAMILY_ONLY",
    "NO_CANDIDATE_FITS",
    "UNSTABLE_MATCH",
    "INVALID_OUTPUT",
}
LANES = (
    "dense_statement_rank",
    "char_rank",
    "word_rank",
    "dense_rank",
    "char_statement_rank",
    "word_statement_rank",
    "rank",
)


def _resolve(path: str | Path, anchor: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else anchor.parent / value


def _unique_rows(paths: Iterable[Path], kind: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            uid = normalize_space(row.get("norm_uid"))
            if not uid:
                raise ValueError(f"{kind} row missing norm_uid in {path}")
            if uid in output:
                raise ValueError(f"duplicate {kind} norm_uid: {uid}")
            output[uid] = row
    return output


def _system_name(path: Path, index: int) -> str:
    stem = path.stem.lower()
    if "adapter" in stem:
        return "adapter"
    if "nemotron" in stem and "base" in stem:
        return "nemotron_base"
    if "bge" in stem:
        return "bge"
    return f"system{index}"


def _candidate_systems(
    paths: list[Path],
) -> list[tuple[str, tuple[Path, ...], dict[str, dict[str, Any]]]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for index, path in enumerate(paths):
        grouped[_system_name(path, index)].append(path)
    return [
        (name, tuple(system_paths), _unique_rows(system_paths, f"candidate:{name}"))
        for name, system_paths in sorted(grouped.items())
    ]


def is_eligible(
    primary: Mapping[str, Any],
    decisions: set[str],
    *,
    include_all_abstentions: bool,
    include_low_confidence: bool,
) -> bool:
    decision = normalize_space(primary.get("decision"))
    if decision == "MATCH":
        return False
    return (
        include_all_abstentions
        or decision in decisions
        or (include_low_confidence and normalize_space(primary.get("confidence")) == "low")
    )


def _lane_value(candidate: Mapping[str, Any], lane: str) -> tuple[int, int, str]:
    raw = candidate.get(lane)
    try:
        rank = int(raw)
    except (TypeError, ValueError):
        rank = 10**9
    try:
        fallback = int(candidate.get("rank"))
    except (TypeError, ValueError):
        fallback = 10**9
    return rank, fallback, normalize_space(candidate.get("metric_id"))


def complementary_blocks(
    candidates: list[dict[str, Any]],
    bank_ids: set[str],
    primary_ids: Iterable[str],
    *,
    block_size: int,
    lanes: tuple[str, ...] = LANES,
    lane_offset: int = 0,
) -> list[dict[str, Any]]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        metric_id = normalize_space(candidate.get("metric_id"))
        if not metric_id or metric_id in by_id:
            raise ValueError(f"missing/duplicate candidate metric_id: {metric_id!r}")
        by_id[metric_id] = candidate
    if set(by_id) != bank_ids:
        raise ValueError(
            f"full-bank candidates required: missing={len(bank_ids - set(by_id))}, "
            f"extra={len(set(by_id) - bank_ids)}"
        )
    seen = {normalize_space(value) for value in primary_ids if normalize_space(value)}
    if not seen <= bank_ids:
        raise ValueError(f"primary slate contains IDs outside bank: {sorted(seen - bank_ids)}")
    initial_seen_count = len(seen)
    blocks: list[dict[str, Any]] = []
    trial = 0
    while seen != bank_ids:
        lane = lanes[(trial + lane_offset) % len(lanes)]
        remaining = [by_id[mid] for mid in bank_ids - seen]
        remaining.sort(key=lambda row: _lane_value(row, lane))
        selected = remaining[:block_size]
        before = len(seen)
        seen.update(normalize_space(row["metric_id"]) for row in selected)
        blocks.append(
            {
                "trial": trial,
                "lane": lane,
                "coverage_before": before,
                "coverage_after": len(seen),
                "coverage_complete": seen == bank_ids,
                "candidates": selected,
            }
        )
        trial += 1
    expected = math.ceil((len(bank_ids) - initial_seen_count) / block_size)
    if len(blocks) != max(expected, 0):
        raise AssertionError(f"unexpected rescue block count: {len(blocks)} != {expected}")
    return blocks


def complementary_blocks_multi(
    systems: list[tuple[str, list[dict[str, Any]]]],
    bank_ids: set[str],
    primary_ids: Iterable[str],
    *,
    block_size: int,
    schedule_offset: int = 0,
) -> list[dict[str, Any]]:
    """Cover the bank using interleaved rankings from independent systems."""
    if not systems:
        raise ValueError("at least one full-bank candidate system is required")
    if len(systems) == 1:
        return complementary_blocks(
            systems[0][1],
            bank_ids,
            primary_ids,
            block_size=block_size,
            lane_offset=schedule_offset,
        )
    by_system: dict[str, dict[str, dict[str, Any]]] = {}
    for name, candidates in systems:
        index = {}
        for candidate in candidates:
            metric_id = normalize_space(candidate.get("metric_id"))
            if not metric_id or metric_id in index:
                raise ValueError(f"{name}: missing/duplicate metric_id {metric_id!r}")
            index[metric_id] = candidate
        if set(index) != bank_ids:
            raise ValueError(
                f"{name}: full-bank candidates required: "
                f"missing={len(bank_ids-set(index))}, extra={len(set(index)-bank_ids)}"
            )
        by_system[name] = index
    seen = {normalize_space(value) for value in primary_ids if normalize_space(value)}
    if not seen <= bank_ids:
        raise ValueError(f"primary slate contains IDs outside bank: {sorted(seen-bank_ids)}")
    initial_seen = len(seen)
    # Cross-system rank is the most directly comparable lane; then interleave
    # evidence/statement dense and lexical views before cycling again.
    lane_order = ("rank", *LANES[:-1])
    schedule = [(name, lane) for lane in lane_order for name, _ in systems]
    blocks = []
    trial = 0
    while seen != bank_ids:
        name, lane = schedule[(trial + schedule_offset) % len(schedule)]
        remaining = [by_system[name][metric_id] for metric_id in bank_ids - seen]
        remaining.sort(key=lambda row: _lane_value(row, lane))
        selected = []
        for candidate in remaining[:block_size]:
            rendered = dict(candidate)
            rendered["retrieval_system"] = name
            selected.append(rendered)
        before = len(seen)
        seen.update(row["metric_id"] for row in selected)
        blocks.append(
            {
                "trial": trial,
                "lane": f"{name}:{lane}",
                "system": name,
                "coverage_before": before,
                "coverage_after": len(seen),
                "coverage_complete": seen == bank_ids,
                "candidates": selected,
            }
        )
        trial += 1
    expected = math.ceil((len(bank_ids) - initial_seen) / block_size)
    if len(blocks) != max(expected, 0):
        raise AssertionError(f"unexpected rescue block count: {len(blocks)} != {expected}")
    return blocks


def repeated_blocks_multi(
    systems: list[tuple[str, list[dict[str, Any]]]],
    bank_ids: set[str],
    primary_ids: Iterable[str],
    *,
    block_size: int,
    coverage_repeats: int,
    reinclude_primary: bool,
) -> list[dict[str, Any]]:
    """Repeat exhaustive coverage with shifted systems/lanes and global trial IDs."""
    if coverage_repeats < 1:
        raise ValueError("coverage_repeats must be positive")
    primary = list(primary_ids)
    output = []
    global_trial = 0
    for capture in range(coverage_repeats):
        capture_primary = [] if reinclude_primary else primary
        blocks = complementary_blocks_multi(
            systems,
            bank_ids,
            capture_primary,
            block_size=block_size,
            schedule_offset=capture,
        )
        for capture_trial, block in enumerate(blocks):
            output.append(
                {
                    **block,
                    "trial": global_trial,
                    "capture": capture,
                    "capture_trial": capture_trial,
                    "coverage_repeats": coverage_repeats,
                    "reincludes_primary": reinclude_primary,
                }
            )
            global_trial += 1
    return output


def build_rescue(
    *,
    manifest_path: Path,
    candidate_paths: list[Path],
    primary_paths: list[Path],
    output_root: Path,
    block_size: int,
    primary_k: int,
    eligible_decisions: set[str],
    include_all_abstentions: bool,
    include_low_confidence: bool,
    coverage_repeats: int = 1,
    reinclude_primary: bool = False,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    banks: dict[str, set[str]] = {}
    for task, meta in manifest["banks"].items():
        payload = json.loads(_resolve(meta["path"], manifest_path).read_text(encoding="utf-8"))
        banks[task] = {normalize_space(row["metric_id"]) for row in payload["metrics"]}
    candidate_systems = _candidate_systems(candidate_paths)
    primary = _unique_rows(primary_paths, "primary")
    trial_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    decisions = Counter()
    task_eligible = Counter()
    task_trials = Counter()
    initial_coverage = Counter()

    for uid, adjudication in sorted(primary.items()):
        decisions[normalize_space(adjudication.get("decision")) or "MISSING"] += 1
        if not is_eligible(
            adjudication,
            eligible_decisions,
            include_all_abstentions=include_all_abstentions,
            include_low_confidence=include_low_confidence,
        ):
            continue
        system_rows = [(name, rows.get(uid)) for name, _, rows in candidate_systems]
        if any(row is None for _, row in system_rows):
            absent = [name for name, row in system_rows if row is None]
            raise ValueError(f"eligible UID lacks candidate systems {absent}: {uid}")
        row = system_rows[0][1]
        assert row is not None
        task = normalize_space(row.get("task"))
        corpus = normalize_space(row.get("corpus"))
        if task != normalize_space(adjudication.get("task")):
            raise ValueError(f"task mismatch for {uid}")
        if corpus != normalize_space(adjudication.get("corpus")):
            raise ValueError(f"corpus mismatch for {uid}")
        bank_sha = normalize_space(row.get("bank_source_sha256"))
        if bank_sha != normalize_space(manifest["banks"][task]["source_sha256"]):
            raise ValueError(f"bank hash mismatch for {uid}")
        for name, system_row in system_rows:
            assert system_row is not None
            if (
                normalize_space(system_row.get("task")) != task
                or normalize_space(system_row.get("corpus")) != corpus
                or normalize_space(system_row.get("bank_source_sha256")) != bank_sha
            ):
                raise ValueError(f"candidate-system routing/hash mismatch for {uid} in {name}")
        shown = adjudication.get("candidate_ids") or [
            candidate["metric_id"] for candidate in row["candidates"][:primary_k]
        ]
        blocks = repeated_blocks_multi(
            [(name, list(system_row["candidates"])) for name, system_row in system_rows if system_row is not None],
            banks[task],
            shown,
            block_size=block_size,
            coverage_repeats=coverage_repeats,
            reinclude_primary=reinclude_primary,
        )
        task_eligible[task] += 1
        task_trials[task] += len(blocks)
        initial_coverage[f"{task}:{len(set(shown))}"] += 1
        for block in blocks:
            trial_rows[block["trial"]].append(
                {
                    "schema_version": manifest["schema_version"],
                    "norm_uid": uid,
                    "corpus": corpus,
                    "task": task,
                    "row": row.get("row", adjudication.get("row")),
                    "bank_source_sha256": bank_sha,
                    "rescue_trial": block["trial"],
                    "rescue_capture": block["capture"],
                    "rescue_capture_trial": block["capture_trial"],
                    "rescue_coverage_repeats": coverage_repeats,
                    "rescue_reincludes_primary": reinclude_primary,
                    "rescue_lane": block["lane"],
                    "rescue_system": block.get("system"),
                    "rescue_coverage_before": block["coverage_before"],
                    "rescue_coverage_after": block["coverage_after"],
                    "rescue_bank_count": len(banks[task]),
                    "rescue_coverage_complete": block["coverage_complete"],
                    "primary_decision": adjudication.get("decision"),
                    "primary_confidence": adjudication.get("confidence"),
                    "primary_candidate_ids": list(shown),
                    "candidates": block["candidates"],
                }
            )

    outputs = {}
    for trial, rows in sorted(trial_rows.items()):
        rows.sort(key=lambda row: (row["task"], row["corpus"], row["norm_uid"]))
        path = output_root / f"trial-{trial:03d}.jsonl"
        write_jsonl(path, rows)
        outputs[str(path)] = {"count": len(rows), "sha256": sha256_file(path)}
    report = {
        "schema_version": "silver-match-v3-abstention-rescue-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "candidate_inputs": {str(path): sha256_file(path) for path in candidate_paths},
        "primary_inputs": {str(path): sha256_file(path) for path in primary_paths},
        "block_size": block_size,
        "fallback_primary_k": primary_k,
        "lanes": list(LANES),
        "candidate_systems": {
            name: [str(path) for path in paths]
            for name, paths, _ in candidate_systems
        },
        "eligible_decisions": sorted(eligible_decisions),
        "include_all_abstentions": include_all_abstentions,
        "include_low_confidence": include_low_confidence,
        "coverage_repeats": coverage_repeats,
        "reinclude_primary": reinclude_primary,
        "primary_decision_counts": dict(sorted(decisions.items())),
        "eligible_by_task": dict(sorted(task_eligible.items())),
        "rescue_trials_by_task": dict(sorted(task_trials.items())),
        "initial_coverage_counts": dict(sorted(initial_coverage.items())),
        "outputs": outputs,
        "coverage_invariant": (
            "every frozen bank metric appears exactly coverage_repeats times in rescue trials"
            if reinclude_primary
            else "every metric outside the primary slate appears exactly coverage_repeats times"
        ),
    }
    report_path = output_root / "rescue_manifest.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidates", action="append", required=True)
    parser.add_argument("--primary", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument("--primary-k", type=int, default=50)
    parser.add_argument("--eligible-decision", action="append", default=[])
    parser.add_argument("--all-abstentions", action="store_true")
    parser.add_argument("--exclude-low-confidence", action="store_true")
    parser.add_argument("--coverage-repeats", type=int, default=1)
    parser.add_argument(
        "--reinclude-primary",
        action="store_true",
        help="repeat the entire bank, including metrics already shown in the primary K",
    )
    args = parser.parse_args()
    decisions = set(args.eligible_decision) or set(DEFAULT_ELIGIBLE)
    report = build_rescue(
        manifest_path=Path(args.manifest).resolve(),
        candidate_paths=[Path(path).resolve() for path in args.candidates],
        primary_paths=[Path(path).resolve() for path in args.primary],
        output_root=Path(args.output_root).resolve(),
        block_size=args.block_size,
        primary_k=args.primary_k,
        eligible_decisions=decisions,
        include_all_abstentions=args.all_abstentions,
        include_low_confidence=not args.exclude_low_confidence,
        coverage_repeats=args.coverage_repeats,
        reinclude_primary=args.reinclude_primary,
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
