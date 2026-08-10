#!/usr/bin/env python3
"""Freeze a dev-only retriever-lane sequence, then evaluate it once on test.

The ``select`` command greedily adds the lane with the largest number of new
gold captures on the selection split.  It stops as soon as the exact one-sided
miss-rate upper bound is below the requested target.  The ``evaluate`` command
rehashes every input and applies only that already-frozen sequence to a
different split.  This keeps test labels out of lane and stopping selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .audit_candidate_capture import CAPTURE_LANES, lane_topk
from .audit_false_abstentions import clopper_pearson_upper
from .common import normalize_space, read_jsonl, sha256_file


SCHEMA = "silver-match-v3-candidate-capture-sequence-v1"
EVALUATION_SCHEMA = "silver-match-v3-candidate-capture-sequence-evaluation-v1"


def parse_named_paths(values: Iterable[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"candidate must be NAME=PATH: {value}")
        name, raw_path = value.split("=", 1)
        name = normalize_space(name)
        if not name or ":" in name:
            raise ValueError(f"invalid candidate system name: {name!r}")
        if name in paths:
            raise ValueError(f"duplicate candidate system name: {name}")
        paths[name] = Path(raw_path).resolve()
    if not paths:
        raise ValueError("at least one --candidate NAME=PATH is required")
    return paths


def _load_labels(paths: Iterable[Path], split: str) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            if normalize_space(row.get("split")) != split or row.get("decision") != "MATCH":
                continue
            uid = normalize_space(row.get("norm_uid"))
            metric_id = normalize_space(row.get("metric_id"))
            if not uid or not metric_id:
                raise ValueError(f"MATCH label missing norm_uid/metric_id in {path}")
            if uid in labels:
                raise ValueError(f"duplicate label norm_uid on split {split}: {uid}")
            labels[uid] = dict(row)
    if not labels:
        raise ValueError(f"no MATCH labels found on split {split}")
    return labels


def _load_candidates(path: Path, required_uids: set[str], system: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        uid = normalize_space(row.get("norm_uid"))
        if uid not in required_uids:
            continue
        if uid in rows:
            raise ValueError(f"duplicate candidate norm_uid in {system}: {uid}")
        rows[uid] = dict(row)
    missing = required_uids - set(rows)
    if missing:
        raise ValueError(
            f"candidate system {system} misses {len(missing)} required UIDs; "
            f"first={sorted(missing)[:3]}"
        )
    return rows


def _capture_sets(
    labels: Mapping[str, Mapping[str, Any]],
    candidate_paths: Mapping[str, Path],
    *,
    k: int,
) -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]]]:
    """Return gold-captured UIDs per lane and candidate IDs per UID/lane."""
    captured: dict[str, set[str]] = {}
    slates: dict[str, dict[str, set[str]]] = {uid: {} for uid in labels}
    required = set(labels)
    for system, path in sorted(candidate_paths.items()):
        candidates = _load_candidates(path, required, system)
        for lane in CAPTURE_LANES:
            key = f"{system}:{lane}"
            captured[key] = set()
            for uid, label in labels.items():
                candidate = candidates[uid]
                if normalize_space(candidate.get("task")) != normalize_space(label.get("task")):
                    raise ValueError(f"task mismatch for {uid} in {system}")
                rows = list(candidate.get("candidates") or [])
                bank_ids = {normalize_space(row.get("metric_id")) for row in rows}
                gold = normalize_space(label.get("metric_id"))
                if gold not in bank_ids:
                    raise ValueError(f"gold metric absent from bank for {uid} in {system}: {gold}")
                slate = lane_topk(rows, lane, k)
                slates[uid][key] = slate
                if gold in slate:
                    captured[key].add(uid)
    return captured, slates


def _union_size_summary(slates: Mapping[str, Mapping[str, set[str]]], sequence: list[str]) -> dict[str, Any]:
    counts = sorted(
        len(set().union(*(by_lane[lane] for lane in sequence))) if sequence else 0
        for by_lane in slates.values()
    )

    def percentile(fraction: float) -> int | None:
        if not counts:
            return None
        return counts[round((len(counts) - 1) * fraction)]

    return {
        "min": min(counts) if counts else None,
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "max": max(counts) if counts else None,
        "mean": sum(counts) / len(counts) if counts else None,
    }


def _prefix_report(
    labels: Mapping[str, Mapping[str, Any]],
    captured: Mapping[str, set[str]],
    slates: Mapping[str, Mapping[str, set[str]]],
    sequence: list[str],
    *,
    alpha: float,
    target: float,
) -> dict[str, Any]:
    union = set().union(*(captured[lane] for lane in sequence)) if sequence else set()
    misses = len(labels) - len(union)
    upper = clopper_pearson_upper(misses, len(labels), alpha=alpha)
    return {
        "trials": len(sequence),
        "sequence": list(sequence),
        "gold_matches": len(labels),
        "union_capture_count": len(union),
        "union_capture_rate": len(union) / len(labels),
        "union_miss_count": misses,
        "union_miss_rate": misses / len(labels),
        "union_miss_upper_bound": upper,
        "confidence_level_one_sided": 1.0 - alpha,
        "target_upper_bound": target,
        "under_target_supported": upper < target,
        "unique_candidate_union_size": _union_size_summary(slates, sequence),
    }


def select_sequence(
    label_paths: list[Path],
    candidate_paths: Mapping[str, Path],
    *,
    split: str = "dev",
    k: int = 50,
    alpha: float = 0.05,
    target: float = 0.05,
) -> dict[str, Any]:
    labels = _load_labels(label_paths, split)
    captured, slates = _capture_sets(labels, candidate_paths, k=k)
    uncovered = set(labels)
    remaining = set(captured)
    sequence: list[str] = []
    prefixes: list[dict[str, Any]] = []
    while remaining:
        scored = sorted(
            ((len(captured[lane] & uncovered), lane) for lane in remaining),
            key=lambda item: (-item[0], item[1]),
        )
        gain, lane = scored[0]
        if gain == 0:
            break
        sequence.append(lane)
        remaining.remove(lane)
        uncovered -= captured[lane]
        prefix = _prefix_report(
            labels, captured, slates, sequence, alpha=alpha, target=target
        )
        prefix["new_captures"] = gain
        prefixes.append(prefix)
        if prefix["under_target_supported"]:
            break
    final = prefixes[-1] if prefixes else _prefix_report(
        labels, captured, slates, sequence, alpha=alpha, target=target
    )
    return {
        "schema_version": SCHEMA,
        "selection_policy": "greedy_max_new_gold_capture_then_lexical_tie_break",
        "selection_split": split,
        "test_labels_used_for_selection": False,
        "k": k,
        "alpha": alpha,
        "target_upper_bound": target,
        "label_inputs": {str(path): sha256_file(path) for path in label_paths},
        "candidate_inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(candidate_paths.items())
        },
        "available_lanes": sorted(captured),
        "selected_sequence": sequence,
        "selection_prefixes": prefixes,
        "selection_result": final,
    }


def evaluate_sequence(
    selection: Mapping[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    if selection.get("schema_version") != SCHEMA:
        raise ValueError("selection schema mismatch")
    if split == selection.get("selection_split"):
        raise ValueError("evaluation split must differ from selection split")
    label_paths = [Path(path) for path in selection["label_inputs"]]
    for path, expected in selection["label_inputs"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"label hash mismatch: {path}")
    candidate_paths: dict[str, Path] = {}
    for name, artifact in selection["candidate_inputs"].items():
        path = Path(artifact["path"])
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"candidate hash mismatch: {name}")
        candidate_paths[name] = path
    labels = _load_labels(label_paths, split)
    captured, slates = _capture_sets(labels, candidate_paths, k=int(selection["k"]))
    sequence = list(selection["selected_sequence"])
    if not sequence or any(lane not in captured for lane in sequence):
        raise ValueError("frozen selected sequence is empty or invalid")
    prefixes = [
        _prefix_report(
            labels,
            captured,
            slates,
            sequence[:index],
            alpha=float(selection["alpha"]),
            target=float(selection["target_upper_bound"]),
        )
        for index in range(1, len(sequence) + 1)
    ]
    return {
        "schema_version": EVALUATION_SCHEMA,
        "selection_schema_version": selection["schema_version"],
        "selection_split": selection["selection_split"],
        "evaluation_split": split,
        "selected_sequence": sequence,
        "test_selection_performed": False,
        "prefixes": prefixes,
        "evaluation_result": prefixes[-1],
    }


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--labels", action="append", required=True)
    select.add_argument("--candidate", action="append", required=True, help="NAME=PATH")
    select.add_argument("--split", default="dev")
    select.add_argument("--k", type=int, default=50)
    select.add_argument("--alpha", type=float, default=0.05)
    select.add_argument("--target", type=float, default=0.05)
    select.add_argument("--output", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--selection", required=True)
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if args.command == "select":
        payload = select_sequence(
            [Path(path).resolve() for path in args.labels],
            parse_named_paths(args.candidate),
            split=args.split,
            k=args.k,
            alpha=args.alpha,
            target=args.target,
        )
    else:
        selection_path = Path(args.selection).resolve()
        payload = evaluate_sequence(
            json.loads(selection_path.read_text(encoding="utf-8")), split=args.split
        )
        payload["selection_artifact"] = {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        }
    _write_new(output, payload)
    print(json.dumps({
        "output": str(output),
        "output_sha256": sha256_file(output),
        "result": payload.get("selection_result") or payload.get("evaluation_result"),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
