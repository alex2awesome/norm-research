#!/usr/bin/env python3
"""Materialize disjoint progressive CE trials without weakening bank coverage.

The input K-primary and complete-bank pair artifacts remain authoritative.  A
candidate enters exactly one trial: the first component-union depth at which
any retrieval lane exposes it, the residual fused-K trial, or the final
complete-bank rescue trial.  Consequently, scoring every trial is byte-for-
byte equivalent in candidate coverage to scoring the complete bank once, but
an execution queue may stop individual norms after a development-authorized
two-seed exact match.

This module is CPU-only, create-only, and label blind.  It does not decide
which trials may stop; that is a separate untouched-development contract.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, TextIO

from .common import normalize_space, read_jsonl, sha256_file
from .materialize_nemotron_ce_production_pairs import (
    META_SCHEMA,
    PAIR_SCHEMA,
)


MANIFEST_SCHEMA = "silver-match-v3-progressive-nemotron-ce-pairs-v1"
TRIAL_PAIR_SCHEMA = PAIR_SCHEMA
STATUS = "FROZEN_DISJOINT_PROGRESSIVE_COMPLETE_BANK_UNIVERSE"
FORBIDDEN_LABEL_FIELDS = frozenset(
    {
        "relation",
        "ce_label",
        "target",
        "class_label",
        "label",
        "gold_relation",
        "decision",
        "acceptable_metric_ids",
        "equivalent_metric_ids",
    }
)


def _artifact(path: Path, *, count: int | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if count is not None:
        result["count"] = count
    return result


def _resolve_ref(ref: Mapping[str, Any], anchor: Path, label: str) -> Path:
    raw = Path(str(ref.get("path") or ""))
    if not str(raw):
        raise ValueError(f"{label} reference lacks a path")
    path = raw.resolve() if raw.is_absolute() else (anchor.parent / raw).resolve()
    if not path.is_file() or sha256_file(path) != normalize_space(ref.get("sha256")):
        raise ValueError(f"{label} artifact changed: {path}")
    return path


def _load_pair_report(path: Path, *, task: str, expected_depth: int | None) -> dict[str, Any]:
    path = path.resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    depth = int(report.get("candidate_depth", -1))
    norm_count = int(report.get("norm_count", -1))
    pair_count = int(report.get("pair_count", -1))
    if (
        report.get("schema_version") != META_SCHEMA
        or report.get("status") != "FROZEN_COMPLETE_UNLABELED_PRODUCTION_PAIR_UNIVERSE"
        or normalize_space(report.get("task")) != task
        or report.get("labels_present") is not False
        or report.get("release_ready") is not False
        or depth < 1
        or norm_count < 1
        or pair_count != norm_count * depth
        or (expected_depth is not None and depth != expected_depth)
    ):
        raise ValueError(f"invalid production pair report: {path}")
    pairs = _resolve_ref(report.get("pairs") or {}, path, "pair")
    universe = _resolve_ref(report.get("norm_universe") or {}, path, "norm universe")
    bank = _resolve_ref(report.get("bank") or {}, path, "bank")
    return {
        "path": path,
        "report": report,
        "pairs": pairs,
        "universe": universe,
        "bank": bank,
        "depth": depth,
        "norm_count": norm_count,
        "pair_count": pair_count,
    }


def _candidate_meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def _load_candidate_union(
    path: Path,
    *,
    task: str,
    expected_k: int,
    expected_norms: int,
    bank_source_sha256: str,
) -> dict[str, Any]:
    path = path.resolve()
    meta_path = _candidate_meta_path(path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lanes = (meta.get("union") or {}).get("lanes") or []
    names = [normalize_space(row.get("name")) for row in lanes if isinstance(row, Mapping)]
    complete_bank_lanes = [
        row
        for row in lanes
        if isinstance(row, Mapping)
        and normalize_space(row.get("kind")) == "complete-bank"
    ]
    if (
        not path.is_file()
        or meta.get("output_sha256") != sha256_file(path)
        or normalize_space(meta.get("task")) != task
        or normalize_space(meta.get("bank_source_sha256")) != bank_source_sha256
        or int(meta.get("input_count", -1)) != expected_norms
        or int(meta.get("output_k", -1)) != expected_k
        or len(lanes) < 2
        or len(complete_bank_lanes) < 2
        or len(names) != len(lanes)
        or "" in names
        or len(names) != len(set(names))
    ):
        raise ValueError("K-primary candidate union contract differs")
    return {
        "path": path,
        "meta_path": meta_path,
        "lane_names": names,
    }


def _groups(path: Path, depth: int) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    iterator = iter(read_jsonl(path))
    for group_index in range(10**18):
        try:
            first = next(iterator)
        except StopIteration:
            return
        uid = normalize_space(first.get("norm_uid"))
        if not uid:
            raise ValueError(f"pair group {group_index} lacks norm_uid: {path}")
        rows = [first]
        for _ in range(depth - 1):
            try:
                rows.append(next(iterator))
            except StopIteration as exc:
                raise ValueError(f"partial pair group for {uid}: {path}") from exc
        if any(normalize_space(row.get("norm_uid")) != uid for row in rows):
            raise ValueError(f"pair group is not contiguous/fixed-depth for {uid}: {path}")
        yield uid, rows


def _component_entry_depth(candidate: Mapping[str, Any], depths: Sequence[int]) -> int | None:
    ranks = candidate.get("lane_ranks") or {}
    values = [int(value) for value in ranks.values() if value is not None]
    if not values:
        return None
    best = min(values)
    return next((depth for depth in depths if best <= depth), None)


def _validate_pair_row(
    row: Mapping[str, Any],
    *,
    task: str,
    uid: str,
    metric_id: str,
    bank_source_sha256: str,
) -> None:
    if (
        row.get("schema_version") != PAIR_SCHEMA
        or row.get("task") != task
        or normalize_space(row.get("norm_uid")) != uid
        or normalize_space(row.get("metric_id")) != metric_id
        or row.get("split") != "production"
        or FORBIDDEN_LABEL_FIELDS.intersection(row)
        or not normalize_space(row.get("source_group"))
        or not normalize_space(row.get("query"))
        or not normalize_space(row.get("metric_card"))
        or normalize_space(row.get("current_bank_source_sha256"))
        != bank_source_sha256
    ):
        raise ValueError(f"invalid/leaky production pair: {uid}/{metric_id}")


def materialize(
    *,
    task: str,
    primary_report_path: Path,
    fullbank_report_path: Path,
    primary_candidates_path: Path,
    output_root: Path,
    component_depths: Sequence[int],
) -> dict[str, Any]:
    """Write a disjoint tier partition whose union is the complete bank."""

    task = normalize_space(task)
    depths = tuple(int(value) for value in component_depths)
    if not task or not depths or any(value < 1 for value in depths):
        raise ValueError("task and positive component depths are required")
    if tuple(sorted(set(depths))) != depths:
        raise ValueError("component depths must be strictly increasing and unique")
    output_root = output_root.resolve()
    manifest_path = output_root / "MANIFEST.json"
    if output_root.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite progressive pair root: {output_root}")

    primary = _load_pair_report(primary_report_path, task=task, expected_depth=None)
    fullbank = _load_pair_report(fullbank_report_path, task=task, expected_depth=None)
    if primary["norm_count"] != fullbank["norm_count"]:
        raise ValueError("primary/fullbank norm counts differ")
    if primary["depth"] >= fullbank["depth"]:
        raise ValueError("primary depth must be smaller than complete-bank depth")
    primary_report = primary["report"]
    full_report = fullbank["report"]
    if (
        primary_report.get("norm_universe") != full_report.get("norm_universe")
        or primary_report.get("bank") != full_report.get("bank")
        or primary_report.get("corpus_order") != full_report.get("corpus_order")
    ):
        # Paths may be different while bytes are identical.  The explicit hash
        # checks below are the relocatable scientific identity.
        if (
            sha256_file(primary["universe"]) != sha256_file(fullbank["universe"])
            or normalize_space((primary_report.get("bank") or {}).get("source_sha256"))
            != normalize_space((full_report.get("bank") or {}).get("source_sha256"))
            or primary_report.get("corpus_order") != full_report.get("corpus_order")
        ):
            raise ValueError("primary/fullbank universe, bank, or corpus scope differs")
    bank_sha = normalize_space((primary_report.get("bank") or {}).get("source_sha256"))
    bank_payload = json.loads(primary["bank"].read_text(encoding="utf-8"))
    bank_ids = {
        normalize_space(row.get("metric_id"))
        for row in bank_payload.get("metrics") or []
    }
    if (
        "" in bank_ids
        or len(bank_ids) != fullbank["depth"]
        or normalize_space(bank_payload.get("source_sha256")) != bank_sha
    ):
        raise ValueError("complete-bank depth/source identity differs from bank artifact")
    candidate = _load_candidate_union(
        primary_candidates_path,
        task=task,
        expected_k=primary["depth"],
        expected_norms=primary["norm_count"],
        bank_source_sha256=bank_sha,
    )

    trial_specs = [
        {
            "trial_id": f"component-union-d{depth}",
            "kind": "component_union_increment",
            "component_depth": depth,
        }
        for depth in depths
    ]
    trial_specs.extend(
        [
            {
                "trial_id": f"fused-k{primary['depth']}-remainder",
                "kind": "fused_primary_remainder",
                "component_depth": None,
            },
            {
                "trial_id": f"fullbank-k{fullbank['depth']}-rescue",
                "kind": "complete_bank_rescue",
                "component_depth": None,
            },
        ]
    )
    temporary = output_root.with_name(output_root.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    pair_paths = {
        spec["trial_id"]: temporary / f"{index:02d}-{spec['trial_id']}.pairs.jsonl"
        for index, spec in enumerate(trial_specs, 1)
    }
    counts = {spec["trial_id"]: 0 for spec in trial_specs}
    norm_counts = {spec["trial_id"]: 0 for spec in trial_specs}
    try:
        with ExitStack() as stack:
            handles: dict[str, TextIO] = {
                name: stack.enter_context(path.open("x", encoding="utf-8"))
                for name, path in pair_paths.items()
            }
            primary_groups = _groups(primary["pairs"], primary["depth"])
            full_groups = _groups(fullbank["pairs"], fullbank["depth"])
            candidates = iter(read_jsonl(candidate["path"]))
            sentinel = object()
            observed_norms = 0
            for primary_group, full_group, candidate_row in zip_longest(
                primary_groups, full_groups, candidates, fillvalue=sentinel
            ):
                if sentinel in (primary_group, full_group, candidate_row):
                    raise ValueError("primary/fullbank/candidate streams have different lengths")
                primary_uid, primary_rows = primary_group
                full_uid, full_rows = full_group
                candidate_uid = normalize_space(candidate_row.get("norm_uid"))
                if not primary_uid or {primary_uid, full_uid, candidate_uid} != {primary_uid}:
                    raise ValueError("primary/fullbank/candidate norm order differs")
                uid = primary_uid
                primary_ids = [normalize_space(row.get("metric_id")) for row in primary_rows]
                full_ids = [normalize_space(row.get("metric_id")) for row in full_rows]
                candidate_values = candidate_row.get("candidates") or []
                candidate_ids = [normalize_space(row.get("metric_id")) for row in candidate_values]
                if (
                    len(set(primary_ids)) != primary["depth"]
                    or len(set(full_ids)) != fullbank["depth"]
                    or primary_ids != candidate_ids
                    or not set(primary_ids) < set(full_ids)
                    or [int(row.get("rank", -1)) for row in candidate_values]
                    != list(range(1, primary["depth"] + 1))
                ):
                    raise ValueError(f"candidate coverage/order contract differs: {uid}")
                primary_by_id = dict(zip(primary_ids, primary_rows, strict=True))
                full_by_id = dict(zip(full_ids, full_rows, strict=True))
                assigned: set[str] = set()
                touched: set[str] = set()
                for candidate_value in candidate_values:
                    metric_id = normalize_space(candidate_value.get("metric_id"))
                    entry = _component_entry_depth(candidate_value, depths)
                    trial_id = (
                        f"component-union-d{entry}"
                        if entry is not None
                        else f"fused-k{primary['depth']}-remainder"
                    )
                    row = dict(primary_by_id[metric_id])
                    _validate_pair_row(
                        row,
                        task=task,
                        uid=uid,
                        metric_id=metric_id,
                        bank_source_sha256=bank_sha,
                    )
                    row.update(
                        {
                            "progressive_trial_id": trial_id,
                            "progressive_trial_kind": next(
                                spec["kind"] for spec in trial_specs if spec["trial_id"] == trial_id
                            ),
                            "progressive_source_candidate_rank": int(row["candidate_rank"]),
                            "progressive_component_entry_depth": entry,
                            "progressive_partition_schema": MANIFEST_SCHEMA,
                        }
                    )
                    handles[trial_id].write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    counts[trial_id] += 1
                    touched.add(trial_id)
                    assigned.add(metric_id)
                rescue_id = f"fullbank-k{fullbank['depth']}-rescue"
                for full_rank, metric_id in enumerate(full_ids, 1):
                    if metric_id in assigned:
                        continue
                    row = dict(full_by_id[metric_id])
                    _validate_pair_row(
                        row,
                        task=task,
                        uid=uid,
                        metric_id=metric_id,
                        bank_source_sha256=bank_sha,
                    )
                    row.update(
                        {
                            "progressive_trial_id": rescue_id,
                            "progressive_trial_kind": "complete_bank_rescue",
                            "progressive_source_candidate_rank": full_rank,
                            "progressive_component_entry_depth": None,
                            "progressive_partition_schema": MANIFEST_SCHEMA,
                        }
                    )
                    handles[rescue_id].write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    counts[rescue_id] += 1
                    touched.add(rescue_id)
                    assigned.add(metric_id)
                if (
                    assigned != set(full_ids)
                    or assigned != bank_ids
                    or len(assigned) != fullbank["depth"]
                ):
                    raise AssertionError(f"progressive partition is not exact full bank: {uid}")
                for trial_id in touched:
                    norm_counts[trial_id] += 1
                observed_norms += 1
            if observed_norms != primary["norm_count"]:
                raise ValueError("progressive partition norm count differs")
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())

        total_pairs = sum(counts.values())
        expected_pairs = fullbank["norm_count"] * fullbank["depth"]
        if total_pairs != expected_pairs:
            raise AssertionError("disjoint trial counts do not sum to complete bank")
        trials = []
        cumulative = 0
        for ordinal, spec in enumerate(trial_specs, 1):
            count = counts[spec["trial_id"]]
            cumulative += count
            trials.append(
                {
                    **spec,
                    "ordinal": ordinal,
                    "pairs": _artifact(pair_paths[spec["trial_id"]], count=count),
                    "norms_with_new_candidates": norm_counts[spec["trial_id"]],
                    "mean_new_candidates_per_all_norms": count / fullbank["norm_count"],
                    "cumulative_pair_count_if_no_norm_exits": cumulative,
                    "early_stop_requires_dev_policy_authorization": spec["kind"]
                    != "complete_bank_rescue",
                    "terminal": spec["kind"] == "complete_bank_rescue",
                }
            )
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "status": STATUS,
            "task": task,
            "source": {
                "primary_report": _artifact(primary["path"]),
                "primary_pairs": _artifact(primary["pairs"], count=primary["pair_count"]),
                "primary_candidates": _artifact(candidate["path"], count=primary["norm_count"]),
                "primary_candidates_meta": _artifact(candidate["meta_path"]),
                "fullbank_report": _artifact(fullbank["path"]),
                "fullbank_pairs": _artifact(fullbank["pairs"], count=fullbank["pair_count"]),
                "norm_universe": _artifact(primary["universe"], count=primary["norm_count"]),
                "bank": _artifact(primary["bank"]),
                "bank_source_sha256": bank_sha,
            },
            "norm_count": primary["norm_count"],
            "primary_depth": primary["depth"],
            "fullbank_depth": fullbank["depth"],
            "component_depths": list(depths),
            "trials": trials,
            "coverage_contract": {
                "trials_are_pairwise_disjoint": True,
                "union_equals_complete_bank_for_every_norm": True,
                "candidate_omission_count_after_terminal_trial": 0,
                "total_pair_count": total_pairs,
                "expected_complete_bank_pair_count": expected_pairs,
                "worst_case_two_seed_pair_evaluations": 2 * expected_pairs,
                "early_exit_changes_coverage": False,
                "labels_or_outcomes_read": False,
            },
            "release_ready": False,
        }
        raw = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        temp_manifest = temporary / "MANIFEST.json"
        temp_manifest.write_text(raw, encoding="utf-8")
        # References were computed under the temporary directory.  Rebind only
        # paths, never hashes, before publication.
        for trial in manifest["trials"]:
            trial["pairs"]["path"] = str(
                output_root / Path(trial["pairs"]["path"]).name
            )
        temp_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_root)
        published = json.loads(manifest_path.read_text(encoding="utf-8"))
        for trial in published["trials"]:
            ref = trial["pairs"]
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                raise AssertionError("published trial hash differs")
        return published
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--primary-report", required=True)
    parser.add_argument("--fullbank-report", required=True)
    parser.add_argument("--primary-candidates", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--component-depth", action="append", type=int, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = materialize(
        task=args.task,
        primary_report_path=Path(args.primary_report),
        fullbank_report_path=Path(args.fullbank_report),
        primary_candidates_path=Path(args.primary_candidates),
        output_root=Path(args.output_root),
        component_depths=args.component_depth,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "manifest": str(Path(args.output_root).resolve() / "MANIFEST.json"),
                "norm_count": result["norm_count"],
                "complete_bank_pair_count": result["coverage_contract"]["total_pair_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
