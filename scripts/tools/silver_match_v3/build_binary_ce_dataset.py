#!/usr/bin/env python3
"""Materialize the frozen binary Humor CE recipe from audited pair slates.

All train EXACT pairs are retained. Train negatives are deterministically fixed
to 3,500 FAMILY siblings, 3,500 retrieval candidates ranked <=10, and 7,000
global/easy negatives. Dev and test are copied in full and remain natural. The
builder validates source-group and UID disjointness across all three roles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import normalize_space, read_jsonl, sha256_file


def _relation(row: Mapping[str, Any]) -> str:
    value = normalize_space(row.get("relation") or row.get("label")).upper()
    if value not in {"EXACT", "FAMILY", "REJECT"}:
        raise ValueError(f"invalid relation: {value!r}")
    return value


def _lanes(row: Mapping[str, Any]) -> set[str]:
    values = row.get("candidate_lanes") or []
    return {normalize_space(value) for value in values if normalize_space(value)}


def _top_retrieval_rank(row: Mapping[str, Any]) -> int | None:
    ranks = []
    for item in row.get("candidate_provenance") or []:
        if not isinstance(item, Mapping):
            continue
        lane = normalize_space(item.get("lane"))
        rank = item.get("rank")
        if lane != "global_balanced_negative" and isinstance(rank, int):
            ranks.append(rank)
    return min(ranks) if ranks else None


def negative_type(row: Mapping[str, Any]) -> str | None:
    relation = _relation(row)
    if relation == "EXACT":
        return None
    if relation == "FAMILY":
        return "hard_family"
    if "global_balanced_negative" in _lanes(row):
        return "easy_global"
    rank = _top_retrieval_rank(row)
    if rank is not None and rank <= 10:
        return "hard_retrieval_top10"
    return "unused_negative"


def _priority(row: Mapping[str, Any], *, seed: int, bucket: str) -> str:
    payload = "\x1f".join(
        (
            str(seed),
            bucket,
            normalize_space(row.get("norm_uid") or row.get("uid")),
            normalize_space(row.get("metric_id") or row.get("candidate_metric_id")),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load(path: Path, role: str) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for line_no, raw in enumerate(read_jsonl(path), 1):
        row = dict(raw)
        uid = normalize_space(row.get("norm_uid") or row.get("uid"))
        metric_id = normalize_space(row.get("metric_id") or row.get("candidate_metric_id"))
        source_group = normalize_space(row.get("source_group"))
        if not uid or not metric_id or not source_group:
            raise ValueError(f"{path}:{line_no}: missing UID, metric ID, or source group")
        key = (uid, metric_id)
        if key in seen:
            raise ValueError(f"{path}:{line_no}: duplicate pair {key}")
        if row.get("split") not in {None, role}:
            raise ValueError(f"{path}:{line_no}: row split disagrees with role {role}")
        seen.add(key)
        rows.append(row)
    if not rows:
        raise ValueError(f"empty {role} input: {path}")
    return rows


def _decorate(row: Mapping[str, Any], *, role: str, selected_type: str | None) -> dict[str, Any]:
    result = dict(row)
    relation = _relation(row)
    result.update(
        {
            "binary_label": 1 if relation == "EXACT" else 0,
            "binary_role": role,
            "binary_negative_type": selected_type,
            "binary_sampling_provenance": (
                "retain_all_known_positive_pairs"
                if relation == "EXACT"
                else selected_type or negative_type(row)
            ),
        }
    )
    return result


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def build(args: argparse.Namespace) -> dict[str, Any]:
    paths = {role: Path(getattr(args, role)).resolve() for role in ("train", "dev", "test")}
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = {role: _load(path, role) for role, path in paths.items()}

    groups = {
        role: {normalize_space(row["source_group"]) for row in values}
        for role, values in rows.items()
    }
    uids = {
        role: {normalize_space(row.get("norm_uid") or row.get("uid")) for row in values}
        for role, values in rows.items()
    }
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if groups[left] & groups[right]:
            raise ValueError(f"source-group leakage between {left} and {right}")
        if uids[left] & uids[right]:
            raise ValueError(f"norm UID leakage between {left} and {right}")

    train = rows["train"]
    positives = [row for row in train if _relation(row) == "EXACT"]
    pools = {
        kind: [row for row in train if negative_type(row) == kind]
        for kind in ("hard_family", "hard_retrieval_top10", "easy_global")
    }
    quotas = {
        "hard_family": args.hard_family,
        "hard_retrieval_top10": args.hard_retrieval,
        "easy_global": args.easy_global,
    }
    selected: dict[str, list[dict[str, Any]]] = {}
    for kind, pool in pools.items():
        if len(pool) < quotas[kind]:
            raise ValueError(f"{kind} pool has {len(pool)} rows, requires {quotas[kind]}")
        selected[kind] = sorted(
            pool, key=lambda row: _priority(row, seed=args.seed, bucket=kind)
        )[: quotas[kind]]

    train_output = [
        *(_decorate(row, role="train", selected_type=None) for row in positives),
        *(
            _decorate(row, role="train", selected_type=kind)
            for kind in ("hard_family", "hard_retrieval_top10", "easy_global")
            for row in selected[kind]
        ),
    ]
    train_output.sort(
        key=lambda row: (
            _priority(row, seed=args.seed, bucket="final_order"),
            normalize_space(row.get("norm_uid")),
            normalize_space(row.get("metric_id")),
        )
    )

    output_paths = {
        role: output / f"binary.{role}.pairs.jsonl" for role in ("train", "dev", "test")
    }
    _write_jsonl(output_paths["train"], train_output)
    for role in ("dev", "test"):
        _write_jsonl(
            output_paths[role],
            (_decorate(row, role=role, selected_type=None) for row in rows[role]),
        )

    report = {
        "schema_version": "silver-match-v3-binary-ce-dataset-v1",
        "status": "COMPLETE_AUDITED_BINARY_DATASET",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "recipe": {
            "retain_all_train_exact": True,
            "hard_family": args.hard_family,
            "hard_retrieval_top10": args.hard_retrieval,
            "easy_global": args.easy_global,
            "dev_test_sampling": "NONE_NATURAL_FULL_SLATES",
            "allows_zero_one_or_multiple_exact_metrics_per_norm": True,
        },
        "inputs": {role: {"path": str(path), "sha256": sha256_file(path)} for role, path in paths.items()},
        "pool_counts": {kind: len(pool) for kind, pool in pools.items()},
        "output_counts": {
            "train": len(train_output),
            "train_binary": {"1": len(positives), "0": sum(quotas.values())},
            "train_negative_provenance": quotas,
            "dev_relations": dict(sorted(Counter(_relation(row) for row in rows["dev"]).items())),
            "test_relations": dict(sorted(Counter(_relation(row) for row in rows["test"]).items())),
        },
        "split_audit": {
            "source_group_overlap_count": 0,
            "norm_uid_overlap_count": 0,
            "source_groups": {role: len(value) for role, value in groups.items()},
            "norm_uids": {role: len(value) for role, value in uids.items()},
        },
        "outputs": {
            role: {"path": str(path), "sha256": sha256_file(path)}
            for role, path in output_paths.items()
        },
    }
    report_path = output / "REPORT.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--hard-family", type=int, default=3500)
    parser.add_argument("--hard-retrieval", type=int, default=3500)
    parser.add_argument("--easy-global", type=int, default=7000)
    args = parser.parse_args(argv)
    for name in ("hard_family", "hard_retrieval", "easy_global"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
