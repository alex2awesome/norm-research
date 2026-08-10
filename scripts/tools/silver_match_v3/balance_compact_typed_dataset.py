#!/usr/bin/env python3
"""Deterministically balance a compact typed training split by decision.

The source rows are never edited.  Every MATCH row is retained exactly once;
every non-MATCH row is retained at least once and deterministic duplicates are
added, proportionally within typed decision, until the two superclasses have
equal row counts.  A dev split is read only to prove UID/group disjointness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TYPED_DECISIONS = {
    "MATCH",
    "MATCH_FAMILY_ONLY",
    "NO_EXPLICIT_CRITERION",
    "CONTEXT_NEEDED",
    "GENERIC_VERDICT",
    "NO_CANDIDATE_FITS",
    "NOISE",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("decision") not in TYPED_DECISIONS:
                raise ValueError(f"untyped decision at {path}:{line_number}")
            messages = row.get("messages")
            if not isinstance(messages, list) or messages[-1].get("role") != "assistant":
                raise ValueError(f"invalid assistant target at {path}:{line_number}")
            target = json.loads(messages[-1]["content"])
            if target.get("decision") != row.get("decision"):
                raise ValueError(f"decision/target mismatch at {path}:{line_number}")
            if not row.get("norm_uid") or not row.get("source_group"):
                raise ValueError(f"missing identity at {path}:{line_number}")
            rows.append(row)
    return rows


def proportional_targets(counts: Counter[str], total: int) -> dict[str, int]:
    original_total = sum(counts.values())
    if not counts or total < original_total:
        raise ValueError("target cannot discard a typed non-MATCH row")
    exact = {key: total * value / original_total for key, value in counts.items()}
    allocated = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(allocated.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - allocated[key]), key))
    for key in order[:remainder]:
        allocated[key] += 1
    if sum(allocated.values()) != total or any(allocated[k] < counts[k] for k in counts):
        raise AssertionError("invalid proportional allocation")
    return allocated


def row_key(row: dict[str, Any], index: int, seed: int) -> str:
    identity = json.dumps(
        [seed, row["decision"], row["norm_uid"], row.get("view"), index],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--dev-sha256", required=True)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--source-report-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-train-rows", type=int, default=34_944)
    parser.add_argument("--expected-match-rows", type=int, default=29_344)
    parser.add_argument("--expected-train-uids", type=int, default=17_472)
    parser.add_argument("--expected-train-groups", type=int, default=12_575)
    parser.add_argument("--expected-dev-rows", type=int, default=2_119)
    parser.add_argument("--seed", type=int, default=94_137)
    args = parser.parse_args()

    train_path = Path(args.train).resolve()
    dev_path = Path(args.dev).resolve()
    source_report_path = Path(args.source_report).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    for path, expected in (
        (train_path, args.train_sha256),
        (dev_path, args.dev_sha256),
        (source_report_path, args.source_report_sha256),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA mismatch for {path}: {actual} != {expected}")

    train = read_jsonl(train_path)
    dev = read_jsonl(dev_path)
    if len(train) != args.expected_train_rows or len(dev) != args.expected_dev_rows:
        raise ValueError(f"unexpected split rows: train={len(train)}, dev={len(dev)}")
    if any(row.get("split") != "train" or row.get("gradient_eligible") is not True for row in train):
        raise ValueError("train role contract differs")
    if any(row.get("split") != "dev" or row.get("gradient_eligible") is not False for row in dev):
        raise ValueError("dev role contract differs")

    train_uids = {str(row["norm_uid"]) for row in train}
    train_groups = {str(row["source_group"]) for row in train}
    dev_uids = {str(row["norm_uid"]) for row in dev}
    dev_groups = {str(row["source_group"]) for row in dev}
    if len(train_uids) != args.expected_train_uids or len(train_groups) != args.expected_train_groups:
        raise ValueError("source train identity cardinality differs")
    if train_uids & dev_uids or train_groups & dev_groups:
        raise ValueError("train/dev leakage")

    match = [(index, row) for index, row in enumerate(train) if row["decision"] == "MATCH"]
    nonmatch = [(index, row) for index, row in enumerate(train) if row["decision"] != "MATCH"]
    if len(match) != args.expected_match_rows:
        raise ValueError(f"expected {args.expected_match_rows} MATCH rows, got {len(match)}")
    nonmatch_counts = Counter(row["decision"] for _, row in nonmatch)
    targets = proportional_targets(nonmatch_counts, len(match))

    by_decision: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for pair in nonmatch:
        by_decision[pair[1]["decision"]].append(pair)
    additions: list[tuple[int, dict[str, Any]]] = []
    for decision in sorted(by_decision):
        ordered = sorted(
            by_decision[decision],
            key=lambda pair: row_key(pair[1], pair[0], args.seed),
        )
        needed = targets[decision] - len(ordered)
        additions.extend(ordered[offset % len(ordered)] for offset in range(needed))

    # Retain every source row once, add only byte-identical source row objects,
    # and deterministically mix the resulting training order.
    indexed = [(index, row, 0) for index, row in enumerate(train)]
    occurrence: Counter[int] = Counter()
    for index, row in additions:
        occurrence[index] += 1
        indexed.append((index, row, occurrence[index]))
    rng = random.Random(args.seed)
    rng.shuffle(indexed)
    balanced = [row for _, row, _ in indexed]

    output_root.mkdir(parents=True, exist_ok=False)
    temporary = output_root / f".train.jsonl.tmp-{os.getpid()}"
    output = output_root / "train.jsonl"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in balanced:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    output_counts = Counter(row["decision"] for row in balanced)
    output_match = output_counts["MATCH"]
    output_nonmatch = len(balanced) - output_match
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    source_train_max = int(source_report["train"]["token_audit"]["maximum"]["tokens"])
    if source_train_max > 2048:
        raise ValueError(f"source compact train exceeds 2048 tokens: {source_train_max}")

    report = {
        "schema_version": "silver-match-v3-compact-typed-decision-balance-v1",
        "status": "PASS_DETERMINISTIC_DECISION_BALANCE_AND_LEAKAGE_AUDIT",
        "seed": args.seed,
        "source": {
            "train": {"path": str(train_path), "sha256": args.train_sha256, "rows": len(train)},
            "dev": {"path": str(dev_path), "sha256": args.dev_sha256, "rows": len(dev)},
            "compact_report": {
                "path": str(source_report_path),
                "sha256": args.source_report_sha256,
            },
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "rows": len(balanced),
            "decision_counts": dict(sorted(output_counts.items())),
            "match_rows": output_match,
            "nonmatch_rows": output_nonmatch,
        },
        "balance": {
            "source_decision_counts": dict(sorted(Counter(row["decision"] for row in train).items())),
            "nonmatch_proportional_targets": dict(sorted(targets.items())),
            "every_source_row_retained_at_least_once": True,
            "match_source_rows_retained_exactly_once": True,
            "assistant_target_bytes_per_source_row_unchanged": True,
            "source_rows_edited": 0,
        },
        "identity_and_leakage_audit": {
            "train_unique_norm_uids": len({str(row["norm_uid"]) for row in balanced}),
            "train_source_groups": len({str(row["source_group"]) for row in balanced}),
            "dev_unique_norm_uids": len(dev_uids),
            "dev_source_groups": len(dev_groups),
            "train_dev_norm_uid_overlap": 0,
            "train_dev_source_group_overlap": 0,
            "heldout_or_blind_read": False,
        },
        "token_audit": {
            "method": "exact-source-row-identity inheritance from frozen compact report",
            "source_train_maximum_tokens": source_train_max,
            "output_rows_are_unedited_source_rows": True,
            "max_allowed_tokens": 2048,
            "all_rows_within_limit": True,
        },
    }
    if (
        len(balanced) != 2 * len(match)
        or output_match != len(match)
        or output_nonmatch != len(match)
        or len({str(row["norm_uid"]) for row in balanced}) != args.expected_train_uids
        or len({str(row["source_group"]) for row in balanced}) != args.expected_train_groups
        or set(nonmatch_counts) != set(targets)
    ):
        raise AssertionError("final balance audit failed")
    report_path = output_root / "BALANCE_REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(report_path), **report["output"]}, sort_keys=True))


if __name__ == "__main__":
    main()
