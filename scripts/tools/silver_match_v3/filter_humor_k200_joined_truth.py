#!/usr/bin/env python3
"""Exclude hash-bound joined Humor truth from production K200 CE scoring.

The filter is deliberately fail-closed.  It streams the complete immutable
pair file once, verifies the expected 77,378 x 200 input rectangle and the
22,090-UID joined-truth overlay, and publishes the 55,288 x 200 complement
only after every identity/count/hash invariant passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_INPUT_ROWS = 15_475_600
EXPECTED_INPUT_UIDS = 77_378
EXPECTED_TRUTH_UIDS = 22_090
EXPECTED_OUTPUT_ROWS = 11_057_600
EXPECTED_OUTPUT_UIDS = 55_288
EXPECTED_K = 200
EXPECTED_BANK_METRICS = 285
EXPECTED_BANK_SOURCE_SHA256 = (
    "1b4a29d34b4ef4d999e0cb0b2d1125286372349ff6dfa21a6adc5bc8e76f0de9"
)
SCHEMA = "silver-match-v3-humor-k200-minus-joined-truth-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _manifest_binds_truth(manifest: Mapping[str, Any], truth_sha: str) -> bool:
    for value in _walk(manifest):
        if not isinstance(value, Mapping):
            continue
        path = str(value.get("path") or "")
        if path.endswith("truth.joined.all.jsonl") and value.get("sha256") == truth_sha:
            return True
    return False


def _manifest_binds_bank(manifest: Mapping[str, Any]) -> bool:
    return any(value == EXPECTED_BANK_SOURCE_SHA256 for value in _walk(manifest))


def _load_bank(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_sha256") != EXPECTED_BANK_SOURCE_SHA256:
        raise ValueError("bank source SHA is not the frozen Humor bank source")
    metric_ids = {str(row.get("metric_id") or "") for row in payload.get("metrics", [])}
    if "" in metric_ids or len(metric_ids) != EXPECTED_BANK_METRICS:
        raise ValueError("bank is not the exact unique 285-metric Humor bank")
    return metric_ids


def _load_truth(path: Path, metric_ids: set[str]) -> tuple[set[str], Counter[str]]:
    uids: set[str] = set()
    decisions: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            uid = str(row.get("norm_uid") or "").strip()
            if not uid or uid in uids:
                raise ValueError(f"truth missing/duplicate UID at line {line_number}: {uid!r}")
            if row.get("task") not in (None, "", "humor"):
                raise ValueError(f"non-Humor truth row: {uid}")
            supplied_bank_sha = str(
                row.get("current_bank_source_sha256")
                or row.get("bank_source_sha256")
                or ""
            )
            if supplied_bank_sha and supplied_bank_sha != EXPECTED_BANK_SOURCE_SHA256:
                raise ValueError(f"truth bank SHA differs for {uid}")
            decision = str(row.get("decision") or "").strip()
            metric_id = row.get("metric_id")
            if decision == "MATCH":
                if str(metric_id or "") not in metric_ids:
                    raise ValueError(f"truth MATCH metric absent from bank: {uid}")
            elif metric_id not in (None, ""):
                raise ValueError(f"truth abstention has a metric_id: {uid}")
            if not decision:
                raise ValueError(f"truth lacks a typed decision: {uid}")
            uids.add(uid)
            decisions[decision] += 1
    if len(uids) != EXPECTED_TRUTH_UIDS:
        raise ValueError(f"expected {EXPECTED_TRUTH_UIDS} truth UIDs, got {len(uids)}")
    return uids, decisions


def build(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_pairs).resolve()
    truth_path = Path(args.joined_truth).resolve()
    truth_manifest_path = Path(args.truth_manifest).resolve()
    bank_path = Path(args.bank).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report_output).resolve()
    if output_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite filtered production artifacts")

    metric_ids = _load_bank(bank_path)
    truth_sha = _sha256(truth_path)
    manifest = json.loads(truth_manifest_path.read_text(encoding="utf-8"))
    if not _manifest_binds_truth(manifest, truth_sha):
        raise ValueError("truth manifest does not bind the copied joined-truth bytes")
    if not _manifest_binds_bank(manifest):
        raise ValueError("truth manifest does not bind the frozen Humor bank source")
    truth_uids, truth_decisions = _load_truth(truth_path, metric_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    input_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    input_uid_counts: Counter[str] = Counter()
    output_uid_counts: Counter[str] = Counter()
    truth_uids_seen: set[str] = set()
    input_rows = 0
    output_rows = 0
    try:
        with input_path.open("rb") as source, temporary.open("xb") as destination:
            for line_number, raw in enumerate(source, 1):
                input_digest.update(raw)
                input_rows += 1
                row = json.loads(raw)
                uid = str(row.get("norm_uid") or "").strip()
                metric_id = str(row.get("metric_id") or "").strip()
                if not uid or metric_id not in metric_ids:
                    raise ValueError(f"invalid pair identity at line {line_number}")
                input_uid_counts[uid] += 1
                if uid in truth_uids:
                    truth_uids_seen.add(uid)
                    continue
                destination.write(raw)
                output_digest.update(raw)
                output_rows += 1
                output_uid_counts[uid] += 1
            destination.flush()
            os.fsync(destination.fileno())

        input_sha = input_digest.hexdigest()
        if input_sha != args.expected_input_sha256:
            raise ValueError(f"input pair SHA differs: {input_sha}")
        if input_rows != EXPECTED_INPUT_ROWS or len(input_uid_counts) != EXPECTED_INPUT_UIDS:
            raise ValueError("input pair rectangle count differs")
        if any(count != EXPECTED_K for count in input_uid_counts.values()):
            raise ValueError("input pair file is not exactly K200 per UID")
        if truth_uids_seen != truth_uids:
            raise ValueError(
                f"joined truth is not a subset of production: missing={len(truth_uids-truth_uids_seen)}"
            )
        if output_rows != EXPECTED_OUTPUT_ROWS or len(output_uid_counts) != EXPECTED_OUTPUT_UIDS:
            raise ValueError("filtered pair rectangle count differs")
        if any(count != EXPECTED_K for count in output_uid_counts.values()):
            raise ValueError("filtered pair file is not exactly K200 per UID")
        if set(output_uid_counts) & truth_uids:
            raise AssertionError("truth UID leaked into filtered output")

        os.replace(temporary, output_path)
        report = {
            "schema_version": SCHEMA,
            "status": "COMPLETE_IMMUTABLE_FILTERED_PAIR_UNIVERSE",
            "task": "humor",
            "input": {
                "path": str(input_path),
                "sha256": input_sha,
                "rows": input_rows,
                "norm_uids": len(input_uid_counts),
                "k": EXPECTED_K,
            },
            "joined_truth": {
                "path": str(truth_path),
                "sha256": truth_sha,
                "norm_uids": len(truth_uids),
                "decision_counts": dict(sorted(truth_decisions.items())),
                "manifest_path": str(truth_manifest_path),
                "manifest_sha256": _sha256(truth_manifest_path),
            },
            "bank": {
                "path": str(bank_path),
                "sha256": _sha256(bank_path),
                "source_sha256": EXPECTED_BANK_SOURCE_SHA256,
                "metrics": len(metric_ids),
            },
            "output": {
                "path": str(output_path),
                "sha256": output_digest.hexdigest(),
                "rows": output_rows,
                "norm_uids": len(output_uid_counts),
                "k": EXPECTED_K,
            },
            "mi_policy": {
                "joined_truth_uids_excluded_from_mi": True,
                "joined_truth_uids_relabelled": False,
            },
        }
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return report
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pairs", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--joined-truth", required=True)
    parser.add_argument("--truth-manifest", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-output", required=True)
    return parser.parse_args(argv)


def main() -> None:
    print(json.dumps(build(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
