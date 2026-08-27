#!/usr/bin/env python3
"""Merge deterministic Gemma inference shards into one audit-compatible artifact.

The Gemma runners can partition work by ``stable_shard(norm_uid)``.  Their
ordinary metadata, however, describes one shard, while the production auditors
expect a single complete artifact.  This merger verifies every shard's output
hash and invariant runtime metadata, proves exact expected-UID coverage, and
emits a deterministic combined JSONL plus normal runner-style metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .retrieve import stable_shard


MUTABLE_META_KEYS = {
    "output",
    "output_sha256",
    "new_count",
    "eligible_count",
    "unique_prompt_inferences",
    "deduplicated_prompt_count",
    "retry_prompt_inferences",
    "invalid_count",
    "possible_exact_bank_match_count",
    "shard_id",
    "num_shards",
    "elapsed_seconds",
}


def _unique_uids(path: Path, label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        uid = str(row.get("norm_uid") or "")
        if not uid or uid in output:
            raise ValueError(f"missing/duplicate {label} norm_uid: {uid!r}")
        output[uid] = row
    return output


def _expected_uids(meta: dict[str, Any]) -> set[str]:
    if meta.get("primary"):
        primary_path = Path(str(meta["primary"]))
        if sha256_file(primary_path) != str(meta.get("primary_sha256") or ""):
            raise ValueError("primary input changed since sharded inference")
        return {
            uid
            for uid, row in _unique_uids(primary_path, "primary").items()
            if row.get("decision") == "MATCH"
        }
    if meta.get("audits"):
        audits_path = Path(str(meta["audits"]))
        if sha256_file(audits_path) != str(meta.get("audits_sha256") or ""):
            raise ValueError("audit input changed since sharded inference")
        return set(_unique_uids(audits_path, "audit"))
    if meta.get("input_candidates"):
        candidates_path = Path(str(meta["input_candidates"]))
        if sha256_file(candidates_path) != str(
            meta.get("input_candidates_sha256") or ""
        ):
            raise ValueError("candidate input changed since sharded inference")
        return set(_unique_uids(candidates_path, "candidate"))
    raise ValueError("cannot infer expected coverage from shard metadata")


def _invariants(meta: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in meta.items() if key not in MUTABLE_META_KEYS}


def merge_shards(*, input_paths: list[Path], output_path: Path) -> dict[str, Any]:
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError(f"refusing to overwrite merged artifact: {output_path}")
    if len(input_paths) < 2 or len(set(input_paths)) != len(input_paths):
        raise ValueError("provide at least two distinct shard outputs")

    shard_records: dict[int, dict[str, Any]] = {}
    invariant: dict[str, Any] | None = None
    num_shards: int | None = None
    all_rows: dict[str, dict[str, Any]] = {}
    raw_retry_count = 0
    raw_elapsed_seconds = 0.0
    counters_complete = True

    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        shard_meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta = json.loads(shard_meta_path.read_text(encoding="utf-8"))
        if meta.get("output_sha256") != sha256_file(path):
            raise ValueError(f"shard output hash mismatch: {path}")
        shard_id = int(meta.get("shard_id", -1))
        current_num_shards = int(meta.get("num_shards", -1))
        if current_num_shards < 2 or not 0 <= shard_id < current_num_shards:
            raise ValueError(f"invalid shard coordinates: {path}")
        if shard_id in shard_records:
            raise ValueError(f"duplicate shard_id={shard_id}")
        if num_shards is None:
            num_shards = current_num_shards
        elif current_num_shards != num_shards:
            raise ValueError("shards disagree on num_shards")
        current_invariant = _invariants(meta)
        if invariant is None:
            invariant = current_invariant
        elif current_invariant != invariant:
            raise ValueError(f"runtime metadata differs across shards: {path}")

        rows = _unique_uids(path, f"shard-{shard_id}")
        if int(meta.get("new_count", -1)) != len(rows):
            # A resumed runner reports only rows newly appended in that process.
            # Coverage remains auditable, but historical inference counters do not.
            counters_complete = False
        for uid, row in rows.items():
            if stable_shard(uid, current_num_shards) != shard_id:
                raise ValueError(f"norm_uid assigned to wrong shard: {uid}")
            if uid in all_rows:
                raise ValueError(f"duplicate norm_uid across shards: {uid}")
            all_rows[uid] = row
        raw_retry_count += int(meta.get("retry_prompt_inferences") or 0)
        raw_elapsed_seconds += float(meta.get("elapsed_seconds") or 0.0)
        shard_records[shard_id] = {
            "output": {"path": str(path), "sha256": sha256_file(path)},
            "meta": {
                "path": str(shard_meta_path),
                "sha256": sha256_file(shard_meta_path),
            },
            "count": len(rows),
        }

    assert num_shards is not None and invariant is not None
    expected_shards = set(range(num_shards))
    if set(shard_records) != expected_shards:
        raise ValueError(
            f"incomplete shard set: missing={sorted(expected_shards - set(shard_records))}"
        )
    expected = _expected_uids(invariant)
    observed = set(all_rows)
    if observed != expected:
        raise ValueError(
            "merged coverage differs from source input: "
            f"missing={len(expected - observed)} extra={len(observed - expected)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, (all_rows[uid] for uid in sorted(all_rows)))
    invalid_count = sum(
        row.get("decision") == "INVALID_OUTPUT" for row in all_rows.values()
    )
    representatives = {
        str(row.get("inference_representative_norm_uid"))
        for row in all_rows.values()
        if row.get("inference_representative_norm_uid")
    }
    combined = {
        **invariant,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "new_count": len(all_rows),
        "eligible_count": len(expected),
        "invalid_count": invalid_count,
        "shard_id": 0,
        "num_shards": 1,
        "combined_from_num_shards": num_shards,
        "combined_shards": {
            str(shard_id): shard_records[shard_id]
            for shard_id in sorted(shard_records)
        },
        "elapsed_seconds_sum": raw_elapsed_seconds,
        "inference_counters_complete": counters_complete,
    }
    if representatives:
        combined["unique_prompt_inferences"] = len(representatives)
        combined["deduplicated_prompt_count"] = len(all_rows) - len(representatives)
    if counters_complete:
        combined["retry_prompt_inferences"] = raw_retry_count
    if any("possible_exact_bank_match" in row for row in all_rows.values()):
        combined["possible_exact_bank_match_count"] = sum(
            bool(row.get("possible_exact_bank_match")) for row in all_rows.values()
        )
    meta_path.write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = merge_shards(
        input_paths=[Path(path).resolve() for path in args.inputs],
        output_path=Path(args.output).resolve(),
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
