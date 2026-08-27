#!/usr/bin/env python3
"""Seal a same-source, order-perturbed verifier pair before finalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


CONFIG_KEYS = (
    "schema_version",
    "manifest_sha256",
    "input_candidates_sha256",
    "primary_sha256",
    "prompt_sha256",
    "model",
    "max_alternatives",
    "batch_size",
    "max_model_len",
    "max_tokens",
    "gpu_memory_utilization",
    "seed",
    "context_chars",
    "description_chars",
    "example_chars",
    "max_examples",
    "shard_id",
    "num_shards",
)


def validate_pair(
    source: Path,
    first: Path,
    second: Path,
    selection: Path,
    *,
    expected_source_sha256: str,
) -> dict[str, Any]:
    source_sha = sha256_file(source)
    if source_sha != expected_source_sha256:
        raise ValueError(f"verifier source pin mismatch: {source_sha}")
    first_meta_path = first.with_suffix(first.suffix + ".meta.json")
    second_meta_path = second.with_suffix(second.suffix + ".meta.json")
    metas = [
        json.loads(first_meta_path.read_text(encoding="utf-8")),
        json.loads(second_meta_path.read_text(encoding="utf-8")),
    ]
    for path, meta in zip((first, second), metas):
        actual = sha256_file(path)
        if actual != meta.get("output_sha256"):
            raise ValueError(f"verifier output/meta hash mismatch: {path}")
    orders = {str(meta.get("order_mode")) for meta in metas}
    if orders != {"hashed", "reverse"}:
        raise ValueError(f"expected hashed+reverse orders, got {orders}")
    mismatches = {
        key: [meta.get(key) for meta in metas]
        for key in CONFIG_KEYS
        if metas[0].get(key) != metas[1].get(key)
    }
    if mismatches:
        raise ValueError(f"verifier pair configuration mismatch: {mismatches}")
    selection_payload = json.loads(selection.read_text(encoding="utf-8"))
    selected_prompt = selection_payload.get("chosen", {}).get("prompt_sha256")
    if selected_prompt != metas[0].get("prompt_sha256"):
        raise ValueError("verifier pair does not use the dev-selected prompt")
    uid_sets = []
    for path in (first, second):
        uids = [str(row["norm_uid"]) for row in read_jsonl(path)]
        if len(uids) != len(set(uids)):
            raise ValueError(f"duplicate verifier UID in {path}")
        uid_sets.append(set(uids))
    if uid_sets[0] != uid_sets[1]:
        raise ValueError("verifier pair UID sets differ")
    return {
        "schema_version": "silver-match-v3-verifier-pair-seal-v1",
        "verifier_source": str(source),
        "verifier_source_sha256": source_sha,
        "orders": sorted(orders),
        "count": len(uid_sets[0]),
        "shared_config": {key: metas[0].get(key) for key in CONFIG_KEYS},
        "outputs": {
            str(first): sha256_file(first),
            str(second): sha256_file(second),
        },
        "meta": {
            str(first_meta_path): sha256_file(first_meta_path),
            str(second_meta_path): sha256_file(second_meta_path),
        },
        "selection_record": str(selection),
        "selection_record_sha256": sha256_file(selection),
        "requires_independent_audit_before_gradient_use": bool(
            selection_payload.get("requires_independent_audit_before_gradient_use", True)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--selection-record", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    payload = validate_pair(
        Path(args.source).resolve(),
        Path(args.first).resolve(),
        Path(args.second).resolve(),
        Path(args.selection_record).resolve(),
        expected_source_sha256=args.expected_source_sha256,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
