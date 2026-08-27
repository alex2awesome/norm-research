#!/usr/bin/env python3
"""Record the post-smoke Math-a12 parser revision without rerunning calls."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Sequence

from methods.metric_seam.verifiers.math_a12_llm_contract import (
    PARSER_VERSION,
    validate_response_envelope,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_receipt(*, old_requests: Path, old_responses: Path, revised_requests: Path) -> dict:
    old = [json.loads(line) for line in old_requests.read_text(encoding="utf-8").splitlines()]
    revised = [json.loads(line) for line in revised_requests.read_text(encoding="utf-8").splitlines()]
    responses = [json.loads(line) for line in old_responses.read_text(encoding="utf-8").splitlines()]
    if len(responses) != 10 or {row.get("request_index") for row in responses} != set(range(10)):
        raise ValueError("source must be the exact ten-row failed smoke")
    old_status = Counter(row.get("status") for row in responses)
    revised_by_digest = {row["request_sha256"]: row for row in revised}
    parse_modes: Counter[str] = Counter()
    for row in responses:
        request = revised_by_digest.get(row["request_sha256"])
        if request is None:
            raise ValueError("revised bundle is missing a source smoke request")
        validated = validate_response_envelope(row, request)
        parse_modes[validated["parse_mode"]] += 1
    same_digests = all(
        old[index]["request_sha256"] == revised_by_digest[old[index]["request_sha256"]]["request_sha256"]
        for index in range(10)
    )
    return {
        "schema": "metric-seam.math-a12-post-smoke-parser-calibration.v1",
        "status": "failed_smoke_recovered_by_cpu_parser_revision",
        "original_smoke": {
            "request_count": 10,
            "status_counts": dict(sorted(old_status.items())),
            "production_launched_before_revision": False,
        },
        "revision": {
            "parser_version": PARSER_VERSION,
            "prompt_changed": False,
            "model_calls_repeated": False,
            "replayed_valid": 10,
            "parse_mode_counts": dict(sorted(parse_modes.items())),
        },
        "digest_boundary": {
            "request_sha256_unchanged": same_digests,
            "reason": "response_contract/parser_version is outside the historical request identity digest",
            "bundle_file_hashes_bind_the_revision": True,
        },
        "sources": {
            "old_requests": {"path": str(old_requests), "sha256": _sha(old_requests)},
            "old_responses": {"path": str(old_responses), "sha256": _sha(old_responses)},
            "revised_requests": {"path": str(revised_requests), "sha256": _sha(revised_requests)},
        },
        "claim_limit": "This is post-smoke instrument calibration, not a successful original smoke and not a new model result.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-requests", type=Path, required=True)
    parser.add_argument("--old-responses", type=Path, required=True)
    parser.add_argument("--revised-requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = build_receipt(
        old_requests=args.old_requests,
        old_responses=args.old_responses,
        revised_requests=args.revised_requests,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
