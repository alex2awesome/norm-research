#!/usr/bin/env python3
"""Freeze a bounded verifier-only GEPA round before inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    out = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in out or len(out) != len(rows):
        raise ValueError(f"empty, missing, or duplicate norm_uid values: {path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--order", action="append", choices=("original", "hashed", "reverse"), required=True)
    parser.add_argument("--model", default="google/gemma-4-31b-it")
    parser.add_argument("--max-api-requests", type=int, required=True)
    parser.add_argument("--fresh-select-freeze", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if len(args.order) != len(set(args.order)):
        raise ValueError("duplicate order mode")
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    truth_path, candidates_path, primary_path, fresh_path = map(
        lambda value: Path(value).resolve(),
        (args.truth, args.candidates, args.primary, args.fresh_select_freeze),
    )
    truth, candidates, primary = map(_index, (truth_path, candidates_path, primary_path))
    if any(
        row.get("task") != args.task
        or (row.get("predeclared_split") or row.get("split")) != "train"
        or row.get("prompt_gradient_eligible") is not True
        for row in truth.values()
    ):
        raise ValueError("GEPA truth is not wholly authoritative-train gradient evidence")
    if not set(primary) <= set(truth) or not set(primary) <= set(candidates):
        raise ValueError("primary proposals are not covered by truth and candidates")
    if any(row.get("decision") != "MATCH" for row in primary.values()):
        raise ValueError("verifier primary contains a non-MATCH proposal")
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    if (
        fresh.get("task") != args.task
        or fresh.get("role") != "select"
        or fresh.get("status") != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
    ):
        raise ValueError("fresh select freeze is invalid")
    components = [Path(value).resolve() for value in args.prompt]
    if any(not path.is_file() for path in components):
        raise FileNotFoundError("prompt component missing")
    combined = "\n\n".join(path.read_text(encoding="utf-8").rstrip() for path in components) + "\n"
    maximum_needed = 2 * len(primary) * len(args.order)
    if args.max_api_requests < maximum_needed:
        raise ValueError(f"API cap {args.max_api_requests} is below fail-safe maximum {maximum_needed}")
    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": "silver-match-v3-verifier-gepa-round-freeze-v1",
        "status": "FROZEN_BEFORE_VERIFIER_GEPA_INFERENCE",
        "task": args.task,
        "scope": "consumed_selection_authoritative_upstream_train_optimize_only",
        "truth_count": len(truth),
        "proposal_count": len(primary),
        "orders": args.order,
        "model": args.model,
        "maximum_api_requests": args.max_api_requests,
        "fail_safe_maximum_requests_with_one_parse_retry": maximum_needed,
        "prompt": {
            "components": [
                {"path": str(path), "sha256": sha256_file(path)} for path in components
            ],
            "combined_sha256": hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        },
        "selection_gate": {
            "minimum_retained_precision": 0.90,
            "minimum_wilson_95_lower": 0.80,
            "minimum_retained_support": 20,
            "thresholds_lowered": False,
        },
        "fresh_select": {"path": str(fresh_path), "sha256": sha256_file(fresh_path)},
        "permanent_blind_consumed": False,
        "inputs": {
            "truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
            "candidates": {"path": str(candidates_path), "sha256": sha256_file(candidates_path)},
            "primary": {"path": str(primary_path), "sha256": sha256_file(primary_path)},
        },
    }
    freeze = output / "FREEZE.json"
    freeze.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "freeze_sha256": sha256_file(freeze)}, sort_keys=True))


if __name__ == "__main__":
    main()
