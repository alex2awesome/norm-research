#!/usr/bin/env python3
"""Freeze a cheap-model verifier benchmark before any API requests.

The benchmark is deliberately optimize-only.  It may establish whether a
smaller model is useful as a proposal checker, but it cannot promote that model
or replace a fresh task-level blind audit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    indexed = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in indexed or len(indexed) != len(rows):
        raise ValueError(f"invalid norm_uid coverage: {path}")
    return indexed


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        name: Path(getattr(args, name)).expanduser().resolve()
        for name in (
            "canonical_manifest",
            "inference_manifest",
            "balanced_report",
            "truth",
            "candidates",
            "primary",
            "targets",
            "prompt",
        )
    }
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    if args.orders != ["original", "hashed"]:
        raise ValueError("checker benchmark requires original and hashed orders")
    if not args.model.startswith("openai/gpt-5-mini"):
        raise ValueError("this benchmark is frozen specifically for gpt-5-mini")

    report = json.loads(paths["balanced_report"].read_text(encoding="utf-8"))
    if report.get("schema_version") != "silver-match-v3-balanced-verifier-gepa-train-v1":
        raise ValueError("unsupported balanced verifier report")
    for name in ("truth", "candidates", "primary", "targets"):
        if (report.get("output_hashes") or {}).get(name) != sha256_file(paths[name]):
            raise ValueError(f"balanced verifier artifact drift: {name}")

    truth = _index(paths["truth"])
    candidates = _index(paths["candidates"])
    primary = _index(paths["primary"])
    targets = _index(paths["targets"])
    uids = set(truth)
    if any(set(rows) != uids for rows in (candidates, primary, targets)):
        raise ValueError("benchmark inputs lack exact UID coverage")
    if len(uids) != int(report.get("count", -1)):
        raise ValueError("balanced report count drift")
    if any(
        row.get("task") != args.task
        or row.get("gepa_role") != "optimize"
        or row.get("split") != "train"
        or row.get("prompt_gradient_eligible") is not True
        for row in truth.values()
    ):
        raise ValueError("truth is not entirely optimize/train evidence")
    groups = [str(row.get("source_group") or "") for row in truth.values()]
    if "" in groups or len(set(groups)) != len(groups):
        raise ValueError("benchmark source groups are missing or duplicated")
    target_counts = Counter(str(row.get("target") or "") for row in targets.values())
    if target_counts != {"CONFIRM_MATCH": len(uids) // 2, "REJECT": len(uids) // 2}:
        raise ValueError(f"benchmark is not exactly balanced: {target_counts}")
    if any(row.get("decision") != "MATCH" for row in primary.values()):
        raise ValueError("every checker proposal must be a MATCH")
    for uid in uids:
        candidate_ids = {
            str(row.get("metric_id"))
            for row in candidates[uid].get("candidates") or []
        }
        if str(primary[uid].get("metric_id")) not in candidate_ids:
            raise ValueError(f"proposal absent from candidate slate: {uid}")

    canonical = json.loads(paths["canonical_manifest"].read_text(encoding="utf-8"))
    inference = json.loads(paths["inference_manifest"].read_text(encoding="utf-8"))
    canonical_bank = (canonical.get("banks") or {}).get(args.task) or {}
    local_bank = (inference.get("banks") or {}).get(args.task) or {}
    if (
        sha256_file(paths["canonical_manifest"])
        != "b614e345a07123f9fe79d9521351886107476d34cf2b09daa50efce71dc1356f"
        or local_bank.get("source_sha256") != canonical_bank.get("source_sha256")
        or int(canonical_bank.get("count", -1)) < 1
    ):
        raise ValueError("canonical manifest or current-bank identity drift")

    maximum_calls = 2 * len(uids) * len(args.orders)
    if args.max_api_requests < maximum_calls:
        raise ValueError(
            f"request cap must allow one parse retry per row/order: {maximum_calls}"
        )
    output_root.mkdir(parents=True, exist_ok=False)
    outputs = {order: str(output_root / f"{order}.jsonl") for order in args.orders}
    payload = {
        "schema_version": "silver-match-v3-small-checker-benchmark-freeze-v1",
        "status": "FROZEN_BEFORE_ANY_API_REQUEST",
        "task": args.task,
        "role": "optimize_only_model_capability_benchmark",
        "model": args.model,
        "count": len(uids),
        "source_group_count": len(set(groups)),
        "target_counts": dict(sorted(target_counts.items())),
        "orders": args.orders,
        "max_api_requests": args.max_api_requests,
        "fail_safe_maximum_calls_with_one_parse_retry": maximum_calls,
        "inference": {
            "max_alternatives": args.max_alternatives,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "temperature": 0.0,
            "reasoning_effort": args.reasoning_effort,
            "reasoning_exclude": True,
            "force_json_object": args.force_json_object,
        },
        "fixed_gate": {
            "minimum_retained_precision": 0.90,
            "minimum_wilson_95_lower": 0.80,
            "minimum_retained_support": 20,
            "two_order_exact_high": True,
        },
        "contracts": {
            "optimize_only": True,
            "sealed_dev_test_or_final_blind_opened": False,
            "prompt_selection_allowed_from_this_run": False,
            "production_promotion_allowed_from_this_run": False,
            "fresh_task_level_blind_confirmation_required": True,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "outputs": outputs,
    }
    freeze_path = output_root / "FREEZE.json"
    freeze_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**payload, "freeze_sha256": sha256_file(freeze_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    for name in (
        "canonical-manifest",
        "inference-manifest",
        "balanced-report",
        "truth",
        "candidates",
        "primary",
        "targets",
        "prompt",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--orders", nargs="+", default=["original", "hashed"])
    parser.add_argument("--max-api-requests", type=int, required=True)
    parser.add_argument("--max-alternatives", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high"),
        default="minimal",
    )
    parser.add_argument("--force-json-object", action="store_true")
    parser.add_argument("--seed", type=int, default=2026071301)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
