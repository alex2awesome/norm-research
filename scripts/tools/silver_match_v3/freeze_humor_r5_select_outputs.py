#!/usr/bin/env python3
"""Freeze complete Humor R5 select predictions before joining sealed truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def load_unique(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    rows = list(read_jsonl(path))
    uids = [str(row.get("norm_uid") or "") for row in rows]
    if not rows or not all(uids) or len(uids) != len(set(uids)):
        raise ValueError(f"empty/duplicate output: {path}")
    return rows, set(uids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--resume-runner")
    parser.add_argument("--failure-log")
    parser.add_argument("--resume-log")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    plan_path = Path(args.plan).resolve()
    run_root = Path(args.run_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("schema_version")
        != "silver-match-v3-humor-fresh-select-openrouter-gemma4-r5-freeze-v1"
        or plan.get("status") != "FROZEN_BEFORE_OPENROUTER_REQUESTS"
        or plan.get("task") != "humor"
        or plan.get("policy", {}).get("no_exploratory_variants") is not True
        or plan.get("fresh_select", {}).get("permanent_blind_consumed") is not False
        or plan.get("fresh_select", {}).get("inference_inputs_include_truth_or_labels")
        is not False
    ):
        raise ValueError("unsupported Humor R5 plan")

    model = plan["api"]["model"]
    candidate_sha = plan["inputs"]["candidates_sha256"]
    adjudicator: dict[str, Any] = {}
    adjudicator_uids: dict[str, set[str]] = {}
    for order in plan["policy"]["adjudicator_orders"]:
        path = run_root / "adjudicator" / f"{order}.jsonl"
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        rows, uids = load_unique(path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            len(rows) != int(plan["fresh_select"]["count"])
            or meta.get("output_sha256") != sha256_file(path)
            or meta.get("input_candidates_sha256") != candidate_sha
            or meta.get("model") != model
            or meta.get("order_mode") != order
            or meta.get("selection_role") != "dev"
            or int(meta.get("invalid_count", -1)) != 0
            or any(row.get("parse_error") is not None for row in rows)
        ):
            raise ValueError(f"incomplete/drifted adjudicator output: {order}")
        adjudicator_uids[order] = uids
        adjudicator[order] = {
            "predictions": artifact(path),
            "meta": artifact(meta_path),
            "count": len(rows),
            "invalid_count": 0,
            "api_request_count": int(meta.get("api_request_count", 0)),
        }
    if len({frozenset(value) for value in adjudicator_uids.values()}) != 1:
        raise ValueError("adjudicator orders lack paired coverage")

    primary = run_root / "adjudicator" / "exact_consensus.proposals.jsonl"
    primary_report = primary.with_suffix(primary.suffix + ".report.json")
    primary_rows, primary_uids = load_unique(primary)
    consensus = json.loads(primary_report.read_text(encoding="utf-8"))
    if (
        int(consensus.get("input_count", -1)) != int(plan["fresh_select"]["count"])
        or int(consensus.get("consensus_match_count", -1)) != len(primary_rows)
        or any(row.get("decision") != "MATCH" or not row.get("metric_id") for row in primary_rows)
    ):
        raise ValueError("invalid exact two-order proposal set")

    verifier: dict[str, Any] = {}
    verifier_uids: dict[str, set[str]] = {}
    for order in plan["policy"]["verifier_orders"]:
        path = run_root / "verifier_r5" / f"{order}.jsonl"
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        rows, uids = load_unique(path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            len(rows) != len(primary_rows)
            or uids != primary_uids
            or meta.get("output_sha256") != sha256_file(path)
            or meta.get("input_candidates_sha256") != candidate_sha
            or meta.get("primary_sha256") != sha256_file(primary)
            or meta.get("model") != model
            or meta.get("order_mode") != order
            or meta.get("selection_role") != "dev"
            or int(meta.get("invalid_count", -1)) != 0
            or any(row.get("parse_error") is not None for row in rows)
        ):
            raise ValueError(f"incomplete/drifted verifier output: {order}")
        verifier_uids[order] = uids
        verifier[order] = {
            "predictions": artifact(path),
            "meta": artifact(meta_path),
            "count": len(rows),
            "invalid_count": 0,
            "final_invocation_api_request_count": int(meta.get("api_request_count", 0)),
        }
    if len({frozenset(value) for value in verifier_uids.values()}) != 1:
        raise ValueError("verifier orders lack paired coverage")

    continuation: dict[str, Any] | None = None
    if args.resume_runner or args.failure_log or args.resume_log:
        if not all((args.resume_runner, args.failure_log, args.resume_log)):
            raise ValueError("continuation provenance must be complete")
        continuation = {
            "order": "reverse",
            "reason": "transport response lacked choices; prompt/model/order unchanged",
            "rows_preserved_before_resume": 96,
            "failed_initial_request_upper_bound": 224,
            "resume_hard_request_cap": 80,
            "runner": artifact(Path(args.resume_runner).resolve()),
            "failed_initial_log": artifact(Path(args.failure_log).resolve()),
            "resume_log": artifact(Path(args.resume_log).resolve()),
        }

    reverse_upper_bound = (
        continuation["failed_initial_request_upper_bound"]
        + continuation["resume_hard_request_cap"]
        if continuation
        else int(plan["request_budget"]["verifier_cap_per_order"])
    )
    request_upper_bound = (
        2 * int(plan["request_budget"]["adjudicator_cap_per_order"])
        + 2 * int(plan["request_budget"]["verifier_cap_per_order"])
        + reverse_upper_bound
    )
    if request_upper_bound > int(plan["request_budget"]["hard_maximum_total_requests"]):
        raise ValueError("continuation could exceed frozen global request budget")

    result = {
        "schema_version": "silver-match-v3-humor-r5-select-output-freeze-v1",
        "status": "FROZEN_COMPLETE_BEFORE_TRUTH_JOIN",
        "task": "humor",
        "role": "select",
        "model": model,
        "adjudicator": adjudicator,
        "exact_two_order_proposals": {
            "predictions": artifact(primary),
            "report": artifact(primary_report),
            "count": len(primary_rows),
        },
        "verifier_r5": verifier,
        "transport_continuation": continuation,
        "global_request_cap": int(plan["request_budget"]["hard_maximum_total_requests"]),
        "conservative_request_upper_bound": request_upper_bound,
        "plan": artifact(plan_path),
        "sealed_truth_sha256_not_opened": plan["fresh_select"][
            "exact_truth_sha256_seal_only"
        ],
        "truth_rows_read_by_freezer": False,
        "permanent_blind_consumed": False,
        "selection_or_prompt_iteration_after_truth_join_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**result, "freeze_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
