#!/usr/bin/env python3
"""Combine production verifier orders under a frozen fail-closed policy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import read_jsonl, sha256_file, write_jsonl


def _unique(path: Path, kind: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        uid = str(row.get("norm_uid") or "")
        if not uid or uid in output:
            raise ValueError(f"missing/duplicate {kind} norm_uid: {uid!r}")
        output[uid] = row
    return output


def _confidence_high(row: dict[str, Any]) -> bool:
    return str(row.get("confidence") or "").lower() == "high"


def combine(
    *,
    primary_path: Path,
    original_path: Path,
    hashed_path: Path,
    selection_path: Path,
    policy_path: Path,
    output_path: Path,
    plan_path: Path | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    primary = _unique(primary_path, "primary")
    original = _unique(original_path, "original verifier")
    hashed = _unique(hashed_path, "hashed verifier")
    expected = {
        uid for uid, row in primary.items() if row.get("decision") == "MATCH"
    }
    if set(original) != expected or set(hashed) != expected:
        raise ValueError(
            "two-order verifier coverage mismatch: "
            f"expected={len(expected)} original={len(original)} hashed={len(hashed)}"
        )

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    chosen = selection.get("chosen") or {}
    selected_prompt = str(chosen.get("prompt_sha256") or "")
    if (
        selection.get("selection_split") not in {"dev", "external_dev_only"}
        or selection.get("calibration_power_status") != "supported"
        or chosen.get("statistically_supported") is not True
        or len(selected_prompt) != 64
    ):
        raise ValueError("verifier selection is not supported and dev-selected")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    selection_ref = (policy.get("inputs") or {}).get("selection") or {}
    if (
        selection_ref.get("sha256") != sha256_file(selection_path)
        or policy.get("may_run_on_production_unlabeled_norms") is not True
        or (policy.get("dev_gate") or {}).get("cleared") is not True
        or str((policy.get("prompt") or {}).get("rendered_prompt_sha256") or "")
        != selected_prompt
    ):
        raise ValueError("verifier policy is not linked to the supported selection")
    acceptance = str((policy.get("order_policy") or {}).get("retain_only_if") or "")
    if "both orders" not in acceptance or "high confidence" not in acceptance:
        raise ValueError("verifier policy is not the expected two-high fail-closed policy")

    plan = None
    plan_sha = None
    runtime_invalid_counts: dict[str, int] = {}
    if plan_path:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_sha = sha256_file(plan_path)
        if (
            plan.get("status") != "FROZEN_READY_FOR_UNLABELED_PRODUCTION"
            or plan["candidate_union"]["sha256"]
            != sha256_file(Path(plan["candidate_union"]["path"]))
            or plan["verifier"]["selection"]["sha256"]
            != sha256_file(selection_path)
            or plan["verifier"]["production_policy"]["sha256"]
            != sha256_file(policy_path)
            or plan["verifier"]["implementation"]["sha256"]
            != sha256_file(Path(plan["verifier"]["implementation"]["path"]))
        ):
            raise ValueError("production plan or verifier implementation changed")
        rendering = plan["verifier"]["rendering"]
        expected_meta = {
            "input_candidates_sha256": plan["candidate_union"]["sha256"],
            "primary_sha256": sha256_file(primary_path),
            "prompt_sha256": selected_prompt,
            "model": rendering["model"],
            "max_alternatives": rendering["max_alternatives"],
            "max_model_len": rendering["max_model_len"],
            "max_tokens": rendering["max_tokens"],
            "seed": rendering["seed"],
            "context_chars": rendering["context_chars"],
            "description_chars": rendering["description_chars"],
            "example_chars": rendering["example_chars"],
            "max_examples": rendering["max_examples"],
        }
        expected_component_hashes = sorted(
            value["sha256"]
            for value in plan["verifier"]["prompt_components"].values()
        )
        for order, path in (("original", original_path), ("hashed", hashed_path)):
            meta_path = path.with_suffix(path.suffix + ".meta.json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            actual_components = sorted(
                (meta.get("prompt_component_sha256") or {}).values()
            )
            order_rows = original if order == "original" else hashed
            actual_invalid = sum(
                row.get("decision") == "INVALID_OUTPUT" for row in order_rows.values()
            )
            runtime_invalid_counts[order] = actual_invalid
            if (
                any(meta.get(key) != value for key, value in expected_meta.items())
                or meta.get("order_mode") != order
                or meta.get("output_sha256") != sha256_file(path)
                or int(meta.get("invalid_count", -1)) != actual_invalid
                or actual_components != expected_component_hashes
            ):
                raise ValueError(f"{order} verifier runtime differs from frozen plan")

    counts: Counter[str] = Counter()
    output = []
    for uid in sorted(expected):
        proposal = primary[uid]
        left, right = original[uid], hashed[uid]
        proposed = str(proposal.get("metric_id") or "")
        if not proposed:
            raise ValueError(f"primary MATCH lacks metric ID: {uid}")
        if {left.get("order_mode"), right.get("order_mode")} != {
            "original",
            "hashed",
        }:
            raise ValueError(f"verifier orders are not original+hashed: {uid}")
        for label, row in (("original", left), ("hashed", right)):
            if (
                row.get("primary_metric_id") != proposed
                or row.get("primary_prompt_sha256") != proposal.get("prompt_sha256")
                or row.get("prompt_sha256") != selected_prompt
                or row.get("candidate_bank_source_sha256")
                != proposal.get("candidate_bank_source_sha256")
            ):
                raise ValueError(f"{label} verifier provenance mismatch: {uid}")
        if left.get("model") != right.get("model"):
            raise ValueError(f"verifier model mismatch: {uid}")
        if set(left.get("alternative_ids") or []) != set(
            right.get("alternative_ids") or []
        ):
            raise ValueError(f"verifier alternative slate mismatch: {uid}")
        left_ok = (
            left.get("decision") == "CONFIRM_MATCH"
            and left.get("metric_id") == proposed
            and _confidence_high(left)
            and not left.get("parse_error")
        )
        right_ok = (
            right.get("decision") == "CONFIRM_MATCH"
            and right.get("metric_id") == proposed
            and _confidence_high(right)
            and not right.get("parse_error")
        )
        accepted = left_ok and right_ok
        if accepted:
            decision, metric_id, confidence, reason = (
                "CONFIRM_MATCH",
                proposed,
                "high",
                "both_orders_high_confidence_confirm_same_id",
            )
            counts["accepted"] += 1
        else:
            decision, metric_id, confidence, reason = (
                "REJECT_MATCH",
                None,
                "low",
                "two_order_verifier_policy_not_satisfied",
            )
            counts["rejected"] += 1
        counts[f"pair:{left.get('decision')}|{right.get('decision')}"] += 1
        output.append(
            {
                "schema_version": "silver-match-v3-two-order-production-verification-v1",
                "norm_uid": uid,
                "corpus": proposal.get("corpus"),
                "task": proposal.get("task"),
                "row": proposal.get("row"),
                "primary_metric_id": proposed,
                "decision": decision,
                "metric_id": metric_id,
                "confidence": confidence,
                "reason": reason,
                "candidate_bank_source_sha256": proposal.get(
                    "candidate_bank_source_sha256"
                ),
                "primary_prompt_sha256": proposal.get("prompt_sha256"),
                "prompt_sha256": selected_prompt,
                "model": left.get("model"),
                "order_mode": "original+hashed",
                "verification_orders": ["original", "hashed"],
                "strict_two_order_acceptance": accepted,
                "verifier_selection_sha256": sha256_file(selection_path),
                "verifier_policy_sha256": sha256_file(policy_path),
                "production_plan_sha256": plan_sha,
                "original": left,
                "hashed": right,
            }
        )
    write_jsonl(output_path, output)
    report = {
        "schema_version": "silver-match-v3-two-order-production-verification-report-v1",
        "count": len(output),
        "expected_count": len(expected),
        "complete": len(output) == len(expected),
        "counts": dict(sorted(counts.items())),
        "inputs": {
            "primary": {"path": str(primary_path), "sha256": sha256_file(primary_path)},
            "original": {
                "path": str(original_path),
                "sha256": sha256_file(original_path),
            },
            "hashed": {"path": str(hashed_path), "sha256": sha256_file(hashed_path)},
            "selection": {
                "path": str(selection_path),
                "sha256": sha256_file(selection_path),
            },
            "policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
        },
        "selected_prompt_sha256": selected_prompt,
        "acceptance": acceptance,
        "production_plan": (
            {"path": str(plan_path), "sha256": plan_sha} if plan_path else None
        ),
        "runtime_invalid_counts": runtime_invalid_counts,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    output_path.with_suffix(output_path.suffix + ".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--hashed", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--plan",
        help="optional frozen primary-production plan; rescue finalist pairs omit it",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = combine(
        primary_path=Path(args.primary).resolve(),
        original_path=Path(args.original).resolve(),
        hashed_path=Path(args.hashed).resolve(),
        selection_path=Path(args.selection).resolve(),
        policy_path=Path(args.policy).resolve(),
        plan_path=Path(args.plan).resolve() if args.plan else None,
        output_path=Path(args.output).resolve(),
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
