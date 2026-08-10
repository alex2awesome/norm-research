#!/usr/bin/env python3
"""Combine every verifier order required by a frozen production policy.

The older production combiner was intentionally limited to original+hashed.
Explicit-role GEPA may select either that policy or the stricter
original+hashed+reverse policy.  This module reads the order set from the
frozen policy and retains a proposal only when *every* required order returns
the same exact metric ID at high confidence without a parse error.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .audit_production_adjudications import _check_plan_artifacts
from .common import read_jsonl, sha256_file, write_jsonl


SUPPORTED_ORDERS = ("original", "hashed", "reverse")


def _unique(path: Path, kind: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        uid = str(row.get("norm_uid") or "")
        if not uid or uid in output:
            raise ValueError(f"missing/duplicate {kind} norm_uid: {uid!r}")
        output[uid] = row
    return output


def _selected_verifier_prompt(selection: dict[str, Any]) -> str:
    chosen = selection.get("chosen") or {}
    schema = str(selection.get("schema_version") or "")
    if schema == "silver-match-v3-explicit-role-verifier-selection-v1":
        if (
            selection.get("status") != "selected"
            or selection.get("selection_role") != "prompt_dev"
            or selection.get("test_or_blind_audit_consumed") is not False
            or selection.get("production_consumed") is not False
            or selection.get("outcomes_or_mi_used") is not False
            or chosen.get("eligible") is not True
        ):
            raise ValueError("explicit-role verifier was not cleanly selected on prompt_dev")
        prompt = str(chosen.get("verifier_prompt_sha256") or "")
    else:
        if (
            selection.get("selection_split") not in {"dev", "external_dev_only"}
            or selection.get("calibration_power_status") != "supported"
            or chosen.get("statistically_supported") is not True
        ):
            raise ValueError("legacy verifier is not supported and dev-selected")
        prompt = str(chosen.get("prompt_sha256") or "")
    if len(prompt) != 64:
        raise ValueError("selected verifier prompt lacks a SHA-256")
    return prompt


def _policy_orders(policy: dict[str, Any]) -> list[str]:
    orders = [str(value) for value in (policy.get("order_policy") or {}).get("orders") or []]
    if (
        orders not in (["original", "hashed"], ["original", "hashed", "reverse"])
        or len(orders) != len(set(orders))
    ):
        raise ValueError(f"unsupported or malformed verifier order policy: {orders}")
    mode = str((policy.get("order_policy") or {}).get("acceptance_mode") or "")
    if mode and mode != "all_orders_exact_high_same_id_no_parse_error":
        raise ValueError(f"unsupported verifier acceptance mode: {mode}")
    if not mode:
        acceptance = str((policy.get("order_policy") or {}).get("retain_only_if") or "")
        if orders != ["original", "hashed"] or "both orders" not in acceptance or "high confidence" not in acceptance:
            raise ValueError("legacy policy is not the established two-order exact/high policy")
    return orders


def _parse_order_paths(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--verification values must be ORDER=PATH")
        order, raw_path = value.split("=", 1)
        if order not in SUPPORTED_ORDERS or order in output or not raw_path:
            raise ValueError(f"duplicate or unsupported verifier order: {order!r}")
        output[order] = Path(raw_path).resolve()
    return output


def combine(
    *,
    primary_path: Path,
    verification_paths: dict[str, Path],
    selection_path: Path,
    policy_path: Path,
    output_path: Path,
    plan_path: Path | None = None,
    rescue_plan_path: Path | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    if plan_path is not None and rescue_plan_path is not None:
        raise ValueError("production and rescue runtime bindings are mutually exclusive")
    primary = _unique(primary_path, "primary")
    expected = {uid for uid, row in primary.items() if row.get("decision") == "MATCH"}

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_prompt = _selected_verifier_prompt(selection)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    orders = _policy_orders(policy)
    if list(verification_paths) != orders:
        raise ValueError(
            f"verification inputs must follow the exact frozen order list: expected={orders} "
            f"observed={list(verification_paths)}"
        )
    selection_ref = (policy.get("inputs") or {}).get("selection") or {}
    if (
        policy.get("task") != selection.get("task")
        or selection_ref.get("sha256") != sha256_file(selection_path)
        or policy.get("may_run_on_production_unlabeled_norms") is not True
        or (policy.get("dev_gate") or {}).get("cleared") is not True
        or str((policy.get("prompt") or {}).get("rendered_prompt_sha256") or "")
        != selected_prompt
    ):
        raise ValueError("verifier policy is not linked to the supported selection")

    per_order = {
        order: _unique(path, f"{order} verifier")
        for order, path in verification_paths.items()
    }
    for order, rows in per_order.items():
        if set(rows) != expected:
            raise ValueError(
                f"{order} verifier coverage mismatch: expected={len(expected)} observed={len(rows)}"
            )

    plan = None
    plan_sha = None
    rescue_plan_sha = None
    runtime_invalid_counts: dict[str, int] = {}
    if plan_path:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_sha = sha256_file(plan_path)
        _check_plan_artifacts(plan)
        if (
            plan.get("status") != "FROZEN_READY_FOR_UNLABELED_PRODUCTION"
            or plan.get("task") != policy.get("task")
            or plan["candidate_union"]["sha256"]
            != sha256_file(Path(plan["candidate_union"]["path"]))
            or plan["verifier"]["selection"]["sha256"] != sha256_file(selection_path)
            or plan["verifier"]["production_policy"]["sha256"] != sha256_file(policy_path)
            or plan["verifier"].get("orders") != orders
        ):
            raise ValueError("production plan differs from verifier selection/policy")
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
        expected_components = sorted(
            value["sha256"] for value in plan["verifier"]["prompt_components"].values()
        )
        for order, path in verification_paths.items():
            meta_path = path.with_suffix(path.suffix + ".meta.json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            actual_invalid = sum(
                row.get("decision") == "INVALID_OUTPUT" for row in per_order[order].values()
            )
            runtime_invalid_counts[order] = actual_invalid
            if (
                any(meta.get(key) != value for key, value in expected_meta.items())
                or meta.get("order_mode") != order
                or meta.get("output_sha256") != sha256_file(path)
                or int(meta.get("invalid_count", -1)) != actual_invalid
                or sorted((meta.get("prompt_component_sha256") or {}).values())
                != expected_components
            ):
                raise ValueError(f"{order} verifier runtime differs from frozen plan")
    elif rescue_plan_path:
        rescue_plan = json.loads(rescue_plan_path.read_text(encoding="utf-8"))
        rescue_plan_sha = sha256_file(rescue_plan_path)
        implementations = rescue_plan.get("implementations") or {}
        combiner_ref = implementations.get("combine_ordered_verifications.py") or {}
        verifier_ref = implementations.get("verify_gemma.py") or {}
        verifier_block = rescue_plan.get("verifier") or {}
        if (
            rescue_plan.get("schema_version") != "silver-match-v3-task-rescue-plan-v3"
            or rescue_plan.get("status")
            != "FROZEN_READY_FOR_REPEATED_FULL_BANK_RESCUE"
            or rescue_plan.get("task") != policy.get("task")
            or verifier_block.get("selection", {}).get("sha256")
            != sha256_file(selection_path)
            or verifier_block.get("production_policy", {}).get("sha256")
            != sha256_file(policy_path)
            or verifier_block.get("orders") != orders
            or combiner_ref.get("sha256") != sha256_file(Path(__file__).resolve())
            or verifier_ref.get("sha256")
            != sha256_file(Path(str(verifier_ref.get("path") or "")))
        ):
            raise ValueError("rescue plan differs from verifier selection/policy/runtime")
        rendering = verifier_block.get("rendering") or {}
        expected_components = sorted(
            value["sha256"]
            for value in (verifier_block.get("prompt_components") or {}).values()
        )
        expected_meta = {
            "primary_sha256": sha256_file(primary_path),
            "prompt_sha256": selected_prompt,
            "model": rendering.get("model"),
            "max_alternatives": max(
                1, int((rescue_plan.get("rescue_policy") or {}).get("max_finalists", 0)) - 1
            ),
            "max_model_len": rendering.get("max_model_len"),
            "max_tokens": rendering.get("max_tokens"),
            "seed": rendering.get("seed"),
            "context_chars": rendering.get("context_chars"),
            "description_chars": rendering.get("description_chars"),
            "example_chars": rendering.get("example_chars"),
            "max_examples": rendering.get("max_examples"),
        }
        if any(value is None for value in expected_meta.values()) or not expected_components:
            raise ValueError("rescue plan lacks exact verifier rendering/components")
        candidate_sha: str | None = None
        for order, path in verification_paths.items():
            meta_path = path.with_suffix(path.suffix + ".meta.json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            candidate_path = Path(str(meta.get("input_candidates") or "")).resolve()
            actual_candidate_sha = str(meta.get("input_candidates_sha256") or "")
            if (
                not candidate_path.is_file()
                or sha256_file(candidate_path) != actual_candidate_sha
                or (candidate_sha is not None and actual_candidate_sha != candidate_sha)
            ):
                raise ValueError(f"{order} rescue verifier candidate binding differs")
            candidate_sha = actual_candidate_sha
            actual_invalid = sum(
                row.get("decision") == "INVALID_OUTPUT"
                for row in per_order[order].values()
            )
            runtime_invalid_counts[order] = actual_invalid
            optional_runtime = {
                key: rendering[key]
                for key in ("batch_size", "gpu_memory_utilization", "enforce_eager")
                if key in rendering
            }
            if (
                any(meta.get(key) != value for key, value in expected_meta.items())
                or any(meta.get(key) != value for key, value in optional_runtime.items())
                or meta.get("order_mode") != order
                or meta.get("output_sha256") != sha256_file(path)
                or int(meta.get("invalid_count", -1)) != actual_invalid
                or sorted((meta.get("prompt_component_sha256") or {}).values())
                != expected_components
            ):
                raise ValueError(f"{order} verifier runtime differs from frozen rescue plan")

    counts: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    for uid in sorted(expected):
        proposal = primary[uid]
        proposed = str(proposal.get("metric_id") or "")
        if not proposed:
            raise ValueError(f"primary MATCH lacks metric ID: {uid}")
        rows = {order: per_order[order][uid] for order in orders}
        models = {str(row.get("model") or "") for row in rows.values()}
        alternative_sets = {
            tuple(sorted(str(value) for value in row.get("alternative_ids") or []))
            for row in rows.values()
        }
        if len(models) != 1 or "" in models:
            raise ValueError(f"verifier model mismatch: {uid}")
        if len(alternative_sets) != 1:
            raise ValueError(f"verifier alternative slate mismatch: {uid}")
        accepted_by_order: dict[str, bool] = {}
        for order, row in rows.items():
            if (
                row.get("order_mode") != order
                or row.get("primary_metric_id") != proposed
                or row.get("primary_prompt_sha256") != proposal.get("prompt_sha256")
                or row.get("prompt_sha256") != selected_prompt
                or row.get("candidate_bank_source_sha256")
                != proposal.get("candidate_bank_source_sha256")
            ):
                raise ValueError(f"{order} verifier provenance mismatch: {uid}")
            accepted_by_order[order] = bool(
                row.get("decision") == "CONFIRM_MATCH"
                and row.get("metric_id") == proposed
                and str(row.get("confidence") or "").lower() == "high"
                and not row.get("parse_error")
            )
        accepted = all(accepted_by_order.values())
        if accepted:
            decision, metric_id, confidence, reason = (
                "CONFIRM_MATCH",
                proposed,
                "high",
                "all_frozen_orders_high_confidence_confirm_same_id",
            )
            counts["accepted"] += 1
        else:
            decision, metric_id, confidence, reason = (
                "REJECT_MATCH",
                None,
                "low",
                "all_order_exact_high_policy_not_satisfied",
            )
            counts["rejected"] += 1
        counts["pattern:" + "|".join(str(rows[o].get("decision")) for o in orders)] += 1
        output.append(
            {
                "schema_version": "silver-match-v3-multi-order-production-verification-v1",
                "norm_uid": uid,
                "corpus": proposal.get("corpus"),
                "task": proposal.get("task"),
                "row": proposal.get("row"),
                "primary_metric_id": proposed,
                "decision": decision,
                "metric_id": metric_id,
                "confidence": confidence,
                "reason": reason,
                "candidate_bank_source_sha256": proposal.get("candidate_bank_source_sha256"),
                "primary_prompt_sha256": proposal.get("prompt_sha256"),
                "prompt_sha256": selected_prompt,
                "model": next(iter(models)),
                "order_mode": "+".join(orders),
                "verification_orders": orders,
                "strict_all_order_acceptance": accepted,
                "accepted_by_order": accepted_by_order,
                "verifier_selection_sha256": sha256_file(selection_path),
                "verifier_policy_sha256": sha256_file(policy_path),
                "production_plan_sha256": plan_sha,
                "rescue_plan_sha256": rescue_plan_sha,
                "orders": rows,
            }
        )
    write_jsonl(output_path, output)
    report = {
        "schema_version": "silver-match-v3-multi-order-production-verification-report-v1",
        "count": len(output),
        "expected_count": len(expected),
        "complete": len(output) == len(expected),
        "verification_orders": orders,
        "counts": dict(sorted(counts.items())),
        "inputs": {
            "primary": {"path": str(primary_path), "sha256": sha256_file(primary_path)},
            "verifications": {
                order: {"path": str(path), "sha256": sha256_file(path)}
                for order, path in verification_paths.items()
            },
            "selection": {"path": str(selection_path), "sha256": sha256_file(selection_path)},
            "policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
        },
        "selected_prompt_sha256": selected_prompt,
        "production_plan": {"path": str(plan_path), "sha256": plan_sha} if plan_path else None,
        "rescue_plan": (
            {"path": str(rescue_plan_path), "sha256": rescue_plan_sha}
            if rescue_plan_path
            else None
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
    parser.add_argument(
        "--verification",
        action="append",
        required=True,
        metavar="ORDER=PATH",
        help="one verifier output per frozen order, in frozen order sequence",
    )
    parser.add_argument("--selection", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--plan")
    parser.add_argument("--rescue-plan")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = combine(
        primary_path=Path(args.primary).resolve(),
        verification_paths=_parse_order_paths(args.verification),
        selection_path=Path(args.selection).resolve(),
        policy_path=Path(args.policy).resolve(),
        plan_path=Path(args.plan).resolve() if args.plan else None,
        rescue_plan_path=(
            Path(args.rescue_plan).resolve() if args.rescue_plan else None
        ),
        output_path=Path(args.output).resolve(),
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
