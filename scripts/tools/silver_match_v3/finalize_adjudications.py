#!/usr/bin/env python3
"""Finalize one corpus with exact coverage and conservative match verification."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .adjudicate_gemma import DECISIONS as ADJUDICATION_DECISIONS
from .audit_production_adjudications import _check_plan_artifacts
from .common import read_jsonl, sha256_file, write_jsonl
from .config import DEFAULT_OUTPUT_ROOT


FINAL_DECISIONS = set(ADJUDICATION_DECISIONS) | {"INVALID_OUTPUT", "UNSTABLE_MATCH"}


def _load_unique_minimal(
    paths: Iterable[Path],
    *,
    kind: str,
    fields: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            uid = str(row.get("norm_uid") or "")
            if not uid:
                raise ValueError(f"{kind} row lacks norm_uid in {path}")
            if uid in output:
                raise ValueError(f"duplicate {kind} norm_uid={uid}")
            output[uid] = {field: row.get(field) for field in fields}
    return output


def _resolve(path: str, manifest_path: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else manifest_path.parent / value


def selected_prompt_sha(path: Path, task: str, role: str) -> tuple[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task") != task:
        raise ValueError(f"{role} selection task mismatch: {path}")
    schema = str(payload.get("schema_version") or "")
    chosen = payload.get("chosen") or payload.get("selected") or payload
    if schema in {
        "silver-match-v3-explicit-role-adjudicator-selection-v1",
        "silver-match-v3-explicit-role-verifier-selection-v1",
    }:
        if (
            payload.get("status") != "selected"
            or payload.get("selection_role") != "prompt_dev"
            or payload.get("test_or_blind_audit_consumed") is not False
            or payload.get("production_consumed") is not False
            or payload.get("outcomes_or_mi_used") is not False
        ):
            raise ValueError(f"{role} explicit selection was not cleanly made on prompt_dev: {path}")
        key = (
            "verifier_prompt_sha256"
            if schema == "silver-match-v3-explicit-role-verifier-selection-v1"
            else "prompt_sha256"
        )
        prompt_sha = str(chosen.get(key) or "")
    else:
        if payload.get("selection_split") not in {"dev", "external_dev_only"}:
            raise ValueError(f"{role} selection was not made on dev: {path}")
        prompt_sha = str(chosen.get("prompt_sha256") or "")
    if len(prompt_sha) != 64:
        raise ValueError(f"{role} selection lacks prompt SHA-256: {path}")
    return prompt_sha, payload


def final_match_decision(
    primary: dict[str, Any],
    order: dict[str, Any] | None,
    verification: dict[str, Any] | None,
) -> tuple[str, str | None, str]:
    """Return final decision, metric ID, and a machine-readable status reason."""
    decision = str(primary.get("decision") or "")
    metric_id = primary.get("metric_id")
    if decision != "MATCH":
        return decision, None, "primary_typed_abstention"
    if not metric_id:
        return "INVALID_OUTPUT", None, "primary_match_missing_metric"
    if order is None:
        return "UNSTABLE_MATCH", None, "missing_order_check"
    if order.get("decision") != "MATCH":
        return "UNSTABLE_MATCH", None, "order_check_abstained"
    if order.get("metric_id") != metric_id:
        return "UNSTABLE_MATCH", None, "order_check_disagreed"
    if verification is None:
        return "UNSTABLE_MATCH", None, "missing_contrastive_verifier"
    if verification.get("decision") != "CONFIRM_MATCH":
        return "UNSTABLE_MATCH", None, "contrastive_verifier_rejected"
    if verification.get("metric_id") != metric_id:
        return "UNSTABLE_MATCH", None, "contrastive_verifier_metric_mismatch"
    if verification.get("confidence") == "low":
        return "UNSTABLE_MATCH", None, "contrastive_verifier_low_confidence"
    return "MATCH", str(metric_id), "verified_exact_match"


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.corpus not in manifest.get("corpora", {}):
        raise KeyError(f"unknown corpus {args.corpus!r}")
    corpus_meta = manifest["corpora"][args.corpus]
    task = corpus_meta["task"]
    bank_meta = manifest["banks"][task]
    bank_path = _resolve(bank_meta["path"], manifest_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))["metrics"]
    bank_ids = {str(metric["metric_id"]) for metric in bank}
    expected_path = _resolve(corpus_meta["path"], manifest_path)
    expected_rows = list(read_jsonl(expected_path))
    expected_uids = {str(row["norm_uid"]) for row in expected_rows}
    if len(expected_uids) != len(expected_rows):
        raise ValueError(f"duplicate norm_uid in canonical corpus {args.corpus}")

    primary_paths = [Path(path) for path in args.primary]
    order_paths = [Path(path) for path in args.order_check]
    verification_paths = [Path(path) for path in args.verification]
    primary = _load_unique_minimal(
        primary_paths,
        kind="primary",
        fields=(
            "corpus", "task", "row", "decision", "metric_id", "confidence", "reason",
            "candidate_ids", "candidate_bank_source_sha256", "prompt_sha256", "model",
            "order_mode", "parse_error",
        ),
    )
    order = _load_unique_minimal(
        order_paths,
        kind="order-check",
        fields=(
            "decision", "metric_id", "confidence", "reason", "prompt_sha256",
            "order_mode", "candidate_ids", "candidate_bank_source_sha256", "model",
        ),
    )
    verification = _load_unique_minimal(
        verification_paths,
        kind="verification",
        fields=(
            "decision", "metric_id", "confidence", "reason", "prompt_sha256",
            "primary_metric_id", "primary_prompt_sha256",
            "candidate_bank_source_sha256", "model",
            "schema_version", "verification_orders",
            "strict_two_order_acceptance", "strict_all_order_acceptance",
            "accepted_by_order", "verifier_selection_sha256",
            "verifier_policy_sha256", "production_plan_sha256",
        ),
    )
    primary_uids = set(primary)
    missing = expected_uids - primary_uids
    extra = primary_uids - expected_uids
    if missing or extra:
        raise ValueError(
            f"primary coverage mismatch for {args.corpus}: missing={len(missing)}, "
            f"extra={len(extra)}, sample_missing={sorted(missing)[:3]}, "
            f"sample_extra={sorted(extra)[:3]}"
        )
    unknown_order = set(order) - expected_uids
    unknown_verification = set(verification) - expected_uids
    if unknown_order or unknown_verification:
        raise ValueError(
            f"verification inputs contain unknown UIDs: order={len(unknown_order)}, "
            f"contrastive={len(unknown_verification)}"
        )

    adjudicator_selection_path = (
        Path(args.adjudicator_selection) if args.adjudicator_selection else None
    )
    verifier_selection_path = (
        Path(args.verifier_selection) if args.verifier_selection else None
    )
    verifier_policy_path = Path(args.verifier_policy) if args.verifier_policy else None
    production_plan_path = (
        Path(args.production_plan) if args.production_plan else None
    )
    if args.strict_production and (
        adjudicator_selection_path is None
        or verifier_selection_path is None
        or verifier_policy_path is None
        or production_plan_path is None
    ):
        raise ValueError(
            "--strict-production requires the production plan, adjudicator/verifier "
            "selections, and verifier policy"
        )
    expected_adjudicator_prompt = None
    adjudicator_selection = None
    if adjudicator_selection_path:
        expected_adjudicator_prompt, adjudicator_selection = selected_prompt_sha(
            adjudicator_selection_path, task, "adjudicator"
        )
    expected_verifier_prompt = None
    verifier_selection = None
    if verifier_selection_path:
        expected_verifier_prompt, verifier_selection = selected_prompt_sha(
            verifier_selection_path, task, "verifier"
        )
    verifier_policy = None
    expected_verifier_orders: list[str] | None = None
    if verifier_policy_path:
        verifier_policy = json.loads(verifier_policy_path.read_text(encoding="utf-8"))
        selection_ref = (verifier_policy.get("inputs") or {}).get("selection") or {}
        if (
            verifier_policy.get("task") != task
            or selection_ref.get("sha256")
            != sha256_file(verifier_selection_path)  # type: ignore[arg-type]
            or verifier_policy.get("may_run_on_production_unlabeled_norms") is not True
            or (verifier_policy.get("dev_gate") or {}).get("cleared") is not True
        ):
            raise ValueError("verifier policy is not task-matched, linked, and dev-cleared")
        expected_verifier_orders = [
            str(value)
            for value in (verifier_policy.get("order_policy") or {}).get("orders") or []
        ]
        if expected_verifier_orders not in (
            ["original", "hashed"],
            ["original", "hashed", "reverse"],
        ):
            raise ValueError("verifier policy has an unsupported order topology")
    production_plan = None
    if production_plan_path:
        production_plan = json.loads(
            production_plan_path.read_text(encoding="utf-8")
        )
        if production_plan.get("schema_version") in {
            "silver-match-v3-task-production-plan-v1",
            "silver-match-v3-task-production-plan-v2",
        }:
            _check_plan_artifacts(production_plan)
        if (
            production_plan.get("status")
            != "FROZEN_READY_FOR_UNLABELED_PRODUCTION"
            or production_plan.get("task") != task
            or production_plan["manifest"]["sha256"] != sha256_file(manifest_path)
            or production_plan["bank_source_sha256"] != bank_meta["source_sha256"]
            or production_plan["adjudicator"]["selection"]["sha256"]
            != sha256_file(adjudicator_selection_path)  # type: ignore[arg-type]
            or production_plan["verifier"]["selection"]["sha256"]
            != sha256_file(verifier_selection_path)  # type: ignore[arg-type]
            or production_plan["verifier"]["production_policy"]["sha256"]
            != sha256_file(verifier_policy_path)  # type: ignore[arg-type]
        ):
            raise ValueError("production plan is not linked to the finalization inputs")
        if (
            expected_verifier_orders is not None
            and production_plan["verifier"].get("orders") != expected_verifier_orders
        ):
            raise ValueError("production plan and verifier policy order topology differ")

    if args.strict_production:
        if set(order) != expected_uids:
            raise ValueError(
                f"strict order-check coverage mismatch: expected={len(expected_uids)} "
                f"observed={len(order)}"
            )
        expected_verification = {
            uid for uid, row in primary.items() if row.get("decision") == "MATCH"
        }
        if set(verification) != expected_verification:
            raise ValueError(
                f"strict verifier coverage mismatch: expected={len(expected_verification)} "
                f"observed={len(verification)}"
            )

    output_rows = []
    decision_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for canonical in expected_rows:
        uid = str(canonical["norm_uid"])
        row = primary[uid]
        if row.get("corpus") != args.corpus or row.get("task") != task:
            raise ValueError(f"primary routing mismatch for {uid}")
        decision = str(row.get("decision") or "")
        if decision not in FINAL_DECISIONS:
            raise ValueError(f"unknown primary decision for {uid}: {decision!r}")
        candidate_sha = str(row.get("candidate_bank_source_sha256") or "")
        if candidate_sha != str(bank_meta["source_sha256"]):
            raise ValueError(f"bank provenance mismatch for {uid}")
        proposed_metric = row.get("metric_id")
        if proposed_metric is not None and str(proposed_metric) not in bank_ids:
            raise ValueError(f"primary metric outside bank for {uid}: {proposed_metric}")
        order_row = order.get(uid)
        if order_row is not None:
            if order_row.get("prompt_sha256") != row.get("prompt_sha256"):
                raise ValueError(f"two-order prompt mismatch for {uid}")
            if order_row.get("model") != row.get("model"):
                raise ValueError(f"two-order model mismatch for {uid}")
            if order_row.get("order_mode") == row.get("order_mode"):
                raise ValueError(f"two-order runs reused the same order for {uid}")
            primary_ids = [str(value) for value in row.get("candidate_ids") or []]
            order_ids = [str(value) for value in order_row.get("candidate_ids") or []]
            if len(primary_ids) != len(order_ids) or set(primary_ids) != set(order_ids):
                raise ValueError(f"two-order candidate-slate mismatch for {uid}")
            if str(order_row.get("candidate_bank_source_sha256") or "") != str(
                bank_meta["source_sha256"]
            ):
                raise ValueError(f"order-check bank provenance mismatch for {uid}")
            if args.strict_production and {
                str(row.get("order_mode")), str(order_row.get("order_mode"))
            } != {"original", "hashed"}:
                raise ValueError(f"strict production requires original+hashed orders: {uid}")
            if args.strict_production:
                depth = int(
                    (production_plan or {}).get("adjudicator", {}).get(
                        "candidate_depth",
                        (adjudicator_selection or {}).get("candidate_depth", -1),
                    )
                )
                if len(primary_ids) != depth:
                    raise ValueError(
                        f"candidate depth differs from adjudicator selection for {uid}"
                    )
        if expected_adjudicator_prompt and row.get("prompt_sha256") != expected_adjudicator_prompt:
            raise ValueError(f"primary prompt differs from dev selection for {uid}")
        verification_row = verification.get(uid)
        if verification_row is not None:
            if verification_row.get("primary_prompt_sha256") != row.get("prompt_sha256"):
                raise ValueError(f"verifier primary-prompt provenance mismatch for {uid}")
            if str(verification_row.get("candidate_bank_source_sha256") or "") != str(
                bank_meta["source_sha256"]
            ):
                raise ValueError(f"verifier bank provenance mismatch for {uid}")
            if expected_verifier_prompt and verification_row.get(
                "prompt_sha256"
            ) != expected_verifier_prompt:
                raise ValueError(f"verifier prompt differs from dev selection for {uid}")
            if args.strict_production:
                schema = verification_row.get("schema_version")
                if schema == "silver-match-v3-two-order-production-verification-v1":
                    if expected_verifier_orders != ["original", "hashed"]:
                        raise ValueError(f"legacy two-order row weakens selected policy: {uid}")
                    acceptance_flag = verification_row.get(
                        "strict_two_order_acceptance"
                    )
                elif schema == "silver-match-v3-multi-order-production-verification-v1":
                    acceptance_flag = verification_row.get(
                        "strict_all_order_acceptance"
                    )
                    accepted_by_order = verification_row.get("accepted_by_order")
                    if (
                        not isinstance(accepted_by_order, dict)
                        or list(accepted_by_order) != expected_verifier_orders
                        or not all(isinstance(value, bool) for value in accepted_by_order.values())
                    ):
                        raise ValueError(f"multi-order acceptance evidence is malformed: {uid}")
                else:
                    raise ValueError(f"unsupported frozen verifier row schema: {uid}")
                if (
                    verification_row.get("verification_orders")
                    != expected_verifier_orders
                    or verification_row.get("verifier_selection_sha256")
                    != sha256_file(verifier_selection_path)  # type: ignore[arg-type]
                    or verification_row.get("verifier_policy_sha256")
                    != sha256_file(verifier_policy_path)  # type: ignore[arg-type]
                    or verification_row.get("production_plan_sha256")
                    != sha256_file(production_plan_path)  # type: ignore[arg-type]
                ):
                    raise ValueError(f"verifier row lacks frozen all-order provenance: {uid}")
                is_confirm = verification_row.get("decision") == "CONFIRM_MATCH"
                if is_confirm != bool(acceptance_flag):
                    raise ValueError(f"verifier acceptance flag mismatch: {uid}")
                if schema == "silver-match-v3-multi-order-production-verification-v1" and (
                    is_confirm != all(accepted_by_order.values())
                ):
                    raise ValueError(f"per-order acceptance evidence differs: {uid}")
                if is_confirm and verification_row.get("confidence") != "high":
                    raise ValueError(f"strict verifier confirmation is not high-confidence: {uid}")
        final_decision, final_metric, status = final_match_decision(
            row, order_row, verification_row
        )
        if final_decision not in FINAL_DECISIONS:
            raise ValueError(f"invalid final decision for {uid}: {final_decision}")
        decision_counts[final_decision] += 1
        status_counts[status] += 1
        output_rows.append(
            {
                "schema_version": manifest["schema_version"],
                "norm_uid": uid,
                "corpus": args.corpus,
                "task": task,
                "row": canonical["row"],
                "decision": final_decision,
                "metric_id": final_metric,
                "confidence": row.get("confidence") if final_decision != "UNSTABLE_MATCH" else "low",
                "reason": row.get("reason") if final_decision != "UNSTABLE_MATCH" else status,
                "verification_status": status,
                "proposed_metric_id": proposed_metric,
                "candidate_ids": row.get("candidate_ids") or [],
                "bank_source_sha256": bank_meta["source_sha256"],
                "primary": {
                    "decision": row.get("decision"),
                    "metric_id": row.get("metric_id"),
                    "confidence": row.get("confidence"),
                    "reason": row.get("reason"),
                    "prompt_sha256": row.get("prompt_sha256"),
                    "model": row.get("model"),
                    "order_mode": row.get("order_mode"),
                    "parse_error": row.get("parse_error"),
                },
                "order_check": order.get(uid),
                "contrastive_verification": verification.get(uid),
            }
        )

    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(output_path)
    write_jsonl(output_path, output_rows)
    report = {
        "schema_version": manifest["schema_version"],
        "corpus": args.corpus,
        "task": task,
        "count": len(output_rows),
        "expected_count": corpus_meta["count"],
        "complete": len(output_rows) == corpus_meta["count"],
        "decision_counts": dict(sorted(decision_counts.items())),
        "verification_status_counts": dict(sorted(status_counts.items())),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "bank_source_sha256": bank_meta["source_sha256"],
        "primary_inputs": {str(path): sha256_file(path) for path in primary_paths},
        "order_check_inputs": {str(path): sha256_file(path) for path in order_paths},
        "verification_inputs": {str(path): sha256_file(path) for path in verification_paths},
        "strict_production": bool(args.strict_production),
        "adjudicator_selection": (
            {
                "path": str(adjudicator_selection_path),
                "sha256": sha256_file(adjudicator_selection_path),
                "prompt_sha256": expected_adjudicator_prompt,
            }
            if adjudicator_selection_path
            else None
        ),
        "verifier_selection": (
            {
                "path": str(verifier_selection_path),
                "sha256": sha256_file(verifier_selection_path),
                "prompt_sha256": expected_verifier_prompt,
            }
            if verifier_selection_path
            else None
        ),
        "verifier_policy": (
            {
                "path": str(verifier_policy_path),
                "sha256": sha256_file(verifier_policy_path),
            }
            if verifier_policy_path
            else None
        ),
        "production_plan": (
            {
                "path": str(production_plan_path),
                "sha256": sha256_file(production_plan_path),
            }
            if production_plan_path
            else None
        ),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    if not report["complete"]:
        raise RuntimeError(f"final count mismatch: {report}")
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifest.json"))
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--primary", nargs="+", required=True)
    parser.add_argument("--order-check", nargs="+", required=True)
    parser.add_argument("--verification", nargs="+", required=True)
    parser.add_argument("--adjudicator-selection")
    parser.add_argument("--verifier-selection")
    parser.add_argument("--verifier-policy")
    parser.add_argument("--production-plan")
    parser.add_argument("--strict-production", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
