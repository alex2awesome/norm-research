#!/usr/bin/env python3
"""Seal the exact verifier policy selected by explicit-role GEPA.

This is intentionally a post-selection, pre-production operation.  It replays
the frozen gate, binds the selected prompt components and the exact prompt-dev
runtime metadata, and maps the selected policy name to its immutable order set.
It consumes no test, blind-audit, production, outcome, or MI information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import sha256_file


PLAN_SCHEMA = "silver-match-v3-explicit-role-task-local-gepa-plan-v1"
FREEZE_SCHEMA = "silver-match-v3-explicit-role-task-local-gepa-freeze-v1"
ADJ_SCHEMA = "silver-match-v3-explicit-role-adjudicator-selection-v1"
VERIFIER_SCHEMA = "silver-match-v3-explicit-role-verifier-selection-v1"
POLICY_FREEZE_SCHEMA = "silver-match-v3-explicit-role-verifier-selection-freeze-v1"
POLICY_ORDERS = {
    "two_order_exact_high": ["original", "hashed"],
    "all_three_order_exact_high": ["original", "hashed", "reverse"],
}
RENDERING_KEYS = (
    "model",
    "max_alternatives",
    "batch_size",
    "max_model_len",
    "max_tokens",
    "gpu_memory_utilization",
    "enforce_eager",
    "seed",
    "context_chars",
    "description_chars",
    "example_chars",
    "max_examples",
)


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _validate_ref(ref: dict[str, Any], label: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    if not path.is_file() or sha256_file(path) != str(ref.get("sha256") or ""):
        raise ValueError(f"{label} is missing or hash-drifted: {path}")
    return path


def _arg(argv: list[str], flag: str) -> str:
    if argv.count(flag) != 1:
        raise ValueError(f"frozen command must contain exactly one {flag}")
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise ValueError(f"frozen command lacks value for {flag}")
    return str(argv[index + 1])


def seal_policy(
    *,
    plan_path: Path,
    role_freeze_path: Path,
    adjudicator_selection_path: Path,
    verifier_selection_path: Path,
) -> dict[str, Any]:
    paths = {
        "command_plan": plan_path.resolve(),
        "role_freeze": role_freeze_path.resolve(),
        "adjudicator_selection": adjudicator_selection_path.resolve(),
        "selection": verifier_selection_path.resolve(),
    }
    plan = json.loads(paths["command_plan"].read_text(encoding="utf-8"))
    role_freeze = json.loads(paths["role_freeze"].read_text(encoding="utf-8"))
    adjudicator = json.loads(paths["adjudicator_selection"].read_text(encoding="utf-8"))
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    plan_sha = sha256_file(paths["command_plan"])
    task = str(plan.get("task") or "")
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "FROZEN_BEFORE_TASK_LOCAL_GEPA_INFERENCE"
        or not task
        or role_freeze.get("schema_version") != FREEZE_SCHEMA
        or role_freeze.get("task") != task
        or (role_freeze.get("command_plan") or {}).get("sha256") != plan_sha
        or role_freeze.get("test_or_blind_audit_consumed") is not False
        or role_freeze.get("production_consumed") is not False
    ):
        raise ValueError("explicit-role plan/FREEZE is not clean and hash-linked")
    if (
        adjudicator.get("schema_version") != ADJ_SCHEMA
        or adjudicator.get("task") != task
        or adjudicator.get("status") != "selected"
        or adjudicator.get("selection_role") != "prompt_dev"
        or adjudicator.get("test_or_blind_audit_consumed") is not False
        or adjudicator.get("production_consumed") is not False
        or adjudicator.get("outcomes_or_mi_used") is not False
        or (adjudicator.get("inputs") or {}).get("command_plan", {}).get("sha256")
        != plan_sha
    ):
        raise ValueError("adjudicator selection is not clean prompt-dev selection")
    if (
        selection.get("schema_version") != VERIFIER_SCHEMA
        or selection.get("task") != task
        or selection.get("status") != "selected"
        or selection.get("selection_role") != "prompt_dev"
        or selection.get("test_or_blind_audit_consumed") is not False
        or selection.get("production_consumed") is not False
        or selection.get("outcomes_or_mi_used") is not False
        or (selection.get("inputs") or {}).get("plan", {}).get("sha256") != plan_sha
        or (selection.get("inputs") or {}).get("role_freeze", {}).get("sha256")
        != sha256_file(paths["role_freeze"])
        or (selection.get("inputs") or {})
        .get("adjudicator_selection", {})
        .get("sha256")
        != sha256_file(paths["adjudicator_selection"])
    ):
        raise ValueError("verifier selection is not clean prompt-dev selection")
    for label, ref in (selection.get("inputs") or {}).items():
        _validate_ref(ref, f"selection input {label}")

    policy_freeze_path = _validate_ref(
        (selection.get("inputs") or {}).get("policy_freeze") or {},
        "verifier policy freeze",
    )
    policy_freeze = json.loads(policy_freeze_path.read_text(encoding="utf-8"))
    if (
        policy_freeze.get("schema_version") != POLICY_FREEZE_SCHEMA
        or policy_freeze.get("status")
        != "FROZEN_BEFORE_ANY_VERIFIER_INFERENCE_OR_SCORE"
        or policy_freeze.get("task") != task
        or policy_freeze.get("test_or_blind_audit_consumed") is not False
        or policy_freeze.get("production_consumed") is not False
        or policy_freeze.get("outcomes_or_mi_used") is not False
        or (policy_freeze.get("inputs") or {}).get("command_plan", {}).get("sha256")
        != plan_sha
    ):
        raise ValueError("verifier policy universe was not cleanly pre-frozen")

    chosen = selection.get("chosen") or {}
    policy_name = str(chosen.get("policy") or "")
    orders = POLICY_ORDERS.get(policy_name)
    prompt_sha = str(chosen.get("verifier_prompt_sha256") or "")
    adj_variant = str(chosen.get("adjudicator_variant") or "")
    verifier_variant = str(chosen.get("verifier_variant") or "")
    if chosen.get("eligible") is not True or orders is None or len(prompt_sha) != 64:
        raise ValueError("selected verifier candidate is not production-eligible")
    if adj_variant != str((adjudicator.get("chosen") or {}).get("name") or ""):
        raise ValueError("verifier selection does not use selected adjudicator branch")

    thresholds = selection.get("thresholds") or {}
    if thresholds != {
        key: (plan.get("thresholds") or {}).get(key)
        for key in (
            "minimum_point_precision",
            "minimum_retained",
            "minimum_wilson_95_lower",
        )
    }:
        raise ValueError("selection thresholds differ from frozen command plan")
    observed = chosen.get("select_metrics") or {}
    interval = observed.get("retained_precision_wilson_95")
    point = observed.get("retained_precision")
    retained = int(observed.get("retained") or 0)
    cleared = bool(
        point is not None
        and isinstance(interval, list)
        and len(interval) == 2
        and float(point) >= float(thresholds["minimum_point_precision"])
        and float(interval[0]) >= float(thresholds["minimum_wilson_95_lower"])
        and retained >= int(thresholds["minimum_retained"])
    )
    if not cleared:
        raise ValueError("selected verifier does not replay the predeclared dev gate")
    for label in ("optimize_score", "select_score"):
        _validate_ref(chosen.get(label) or {}, f"chosen verifier {label}")

    variants = {
        str(row.get("name") or ""): row for row in plan.get("verifier_variants") or []
    }
    variant = variants.get(verifier_variant) or {}
    if str(variant.get("combined_prompt_sha256") or "") != prompt_sha:
        raise ValueError("selected verifier prompt differs from frozen variant")
    components: list[dict[str, str]] = []
    for raw in variant.get("components") or []:
        path = _validate_ref(raw, "verifier prompt component")
        components.append({"path": str(path), "sha256": sha256_file(path)})
    if not components:
        raise ValueError("selected verifier has no prompt components")

    cells: dict[str, dict[str, Any]] = {}
    for cell in plan.get("commands") or []:
        if (
            cell.get("stage") != "verifier"
            or cell.get("role") != "prompt_dev"
            or cell.get("adjudicator_variant") != adj_variant
            or cell.get("verifier_variant") != verifier_variant
        ):
            continue
        order = str(cell.get("order") or "")
        if order in cells:
            raise ValueError(f"duplicate verifier prompt-dev inference cell: {order}")
        cells[order] = cell
    if set(cells) != {"original", "hashed", "reverse"}:
        raise ValueError("selected branch lacks the frozen three-order verifier universe")

    runtime: dict[str, Any] | None = None
    dev_runs: dict[str, Any] = {}
    input_candidates_sha: str | None = None
    primary_sha: str | None = None
    for order in orders:
        cell = cells[order]
        direct = cell.get("direct_batch_command") or {}
        if direct.get("module") != "scripts.tools.silver_match_v3.verify_gemma":
            raise ValueError(f"{order} verifier was not frozen for direct batch execution")
        argv = [str(value) for value in direct.get("argv") or []]
        if _arg(argv, "--order-mode") != order:
            raise ValueError(f"{order} frozen command order differs")
        output = Path(_arg(argv, "--output")).resolve()
        meta_path = output.with_suffix(output.suffix + ".meta.json")
        if not output.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"selected {order} verifier dev output is incomplete")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        actual_runtime = {key: meta.get(key) for key in RENDERING_KEYS}
        if runtime is None:
            runtime = actual_runtime
            input_candidates_sha = str(meta.get("input_candidates_sha256") or "")
            primary_sha = str(meta.get("primary_sha256") or "")
        if (
            actual_runtime != runtime
            or meta.get("order_mode") != order
            or meta.get("output_sha256") != sha256_file(output)
            or meta.get("prompt_sha256") != prompt_sha
            or str(meta.get("input_candidates_sha256") or "") != input_candidates_sha
            or str(meta.get("primary_sha256") or "") != primary_sha
            or int(meta.get("invalid_count", -1)) != 0
            or (meta.get("prompt_component_sha256") or {})
            != {row["path"]: row["sha256"] for row in components}
            or _arg(argv, "--model") != str(meta.get("model") or "")
            or int(_arg(argv, "--max-alternatives"))
            != int(meta.get("max_alternatives", -1))
        ):
            raise ValueError(f"selected {order} verifier dev runtime differs from plan")
        dev_runs[order] = {
            "output": _artifact(output),
            "meta": _artifact(meta_path),
            "frozen_direct_command": direct,
        }
    assert runtime is not None

    return {
        "schema_version": "silver-match-v3-verifier-production-policy-v2",
        "task": task,
        "status": "prompt_dev_supported_requires_blind_final_match_audit",
        "selection_role": "prompt_dev",
        "test_or_blind_audit_consumed": False,
        "production_consumed": False,
        "outcomes_or_mi_used": False,
        "may_run_on_production_unlabeled_norms": True,
        "may_be_used_for_gradient_labels_before_blind_audit": False,
        "selected_policy": policy_name,
        "selected_adjudicator_variant": adj_variant,
        "selected_verifier_variant": verifier_variant,
        "prompt": {
            "base_path": components[0]["path"],
            "base_sha256": components[0]["sha256"],
            "addon_paths": [row["path"] for row in components[1:]],
            "addon_sha256": {
                row["path"]: row["sha256"] for row in components[1:]
            },
            "components": components,
            "rendered_prompt_sha256": prompt_sha,
        },
        "rendering": runtime,
        "order_policy": {
            "orders": orders,
            "acceptance_mode": "all_orders_exact_high_same_id_no_parse_error",
            "retain_only_if": (
                "every frozen order returns CONFIRM_MATCH for the identical proposed "
                "metric_id with high confidence and no parse error"
            ),
            "corrections_are_retained": False,
            "all_disagreement_or_abstention_is_dropped": True,
        },
        "dev_gate": {
            "minimum_point_precision": thresholds["minimum_point_precision"],
            "minimum_wilson_lower": thresholds["minimum_wilson_95_lower"],
            "minimum_retained": thresholds["minimum_retained"],
            "observed": observed,
            "cleared": True,
        },
        "independent_audit_requirement": {
            "required": True,
            "scope": "blind stratified sample of final retained production MATCH labels",
            "promotion_blocked_until_pass": True,
        },
        "selected_prompt_dev_runs": dev_runs,
        "inputs": {name: _artifact(path) for name, path in paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--role-freeze")
    parser.add_argument("--adjudicator-selection", required=True)
    parser.add_argument("--verifier-selection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = seal_policy(
        plan_path=plan_path,
        role_freeze_path=(
            Path(args.role_freeze).resolve()
            if args.role_freeze
            else plan_path.with_name("FREEZE.json")
        ),
        adjudicator_selection_path=Path(args.adjudicator_selection).resolve(),
        verifier_selection_path=Path(args.verifier_selection).resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "output_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
