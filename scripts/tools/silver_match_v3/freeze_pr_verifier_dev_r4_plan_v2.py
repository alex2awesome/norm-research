#!/usr/bin/env python3
"""Freeze a fresh PR R4 plan after an output-interface-only repair."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .common import sha256_file


OLD_PROMPT_SHA = "664f16dc6f459531fd1bcec98cd06130625ea28da6d4da2e042eb67e8db7d9c7"
OLD_PLAN_SHA = "05b6be3f58cd4a576ee1773e25cad2c3ed143ba53edcc9ae7c540d5b6d7448f6"
NEW_SEED = 2026071321
NEW_MAX_TOKENS = 512
ORDERS = ("original", "hashed", "reverse")


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def freeze(args: argparse.Namespace) -> dict:
    failed_plan_path = Path(args.failed_plan).resolve()
    failure_audit_path = Path(args.failure_audit).resolve()
    repaired_prompt_path = Path(args.repaired_prompt).resolve()
    repair_audit_path = Path(args.repair_audit).resolve()
    output_root = Path(args.output_root).resolve()
    diff_audit_path = Path(args.diff_audit).resolve()
    output_path = Path(args.output).resolve()
    if output_root.exists() or diff_audit_path.exists() or output_path.exists():
        raise FileExistsError("refusing to overwrite or reuse a repaired R4 run")

    if sha256_file(failed_plan_path) != OLD_PLAN_SHA:
        raise ValueError("unexpected failed-plan identity")
    failed = json.loads(failed_plan_path.read_text(encoding="utf-8"))
    failure = json.loads(failure_audit_path.read_text(encoding="utf-8"))
    repair = json.loads(repair_audit_path.read_text(encoding="utf-8"))
    repaired_prompt_sha = sha256_file(repaired_prompt_path)

    if (
        failure.get("status") != "REJECTED_INTERFACE_CONTRACT_BEFORE_TRUTH_JOIN"
        or (failure.get("plan") or {}).get("sha256") != OLD_PLAN_SHA
        or int((failure.get("partial_output") or {}).get("count") or -1) != 288
        or int((failure.get("partial_output") or {}).get("invalid_count") or -1)
        != 288
        or (failure.get("contracts") or {}).get("truth_or_truth_predictions_read")
        is not False
        or (failure.get("contracts") or {}).get("partial_output_must_never_be_reused")
        is not True
    ):
        raise ValueError("failed run is not quarantined by the expected audit")
    checks = repair.get("machine_checks") or {}
    if (
        repair.get("status") != "INTERFACE_ONLY_REPAIR_BEFORE_TRUTH_JOIN"
        or (repair.get("source") or {}).get("sha256") != OLD_PROMPT_SHA
        or (repair.get("output") or {}).get("sha256") != repaired_prompt_sha
        or repair.get("truth_or_predictions_read") is not False
        or checks.get("reverse_substitution_recovers_source_bytes") is not True
        or checks.get("only_changed_lines") != [6]
        or checks.get("decision_vocabulary_unchanged") is not True
        or checks.get("semantic_adjudication_instructions_unchanged") is not True
    ):
        raise ValueError("prompt repair is not the accepted interface-only delta")

    old_rendering = failed.get("rendering") or {}
    if (
        failed.get("status") != "FROZEN_BEFORE_DEV_PROPOSAL_INFERENCE"
        or failed.get("orders") != list(ORDERS)
        or int(failed.get("row_count") or -1) != 300
        or int(failed.get("candidate_depth") or -1) != 50
        or int(old_rendering.get("seed") or -1) != 2026071311
        or int(old_rendering.get("max_tokens") or -1) != 220
        or ((failed.get("inputs") or {}).get("adjudicator_prompt") or {}).get(
            "sha256"
        )
        != OLD_PROMPT_SHA
    ):
        raise ValueError("failed plan differs from the expected pre-truth plan")

    output_root.mkdir(parents=True, exist_ok=False)
    outputs = {order: str(output_root / f"{order}.jsonl") for order in ORDERS}
    old_outputs = {str(Path(value).resolve()) for value in failed["outputs"].values()}
    if old_outputs & {str(Path(value).resolve()) for value in outputs.values()}:
        raise ValueError("new outputs overlap the quarantined run")

    new_rendering = copy.deepcopy(old_rendering)
    new_rendering["seed"] = NEW_SEED
    new_rendering["max_tokens"] = NEW_MAX_TOKENS
    diff_audit = {
        "schema_version": "silver-match-v3-pr-r4-repaired-plan-diff-audit-v2",
        "status": "FROZEN_AND_MAIN_JUDGE_ACCEPTED_BEFORE_INFERENCE_OR_TRUTH_JOIN",
        "task": "press-releases",
        "main_judge_verdict": "ACCEPT_INTERFACE_AND_RUNTIME_CONTRACT_REPAIR",
        "failed_plan": _artifact(failed_plan_path),
        "failure_audit": _artifact(failure_audit_path),
        "prompt_repair_audit": _artifact(repair_audit_path),
        "allowed_deltas": {
            "prompt": {
                "old_sha256": OLD_PROMPT_SHA,
                "new_sha256": repaired_prompt_sha,
                "surface": "line 6 confidence output type only",
            },
            "max_tokens": {"old": 220, "new": NEW_MAX_TOKENS},
            "seed": {"old": 2026071311, "new": NEW_SEED},
            "output_root": {
                "old": str(Path(next(iter(failed["outputs"].values()))).parent),
                "new": str(output_root),
            },
        },
        "unchanged_contracts": {
            "adjudication_semantics": True,
            "decision_vocabulary": True,
            "temperature_zero": True,
            "orders": list(ORDERS),
            "model": failed["model"],
            "candidate_subset": failed["inputs"]["candidate_subset"],
            "candidate_depth": 50,
            "row_count": 300,
            "runner": failed["inputs"]["runner"],
            "all_other_rendering": {
                key: value
                for key, value in old_rendering.items()
                if key not in {"seed", "max_tokens"}
            },
        },
        "contracts": {
            "quarantined_rows_reused": False,
            "truth_or_truth_predictions_read": False,
            "candidate_slate_or_identity_changed": False,
        },
    }
    diff_audit_path.parent.mkdir(parents=True, exist_ok=True)
    diff_audit_path.write_text(
        json.dumps(diff_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    inputs = copy.deepcopy(failed["inputs"])
    inputs["source_adjudicator_prompt"] = inputs.pop("adjudicator_prompt")
    inputs["source_adjudicator_prompt_meta"] = inputs.pop(
        "adjudicator_prompt_meta"
    )
    inputs["adjudicator_prompt"] = _artifact(repaired_prompt_path)
    inputs["prompt_repair_audit"] = _artifact(repair_audit_path)
    inputs["rejected_interface_run_audit"] = _artifact(failure_audit_path)
    inputs["plan_diff_audit"] = _artifact(diff_audit_path)
    plan = {
        "schema_version": "silver-match-v3-pr-verifier-dev-r4-proposal-plan-v2",
        "status": "FROZEN_BEFORE_REPAIRED_DEV_PROPOSAL_INFERENCE",
        "task": failed["task"],
        "role": failed["role"],
        "row_count": failed["row_count"],
        "candidate_depth": failed["candidate_depth"],
        "orders": list(ORDERS),
        "model": failed["model"],
        "rendering": new_rendering,
        "inputs": inputs,
        "outputs": outputs,
        "contracts": {
            **failed["contracts"],
            "quarantined_v1_rows_reused": False,
            "truth_or_truth_predictions_read_by_repair": False,
            "only_interface_runtime_seed_and_output_path_changed": True,
        },
    }
    output_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        **plan,
        "plan_sha256": sha256_file(output_path),
        "diff_audit_sha256": sha256_file(diff_audit_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "failed-plan",
        "failure-audit",
        "repaired-prompt",
        "repair-audit",
        "output-root",
        "diff-audit",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True)
    print(json.dumps(freeze(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
