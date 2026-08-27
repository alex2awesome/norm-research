#!/usr/bin/env python3
"""Fail closed on a frozen advisory full-bank direct-batch model-vote queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


def _resolve(entry: dict[str, Any], repo_root: Path) -> Path:
    path = Path(str(entry.get("path") or ""))
    path = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    if not path.is_file() or sha256_file(path) != entry.get("sha256"):
        raise ValueError(f"missing or hash-mismatched artifact: {path}")
    return path


def audit(queue_path: Path, repo_root: Path) -> dict[str, Any]:
    queue_path, repo_root = queue_path.resolve(), repo_root.resolve()
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    contract = queue.get("scientific_contract") or {}
    if (
        queue.get("schema_version")
        != "silver-match-v3-pr-verifier-dev-gemma4-advisory-full-bank-queue-v1"
        or queue.get("status") != "FROZEN_QUEUED_WAITING_FOR_PROJECT_GPU_SLOT"
        or queue.get("task") != "press-releases"
        or queue.get("role") != "independent_advisory_model_vote"
        or contract.get("gemma_vote_is_advisory_and_diagnostic_only") is not True
        or contract.get("gemma_vote_may_not_select_its_own_verifier") is not True
        or contract.get("gemma_vote_may_not_outvote_replace_or_resolve_codex_truth")
        is not True
        or contract.get("gemma_may_corroborate_codex_truth_only") is not True
        or contract.get("permanently_excluded_from_retriever_verifier_and_prompt_gradients")
        is not True
    ):
        raise ValueError("queue role or advisory-only scientific contract is invalid")

    inputs = queue.get("inputs") or {}
    paths = {
        name: _resolve(inputs[name], repo_root)
        for name in (
            "manifest",
            "verifier_dev_policy",
            "permuted_pack_validation",
            "candidate_freeze",
            "candidates",
            "prompt",
            "runner",
            "model_inventory",
        )
    }
    independence = queue.get("independence") or {}
    audits = [
        _resolve(independence[name], repo_root)
        for name in ("pass_a_vs_gemma_audit", "pass_b_vs_gemma_audit")
    ]
    for path in audits:
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("status")
            != "FROZEN_MUTUALLY_PREDICTION_HIDDEN_BEFORE_LABELING"
            or report.get("same_uid_set") is not True
            or report.get("same_canonical_item_content_by_uid") is not True
            or report.get("same_bank_leaf_set") is not True
            or report.get("candidate_proposals_exposed_to_either_pass") is not False
            or report.get("prior_truth_or_predictions_exposed_to_either_pass") is not False
            or report.get("pass_predictions_mutually_visible") is not False
        ):
            raise ValueError(f"independence audit is invalid: {path}")

    validation = json.loads(paths["permuted_pack_validation"].read_text(encoding="utf-8"))
    candidate_freeze = json.loads(paths["candidate_freeze"].read_text(encoding="utf-8"))
    if (
        validation.get("truth_hidden") is not True
        or int(validation.get("seed", -1)) != int(independence.get("permutation_seed", -2))
        or candidate_freeze.get("status") != "FROZEN_BEFORE_INFERENCE"
        or candidate_freeze.get("truth_hidden") is not True
        or candidate_freeze.get(
            "prior_decisions_metric_ids_predictions_and_proposals_read"
        )
        is not False
        or candidate_freeze.get("output", {}).get("sha256")
        != sha256_file(paths["candidates"])
    ):
        raise ValueError("permuted pack or candidate freeze is invalid")

    bank_path = Path(validation["outputs"]["bank"]["path"]).resolve()
    if (
        not bank_path.is_file()
        or sha256_file(bank_path) != validation["outputs"]["bank"]["sha256"]
    ):
        raise ValueError("permuted bank is missing or hash-mismatched")
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_ids = [str(row["metric_id"]) for row in bank.get("metrics") or []]
    expected_count = int(contract.get("fresh_identity_count", -1))
    expected_depth = int(contract.get("candidate_depth", -1))
    if len(bank_ids) != expected_depth or len(bank_ids) != len(set(bank_ids)):
        raise ValueError("full bank leaf count/uniqueness mismatch")
    rows = list(read_jsonl(paths["candidates"]))
    uids = [str(row.get("norm_uid") or "") for row in rows]
    if (
        len(rows) != expected_count
        or "" in uids
        or len(uids) != len(set(uids))
        or any(
            row.get("task") != queue["task"]
            or row.get("truth_hidden") is not True
            or row.get("prior_predictions_hidden") is not True
            or [str(card.get("metric_id")) for card in row.get("candidates") or []]
            != bank_ids
            for row in rows
        )
    ):
        raise ValueError("candidate rows are not an exact truth-hidden full-bank pack")

    inventory = json.loads(paths["model_inventory"].read_text(encoding="utf-8"))
    runtime = queue.get("runtime") or {}
    if (
        inventory.get("status") != "FROZEN_CONTENT_HASH_INVENTORY"
        or inventory.get("content_inventory_sha256")
        != inputs["model_inventory"].get("content_inventory_sha256")
        or Path(inventory.get("root") or "").resolve()
        != Path(runtime.get("model") or "").resolve()
    ):
        raise ValueError("model snapshot inventory binding is invalid")

    preflight = queue.get("prompt_token_preflight") or {}
    if (
        preflight.get("all_rows_fit") is not True
        or int(preflight.get("count", -1)) != expected_count
        or int(preflight.get("candidate_depth", -1)) != expected_depth
        or int(preflight.get("max_total_budget", 10**9))
        > int(preflight.get("max_model_len", -1))
        or int(preflight.get("max_model_len", -1)) != int(runtime.get("max_model_len", -2))
        or int(preflight.get("max_tokens", -1)) != int(runtime.get("max_tokens", -2))
    ):
        raise ValueError("prompt token-budget preflight is invalid")

    command = [str(token) for token in queue.get("command") or []]
    rendered = " ".join(command).lower()
    output_path = Path(str((queue.get("output") or {}).get("path") or "")).resolve()
    expected_module = "scripts.tools.silver_match_v3.adjudicate_gemma"
    if (
        len(command) < 5
        or command[:4] != [runtime["python"], "-u", "-m", expected_module]
        or any(term in rendered for term in ("openrouter", "_api", "server", "proposal"))
        or "--keep-raw" not in command
        or output_path.exists()
        or (queue.get("output") or {}).get("existed_before_freeze") is not False
    ):
        raise ValueError("command is not a clean unstarted direct-batch execution")

    def _arg(name: str) -> str:
        index = command.index(name)
        return command[index + 1]

    checks = {
        "--manifest": paths["manifest"],
        "--candidates": paths["candidates"],
        "--output": output_path,
        "--prompt": paths["prompt"],
        "--model": Path(runtime["model"]).resolve(),
    }
    if any(Path(_arg(flag)).resolve() != expected for flag, expected in checks.items()):
        raise ValueError("command artifact path differs from frozen queue binding")
    numeric = {
        "--max-candidates": "max_candidates",
        "--context-chars": "context_chars",
        "--description-chars": "description_chars",
        "--example-chars": "example_chars",
        "--max-examples": "max_examples",
        "--batch-size": "batch_size",
        "--max-model-len": "max_model_len",
        "--max-tokens": "max_tokens",
        "--seed": "seed",
    }
    if any(int(_arg(flag)) != int(runtime[key]) for flag, key in numeric.items()):
        raise ValueError("command numeric setting differs from frozen runtime")
    if _arg("--order-mode") != runtime["order_mode"]:
        raise ValueError("command order differs from frozen runtime")

    return {
        "schema_version": "silver-match-v3-advisory-full-bank-queue-audit-v1",
        "status": "EXACT_HASH_PINNED_ADVISORY_QUEUE_PASS_WAITING_FOR_GPU",
        "queue": {"path": str(queue_path), "sha256": sha256_file(queue_path)},
        "task": queue["task"],
        "role": queue["role"],
        "row_count": len(rows),
        "bank_metric_count": len(bank_ids),
        "truth_and_candidate_model_proposals_hidden": True,
        "independent_codex_truth_remains_decisive": True,
        "gemma_vote_is_advisory_only": True,
        "output_absent_before_launch": True,
        "all_bound_artifact_hashes_pass": True,
        "direct_batch_only": True,
        "prompt_token_budget_pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = audit(Path(args.queue), Path(args.repo_root))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**result, "audit_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
