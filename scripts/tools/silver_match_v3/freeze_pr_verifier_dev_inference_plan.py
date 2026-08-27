#!/usr/bin/env python3
"""Freeze the one-variant, two-order PR verifier-dev inference plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import sha256_file


POLICY_SCHEMA = "silver-match-v3-press-releases-verifier-dev-policy-amendment-v3"
BASE_POLICY_SCHEMA = "silver-match-v3-press-releases-verifier-dev-policy-v2"


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def freeze_plan(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        name: Path(getattr(args, name.replace("-", "_"))).resolve()
        for name in (
            "policy",
            "base-policy",
            "pair-freeze",
            "manifest",
            "prompt",
            "prompt-meta",
            "author-output-freeze",
            "model-inventory",
            "r4-proposal-plan",
            "runner",
        )
    }
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)

    policy = json.loads(paths["policy"].read_text(encoding="utf-8"))
    base = json.loads(paths["base-policy"].read_text(encoding="utf-8"))
    caps = base.get("gepa_and_selection_caps") or {}
    if (
        policy.get("schema_version") != POLICY_SCHEMA
        or base.get("schema_version") != BASE_POLICY_SCHEMA
        or policy.get("task") != args.task
        or base.get("task") != args.task
        or (policy.get("base_policy") or {}).get("sha256")
        != sha256_file(paths["base-policy"])
        or int(caps.get("maximum_authored_verifier_variants", -1)) < 1
        or int(caps.get("maximum_order_variants_per_verifier", -1)) < 2
        or int(caps.get("maximum_fresh_dev_batch_inference_runs", -1)) < 2
        or caps.get("no_prompt_edits_after_first_fresh_dev_score") is not True
    ):
        raise ValueError("unsupported or drifted verifier-dev policy/caps")

    pair_freeze = json.loads(paths["pair-freeze"].read_text(encoding="utf-8"))
    if (
        pair_freeze.get("schema_version")
        != "silver-match-v3-pr-verifier-dev-pair-universe-v1"
        or pair_freeze.get("status") != "FROZEN_BALANCED_BEFORE_VERIFIER_INFERENCE"
        or pair_freeze.get("task") != args.task
        or pair_freeze.get("role") != "verifier_dev"
        or int(pair_freeze.get("selected_count", -1)) < 1
        or pair_freeze.get("selected_target_counts", {}).get("CONFIRM_MATCH")
        != pair_freeze.get("selected_target_counts", {}).get("REJECT")
    ):
        raise ValueError("invalid frozen verifier-dev pair universe")
    pair_outputs: dict[str, dict[str, str]] = pair_freeze.get("outputs") or {}
    for name in ("truth", "primary", "candidates", "targets"):
        ref = pair_outputs.get(name) or {}
        path = Path(str(ref.get("path") or "")).resolve()
        if not path.is_file() or ref.get("sha256") != sha256_file(path):
            raise ValueError(f"pair output drift: {name}")

    prompt_meta = json.loads(paths["prompt-meta"].read_text(encoding="utf-8"))
    author_freeze = json.loads(
        paths["author-output-freeze"].read_text(encoding="utf-8")
    )
    if (
        prompt_meta.get("schema_version")
        != "silver-match-v3-materialized-frozen-verifier-author-prompt-v1"
        or prompt_meta.get("status") != "MATERIALIZED_WITHOUT_PROMPT_MUTATION"
        or prompt_meta.get("task") != args.task
        or int(prompt_meta.get("variant_count", -1)) != 1
        or prompt_meta.get("verifier_dev_truth_joined_at_authoring") is not False
        or (prompt_meta.get("prompt") or {}).get("sha256")
        != sha256_file(paths["prompt"])
        or (prompt_meta.get("author_output_freeze") or {}).get("sha256")
        != sha256_file(paths["author-output-freeze"])
        or author_freeze.get("status")
        != "FROZEN_CONTEXT_ISOLATED_BEFORE_VERIFIER_DEV_TRUTH_JOIN"
        or author_freeze.get("verifier_dev_truth_joined_to_author") is not False
        or int(author_freeze.get("variant_count", -1)) != 1
        or (author_freeze.get("promotion_contract") or {}).get(
            "fresh_verifier_dev_cannot_mutate_prompt"
        )
        is not True
    ):
        raise ValueError("accepted verifier prompt is not a pre-dev frozen variant")

    manifest_hash = sha256_file(paths["manifest"])
    if (
        (base.get("canonical_inputs") or {}).get("manifest", {}).get("sha256")
        != manifest_hash
    ):
        raise ValueError("manifest differs from verifier-dev policy")
    inventory = json.loads(paths["model-inventory"].read_text(encoding="utf-8"))
    r4_plan = json.loads(paths["r4-proposal-plan"].read_text(encoding="utf-8"))
    if (
        inventory.get("schema_version")
        != "silver-match-v3-directory-content-inventory-v1"
        or inventory.get("status") != "FROZEN_CONTENT_HASH_INVENTORY"
        or inventory.get("root") != args.model
        or r4_plan.get("model") != args.model
        or (r4_plan.get("inputs") or {}).get("model_inventory", {}).get("sha256")
        != sha256_file(paths["model-inventory"])
    ):
        raise ValueError("Gemma snapshot/inventory differs from frozen R4 runtime")

    if args.orders != ["original", "hashed"]:
        raise ValueError("the predeclared verifier orders must be original then hashed")
    if args.max_alternatives < 1 or args.max_tokens < 128:
        raise ValueError("invalid verifier rendering limits")

    output_root.mkdir(parents=True)
    runs_root = output_root / "runs"
    runs_root.mkdir()
    common = [
        args.python,
        "-u",
        "-m",
        "scripts.tools.silver_match_v3.verify_gemma",
        "--manifest",
        str(paths["manifest"]),
        "--candidates",
        str(Path(pair_outputs["candidates"]["path"]).resolve()),
        "--primary",
        str(Path(pair_outputs["primary"]["path"]).resolve()),
        "--prompt",
        str(paths["prompt"]),
        "--model",
        args.model,
        "--max-alternatives",
        str(args.max_alternatives),
        "--batch-size",
        str(args.batch_size),
        "--max-model-len",
        str(args.max_model_len),
        "--max-tokens",
        str(args.max_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--context-chars",
        str(args.context_chars),
        "--description-chars",
        str(args.description_chars),
        "--example-chars",
        str(args.example_chars),
        "--max-examples",
        str(args.max_examples),
        "--keep-raw",
    ]
    commands = {}
    outputs = {}
    for order in args.orders:
        output = runs_root / f"{order}.jsonl"
        commands[order] = [
            *common,
            "--order-mode",
            order,
            "--output",
            str(output),
        ]
        outputs[order] = str(output)

    plan = {
        "schema_version": "silver-match-v3-pr-verifier-dev-inference-plan-v1",
        "status": "FROZEN_BEFORE_FIRST_FRESH_DEV_VERIFIER_INFERENCE",
        "task": args.task,
        "role": "verifier_dev_selection",
        "variant_count": 1,
        "orders": args.orders,
        "fresh_dev_batch_inference_runs": len(args.orders),
        "selected_pair_count": int(pair_freeze["selected_count"]),
        "model": args.model,
        "rendering": {
            "max_alternatives": args.max_alternatives,
            "batch_size": args.batch_size,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "seed": args.seed,
            "context_chars": args.context_chars,
            "description_chars": args.description_chars,
            "example_chars": args.example_chars,
            "max_examples": args.max_examples,
        },
        "gates": {
            "minimum_retained_proposals": int(caps["minimum_retained_proposals"]),
            "minimum_exact_precision": float(caps["minimum_exact_precision"]),
            "minimum_wilson_lower_95": float(caps["minimum_wilson_lower_95"]),
            "failed_gate_action": caps["failed_gate_action"],
            "promotion_requires_new_independent_blind_audit": caps[
                "promotion_requires_new_independent_blind_audit"
            ],
        },
        "contracts": {
            "prompt_frozen_before_truth_join": True,
            "prompt_may_not_be_edited_from_fresh_dev_results": True,
            "only_one_predeclared_prompt_variant": True,
            "each_order_scored_once": True,
            "fresh_dev_excluded_from_gradients": True,
            "fresh_dev_excluded_from_mi_or_outcome_estimation": True,
            "successful_dev_gate_is_not_final_blind_evidence": True,
        },
        "inputs": {name: _ref(path) for name, path in paths.items()},
        "pair_outputs": pair_outputs,
        "commands": commands,
        "outputs": outputs,
    }
    plan_path = output_root / "PLAN.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**plan, "plan_sha256": sha256_file(plan_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="press-releases")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--base-policy", required=True)
    parser.add_argument("--pair-freeze", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-meta", required=True)
    parser.add_argument("--author-output-freeze", required=True)
    parser.add_argument("--model-inventory", required=True)
    parser.add_argument("--r4-proposal-plan", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument(
        "--orders", nargs="+", default=["original", "hashed"]
    )
    parser.add_argument("--max-alternatives", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--seed", type=int, default=2026071326)
    parser.add_argument("--context-chars", type=int, default=1400)
    parser.add_argument("--description-chars", type=int, default=520)
    parser.add_argument("--example-chars", type=int, default=180)
    parser.add_argument("--max-examples", type=int, default=2)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_plan(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
