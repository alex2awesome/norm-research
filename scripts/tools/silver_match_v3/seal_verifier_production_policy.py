#!/usr/bin/env python3
"""Seal a dev-supported, fail-closed two-order verifier policy."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .common import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--two-order-score", required=True)
    parser.add_argument("--original-meta", required=True)
    parser.add_argument("--hashed-meta", required=True)
    parser.add_argument("--base-prompt", required=True)
    parser.add_argument("--prompt-addon", action="append", default=[])
    parser.add_argument("--min-point-precision", type=float, default=0.90)
    parser.add_argument("--min-wilson-lower", type=float, default=0.80)
    parser.add_argument("--min-retained", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    paths = {
        "selection": Path(args.selection).resolve(),
        "two_order_score": Path(args.two_order_score).resolve(),
        "original_meta": Path(args.original_meta).resolve(),
        "hashed_meta": Path(args.hashed_meta).resolve(),
        "base_prompt": Path(args.base_prompt).resolve(),
    }
    addons = [Path(value).resolve() for value in args.prompt_addon]
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    score = json.loads(paths["two_order_score"].read_text(encoding="utf-8"))
    original = json.loads(paths["original_meta"].read_text(encoding="utf-8"))
    hashed = json.loads(paths["hashed_meta"].read_text(encoding="utf-8"))
    if selection.get("task") != args.task or selection.get("selection_split") != "dev":
        raise ValueError("selection is not the requested task's dev selection")
    if score.get("selection_split") != "dev":
        raise ValueError("two-order score is not dev-only")
    chosen_sha = selection["chosen"]["prompt_sha256"]
    if original.get("prompt_sha256") != chosen_sha or hashed.get("prompt_sha256") != chosen_sha:
        raise ValueError("verifier metadata does not use the selected prompt")
    if original.get("order_mode") != "original" or hashed.get("order_mode") != "hashed":
        raise ValueError("verifier metadata does not cover original+hashed orders")
    if original.get("input_candidates_sha256") != hashed.get("input_candidates_sha256"):
        raise ValueError("two orders used different candidate slates")
    if original.get("primary_sha256") != hashed.get("primary_sha256"):
        raise ValueError("two orders used different proposals")

    policy = score["policies"]["high_only"]
    lower = policy["retained_precision_wilson_95"][0]
    supported = (
        policy["retained"] >= args.min_retained
        and policy["retained_precision"] >= args.min_point_precision
        and lower >= args.min_wilson_lower
    )
    if not supported:
        raise ValueError("strict two-order verifier does not clear requested dev gate")
    rendering_keys = (
        "max_alternatives",
        "context_chars",
        "description_chars",
        "example_chars",
        "max_examples",
        "max_model_len",
        "max_tokens",
        "seed",
        "model",
    )
    rendering = {key: original.get(key) for key in rendering_keys}
    if any(hashed.get(key) != value for key, value in rendering.items()):
        raise ValueError("rendering parameters differ across verifier orders")
    payload = {
        "schema_version": "silver-match-v3-verifier-production-policy-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "status": "dev_supported_requires_blind_final_match_audit",
        "selection_split": "dev",
        "frozen_test_consumed_by_this_policy": False,
        "may_run_on_production_unlabeled_norms": True,
        "may_be_used_for_gradient_labels_before_blind_audit": False,
        "prompt": {
            "base_path": str(paths["base_prompt"]),
            "base_sha256": sha256_file(paths["base_prompt"]),
            "addon_paths": [str(path) for path in addons],
            "addon_sha256": {str(path): sha256_file(path) for path in addons},
            "rendered_prompt_sha256": chosen_sha,
        },
        "rendering": rendering,
        "order_policy": {
            "orders": ["original", "hashed"],
            "retain_only_if": (
                "both orders return CONFIRM_MATCH for the identical proposed metric_id "
                "with high confidence and neither output has a parse error"
            ),
            "corrections_are_retained": False,
            "all_disagreement_or_abstention_is_dropped": True,
        },
        "dev_gate": {
            "minimum_point_precision": args.min_point_precision,
            "minimum_wilson_lower": args.min_wilson_lower,
            "minimum_retained": args.min_retained,
            "observed": policy,
            "cleared": supported,
        },
        "independent_audit_requirement": {
            "required": True,
            "scope": "blind stratified sample of final retained production MATCH labels",
            "promotion_blocked_until_pass": True,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "output_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
