#!/usr/bin/env python3
"""Freeze the one-shot two-order PR R4 select evaluation plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-meta", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--truth-release", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--parser-implementation", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prompt_meta_path = Path(args.prompt_meta).resolve()
    manifest_path = Path(args.manifest).resolve()
    pack_root = Path(args.pack_root).resolve()
    validation_path = pack_root / "validation.json"
    candidates_path = Path(args.candidates).resolve()
    truth_release_path = Path(args.truth_release).resolve()
    runner_path = Path(args.runner).resolve()
    parser_path = Path(args.parser_implementation).resolve()
    output_root = Path(args.output_root).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists() or output_root.exists():
        raise FileExistsError("refusing to overwrite/freeze an already-started R4 select run")

    prompt_meta = json.loads(prompt_meta_path.read_text(encoding="utf-8"))
    prompt_path = Path(prompt_meta["prompt"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    release = json.loads(truth_release_path.read_text(encoding="utf-8"))
    candidate_rows = list(read_jsonl(candidates_path))
    item_uids = {str(row["norm_uid"]) for row in read_jsonl(pack_root / "items.jsonl")}
    candidate_uids = [str(row["norm_uid"]) for row in candidate_rows]
    if (
        prompt_meta.get("status") != "MATERIALIZED_WITHOUT_PROMPT_MUTATION"
        or prompt_meta.get("variant_count") != 1
        or prompt_meta["prompt"]["sha256"] != sha256_file(prompt_path)
        or manifest.get("schema_version") != "silver-match-v3-task-local-inference-manifest-v1"
        or manifest.get("truth_or_label_fields_in_manifest") is not False
        or validation.get("task") != "press-releases"
        or validation.get("truth_hidden") is not True
        or int(validation.get("count", -1)) != 240
        or len(candidate_uids) != 240
        or len(set(candidate_uids)) != 240
        or set(candidate_uids) != item_uids
        or any(len(row.get("candidates") or []) != 50 for row in candidate_rows)
        or release.get("status") != "FROZEN_COMPLETE_EXACT_TRUTH"
        or release.get("task") != "press-releases"
        or release.get("role") != "select"
        or int(release.get("count", -1)) != 240
    ):
        raise ValueError("R4 select plan inputs violate the frozen one-shot contract")

    model = "google/gemma-4-31b-it"
    orders = ["original", "hashed"]
    output_root.mkdir(parents=True)
    plan = {
        "schema_version": "silver-match-v3-pr-r4-select-openrouter-plan-v1",
        "status": "FROZEN_BEFORE_SELECT_INFERENCE",
        "task": "press-releases",
        "role": "select",
        "variant_count": 1,
        "orders": orders,
        "model": model,
        "api_base_url": "https://openrouter.ai/api/v1",
        "max_api_requests_per_order": 250,
        "max_total_api_requests": 500,
        "transport_retries": 0,
        "row_count": 240,
        "candidate_depth": 50,
        "rendering": {
            "context_chars": 1200,
            "description_chars": 260,
            "example_chars": 80,
            "max_examples": 0,
            "max_tokens": 220,
            "seed": 17,
        },
        "inputs": {
            "prompt_meta": _artifact(prompt_meta_path),
            "prompt": _artifact(prompt_path),
            "manifest": _artifact(manifest_path),
            "pack_validation": _artifact(validation_path),
            "pack_items": _artifact(pack_root / "items.jsonl"),
            "pack_bank": _artifact(pack_root / "bank.json"),
            "candidates": _artifact(candidates_path),
            "truth_release": _artifact(truth_release_path),
            "runner": _artifact(runner_path),
            "parser_implementation": _artifact(parser_path),
        },
        "outputs": {
            order: str(output_root / f"{order}.jsonl") for order in orders
        },
        "contracts": {
            "exactly_one_frozen_prompt": True,
            "both_orders_predeclared": True,
            "no_exploratory_transport_retries": True,
            "keep_raw_responses": True,
            "join_truth_only_after_both_outputs_frozen": True,
            "no_new_prompt_variant_after_score": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**plan, "plan_sha256": sha256_file(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
