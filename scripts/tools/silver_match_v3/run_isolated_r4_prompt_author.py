#!/usr/bin/env python3
"""Run and immediately freeze one isolated fresh R4 Codex prompt author."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .common import sha256_file


EXPECTED = {
    "HANDOFF_FREEZE.json": "3daf7f9e897010e2f7520ee4fc2d01e1be2adb7388c9240a945bbb20b03516fa",
    "gradient_errors.jsonl": "0219d48521a2079864f743b414ac4cbe82dd3643b0d413494f934509b8d05c99",
    "retriever_misses_at_50.jsonl": "ddeda0f1d921e51f5fcbba77e31c4071be2a997d9f704bae822e28931d7b0d32",
    "r3_score.json": "9b0cf8943c2371b16202058206a534672398cd286be555481eebfbee56e26d36",
    "r3_prompt.txt": "3a8cd2c94559c22f639fe9a207e0ebba029f66508d019b6eddd14c54c04b48e9",
}

EXPECTED_DECISIONS = [
    "MATCH",
    "MATCH_FAMILY_ONLY",
    "NO_CANDIDATE_FITS",
    "NO_EXPLICIT_CRITERION",
    "GENERIC_VERDICT",
    "CONTEXT_NEEDED",
    "NOISE",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    root = Path(args.workspace).resolve()
    inputs = root / "inputs"
    output = root / "author_output.json"
    log = root / "author.log"
    input_freeze = root / "INPUT_FREEZE.json"
    output_freeze = root / "OUTPUT_FREEZE.json"
    schema = root / "pr_r4_prompt_author_v1.schema.json"
    instructions = root / "AUTHOR_INSTRUCTIONS.md"
    if any(path.exists() for path in (output, log, input_freeze, output_freeze)):
        raise FileExistsError("refusing to overwrite isolated R4 author artifacts")
    if "select" in str(root).lower():
        raise ValueError("isolated R4 author workspace path must not mention select")
    observed = {}
    for name, expected_sha in EXPECTED.items():
        path = inputs / name
        observed[name] = {"path": str(path), "sha256": sha256_file(path)}
        if observed[name]["sha256"] != expected_sha:
            raise ValueError(f"R4 author input hash mismatch: {name}")
    for path in (schema, instructions):
        observed[path.name] = {"path": str(path), "sha256": sha256_file(path)}
    input_record = {
        "schema_version": "silver-match-v3-pr-r4-author-input-freeze-v1",
        "status": "FROZEN_BEFORE_FRESH_AUTHOR_CALL",
        "select_material_joined": False,
        "variant_count": 1,
        "inputs": observed,
    }
    input_freeze.write_text(json.dumps(input_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prompt = (
        "Read AUTHOR_INSTRUCTIONS.md, inputs/HANDOFF_FREEZE.json, "
        "inputs/gradient_errors.jsonl, inputs/retriever_misses_at_50.jsonl, "
        "inputs/r3_score.json, and inputs/r3_prompt.txt. Author exactly one R4 "
        "prompt under the fixed contract. Use no files outside this isolated "
        "workspace. Return only schema-conforming JSON."
    )
    command = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--dangerously-bypass-hook-trust",
        "-m",
        args.model,
        "-c",
        f'model_reasoning_effort="{args.reasoning_effort}"',
        "--output-schema",
        str(schema),
        "-o",
        str(output),
        prompt,
    ]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("OLDPWD", None)
    env["PWD"] = str(root)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
    if completed.returncode != 0 or not output.exists():
        raise RuntimeError(f"fresh R4 author failed: returncode={completed.returncode}")
    value = json.loads(output.read_text(encoding="utf-8"))
    provenance = value.get("provenance") or {}
    selection = value.get("selection_rule") or {}
    parse_rule = value.get("parse_rule") or {}
    if (
        value.get("schema_version") != "silver-match-v3-pr-r4-fresh-author-v1"
        or provenance.get("used_only_frozen_handoff") is not True
        or provenance.get("select_material_read") is not False
        or provenance.get("select_scores_read") is not False
        or provenance.get("mi_or_outcomes_read") is not False
        or selection.get("variant_count") != 1
        or selection.get("choose_without_select_scores") is not True
        or parse_rule.get("allowed_decisions") != EXPECTED_DECISIONS
        or parse_rule.get("match_requires_evidence_span") is not True
        or parse_rule.get("match_requires_exact_leaf_contrast") is not True
        or parse_rule.get("nonmatch_metric_id_must_be_null") is not True
    ):
        raise ValueError("fresh R4 author provenance/selection assertion failed")
    freeze = {
        "schema_version": "silver-match-v3-pr-r4-author-output-freeze-v1",
        "status": "FROZEN_BEFORE_ANY_SELECT_JOIN",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "variant_count": 1,
        "select_material_joined": False,
        "input_freeze": {"path": str(input_freeze), "sha256": sha256_file(input_freeze)},
        "author_output": {"path": str(output), "sha256": sha256_file(output)},
        "author_log": {"path": str(log), "sha256": sha256_file(log)},
        "provenance_assertion": provenance,
        "selection_rule": selection,
    }
    output_freeze.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**freeze, "output_freeze_sha256": sha256_file(output_freeze)}, sort_keys=True))


if __name__ == "__main__":
    main()
