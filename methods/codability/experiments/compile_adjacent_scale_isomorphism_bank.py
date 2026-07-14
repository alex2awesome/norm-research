#!/usr/bin/env python
"""Freeze the intact articulation bank for the within-Llama 1B-to-3B target pair."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from methods.codability.experiments.build_fresh_item_partitions import sha256_file


CELL_ID = "N_humor_49"


def compile_scale_pair_bank(
        source_bank_path: str, *, schema: str, status: str, objective: str,
        model_family: str, target_policy: str, executor: str,
        version_caveat: str | None = None) -> dict:
    """Freeze one within-family scale-pair view of the shared intact articulation bank."""
    source = json.loads(Path(source_bank_path).read_text())
    cell = next(cell for cell in source["cells"] if cell["id"] == CELL_ID)
    payload = {
        "schema": schema,
        "status": status,
        "objective": objective,
        "model_family": model_family,
        "target_policy": target_policy,
        "executor": executor,
        "anchor_policy": "no human, corpus, compiler, or external evaluator target",
        "source_bank": {"path": source_bank_path, "sha256": sha256_file(source_bank_path)},
        "treatment": ("construct name plus frozen intact definitions, explanations, rules, "
                      "examples, and their source-faithful compositions"),
        "partitions": ["residual_prompt_selection", "residual_unit_certification"],
        "margins": {"mae": 0.02, "rho": 0.05, "flip": 0.02, "bias": 0.02},
        "cells": [cell],
    }
    if version_caveat is not None:
        payload["version_caveat"] = version_caveat
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["bank_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def compile_bank(source_bank_path: str) -> dict:
    return compile_scale_pair_bank(
        source_bank_path,
        schema="adjacent_scale_isomorphism_bank/v1",
        status="frozen-before-1b-executor-public-scoring",
        objective="exact Llama-3.2-1B reconstruction of Llama-3.2-3B name-only policy",
        model_family="Llama-3.2 only",
        target_policy="Llama-3.2-3B-Instruct name-only three-form soft policy",
        executor="Llama-3.2-1B-Instruct",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = compile_bank(args.source_bank)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(out), "sha256": sha256_file(out),
                      "bank_content_sha256": result["bank_content_sha256"],
                      "n_arms": len(result["cells"][0]["arms"])}, indent=1))


if __name__ == "__main__":
    main()
