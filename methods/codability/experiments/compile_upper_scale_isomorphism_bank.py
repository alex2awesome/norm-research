#!/usr/bin/env python
"""Freeze intact articulations for the within-Llama 8B-to-70B target pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from methods.codability.experiments.build_fresh_item_partitions import sha256_file
from methods.codability.experiments.compile_adjacent_scale_isomorphism_bank import (
    compile_scale_pair_bank,
)


def compile_bank(source_bank_path: str) -> dict:
    return compile_scale_pair_bank(
        source_bank_path,
        schema="upper_scale_isomorphism_bank/v1",
        status="frozen-before-8b-executor-public-scoring",
        objective="exact Llama-3.1-8B reconstruction of Llama-3.3-70B name-only policy",
        model_family="Llama only; 3.1-to-3.3 version caveat retained",
        target_policy="Llama-3.3-70B-Instruct-FP8 name-only three-form soft policy",
        executor="Llama-3.1-8B-Instruct",
        version_caveat=(
            "This is within the Llama family but not a same-version parameter-only contrast; "
            "no scalar scale-law estimate may treat it as pure size."
        ),
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
