#!/usr/bin/env python3
"""Compile the independent Math-a12 LLM arm from frozen TRAIN pair inputs.

Pairs are re-extracted from the compiler bundle.  The compiler never reads the
symbolic TRAIN readout, so neither symbolic applicability nor polarity can
select the LLM workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from methods.metric_seam.verifiers.math_a12_llm_contract import (
    RationalExpressionPair,
    compile_request,
)
from methods.metric_seam.verifiers.math_a12_symbolic import extract_equality_pairs
from methods.metric_seam.verifiers.schema import Span


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / (
    "outputs/metric_seam_pilot/reconstruction_v2/"
    "math_a12_symbolic_step_retrospective_prepare_001/compiler_bundle.json"
)
DEFAULT_OUTPUT_DIR = ROOT / (
    "outputs/metric_seam_pilot/hierarchy_r123/requests/"
    "math_a12_llm_train_v1"
)
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_bundle(
    *, rows: Sequence[dict], model: str, pass_indices: Sequence[int] = (1, 2)
) -> list[dict]:
    if not pass_indices or any(index not in (1, 2) for index in pass_indices):
        raise ValueError("pass_indices must be a nonempty subset of (1, 2)")
    if len(set(pass_indices)) != len(pass_indices):
        raise ValueError("pass_indices must not contain duplicates")
    requests: list[dict] = []
    for row in rows:
        if set(row) != {"ctext", "item_key"}:
            raise ValueError("TRAIN row exceeds the ctext/item_key allowlist")
        for pair in extract_equality_pairs(row["ctext"], item_key=row["item_key"]):
            # The two expressions occupy the same displayed source region in
            # this extractor.  Remove the V_symbolic-only node id before the
            # independent contract is compiled.
            source = Span(
                pair.witness.path,
                pair.witness.start_line,
                pair.witness.end_line,
            )
            llm_pair = RationalExpressionPair(
                pair_id=f"{pair.item_key}.{pair.pair_id}",
                lhs_display=pair.lhs,
                rhs_display=pair.rhs,
                lhs_span=source,
                rhs_span=source,
            )
            for pass_index in pass_indices:
                requests.append(
                    compile_request(
                        pair=llm_pair,
                        pass_index=pass_index,
                        model=model,
                        split="compiler_train",
                    )
                )
    if len({row["request_sha256"] for row in requests}) != len(requests):
        raise ValueError("compiled request digests are not unique")
    return requests


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--passes", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args(argv)
    if "heldout" in str(args.compiler_bundle).casefold():
        raise ValueError("TRAIN request compiler refuses held-out paths")
    bundle = json.loads(args.compiler_bundle.read_text(encoding="utf-8"))
    if bundle.get("schema") != "metric-seam.sanitized-ctext-train-compiler-view.v2":
        raise ValueError("unexpected compiler bundle schema")
    if bundle.get("objective", {}).get("external_supervised_anchor") is not False:
        raise ValueError("external supervised anchors are forbidden")
    requests = compile_bundle(
        rows=bundle["train_items"], model=args.model, pass_indices=args.passes
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    requests_path = args.output_dir / "requests.jsonl"
    with requests_path.open("x", encoding="utf-8") as handle:
        for request in requests:
            handle.write(
                json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    manifest = {
        "schema": "metric-seam.math-a12-llm-train-request-bundle.v1",
        "status": "frozen_before_llm_execution",
        "task": "math",
        "criterion_id": "a12",
        "relation_id": "explicit_rational_equality_preservation",
        "split": "compiler_train",
        "model": args.model,
        "pair_count": len(requests) // len(args.passes),
        "request_count": len(requests),
        "pass_count": len(args.passes),
        "pass_indices": args.passes,
        "selection": "all structurally extracted adjacent equality pairs; no symbolic output loaded",
        "symbolic_train_readout_loaded": False,
        "model_calls_performed": False,
        "gpu_used": False,
        "inputs": {
            "compiler_bundle": {
                "path": str(args.compiler_bundle),
                "sha256": _sha(args.compiler_bundle),
            }
        },
        "artifacts": {
            "requests": {"path": str(requests_path), "sha256": _sha(requests_path)}
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
