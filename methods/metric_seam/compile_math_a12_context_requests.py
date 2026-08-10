#!/usr/bin/env python3
"""Freeze the 443-request full-document Math-a12 context arm."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from methods.metric_seam.verifiers.math_a12_context_contract import compile_request
from methods.metric_seam.verifiers.math_a12_llm_contract import RationalExpressionPair
from methods.metric_seam.verifiers.math_a12_symbolic import extract_equality_pairs
from methods.metric_seam.verifiers.schema import Span

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "outputs/metric_seam_pilot/reconstruction_v2/math_a12_symbolic_step_retrospective_prepare_001/compiler_bundle.json"
DEFAULT_OUT = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/requests/math_a12_context_train_v1"
MODEL = "claude-sonnet-4-5-20250929"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiler-bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if "heldout" in str(args.compiler_bundle).lower():
        raise ValueError("context compiler refuses held-out paths")
    bundle = json.loads(args.compiler_bundle.read_text())
    if bundle.get("objective", {}).get("external_supervised_anchor") is not False:
        raise ValueError("external supervised anchors are forbidden")
    requests = []
    for row in bundle["train_items"]:
        if set(row) != {"ctext", "item_key"}:
            raise ValueError("TRAIN row exceeds ctext/item_key allowlist")
        for pair in extract_equality_pairs(row["ctext"], item_key=row["item_key"]):
            span = Span(pair.witness.path, pair.witness.start_line, pair.witness.end_line)
            llm_pair = RationalExpressionPair(
                pair_id=f"{pair.item_key}.{pair.pair_id}", lhs_display=pair.lhs,
                rhs_display=pair.rhs, lhs_span=span, rhs_span=span)
            requests.append(compile_request(
                pair=llm_pair, ctext=row["ctext"], item_key=row["item_key"], model=MODEL))
    if len(requests) != 443:
        raise AssertionError(f"frozen context arm expects 443 pairs, got {len(requests)}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    req_path = args.output_dir / "requests.jsonl"
    req_path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in requests))
    manifest = {
        "schema": "metric-seam.math-a12-context-preregistration.v1",
        "status": "frozen_before_model_execution",
        "task": "math", "criterion_id": "a12", "split": "compiler_train",
        "model": MODEL, "request_count": 443, "pass_count": 1,
        "old_pair_only_arm": "retained as a context-free control; its kappa=0.445 is not a validity result",
        "prediction_frozen_before_calls": (
            "Full-document role classification will reduce joint applicability because definitions, "
            "hypotheses, constraints, and equations-to-solve become not_applicable; polarity agreement "
            "is recomputed only on jointly asserted_identity_step pairs."),
        "primary_readouts": ["role_distribution", "role_conditioned_individuation_gap",
                             "joint_asserted_identity_polarity_agreement"],
        "sympy_implementation": "unchanged frozen symbolic arm",
        "external_supervised_anchor": False, "gpu_used": False,
        "inputs": {"compiler_bundle": str(args.compiler_bundle), "sha256": file_sha(args.compiler_bundle)},
        "requests": {"path": str(req_path), "sha256": file_sha(req_path)},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
