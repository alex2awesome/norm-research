#!/usr/bin/env python3
"""Prepare and summarize the TRAIN-only retrospective math-a12 program.

The trusted sealer reads the label-bearing source but emits only sanitized ctext
under opaque TRAIN aliases.  This downstream script consumes that compiler view,
runs the relation-local symbolic analyzer, and writes aggregate counts only.  It
never accepts a reference, heldout, field-result, or model-output path.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics
from typing import Any, Mapping

from methods.metric_seam.battery.seal_ctext_items_v2 import canonical_bytes
from methods.metric_seam.battery.seal_ctext_train_view_v3 import prepare_train_view
from methods.metric_seam.hybrids.ops_symbolic_steps_v1 import analyze_document


SUMMARY_SCHEMA = "metric-seam.math-a12-symbolic-step-train-summary.v1"
AUTHORSHIP_STATUS = (
    "selected_retrospective_seed_with_aggregate_train_summary_exposure"
)

_COUNT_FIELDS = (
    "equality_rows_seen",
    "pair_candidate_count",
    "parsed_rational_pair_count",
    "verified_rational_identity_count",
    "exact_nonidentity_witness_count",
    "universal_identity_counterexample_count",
    "symbolically_unresolved_count",
    "parse_noncoverage_count",
    "positive_code_witness_count",
    "criterion_defect_witness_count",
)


def _write_exclusive_readonly(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_bytes(dict(value))
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def _write_text_exclusive_readonly(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def summarize_train_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Execute aggregate relation analysis over a sanitized compiler bundle."""

    if bundle.get("task") != "math" or bundle.get("criterion_id") != "a12":
        raise ValueError("compiler bundle is not math a12")
    interface = bundle.get("interface", {})
    if interface.get("reference_values_available") is not False:
        raise ValueError("reference values must be unavailable")
    if interface.get("heldout_items_available") is not False:
        raise ValueError("heldout items must be unavailable")
    items = bundle.get("train_items")
    if not isinstance(items, list) or not items:
        raise ValueError("TRAIN compiler rows are unavailable")

    counts: Counter[str] = Counter()
    parsed_pairs_per_row: list[int] = []
    for expected_index, item in enumerate(items, 1):
        if not isinstance(item, dict) or set(item) != {"ctext", "item_key"}:
            raise ValueError("compiler row exceeds the ctext/item_key allowlist")
        if item["item_key"] != f"train_{expected_index:04d}":
            raise ValueError("TRAIN alias sequence is not canonical")
        if not isinstance(item["ctext"], str):
            raise ValueError("TRAIN ctext must be a string")
        result = analyze_document(item["ctext"])
        counts["rows"] += 1
        counts["rows_abstained"] += int(result["abstained"])
        counts["rows_with_executable_pair"] += int(not result["abstained"])
        counts["rows_with_identity_witness"] += int(
            result["verified_rational_identity_count"] > 0
        )
        counts["rows_with_exact_nonidentity_witness"] += int(
            result["exact_nonidentity_witness_count"] > 0
        )
        counts["rows_pair_budget_exhausted"] += int(
            result["pair_budget_exhausted"]
        )
        for field in _COUNT_FIELDS:
            counts[field] += int(result[field])
        parsed_pairs_per_row.append(int(result["parsed_rational_pair_count"]))

    return {
        "schema": SUMMARY_SCHEMA,
        "task": "math",
        "criterion_id": "a12",
        "train_only": True,
        "train_row_count": len(items),
        "coverage": dict(sorted(counts.items())),
        "parsed_pairs_per_row": {
            "median": statistics.median(parsed_pairs_per_row),
            "max": max(parsed_pairs_per_row),
        },
        "authorship_status": AUTHORSHIP_STATUS,
        "authorship_boundary": (
            "retrospective-seed procedural blindness; a legacy h0 docstring "
            "exposed one aggregate TRAIN correlation and qualitative historical "
            "rationale, neither used for relation selection, implementation, "
            "thresholds, or tuning; no per-item, reference, heldout, or residual exposure"
        ),
        "strict_pristine_outcome_blind_authorship_claimed": False,
        "reference_accessed": False,
        "heldout_accessed": False,
        "model_calls": False,
        "gpu_used": False,
        "whole_criterion_fidelity": "UNAVAILABLE",
        "whole_criterion_scalar": None,
        "interpretation": (
            "Identity and exact nonidentity certificates are relation-local. "
            "Exact nonidentity becomes a universal-identity counterexample and "
            "criterion defect only after a separately frozen true universal-scope "
            "determination; TRAIN document analysis makes no such determination."
        ),
    }


def render_report(summary: Mapping[str, Any]) -> str:
    coverage = summary["coverage"]
    return f"""# Math a12 symbolic-step retrospective — TRAIN-only report

## Result

The selected criterion is **a12, Precision and rigor in statements and proofs**. The
executable sub-relation is deliberately narrower than the parent criterion: an explicitly
presented rational-algebra equality step should preserve the same expression on its declared
domain.

The program reuses the existing manually selected `MathOps.extract_math_spans` capability as
a retrospective pipeline seed, then parses bounded answer-side equality pairs with SymPy's
Lark LaTeX parser. It exactly reduces each rational-expression difference. A zero difference
certifies identity preservation and reports denominator-nonzero obligations; an exact rational
assignment with unequal values witnesses that the pair is not an identity. This is deeper
executable checking, not a keyword proxy.

On {summary['train_row_count']} sanitized, opaque, unlabeled TRAIN rows:

- {coverage['rows_with_executable_pair']} rows contained at least one parsed rational pair;
  {coverage['rows_abstained']} abstained.
- {coverage['verified_rational_identity_count']} exact identity certificates occurred across
  {coverage['rows_with_identity_witness']} rows.
- {coverage['exact_nonidentity_witness_count']} exact nonidentity witnesses occurred across
  {coverage['rows_with_exact_nonidentity_witness']} rows. None is labeled a universal-identity
  counterexample because document analysis does not infer universal claim scope.
- {coverage['symbolically_unresolved_count']} parsed pair remained symbolically unresolved;
  {coverage['parse_noncoverage_count']} candidate pairs were parse noncoverage.
- {coverage['rows_pair_budget_exhausted']} rows exhausted the candidate-pair budget.
- {coverage['criterion_defect_witness_count']} document-level rigor defects were asserted. A
  nonidentity witness is a defect only if a separately frozen scope judgment says the document
  presents the equation as universal, rather than as a definition, special solution,
  assumption, or conditional step.

Whole-criterion fidelity and a whole-criterion scalar are **UNAVAILABLE**. The code does not
verify theorem applicability, omitted hypotheses, branch conditions, terminology, or proof
completeness. No executable pair means abstention; it is never evidence of low rigor or
tacitness.

Authorship status is
`selected_retrospective_seed_with_aggregate_train_summary_exposure`, not pristine
outcome-blind authorship. During seed selection the legacy `a12_h0.py` docstring exposed one
aggregate TRAIN correlation and qualitative historical rationale. No per-item outcome,
held-out/reference value, residual, or evaluation output was opened, and the aggregate value
was not used to select the symbolic relation, set a threshold, or tune the program. A fresh
context should re-author from the contract and sanitized TRAIN bundle if the strongest
blindness claim is required.

## Construct adversary

Ten construct-derived tests pass. They cover polynomial expansion, variable-renaming and
side-reversal metamorphisms, denominator-domain obligations, an exact counterexample to a false
universal identity, claim-scope gating, branch-sensitive transcendental abstention,
malformed-LaTeX abstention, answer-only selection, empty noncoverage, and equality chains.

## Next sealed step

Freeze this relation contract and program, project the held-out split through the identical
credential sanitizer, and execute the code before any reference is opened. Report identity,
nonidentity, and abstention coverage first. A later fresh prompt-side scope field may declare
whether each pair is intended as universal and whether its reported domain obligations are
discharged; only that typed hybrid can convert a nonidentity witness into a criterion-local
defect. Any comparison with the historical holistic a12 prompt reference is secondary
reconstruction agreement, never whole-criterion isomorphism or external correctness.

No held-out item, reference value, per-item outcome label, residual, model, API, or GPU was
used.
"""


def prepare(
    *,
    source: Path,
    construct_contract: Path,
    relation_contract: Path,
    out_dir: Path,
    train_count: int = 150,
    split_seed: int = 7,
) -> tuple[Path, Path, Path, Path]:
    """Create a sanitized TRAIN view and its deterministic aggregate summary."""

    bundle_path, manifest_path = prepare_train_view(
        source=source,
        contract_path=construct_contract,
        out_dir=out_dir,
        task="math",
        criterion_id="a12",
        train_count=train_count,
        split_seed=split_seed,
        dependency_files={
            "preparer": Path(__file__),
            "relation_contract": relation_contract,
            "symbolic_operation": Path(analyze_document.__code__.co_filename),
        },
        dependency_packages=("sympy", "lark"),
    )
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    summary = summarize_train_bundle(bundle)
    summary_path = out_dir / "train_symbolic_step_summary.json"
    _write_exclusive_readonly(summary_path, summary)
    report_path = out_dir / "REPORT.md"
    _write_text_exclusive_readonly(report_path, render_report(summary))
    return bundle_path, manifest_path, summary_path, report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--construct-contract", type=Path, required=True)
    parser.add_argument("--relation-contract", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=150)
    parser.add_argument("--split-seed", type=int, default=7)
    args = parser.parse_args()
    bundle, manifest, summary, report = prepare(
        source=args.source,
        construct_contract=args.construct_contract,
        relation_contract=args.relation_contract,
        out_dir=args.out_dir,
        train_count=args.train_count,
        split_seed=args.split_seed,
    )
    # Paths and aggregate counts only; no ctext or source identifier is printed.
    result = json.loads(summary.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "bundle": str(bundle),
                "manifest": str(manifest),
                "summary": str(summary),
                "report": str(report),
                "coverage": result["coverage"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
