#!/usr/bin/env python3
"""Consolidate the proposal-first verifier repair into one bounded result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha(path)}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _real_code_review_rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("real_units", "real_candidates", "candidates", "candidate_relations"):
        rows = value.get(key)
        if isinstance(rows, list):
            real = [row for row in rows if isinstance(row, dict) and row.get("aspect_id") is not None]
            if real:
                return real
    raise ValueError("code-review readout has no real candidate rows")


def build_summary(
    *,
    a12: dict[str, Any],
    math_agreement: dict[str, Any],
    code_review: dict[str, Any],
    patent_probe: dict[str, Any],
    patent_code: dict[str, Any],
    sources: dict[str, dict[str, str]],
) -> dict[str, Any]:
    cr_rows = _real_code_review_rows(code_review)
    cr_passed = sum(row.get("gate", {}).get("passed") is True for row in cr_rows)
    cr_probe_correct = sum(row.get("capability_diagnostic", {}).get("probe_correct", 0) for row in cr_rows)
    cr_probe_total = sum(row.get("capability_diagnostic", {}).get("probe_total", 0) for row in cr_rows)
    p_states = patent_probe["state_counts"]
    c_states = patent_code["natural_train"]["state_counts"]
    applicable_code = c_states["satisfied"] + c_states["violated"]
    return {
        "schema": "metric-seam.construct-validity-repair-summary.v1",
        "status": "bounded_first_release_complete",
        "workflow": [
            "propose", "base_rate_probe", "author_or_import", "construct_challenge",
            "per_node_gate", "select", "freeze", "transcribe", "evaluate",
        ],
        "results": {
            "math_a12": {
                "stop_stage": "construct_challenge",
                "disposition": "rejected_construct_misconstrual",
                "controls": a12["summary"]["controls"],
                "construct_correct": a12["summary"]["old_verifier_construct_correct"],
                "proxy_on_construct_satisfied": a12["summary"]["proxy_on_construct_satisfied"],
                "proxy_on_called_violated": a12["summary"]["proxy_on_called_violated"],
                "prior_agreement": {
                    "valid_responses": math_agreement["valid_response_count"],
                    "requested": math_agreement["expected_request_count"],
                    "jointly_applicable_n": math_agreement["by_pass"]["1"]["jointly_applicable_polarity_agreement"]["n"],
                    "polarity_kappa": math_agreement["by_pass"]["1"]["jointly_applicable_polarity_agreement"]["kappa"],
                    "interpretation": "agreement_under_shared_context_free_identity_misconstrual_not_construct_validity",
                },
                "narrow_capability_retained": "context_free_rational_expression_identity",
            },
            "code_review": {
                "stop_stage": "natural_train_gate",
                "disposition": "corpus_unmeasurable_on_merged_prs",
                "candidate_units": len(cr_rows),
                "natural_gate_passed": cr_passed,
                "plant_detection": {"numerator": cr_probe_correct, "denominator": cr_probe_total},
                "prompt_reconstruction_launched": False,
            },
            "patent_antecedent": {
                "stop_stage": "natural_train_gate_after_import",
                "disposition": "prompt_corpus_supported_but_binary_code_degenerate",
                "pre_authoring_prompt_probe": {
                    "requested": patent_probe["sample_size"],
                    "valid": patent_probe["sample_size"] - patent_probe["execution_errors"],
                    "state_counts": p_states,
                    "passed": patent_probe["passed"],
                },
                "imported_binary_code_train": {
                    "n": patent_code["natural_train"]["n"],
                    "state_counts": c_states,
                    "violated_given_applies": {
                        "numerator": c_states["violated"], "denominator": applicable_code,
                    },
                    "passes_max_90_percent_violation_gate": c_states["violated"] * 10 <= applicable_code * 9,
                },
                "construct_controls_before_blind_adjudication": {
                    "code_correct": patent_code["construct_challenge"]["code_correct"],
                    "n": patent_code["construct_challenge"]["n"],
                    "status": "secondary_diagnostic_only_natural_gate_already_preempted_selection",
                },
                "transcription_calls_not_launched": 150,
                "heldout_accessed": False,
            },
            "a34": {
                "stop_stage": "base_rate_probe",
                "disposition": "audit_reported_dead_subtree_not_independently_bound",
                "reported_dead_nodes": 2,
                "local_node_artifact": None,
                "claim_status": "provenance_gap_explicit_no_independent_reproduction_claim",
            },
        },
        "headline": (
            "Agreement is downstream of construct validity: a12 agreed perfectly on a wrong proxy, "
            "code review lacked natural violations, and the Patent prompt probe found both polarities "
            "while the imported binary code collapsed to 148/149 violated."
        ),
        "claim_limits": [
            "These are stage-specific bounded dispositions, not whole-metric codability rates.",
            "The Patent base-rate probe is prompt-based articulability screening, not external ground truth.",
            "The Patent draft was preexisting manual code imported only after the prompt probe passed.",
            "The Patent construct-control expected states have not received the planned two-pass blind adjudication because the natural code gate already preempted selection.",
            "No failed unit establishes unqualified tacitness.",
        ],
        "sources": sources,
    }


def _report(value: dict[str, Any]) -> str:
    a12 = value["results"]["math_a12"]
    cr = value["results"]["code_review"]
    pa = value["results"]["patent_antecedent"]
    ps = pa["pre_authoring_prompt_probe"]["state_counts"]
    cs = pa["imported_binary_code_train"]["state_counts"]
    return f"""# Construct-valid verifier repair — bounded first release

## Result

{value['headline']}

| Unit | Last completed stage | Result |
|---|---|---|
| Math a12 rigor | construct challenge | 0/{a12['controls']} controls correct; all {a12['proxy_on_called_violated']} rigorous proxy-positive controls called violations |
| Code review | natural TRAIN gate | {cr['natural_gate_passed']}/{cr['candidate_units']} measurable; plants {cr['plant_detection']['numerator']}/{cr['plant_detection']['denominator']} |
| Patent antecedent basis | imported-code natural TRAIN gate | prompt probe S={ps['satisfied']}, V={ps['violated']}; code NA={cs['not_applicable']}, S={cs['satisfied']}, V={cs['violated']} |
| a34 | audit-reported base-rate stop | two dead nodes reported; local node artifact not found, so not independently reproduced here |

The prior a12 91/91 conditional polarity agreement is retained as a counterexample: two implementations can
agree perfectly while measuring context-free identity instead of rigor. The Patent result is prospective:
the proposal-only prompt probe passed 32/32 transport-valid judgments with both polarities, but the imported
binary code relation was 148/149 violated among applicable documents. Selection stopped before prompt
transcription, avoiding a 150-item reconstruction run.

## Licensed interpretation

The repaired ordering distinguishes construct misconstrual, corpus inadequacy, and code-side degeneracy.
It does not estimate whole-metric codability and does not infer tacitness from any stop.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--math-agreement", type=Path, required=True)
    parser.add_argument("--code-review", type=Path, required=True)
    parser.add_argument("--patent-probe", type=Path, required=True)
    parser.add_argument("--patent-code", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {"a12": args.a12, "math_agreement": args.math_agreement, "code_review": args.code_review, "patent_probe": args.patent_probe, "patent_code": args.patent_code}
    value = build_summary(
        a12=_load(args.a12), math_agreement=_load(args.math_agreement),
        code_review=_load(args.code_review), patent_probe=_load(args.patent_probe),
        patent_code=_load(args.patent_code), sources={name: _source(path) for name, path in paths.items()},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "readout.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "report.md").write_text(_report(value))
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
