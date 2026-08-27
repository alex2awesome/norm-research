#!/usr/bin/env python3
"""Build the corrected cross-domain metric-seam results summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "percent": 100.0 * numerator / denominator if denominator else None,
    }


def build_summary(
    *,
    code_review_funnel: dict,
    math_static: dict,
    patents_static: dict,
    science_static: dict,
    code_review_train: dict,
    math_train: dict,
    technical_resolution: dict,
    math_agreement: dict | None = None,
) -> dict[str, Any]:
    cr_static = code_review_funnel["corrected_readout"]["stages"][
        "relation_local_static_fidelity"
    ]["balanced_panel"]
    legacy = [
        {
            "task": "code-review",
            "rate": _rate(cr_static["n_positive"], cr_static["denominator"]),
            "depth_counts": code_review_funnel["corrected_readout"]["by_depth"][
                "relation_local_static_fidelity"
            ],
        },
        {
            "task": "math-stackexchange",
            "rate": _rate(
                math_static["summary"]["eligible_for_relation_local_execution"],
                math_static["summary"]["n_cells"],
            ),
            "depth_counts": math_static["summary"]["eligible_audited_depths"],
        },
        {
            "task": "patents",
            "rate": _rate(patents_static["summary"]["n_partial_relation_local"], 90),
            "depth_counts": patents_static["summary"][
                "maximum_matching_relation_depth_counts"
            ],
        },
        {
            "task": "peer-review-fullarticle",
            "rate": _rate(science_static["summary"]["n_partial_relation_local"], 90),
            "depth_counts": science_static["summary"][
                "maximum_matching_relation_depth_counts"
            ],
        },
    ]
    for row in legacy:
        row["measurement_status"] = (
            "legacy static/manual sub-relation witness incidence; descriptive only"
        )
        row["whole_construct_codability_rate"] = None

    real = code_review_train["real_units"]
    cr_probe_correct = sum(row["gate"]["probe_correct"] for row in real)
    cr_probe_total = sum(row["gate"]["probe_total"] for row in real)
    document_counts = math_train["document_pair_counts"]
    pairs_by_document: dict[str, list[dict]] = {}
    for pair in math_train["natural_pairs"]:
        pairs_by_document.setdefault(pair["item_key"], []).append(pair)
    math_depth = {
        "static_max_relation_depth": 3,
        "documents_total": len(document_counts),
        "documents_reaching_formal_solver": sum(
            row["pair_candidate_count"] > 0 for row in document_counts
        ),
        "documents_with_exact_relation_evidence": sum(
            any(pair["state"] != "not_applicable" for pair in pairs_by_document.get(row["item_key"], []))
            for row in document_counts
        ),
        "pair_level_formal_attempts": math_train["natural_pair_count"],
        "pair_level_exact_results": math_train["gate"]["natural_applies"],
    }

    verifier_results = {
        "code_review": {
            "candidate_relations": len(real),
            "natural_train_measurable": sum(row["gate"]["passed"] for row in real),
            "natural_train_corpus_unmeasurable": sum(
                row["gate"]["failure_reason"] == "corpus_unmeasurable" for row in real
            ),
            "planted_probe_detection": _rate(cr_probe_correct, cr_probe_total),
            "heldout_or_llm_launched": False,
            "interpretation": "merged-PR corpus inadequacy for the selected relations",
        },
        "math_a12": {
            "candidate_relations": 1,
            "natural_train_measurable": int(math_train["gate"]["passed"]),
            "measurement_unit": math_train["measurement_unit"],
            "natural_states": math_train["natural_state_counts"],
            "applies": _rate(
                math_train["gate"]["natural_applies"], math_train["natural_pair_count"]
            ),
            "violated_given_applies": _rate(
                math_train["gate"]["natural_violated"],
                math_train["gate"]["natural_applies"],
            ),
            "planted_probe_detection": _rate(
                math_train["gate"]["probe_correct"], math_train["gate"]["probe_total"]
            ),
            "relation_depth": math_depth,
            "train_prompt_code_agreement": math_agreement,
        },
    }
    return {
        "schema": "metric-seam.corrected-verifier-results-summary.v1",
        "status": (
            "math_train_agreement_complete"
            if math_agreement and math_agreement.get("status") == "complete"
            else "math_train_agreement_pending"
        ),
        "primary_terms": {
            "articulability": "prompt-based implementation",
            "verifiability": "code-based executable implementation",
            "isomorphism": "agreement between implementations of the same frozen sub-relation",
        },
        "legacy_static_subrelation_witness_incidence": legacy,
        "verifier_native_results": verifier_results,
        "technical_target_resolution_diagnostic": technical_resolution["summary_by_task"],
        "claim_limits": [
            "No percentage in the legacy static table is a percentage of whole metrics that code can verify.",
            "Legacy static rates were not measured by the new independent verifier certificate and remain descriptive.",
            "The code-review and Math verifier pilots have different candidate denominators and are not prevalence estimates.",
            "Failed discovery or certification is bounded by corpus, verifier class, and budget; it does not establish tacitness.",
            "No supervised external ground-truth anchor is introduced.",
        ],
    }


def _markdown(value: dict[str, Any]) -> str:
    lines = [
        "# Metric-seam corrected verifier results",
        "",
        f"Status: **{value['status']}**.",
        "",
        "## Descriptive legacy static incidence (not whole-metric codability)",
        "",
        "| task | named sub-relation witness | percent | eligible depth counts |",
        "|---|---:|---:|---|",
    ]
    for row in value["legacy_static_subrelation_witness_incidence"]:
        rate = row["rate"]
        lines.append(
            f"| {row['task']} | {rate['numerator']}/{rate['denominator']} | "
            f"{rate['percent']:.1f}% | `{json.dumps(row['depth_counts'], sort_keys=True)}` |"
        )
    cr = value["verifier_native_results"]["code_review"]
    math = value["verifier_native_results"]["math_a12"]
    lines.extend(
        [
            "",
            "## Verifier-native results",
            "",
            f"- Code review: **{cr['natural_train_measurable']}/{cr['candidate_relations']}** "
            f"relations naturally measurable; plants {cr['planted_probe_detection']['numerator']}/"
            f"{cr['planted_probe_detection']['denominator']}. The result is corpus inadequacy on merged PRs.",
            f"- Math a12: **{math['natural_train_measurable']}/1** relation naturally measurable. "
            f"States: `{json.dumps(math['natural_states'], sort_keys=True)}`; probes "
            f"{math['planted_probe_detection']['numerator']}/{math['planted_probe_detection']['denominator']}.",
            f"- Math depth: {math['relation_depth']['documents_reaching_formal_solver']}/"
            f"{math['relation_depth']['documents_total']} TRAIN documents reach the depth-3 formal path; "
            f"{math['relation_depth']['documents_with_exact_relation_evidence']} yield exact relation evidence.",
            "",
            "The static percentages answer only whether some named sub-relation had a manually adjudicated executable witness. "
            "They do not say that 55.6%, 36.7%, or 6.7% of whole metrics are code-verifiable.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--math-agreement", type=Path)
    args = parser.parse_args(argv)
    h = args.root / "outputs/metric_seam_pilot/hierarchy_r123"
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    agreement = load(args.math_agreement) if args.math_agreement else None
    value = build_summary(
        code_review_funnel=load(h / "code_review_corrected_funnel_v1.json"),
        math_static=load(h / "math_stackexchange_construct_fidelity_merged_v1.json"),
        patents_static=load(h / "patents_construct_fidelity_v1.json"),
        science_static=load(h / "peer_review_science_claim_construct_fidelity_v1.json"),
        code_review_train=load(h / "results/code_review_ast_train_v2/readout.json"),
        math_train=load(h / "results/math_a12_symbolic_train_v1/readout.json"),
        technical_resolution=load(h / "results/technical_target_discrimination_v1/readout.json"),
        math_agreement=agreement,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "readout.json").write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(_markdown(value), encoding="utf-8")
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
