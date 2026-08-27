#!/usr/bin/env python3
"""Run the additive v2.3 relation-fidelity science comparator on continuous text."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core_relation_strict import metamorphic_self_check, verify_document
from .evaluate import _summarize, load_unlabelled


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "datasets/peer-review/peer_review_cv_evidence.jsonl"
DEFAULT_BASELINE = (
    ROOT / "outputs/metric_seam_pilot/science_claims_v2_corrected_v2/results.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs/metric_seam_pilot/science_claims_v2_relation_strict_v23"
)

_COUNTEREXAMPLES = {
    "percentage_metric_collision": "iclr_RVPZJpmyGU",
    "large_count_entity_collision": "iclr_IA3wm5vwUl",
    "nonbijective_near_value_reuse": "iclr_PIpGN5Ko3v",
    "codec_identifier_collision": "iclr_Z7aq3djHZw",
    "interrogative_comparison": "iclr_LRrbD8EZJl",
}

_EXPECTED_ADDED_REVIEW = {
    "iclr_dRel8fuUK4": {
        "executed_subrelation": "one_reference_model_availability",
        "scope_guard": (
            "does_not_execute_the_neighboring_exceptional_performance_or_random_guess_claim"
        ),
    },
    "iclr_lgsyLSsDRe": {
        "executed_subrelation": "evaluation_across_56_embedding_tasks",
        "scope_guard": (
            "does_not_execute_the_neighboring_top_rank_or_sustained_effectiveness_claim"
        ),
    },
    "iclr_uOb7rij7sR": {
        "executed_subrelation": "approximately_10x_training_speedup",
        "scope_guard": (
            "does_not_execute_the_neighboring_resolution_anisotropy_or_averaging_claim"
        ),
    },
}


def _certificate_key(paper_id: str, certificate: dict[str, Any]) -> tuple[str, ...]:
    return (
        paper_id,
        certificate["claim"]["text"],
        certificate["evidence"]["text"],
        certificate["claim"]["relation"],
        certificate["decision"],
    )


def _certificate_counter(records: list[dict[str, Any]]) -> Counter[tuple[str, ...]]:
    return Counter(
        _certificate_key(record["paper_id"], certificate)
        for record in records
        for certificate in record.get("certificates", [])
    )


def _status(records: list[dict[str, Any]], paper_id: str) -> dict[str, Any]:
    record = next(record for record in records if record["paper_id"] == paper_id)
    return {
        "status": record["status"],
        "certificates": record.get("certificate_count", 0),
        "reasons": [match["reason"] for match in record.get("matches", [])],
    }


def run(input_path: Path, baseline_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output_dir}")
    checks = metamorphic_self_check()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_records = baseline["records"]
    records = [
        verify_document(row["paper_id"], row["abstract"], row["body"])
        for row in load_unlabelled(input_path)
    ]
    before = _certificate_counter(baseline_records)
    after = _certificate_counter(records)
    added_keys = list((after - before).elements())
    added_paper_ids = {key[0] for key in added_keys}
    if added_paper_ids != set(_EXPECTED_ADDED_REVIEW) or len(added_keys) != len(
        _EXPECTED_ADDED_REVIEW
    ):
        raise AssertionError(
            "strict assignment introduced an unreviewed certificate identity: "
            f"{sorted(added_paper_ids)}"
        )
    before_summary = baseline["summary"]
    summary = _summarize(records)
    payload = {
        "schema_version": "science-claims-v2.3-relation-strict",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "unsupervised_code_reconstruction_relation_local",
        "external_supervision": "none",
        "input_fields": ["paper_id", "abstract", "body"],
        "method_origin": "manually_constructed_retrospective_seed",
        "certificate_scope": (
            "document_local_parser_witness_not_external_scientific_truth"
        ),
        "corrections": [
            "one_to_one_quantity_obligation_matching",
            "exact_normalized_value_and_unit_matching",
            "local_quantity_metric_or_entity_binding",
            "numeric_change_direction_when_articulated",
            "codec_identifier_exclusion",
            "assertive_comparison_requirement",
            "stronger_weaker_directed_role_parsing",
        ],
        "metamorphic_checks": checks,
        "summary": summary,
        "comparison_to_v22": {
            "baseline_certificates": sum(before.values()),
            "strict_certificates": sum(after.values()),
            "retained": sum((before & after).values()),
            "removed": sum((before - after).values()),
            "added": sum((after - before).values()),
            "added_identity_review": [
                {
                    "paper_id": key[0],
                    "relation": key[3],
                    "decision": key[4],
                    **_EXPECTED_ADDED_REVIEW[key[0]],
                    "interpretation": (
                        "retained_as_parser_accepted_relation_local_witness_not_whole_claim_support"
                    ),
                }
                for key in sorted(added_keys)
            ],
            "baseline_status_counts": before_summary["status_counts"],
            "counterexamples": {
                name: {
                    "paper_id": paper_id,
                    "v22": _status(baseline_records, paper_id),
                    "v23": _status(records, paper_id),
                }
                for name, paper_id in _COUNTEREXAMPLES.items()
            },
        },
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(render_report(payload), encoding="utf-8")
    return payload


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    comparison = payload["comparison_to_v22"]
    certificate_count = sum(summary["certificate_decisions"].values())
    certified_papers = sum(
        record.get("certificate_count", 0) > 0 for record in payload["records"]
    )
    cases = "\n".join(
        f"| {name} | `{case['paper_id']}` | {case['v22']['status']} / "
        f"{case['v22']['certificates']} | {case['v23']['status']} / "
        f"{case['v23']['certificates']} |"
        for name, case in comparison["counterexamples"].items()
    )
    additions = "\n".join(
        f"| `{row['paper_id']}` | {row['executed_subrelation']} | "
        f"{row['scope_guard']} |"
        for row in comparison["added_identity_review"]
    )
    return f"""# Science continuous-text relation comparator v2.3

Status: **completed on CPU**, with no model/API calls and no GPU use.

This is an additive correction; v2.2 remains unchanged. It runs the same manually selected
claim-selection, document-local retrieval, and exact claim↔evidence matching decomposition, but
tightens the executable relation predicate. Numeric obligations now require one-to-one value,
unit, local entity/metric, and articulated change-direction agreement. Comparative support must
be assertive and must preserve entity roles, baseline, and direction. Questions and hypothetical
comparisons cannot certify support.

## Result

- Parser-accepted strong relation witnesses: **{certificate_count}** across
  **{certified_papers}** papers
- Numeric: **{summary['certificate_relations'].get('numeric', 0)}**; comparative:
  **{summary['certificate_relations'].get('comparative', 0)}**
- Weak evidence links (kept separate): **{summary['evidence_link_count']}**
- Statuses: `{json.dumps(summary['status_counts'], sort_keys=True)}`
- v2.2 certificate identities retained/removed/added:
  **{comparison['retained']} / {comparison['removed']} / {comparison['added']}**
- Executable regression checks: **{sum(payload['metamorphic_checks'].values())} /
  {len(payload['metamorphic_checks'])}**

## Corpus counterexamples

| Failure mode | Paper | v2.2 status / certs | v2.3 status / certs |
|---|---|---:|---:|
{cases}

## Review of every newly introduced identity

| Paper | Executed sub-relation | Explicitly not established |
|---|---|---|
{additions}

These are the only identities added relative to v2.2. The evaluator fails if an unreviewed added
identity appears. Retention means only that the named sub-relation is executable under this
parser; it is not a manual truth label for the paper's whole claim.

The corrected certificates are relation-local parser witnesses over one paper. They do not
establish external scientific truth, and the manually built program is a retrospective mock of
a discovered decomposition rather than an automatically discovered verifier.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    payload = run(args.input.resolve(), args.baseline.resolve(), args.out.resolve())
    print(
        json.dumps(
            {
                "certificates": sum(payload["summary"]["certificate_decisions"].values()),
                "relations": payload["summary"]["certificate_relations"],
                "statuses": payload["summary"]["status_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
