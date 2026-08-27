"""Summarize frozen patent claim-structure heldout execution pre-reference.

The producer verifies the v1 heldout execution against the already-frozen
five-cell train gate.  It reports only relation-local code measurement and
finite certificate support.  Prompt articulability, reference reconstruction,
prompt/code isomorphism, whole-criterion codability, and external validity are
outside this artifact and remain unmeasured.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from methods.metric_seam.hierarchy_patent_claim_structure_gate import (
    DEFAULT_EXECUTION as DEFAULT_TRAIN_EXECUTION,
    DEFAULT_FIDELITY,
    DEFAULT_OUTPUT as DEFAULT_TRAIN_GATE,
    EXECUTION_SCHEMA,
    PROGRAM_SCHEMA,
    RELATION_IDS,
    SELECTED_IDS,
    STATIC_ONLY_IDS,
    build_patent_train_gate,
)


SCHEMA = "metric-seam.hierarchy-patent-claim-structure-operational-summary.v1"
STATUS = "heldout-relation-measurement-complete-pre-reference"
HELDOUT_BASENAME = "patents_claim_structure_heldout_pre_reference_v1.json"
DEFAULT_BASE = Path("outputs/metric_seam_pilot/hierarchy_r123")
DEFAULT_HELDOUT_EXECUTION = DEFAULT_BASE / HELDOUT_BASENAME
DEFAULT_OUTPUT = DEFAULT_BASE / "patents_claim_structure_operational_summary_v1.json"

_EXECUTION_FIELDS = {"schema", "program_schema", "phase", "design", "summary", "rows"}
_ROW_FIELDS = {
    "item_key", "status", "error_type", "representation",
    "relation_applicability", "result",
}
_REPRESENTATION_FIELDS = {
    "ctext_chars", "declared_max_chars", "at_declared_character_cap",
    "possibly_truncated_by_declared_character_cap",
    "whole_source_claim_set_completeness_established",
}
_APPLICABILITY_FIELDS = {
    "finite_witnesses_replayable_on_presented_bytes",
    "absence_or_whole_claim_set_inference_permitted",
    "train_gate_scope",
}


class PatentOperationalSummaryError(ValueError):
    """Raised when a heldout receipt or frozen binding fails closed."""


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _relation_profile(rows: Sequence[Mapping], relation_id: str) -> dict[str, Any]:
    values = []
    for row in rows:
        relation = row["result"]["relation_values"].get(relation_id, {})
        value = relation.get("value") if isinstance(relation, Mapping) else None
        if _is_number(value):
            values.append(float(value))
    return {
        "n_rows": len(rows),
        "n_measured": len(values),
        "n_abstained": len(rows) - len(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "n_unique_values": len(set(values)),
        "nonconstant": bool(values and min(values) < max(values)),
        "n_positive": sum(value > 0 for value in values),
        "n_zero": sum(value == 0 for value in values),
    }


def _runner_relation_summary(rows: Sequence[Mapping], relation_id: str) -> dict[str, Any]:
    profile = _relation_profile(rows, relation_id)
    return {
        key: profile[key]
        for key in ("n_measured", "n_abstained", "minimum", "maximum", "nonconstant")
    }


def _validate_selected_certificate(certificate: Mapping) -> None:
    relation = certificate.get("relation")
    kind = certificate.get("kind")
    if kind not in {"positive_witness", "counter_witness"}:
        raise PatentOperationalSummaryError("absence-like heldout certificate")
    if relation == "claim_dependency_well_formedness":
        if kind == "positive_witness":
            if set(certificate) != {"relation", "kind", "child_claim", "parent_claim"}:
                raise PatentOperationalSummaryError("invalid heldout dependency witness")
            child, parent = certificate["child_claim"], certificate["parent_claim"]
            if (
                not isinstance(child, int)
                or isinstance(child, bool)
                or not isinstance(parent, int)
                or isinstance(parent, bool)
                or not 0 < parent < child
            ):
                raise PatentOperationalSummaryError("invalid heldout dependency edge")
        elif (
            set(certificate) != {"relation", "kind", "child", "parent", "reasons"}
            or not isinstance(certificate.get("child"), int)
            or isinstance(certificate.get("child"), bool)
            or certificate["child"] <= 0
            or not isinstance(certificate.get("parent"), int)
            or isinstance(certificate.get("parent"), bool)
            or certificate["parent"] <= 0
            or not isinstance(certificate.get("reasons"), list)
            or not certificate["reasons"]
            or not all(
                isinstance(reason, str) and reason for reason in certificate["reasons"]
            )
        ):
            raise PatentOperationalSummaryError("invalid heldout local counter-witness")
    elif relation == "statutory_category_surface_coverage":
        span = certificate.get("span")
        if (
            kind != "positive_witness"
            or set(certificate)
            != {"relation", "kind", "claim", "category", "surface", "span"}
            or not isinstance(certificate.get("claim"), int)
            or isinstance(certificate.get("claim"), bool)
            or certificate["claim"] <= 0
            or not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in span)
            or not 0 <= span[0] < span[1]
            or not isinstance(certificate.get("surface"), str)
            or not certificate["surface"]
            or not isinstance(certificate.get("category"), str)
            or not certificate["category"]
        ):
            raise PatentOperationalSummaryError("invalid heldout category surface/span certificate")
    elif relation == "functional_limitation_incidence" and (
        kind != "positive_witness"
        or set(certificate) != {"relation", "kind", "claim", "surface"}
        or not isinstance(certificate.get("claim"), int)
        or isinstance(certificate.get("claim"), bool)
        or certificate["claim"] <= 0
        or not isinstance(certificate.get("surface"), str)
        or not certificate["surface"]
    ):
        raise PatentOperationalSummaryError("invalid heldout functional marker certificate")


def _validate_heldout_execution(
    execution: Mapping, *, source: str | Path
) -> list[Mapping]:
    if Path(str(source)).name != HELDOUT_BASENAME:
        raise PatentOperationalSummaryError("summary is frozen to heldout execution v1")
    if set(execution) != _EXECUTION_FIELDS:
        raise PatentOperationalSummaryError("heldout receipt fields drifted")
    if (
        execution.get("schema") != EXECUTION_SCHEMA
        or execution.get("program_schema") != PROGRAM_SCHEMA
        or execution.get("phase") != "heldout_pre_reference"
    ):
        raise PatentOperationalSummaryError("unexpected heldout runner, program, or phase")
    design = execution.get("design")
    if not isinstance(design, Mapping) or design.get("input_fields") != ["item_key", "ctext"]:
        raise PatentOperationalSummaryError("heldout receipt violates the text-only contract")
    for field in (
        "outcome_or_reference_values_loaded",
        "prompt_outputs_loaded",
        "prior_art_or_examiner_evidence_loaded",
        "external_supervision_used",
        "whole_patent_score_emitted",
        "absence_certificate_permitted",
    ):
        if design.get(field) is not False:
            raise PatentOperationalSummaryError(f"heldout execution violated {field}")
    if (
        design.get("declared_representation_max_chars") != 4000
        or design.get("finite_local_counter_witness_permitted") is not True
        or design.get("at_cap_is_treated_as_possible_truncation") is not True
    ):
        raise PatentOperationalSummaryError("heldout cap contract drifted")

    rows = execution.get("rows")
    if not isinstance(rows, list) or len(rows) != 150:
        raise PatentOperationalSummaryError("heldout v1 must contain 150 rows")
    keys = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
            raise PatentOperationalSummaryError("invalid heldout row")
        key = row.get("item_key")
        if not isinstance(key, str) or not key.startswith("heldout_"):
            raise PatentOperationalSummaryError("invalid opaque heldout key")
        keys.append(key)
        if row.get("error_type") is not None or row.get("status") == "failed":
            raise PatentOperationalSummaryError("heldout receipt contains an execution failure")
        representation = row.get("representation")
        applicability = row.get("relation_applicability")
        if not isinstance(representation, Mapping) or set(representation) != (
            _REPRESENTATION_FIELDS
        ):
            raise PatentOperationalSummaryError("invalid heldout representation receipt")
        if not isinstance(applicability, Mapping) or set(applicability) != (
            _APPLICABILITY_FIELDS
        ):
            raise PatentOperationalSummaryError("invalid heldout applicability receipt")
        chars = representation.get("ctext_chars")
        at_cap = representation.get("at_declared_character_cap")
        if (
            not isinstance(chars, int)
            or isinstance(chars, bool)
            or not 0 < chars <= 4000
            or representation.get("declared_max_chars") != 4000
            or at_cap is not (chars == 4000)
            or representation.get("possibly_truncated_by_declared_character_cap") is not at_cap
            or representation.get("whole_source_claim_set_completeness_established") is not False
        ):
            raise PatentOperationalSummaryError("heldout representation accounting drifted")
        if (
            applicability.get("finite_witnesses_replayable_on_presented_bytes") is not True
            or applicability.get("absence_or_whole_claim_set_inference_permitted") is not False
            or applicability.get("train_gate_scope")
            != ("finite_witnesses_only" if at_cap else "presented_text_relations_and_finite_witnesses")
        ):
            raise PatentOperationalSummaryError("heldout applicability policy drifted")
        result = row.get("result")
        if not isinstance(result, Mapping) or (
            result.get("schema") != PROGRAM_SCHEMA
            or result.get("channel") != "pure_code"
            or result.get("maximum_decision_contributing_depth") != 2
            or result.get("aggregation_rule") is not None
        ):
            raise PatentOperationalSummaryError("heldout program result contract drifted")
        scope = result.get("scope")
        if not isinstance(scope, Mapping):
            raise PatentOperationalSummaryError("heldout result lacks scope limits")
        for field in (
            "external_supervision_used",
            "prior_art_or_examiner_evidence_used",
            "whole_patent_construct_established",
            "legal_validity_or_patentability_established",
            "verified_absence_established",
        ):
            if scope.get(field) is not False:
                raise PatentOperationalSummaryError(f"heldout result violated {field}")
        relations = result.get("relation_values")
        if not isinstance(relations, Mapping) or set(relations) != set(RELATION_IDS):
            raise PatentOperationalSummaryError("heldout relation inventory drifted")
        for relation in relations.values():
            if (
                not isinstance(relation, Mapping)
                or set(relation) != {"value", "support"}
                or (relation["value"] is not None and not _is_number(relation["value"]))
                or not isinstance(relation["support"], Mapping)
            ):
                raise PatentOperationalSummaryError("invalid heldout relation value")
        certificates = result.get("certificates")
        if not isinstance(certificates, list):
            raise PatentOperationalSummaryError("invalid heldout certificate list")
        for certificate in certificates:
            if not isinstance(certificate, Mapping):
                raise PatentOperationalSummaryError("invalid heldout certificate")
            if certificate.get("relation") in {
                "claim_dependency_well_formedness",
                "statutory_category_surface_coverage",
                "functional_limitation_incidence",
            }:
                _validate_selected_certificate(certificate)
            elif certificate.get("kind") != "positive_witness":
                raise PatentOperationalSummaryError("unexpected non-positive auxiliary certificate")
        claims = result.get("claims")
        expected_status = (
            "relation_abstained"
            if not claims
            else "measured_with_possible_truncation"
            if at_cap
            else "measured"
        )
        if row.get("status") != expected_status:
            raise PatentOperationalSummaryError("heldout row status drifted")
    if len(set(keys)) != len(keys):
        raise PatentOperationalSummaryError("duplicate heldout item key")

    cap_count = sum(row["representation"]["at_declared_character_cap"] for row in rows)
    if cap_count != 123:
        raise PatentOperationalSummaryError("heldout v1 cap-contact count drifted")
    statuses = Counter(row["status"] for row in rows)
    certificates = Counter(
        certificate["relation"]
        for row in rows
        for certificate in row["result"]["certificates"]
    )
    expected_summary = {
        "n_items": 150,
        "status_counts": dict(sorted(statuses.items())),
        "failure_types": {},
        "items_at_declared_character_cap": 123,
        "items_measured_with_possible_truncation": statuses.get(
            "measured_with_possible_truncation", 0
        ),
        "relation_measurement": {
            relation_id: _runner_relation_summary(rows, relation_id)
            for relation_id in RELATION_IDS
        },
        "certificate_counts": dict(sorted(certificates.items())),
    }
    if execution.get("summary") != expected_summary:
        raise PatentOperationalSummaryError("heldout summary does not replay from its rows")
    return rows


def _certificate_profile(
    rows: Sequence[Mapping], relation_id: str
) -> dict[str, Any]:
    certificates = []
    row_keys = set()
    for row in rows:
        matched = [
            certificate for certificate in row["result"]["certificates"]
            if certificate["relation"] == relation_id
        ]
        if matched:
            row_keys.add(row["item_key"])
            certificates.extend(matched)
    normalized = {
        json.dumps(certificate, sort_keys=True, separators=(",", ":"))
        for certificate in certificates
    }
    return {
        "n_rows": len(rows),
        "n_rows_with_positive_or_local_counter_certificate": len(row_keys),
        "n_rows_without_certificate_treated_as_abstention_not_absence": len(rows) - len(row_keys),
        "n_certificates": len(certificates),
        "n_distinct_certificate_payloads": len(normalized),
        "certificate_kind_counts": dict(sorted(Counter(
            certificate["kind"] for certificate in certificates
        ).items())),
    }


def _certificate_diversity(rows: Sequence[Mapping], relation_id: str) -> dict[str, Any]:
    certificates = [
        certificate
        for row in rows
        for certificate in row["result"]["certificates"]
        if certificate["relation"] == relation_id
    ]
    return {
        "n_distinct_surfaces": len({
            certificate.get("surface") for certificate in certificates
            if certificate.get("surface")
        }),
        "n_distinct_categories": len({
            certificate.get("category") for certificate in certificates
            if certificate.get("category")
        }),
    }


def _local_layering_profile(rows: Sequence[Mapping]) -> dict[str, Any]:
    positive = 0
    for row in rows:
        relation = row["result"]["relation_values"]["claim_set_layering"]
        if relation["value"] != 1.0:
            continue
        support = relation["support"]
        edges = [
            certificate for certificate in row["result"]["certificates"]
            if certificate.get("relation") == "claim_dependency_well_formedness"
            and certificate.get("kind") == "positive_witness"
        ]
        if (
            not _is_number(support.get("independent_claims"))
            or support["independent_claims"] < 1
            or not support.get("validly_linked_dependent_claims")
            or not edges
        ):
            raise PatentOperationalSummaryError(
                "heldout positive layering value lacks a finite local witness"
            )
        positive += 1
    return {
        "n_rows": len(rows),
        "n_positive_local_root_plus_edge_witnesses": positive,
        "n_rows_without_positive_witness_treated_as_abstention_not_source_absence": (
            len(rows) - positive
        ),
    }


def _validate_gate_binding(
    train_execution: Mapping,
    fidelity: Mapping,
    gate: Mapping,
    *,
    gate_source: str | Path,
) -> None:
    if Path(str(gate_source)).name != DEFAULT_TRAIN_GATE.name:
        raise PatentOperationalSummaryError("unexpected frozen train-gate source")
    bindings = gate.get("bindings")
    if not isinstance(bindings, Mapping):
        raise PatentOperationalSummaryError("frozen gate has no source bindings")
    rebuilt = build_patent_train_gate(
        train_execution,
        fidelity,
        execution_source=bindings.get("compiler_train_source", ""),
        fidelity_source=bindings.get("construct_fidelity_source", ""),
    )
    if gate != rebuilt:
        raise PatentOperationalSummaryError("frozen train gate does not replay exactly")


def build_operational_summary(
    train_execution: Mapping,
    fidelity: Mapping,
    gate: Mapping,
    heldout_execution: Mapping,
    *,
    train_gate_source: str | Path = DEFAULT_TRAIN_GATE,
    heldout_execution_source: str | Path = DEFAULT_HELDOUT_EXECUTION,
) -> dict[str, Any]:
    """Build a five-cell heldout relation-measurement summary pre-reference."""

    _validate_gate_binding(
        train_execution, fidelity, gate, gate_source=train_gate_source
    )
    rows = _validate_heldout_execution(
        heldout_execution, source=heldout_execution_source
    )
    cap_rows = [row for row in rows if row["representation"]["at_declared_character_cap"]]
    below_rows = [row for row in rows if not row["representation"]["at_declared_character_cap"]]
    all_profiles = {
        relation_id: _relation_profile(rows, relation_id) for relation_id in RELATION_IDS
    }
    below_profiles = {
        relation_id: _relation_profile(below_rows, relation_id) for relation_id in RELATION_IDS
    }
    dependency = _certificate_profile(rows, "claim_dependency_well_formedness")
    dependency_cap = _certificate_profile(cap_rows, "claim_dependency_well_formedness")
    category = _certificate_profile(rows, "statutory_category_surface_coverage")
    category_cap = _certificate_profile(cap_rows, "statutory_category_surface_coverage")
    category_diversity = _certificate_diversity(
        rows, "statutory_category_surface_coverage"
    )
    functional = _certificate_profile(rows, "functional_limitation_incidence")
    functional_cap = _certificate_profile(cap_rows, "functional_limitation_incidence")
    functional_diversity = _certificate_diversity(
        rows, "functional_limitation_incidence"
    )
    layering = _local_layering_profile(rows)
    layering_cap = _local_layering_profile(cap_rows)

    selected_by_id = {
        row["cell_id"]: row for row in gate["selected_operational_cells"]
    }
    if tuple(selected_by_id) != SELECTED_IDS or set(selected_by_id) != set(SELECTED_IDS):
        raise PatentOperationalSummaryError("frozen five-cell gate order drifted")
    operational_cells = []
    for cell_id in SELECTED_IDS:
        frozen = selected_by_id[cell_id]
        relation_results = []
        for specification in frozen["relations"]:
            relation_id = specification["relation_id"]
            mode = specification["output_mode"]
            if mode == "finite_dependency_edge_and_local_counter_witnesses":
                evidence = {
                    "all_presented_rows": dependency,
                    "cap_contact_rows": dependency_cap,
                }
                nonconstant = dependency["n_distinct_certificate_payloads"] >= 2
            elif mode == "positive_local_root_plus_dependent_edge_witness":
                evidence = {
                    "all_presented_rows": layering,
                    "cap_contact_rows": layering_cap,
                    "below_cap_relation_profile": below_profiles[relation_id],
                }
                nonconstant = below_profiles[relation_id]["nonconstant"] is True
            elif mode == "positive_category_surface_span_certificates_only":
                evidence = {
                    "all_presented_rows": category,
                    "cap_contact_rows": category_cap,
                    "certificate_diversity": category_diversity,
                    "coverage_scalar_operationally_used": False,
                }
                nonconstant = (
                    category["n_certificates"] > 0
                    and category_diversity["n_distinct_categories"] >= 2
                    and category_diversity["n_distinct_surfaces"] >= 2
                )
            elif mode == (
                "positive_marker_certificates_plus_below_cap_presented_text_incidence"
            ):
                evidence = {
                    "all_presented_rows": functional,
                    "cap_contact_rows": functional_cap,
                    "certificate_diversity": functional_diversity,
                    "below_cap_relation_profile": below_profiles[relation_id],
                }
                nonconstant = (
                    functional["n_certificates"] > 0
                    and below_profiles[relation_id]["nonconstant"] is True
                )
            elif mode == "exact_named_presented_abstract_word_count":
                evidence = {
                    "all_presented_rows_relation_profile": all_profiles[relation_id],
                    "below_cap_relation_profile": below_profiles[relation_id],
                }
                nonconstant = all_profiles[relation_id]["nonconstant"] is True
            else:
                raise PatentOperationalSummaryError(f"unknown frozen output mode: {mode}")
            if not nonconstant:
                raise PatentOperationalSummaryError(
                    f"heldout output is not nonconstant under the frozen mode: {cell_id}"
                )
            relation_results.append({
                "relation_id": relation_id,
                "effective_code_depth": specification["effective_code_depth"],
                "output_mode": mode,
                "heldout_relation_measurable": True,
                "nonconstant_under_frozen_output_contract": True,
                "evidence": evidence,
                "absence_or_whole_source_inference_permitted": False,
            })
        operational_cells.append({
            "cell_id": cell_id,
            "level": frozen["level"],
            "selection_rank": frozen["selection_rank"],
            "construct": frozen["construct"],
            "maximum_decision_contributing_depth": frozen[
                "maximum_decision_contributing_depth"
            ],
            "ordered_relation_ids": frozen["ordered_relation_ids"],
            "heldout_relation_measurable": True,
            "relations": relation_results,
            "prompt_articulability_measured": False,
            "reference_reconstruction_measured": False,
            "prompt_code_isomorphism_measured": False,
            "whole_criterion_codability_established": False,
        })

    static_only = []
    gate_static_by_id = {row["cell_id"]: row for row in gate["static_only_cells"]}
    if tuple(gate_static_by_id) != STATIC_ONLY_IDS:
        raise PatentOperationalSummaryError("static-only gate order drifted")
    heldout_section = all_profiles["application_section_presence"]
    if heldout_section["nonconstant"] is not False or heldout_section["n_measured"] != 150:
        raise PatentOperationalSummaryError("heldout section relation is no longer constant")
    for cell_id in STATIC_ONLY_IDS:
        frozen = gate_static_by_id[cell_id]
        static_only.append({
            "cell_id": cell_id,
            "level": frozen["level"],
            "selection_rank": frozen["selection_rank"],
            "construct": frozen["construct"],
            "relation_id": "application_section_presence",
            "heldout_relation_profile": heldout_section,
            "heldout_relation_measurable": False,
            "status": "static-fidelity-only; constant on both train and heldout",
        })

    return {
        "schema": SCHEMA,
        "status": STATUS,
        "task": "patents",
        "bindings": {
            "frozen_train_gate_source": str(train_gate_source),
            "frozen_train_gate_schema": gate["schema"],
            "compiler_train_source": gate["bindings"]["compiler_train_source"],
            "construct_fidelity_source": gate["bindings"]["construct_fidelity_source"],
            "heldout_execution_source": str(heldout_execution_source),
            "runner_schema": EXECUTION_SCHEMA,
            "program_schema": PROGRAM_SCHEMA,
        },
        "channel_boundaries": {
            "heldout_input_fields": ["item_key", "ctext"],
            "selection_frozen_before_heldout_execution": True,
            "reference_or_prompt_values_loaded": False,
            "outcomes_loaded": False,
            "prior_art_or_examiner_evidence_loaded": False,
            "external_supervision_loaded": False,
            "models_or_apis_called": False,
            "whole_patent_score_emitted": False,
        },
        "heldout_receipt": {
            "n_rows": len(rows),
            "n_cap_contact_rows": len(cap_rows),
            "n_cap_contact_rows_with_claim_measurement": sum(
                row["status"] == "measured_with_possible_truncation" for row in cap_rows
            ),
            "n_below_cap_rows": len(below_rows),
            "status_counts": heldout_execution["summary"]["status_counts"],
            "failure_types": heldout_execution["summary"]["failure_types"],
        },
        "cap_policy": {
            "cap_contact_output_modes": [
                "finite_dependency_edge_and_local_counter_witnesses",
                "positive_local_root_plus_dependent_edge_witness",
                "positive_category_surface_span_certificates_only",
                "positive functional marker certificates",
                "exact named presented ABSTRACT word count",
            ],
            "zero_or_missing_certificate_on_cap_contact_is": "abstention, never verified absence",
            "whole_source_completeness_or_compliance_permitted": False,
            "category_coverage_scalar_operationally_used": False,
        },
        "stage_summary": {
            "n_static_relation_local_cells": 8,
            "n_train_operational_cells": 5,
            "n_heldout_relation_measurable_cells": 5,
            "n_static_only_constant_cells": 3,
            "n_prompt_articulability_measured_cells": 0,
            "n_reference_reconstruction_measured_cells": 0,
            "n_prompt_code_isomorphism_evaluable_cells": 0,
            "n_whole_criterion_codability_established_cells": 0,
            "heldout_relation_measurable_by_level": {"R2": 1, "R3": 4},
            "heldout_relation_measurable_by_maximum_depth": {"1": 4, "2": 1},
        },
        "heldout_operational_cells": operational_cells,
        "static_only_cells": static_only,
        "interpretation_limits": [
            "Five cells have nonconstant heldout outputs under the frozen relation-local code contracts; this is not five whole metrics reconstructed or coded.",
            "Articulability is prompt-based and remains unmeasured because no prompt outputs were loaded.",
            "Reference reconstruction and prompt/code isomorphism remain unmeasured.",
            "The frozen LLM judgement is a future reconstruction reference, not external ground truth.",
            "No whole-criterion codability percentage or tacitness conclusion is licensed by this artifact.",
            "Cap-contact rows contribute finite/local positive or counter evidence only; missing evidence is an abstention, never verified absence.",
        ],
    }


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite operational summary: {path}")
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-execution", type=Path, default=DEFAULT_TRAIN_EXECUTION)
    parser.add_argument("--fidelity", type=Path, default=DEFAULT_FIDELITY)
    parser.add_argument("--train-gate", type=Path, default=DEFAULT_TRAIN_GATE)
    parser.add_argument("--heldout-execution", type=Path, default=DEFAULT_HELDOUT_EXECUTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_operational_summary(
        _load(args.train_execution),
        _load(args.fidelity),
        _load(args.train_gate),
        _load(args.heldout_execution),
        train_gate_source=args.train_gate,
        heldout_execution_source=args.heldout_execution,
    )
    _write_new(args.output, payload)
    print(json.dumps(payload["stage_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
