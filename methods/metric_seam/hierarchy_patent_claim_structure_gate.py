"""Freeze the patent claim-structure operational contract on compiler train.

This gate is deliberately relation-local.  It binds the conservative eight-cell
construct map to the v14 compiler-train receipt, retains three constant section
relations as static fidelity findings, and selects five cells with replayable
outputs.  It never reads heldout items, prompt/reference values, outcomes,
examiner evidence, or prior art.

Rows at the 4,000-character representation cap contribute only finite positive
witnesses, finite local counter-witnesses, or an exact measurement of a closed
named section in the presented bytes.  In particular, a zero scalar on a capped
row is never treated as verified absence or whole-claim-set evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "metric-seam.hierarchy-patent-claim-structure-train-gate.v1"
STATUS = "frozen-before-heldout-pre-reference-execution"
TASK = "patents"
EXECUTION_SCHEMA = "metric-seam.hierarchy-patent-claim-structure-execution.v3"
PROGRAM_SCHEMA = "metric-seam.patent-claim-structure.v13"
FIDELITY_SCHEMA = "metric-seam.hierarchy-patent-claim-structure-fidelity.v1"
EXECUTION_BASENAME = "patents_claim_structure_compiler_train_v14.json"
FIDELITY_BASENAME = "patents_claim_structure_construct_fidelity_v1.json"
DEFAULT_BASE = Path("outputs/metric_seam_pilot/hierarchy_r123")
DEFAULT_EXECUTION = DEFAULT_BASE / EXECUTION_BASENAME
DEFAULT_FIDELITY = DEFAULT_BASE / FIDELITY_BASENAME
DEFAULT_OUTPUT = DEFAULT_BASE / "patents_claim_structure_train_gate_v1.json"

RELATION_IDS = (
    "application_section_presence",
    "claim_number_contiguity",
    "claim_dependency_well_formedness",
    "claim_set_layering",
    "antecedent_reference_surface_coverage",
    "statutory_category_surface_coverage",
    "functional_limitation_incidence",
    "numerical_limitation_incidence",
    "abstract_word_count",
)

R1_SECTION = "TB::patents::specific::R1::merged_tree::151::a6737bddab8d451d7ae9"
R2_FUNCTIONAL = "TB::patents::specific::R2::grandparent::10::41a099074657b4acc7f5"
R2_SECTION = "TB::patents::specific::R2::merged_group::40::bb89d6d56dcc9ea9c238"
R3_FUNCTIONAL = "TB::patents::specific::R3::grandparent::0::ed76386d4408681be502"
R3_SECTION = "TB::patents::specific::R3::merged_group::12::4a62e79af29087e6ff96"
R3_ABSTRACT = "TB::patents::specific::R3::merged_group::3::6d907639386384acc1da"
R3_ARCHITECTURE = "TB::patents::specific::R3::grandparent::14::b26fd00c6c47f2854678"
R3_CATEGORY = "TB::patents::specific::R3::merged_group::7::ac30b4e148a5c6a11ec7"

EXPECTED_MATCHES = {
    R1_SECTION: ("application_section_presence",),
    R2_FUNCTIONAL: ("functional_limitation_incidence",),
    R2_SECTION: ("application_section_presence",),
    R3_FUNCTIONAL: ("functional_limitation_incidence",),
    R3_SECTION: ("application_section_presence",),
    R3_ABSTRACT: ("abstract_word_count",),
    R3_ARCHITECTURE: (
        "claim_dependency_well_formedness",
        "claim_set_layering",
    ),
    R3_CATEGORY: ("statutory_category_surface_coverage",),
}
STATIC_ONLY_IDS = (R1_SECTION, R2_SECTION, R3_SECTION)
SELECTED_IDS = (
    R2_FUNCTIONAL,
    R3_FUNCTIONAL,
    R3_ABSTRACT,
    R3_ARCHITECTURE,
    R3_CATEGORY,
)

_EXECUTION_FIELDS = {"schema", "program_schema", "phase", "design", "summary", "rows"}
_ROW_FIELDS = {
    "item_key",
    "status",
    "error_type",
    "representation",
    "relation_applicability",
    "result",
}
_REPRESENTATION_FIELDS = {
    "ctext_chars",
    "declared_max_chars",
    "at_declared_character_cap",
    "possibly_truncated_by_declared_character_cap",
    "whole_source_claim_set_completeness_established",
}
_APPLICABILITY_FIELDS = {
    "finite_witnesses_replayable_on_presented_bytes",
    "absence_or_whole_claim_set_inference_permitted",
    "train_gate_scope",
}
_FIDELITY_ROW_FIELDS = {
    "cell_id",
    "task",
    "level",
    "selection_rank",
    "construct",
    "description",
    "verdict",
    "matched_relations",
    "eligible_relation_local_depths",
    "maximum_matching_relation_depth",
    "exact_whole_construct_fidelity",
    "rejection_or_demotion_reason",
    "sensitivity_near_miss",
}
_MATCH_FIELDS = {
    "relation_id",
    "partial_scope",
    "exclusions",
    "certificate_policy",
    "implemented_relation",
    "channel",
    "effective_code_depth",
    "train_operational_applicability",
}


class PatentTrainGateError(ValueError):
    """Raised when a frozen train input fails the operational contract."""


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _source_name(source: str | Path | None) -> str:
    return Path(str(source or "")).name


def _relation_profile(rows: Sequence[Mapping], relation_id: str) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        result = row.get("result")
        if not isinstance(result, Mapping):
            continue
        relation = result.get("relation_values", {}).get(relation_id)
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


def _validate_certificate(certificate: Mapping) -> None:
    relation = certificate.get("relation")
    kind = certificate.get("kind")
    if relation not in RELATION_IDS or kind not in {"positive_witness", "counter_witness"}:
        raise PatentTrainGateError("invalid or absence-like certificate")
    if relation == "claim_dependency_well_formedness":
        if kind == "positive_witness":
            if set(certificate) != {"relation", "kind", "child_claim", "parent_claim"}:
                raise PatentTrainGateError("invalid finite dependency witness")
            child, parent = certificate["child_claim"], certificate["parent_claim"]
            if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0
                       for value in (child, parent)) or parent >= child:
                raise PatentTrainGateError("invalid finite dependency edge")
        else:
            if set(certificate) != {"relation", "kind", "child", "parent", "reasons"}:
                raise PatentTrainGateError("invalid local dependency counter-witness")
            reasons = certificate["reasons"]
            if (
                not isinstance(certificate["child"], int)
                or not isinstance(certificate["parent"], int)
                or not isinstance(reasons, list)
                or not reasons
                or not all(isinstance(reason, str) and reason for reason in reasons)
            ):
                raise PatentTrainGateError("invalid local dependency counter-witness")
    elif relation == "statutory_category_surface_coverage":
        if set(certificate) != {
            "relation", "kind", "claim", "category", "surface", "span"
        } or kind != "positive_witness":
            raise PatentTrainGateError("category channel permits positive span certificates only")
        span = certificate["span"]
        if (
            not isinstance(certificate["claim"], int)
            or certificate["claim"] <= 0
            or not isinstance(certificate["category"], str)
            or not certificate["category"]
            or not isinstance(certificate["surface"], str)
            or not certificate["surface"]
            or not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in span)
            or not 0 <= span[0] < span[1]
        ):
            raise PatentTrainGateError("invalid category surface/span certificate")
    elif relation == "functional_limitation_incidence":
        if set(certificate) != {"relation", "kind", "claim", "surface"} or kind != (
            "positive_witness"
        ):
            raise PatentTrainGateError("functional channel permits positive markers only")
        if (
            not isinstance(certificate["claim"], int)
            or certificate["claim"] <= 0
            or not isinstance(certificate["surface"], str)
            or not certificate["surface"]
        ):
            raise PatentTrainGateError("invalid functional marker certificate")


def _validate_execution(execution: Mapping, *, source: str | Path | None) -> list[Mapping]:
    if _source_name(source) != EXECUTION_BASENAME:
        raise PatentTrainGateError("gate is frozen to compiler-train receipt v14")
    if set(execution) != _EXECUTION_FIELDS:
        raise PatentTrainGateError("compiler-train receipt fields drifted")
    if (
        execution.get("schema") != EXECUTION_SCHEMA
        or execution.get("program_schema") != PROGRAM_SCHEMA
        or execution.get("phase") != "compiler_train"
    ):
        raise PatentTrainGateError("unexpected runner, program, or phase")
    design = execution.get("design")
    if not isinstance(design, Mapping):
        raise PatentTrainGateError("missing compiler-train design receipt")
    if design.get("input_fields") != ["item_key", "ctext"]:
        raise PatentTrainGateError("compiler train did not use the text-only contract")
    for field in (
        "outcome_or_reference_values_loaded",
        "prompt_outputs_loaded",
        "prior_art_or_examiner_evidence_loaded",
        "external_supervision_used",
        "whole_patent_score_emitted",
        "absence_certificate_permitted",
    ):
        if design.get(field) is not False:
            raise PatentTrainGateError(f"compiler train violated {field}")
    if (
        design.get("declared_representation_max_chars") != 4000
        or design.get("finite_local_counter_witness_permitted") is not True
        or design.get("at_cap_is_treated_as_possible_truncation") is not True
    ):
        raise PatentTrainGateError("compiler-train cap contract drifted")

    rows = execution.get("rows")
    if not isinstance(rows, list) or len(rows) != 150:
        raise PatentTrainGateError("v14 must contain exactly 150 compiler-train rows")
    keys: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
            raise PatentTrainGateError("invalid compiler-train row")
        key = row.get("item_key")
        if not isinstance(key, str) or not key.startswith("train_"):
            raise PatentTrainGateError("invalid opaque compiler-train key")
        keys.append(key)
        if row.get("error_type") is not None or row.get("status") == "failed":
            raise PatentTrainGateError("v14 contains an execution failure")
        representation = row.get("representation")
        applicability = row.get("relation_applicability")
        if not isinstance(representation, Mapping) or set(representation) != (
            _REPRESENTATION_FIELDS
        ):
            raise PatentTrainGateError("invalid representation receipt")
        if not isinstance(applicability, Mapping) or set(applicability) != (
            _APPLICABILITY_FIELDS
        ):
            raise PatentTrainGateError("invalid relation-applicability receipt")
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
            raise PatentTrainGateError("representation cap accounting drifted")
        if (
            applicability.get("finite_witnesses_replayable_on_presented_bytes") is not True
            or applicability.get("absence_or_whole_claim_set_inference_permitted") is not False
            or applicability.get("train_gate_scope")
            != ("finite_witnesses_only" if at_cap else "presented_text_relations_and_finite_witnesses")
        ):
            raise PatentTrainGateError("row relation-applicability policy drifted")
        result = row.get("result")
        if not isinstance(result, Mapping):
            raise PatentTrainGateError("v14 row has no program result")
        if (
            result.get("schema") != PROGRAM_SCHEMA
            or result.get("channel") != "pure_code"
            or result.get("maximum_decision_contributing_depth") != 2
            or result.get("aggregation_rule") is not None
        ):
            raise PatentTrainGateError("program result contract drifted")
        scope = result.get("scope")
        if not isinstance(scope, Mapping):
            raise PatentTrainGateError("program result lacks scope limits")
        for field in (
            "external_supervision_used",
            "prior_art_or_examiner_evidence_used",
            "whole_patent_construct_established",
            "legal_validity_or_patentability_established",
            "verified_absence_established",
        ):
            if scope.get(field) is not False:
                raise PatentTrainGateError(f"program result violated {field}")
        relations = result.get("relation_values")
        if not isinstance(relations, Mapping) or set(relations) != set(RELATION_IDS):
            raise PatentTrainGateError("program relation inventory drifted")
        for relation in relations.values():
            if (
                not isinstance(relation, Mapping)
                or set(relation) != {"value", "support"}
                or (relation["value"] is not None and not _is_number(relation["value"]))
                or not isinstance(relation["support"], Mapping)
            ):
                raise PatentTrainGateError("invalid compiler-train relation value")
        certificates = result.get("certificates")
        if not isinstance(certificates, list):
            raise PatentTrainGateError("invalid certificate list")
        for certificate in certificates:
            if not isinstance(certificate, Mapping):
                raise PatentTrainGateError("invalid certificate")
            _validate_certificate(certificate)
    if len(set(keys)) != len(keys):
        raise PatentTrainGateError("duplicate compiler-train item key")

    at_cap_count = sum(row["representation"]["at_declared_character_cap"] for row in rows)
    if at_cap_count != 119:
        raise PatentTrainGateError("v14 cap-contact count drifted")
    status_counts = Counter(row["status"] for row in rows)
    certificate_counts = Counter(
        certificate["relation"]
        for row in rows
        for certificate in row["result"]["certificates"]
    )
    expected_summary = {
        "n_items": 150,
        "status_counts": dict(sorted(status_counts.items())),
        "failure_types": {},
        "items_at_declared_character_cap": 119,
        "items_measured_with_possible_truncation": status_counts.get(
            "measured_with_possible_truncation", 0
        ),
        "relation_measurement": {
            relation_id: _runner_relation_summary(rows, relation_id)
            for relation_id in RELATION_IDS
        },
        "certificate_counts": dict(sorted(certificate_counts.items())),
    }
    if execution.get("summary") != expected_summary:
        raise PatentTrainGateError("v14 summary does not replay from its rows")
    return rows


def _validate_fidelity(
    fidelity: Mapping,
    rows: Sequence[Mapping],
    *,
    source: str | Path | None,
) -> dict[str, Mapping]:
    if _source_name(source) != FIDELITY_BASENAME:
        raise PatentTrainGateError("gate is frozen to construct-fidelity v1")
    if (
        fidelity.get("schema") != FIDELITY_SCHEMA
        or fidelity.get("status") != "conservative-static-adjudication-complete"
        or fidelity.get("task") != TASK
        or fidelity.get("n_cells") != 90
        or fidelity.get("program_schema") != PROGRAM_SCHEMA
        or fidelity.get("train_receipt_schema") != EXECUTION_SCHEMA
        or fidelity.get("forbidden_inputs_loaded") is not False
        or fidelity.get("execution_performed_by_this_audit") is not False
    ):
        raise PatentTrainGateError("construct-fidelity contract drifted")
    map_rows = fidelity.get("rows")
    if not isinstance(map_rows, list) or len(map_rows) != 90:
        raise PatentTrainGateError("construct-fidelity map must contain 90 rows")
    by_id: dict[str, Mapping] = {}
    for row in map_rows:
        if not isinstance(row, Mapping) or set(row) != _FIDELITY_ROW_FIELDS:
            raise PatentTrainGateError("invalid construct-fidelity row")
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or cell_id in by_id or row.get("task") != TASK:
            raise PatentTrainGateError("invalid or duplicate construct-fidelity identity")
        by_id[cell_id] = row
        if row.get("exact_whole_construct_fidelity") is not False:
            raise PatentTrainGateError("claim-structure map promoted a whole construct")
    partial = {
        cell_id: row for cell_id, row in by_id.items()
        if row.get("verdict") == "partial_relation_local"
    }
    if set(partial) != set(EXPECTED_MATCHES):
        raise PatentTrainGateError("the conservative eight-cell map drifted")
    full_profiles = {
        relation_id: _runner_relation_summary(rows, relation_id)
        for relation_id in RELATION_IDS
    }
    for cell_id, relation_ids in EXPECTED_MATCHES.items():
        row = partial[cell_id]
        matches = row.get("matched_relations")
        if not isinstance(matches, list) or tuple(
            match.get("relation_id") for match in matches if isinstance(match, Mapping)
        ) != relation_ids:
            raise PatentTrainGateError(f"{cell_id}: matched relation order drifted")
        expected_depth = 2 if cell_id == R3_ARCHITECTURE else 1
        if (
            row.get("maximum_matching_relation_depth") != expected_depth
            or row.get("eligible_relation_local_depths") != [expected_depth]
        ):
            raise PatentTrainGateError(f"{cell_id}: relation depth drifted")
        for match in matches:
            if set(match) != _MATCH_FIELDS or match.get("channel") != "code":
                raise PatentTrainGateError(f"{cell_id}: invalid matched relation")
            relation_id = match["relation_id"]
            if match.get("effective_code_depth") != expected_depth:
                raise PatentTrainGateError(f"{cell_id}: effective depth drifted")
            applicability = match.get("train_operational_applicability")
            if not isinstance(applicability, Mapping):
                raise PatentTrainGateError(f"{cell_id}: missing train applicability")
            if applicability.get("absence_or_whole_source_inference_permitted") is not False:
                raise PatentTrainGateError(f"{cell_id}: forbidden absence inference")
            for field, expected in full_profiles[relation_id].items():
                if applicability.get(field) != expected:
                    raise PatentTrainGateError(
                        f"{cell_id}: train applicability differs from v14 for {relation_id}"
                    )
    return partial


def _certificate_profile(rows: Sequence[Mapping], relation_id: str) -> dict[str, Any]:
    subsets = {
        "all_presented_rows": list(rows),
        "cap_contact_rows": [
            row for row in rows if row["representation"]["at_declared_character_cap"]
        ],
        "below_cap_rows": [
            row for row in rows if not row["representation"]["at_declared_character_cap"]
        ],
    }
    output: dict[str, Any] = {}
    for name, subset in subsets.items():
        certificates = [
            certificate
            for row in subset
            for certificate in row["result"]["certificates"]
            if certificate["relation"] == relation_id
        ]
        row_keys = {
            row["item_key"]
            for row in subset
            if any(
                certificate["relation"] == relation_id
                for certificate in row["result"]["certificates"]
            )
        }
        output[name] = {
            "n_rows": len(subset),
            "n_rows_with_certificate": len(row_keys),
            "n_certificates": len(certificates),
            "certificate_kind_counts": dict(sorted(Counter(
                certificate["kind"] for certificate in certificates
            ).items())),
        }
    return output


def _local_layering_profile(rows: Sequence[Mapping]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for name, subset in (
        ("all_presented_rows", list(rows)),
        ("cap_contact_rows", [
            row for row in rows if row["representation"]["at_declared_character_cap"]
        ]),
        ("below_cap_rows", [
            row for row in rows if not row["representation"]["at_declared_character_cap"]
        ]),
    ):
        witnessed = 0
        for row in subset:
            relation = row["result"]["relation_values"]["claim_set_layering"]
            if relation.get("value") != 1.0:
                continue
            support = relation.get("support", {})
            positive_edges = [
                certificate for certificate in row["result"]["certificates"]
                if certificate.get("relation") == "claim_dependency_well_formedness"
                and certificate.get("kind") == "positive_witness"
            ]
            if (
                not isinstance(support, Mapping)
                or not _is_number(support.get("independent_claims"))
                or support["independent_claims"] < 1
                or not isinstance(support.get("validly_linked_dependent_claims"), list)
                or not support["validly_linked_dependent_claims"]
                or not positive_edges
            ):
                raise PatentTrainGateError("positive layering value lacks a finite local witness")
            witnessed += 1
        counts[name] = {"n_rows": len(subset), "n_positive_local_witnesses": witnessed}
    return counts


def _relation_spec(
    match: Mapping,
    *,
    output_mode: str,
    allowed_output: Sequence[str],
    below_cap_profile: Mapping,
    scalar_policy: str,
) -> dict[str, Any]:
    return {
        "relation_id": match["relation_id"],
        "effective_code_depth": match["effective_code_depth"],
        "partial_scope": match["partial_scope"],
        "exclusions": match["exclusions"],
        "output_mode": output_mode,
        "allowed_output": list(allowed_output),
        "scalar_policy": scalar_policy,
        "below_cap_relation_profile": dict(below_cap_profile),
        "absence_or_whole_source_inference_permitted": False,
    }


def build_patent_train_gate(
    execution: Mapping,
    fidelity: Mapping,
    *,
    execution_source: str | Path = DEFAULT_EXECUTION,
    fidelity_source: str | Path = DEFAULT_FIDELITY,
) -> dict[str, Any]:
    """Build the fixed five-cell train gate from v14 and the conservative map."""

    rows = _validate_execution(execution, source=execution_source)
    partial = _validate_fidelity(fidelity, rows, source=fidelity_source)
    cap_rows = [row for row in rows if row["representation"]["at_declared_character_cap"]]
    below_rows = [row for row in rows if not row["representation"]["at_declared_character_cap"]]
    below_profiles = {
        relation_id: _relation_profile(below_rows, relation_id)
        for relation_id in {
            relation_id for relations in EXPECTED_MATCHES.values() for relation_id in relations
        }
    }
    section_profile = below_profiles["application_section_presence"]
    if not (
        section_profile["n_rows"] == 31
        and section_profile["n_measured"] == 31
        and section_profile["minimum"] == 1.0
        and section_profile["maximum"] == 1.0
        and section_profile["nonconstant"] is False
    ):
        raise PatentTrainGateError("section relation is not the frozen constant 31/31 result")
    for relation_id in (
        "claim_dependency_well_formedness",
        "claim_set_layering",
        "functional_limitation_incidence",
        "abstract_word_count",
    ):
        if below_profiles[relation_id]["nonconstant"] is not True:
            raise PatentTrainGateError(f"{relation_id} lost below-cap nonconstancy")

    dependency_certificates = _certificate_profile(
        rows, "claim_dependency_well_formedness"
    )
    category_certificates = _certificate_profile(
        rows, "statutory_category_surface_coverage"
    )
    functional_certificates = _certificate_profile(
        rows, "functional_limitation_incidence"
    )
    layering_witnesses = _local_layering_profile(rows)
    if category_certificates["all_presented_rows"]["n_certificates"] <= 0:
        raise PatentTrainGateError("category certificate channel is empty")
    if functional_certificates["below_cap_rows"]["n_certificates"] <= 0:
        raise PatentTrainGateError("functional channel has no below-cap positive marker")

    selected = []
    for cell_id in SELECTED_IDS:
        row = partial[cell_id]
        relation_by_id = {match["relation_id"]: match for match in row["matched_relations"]}
        specs = []
        if cell_id in {R2_FUNCTIONAL, R3_FUNCTIONAL}:
            relation_id = "functional_limitation_incidence"
            specs.append(_relation_spec(
                relation_by_id[relation_id],
                output_mode="positive_marker_certificates_plus_below_cap_presented_text_incidence",
                allowed_output=(
                    "positive marker certificate with claim number and matched surface",
                    "below-cap scalar incidence on the presented claim text",
                ),
                below_cap_profile=below_profiles[relation_id],
                scalar_policy=(
                    "below-cap presented-text incidence only; zero is not verified absence "
                    "from a source patent or a legal section-112(f) conclusion"
                ),
            ))
        elif cell_id == R3_ABSTRACT:
            relation_id = "abstract_word_count"
            specs.append(_relation_spec(
                relation_by_id[relation_id],
                output_mode="exact_named_presented_abstract_word_count",
                allowed_output=("integer word count for the closed named ABSTRACT span",),
                below_cap_profile=below_profiles[relation_id],
                scalar_policy=(
                    "exact count of the named presented abstract only; no universal office "
                    "compliance threshold or abstract-quality conclusion"
                ),
            ))
        elif cell_id == R3_ARCHITECTURE:
            dependency_id = "claim_dependency_well_formedness"
            layering_id = "claim_set_layering"
            specs.append(_relation_spec(
                relation_by_id[dependency_id],
                output_mode="finite_dependency_edge_and_local_counter_witnesses",
                allowed_output=(
                    "positive child-claim to earlier parent-claim edge certificate",
                    "local invalid-edge counter-witness with explicit reason",
                ),
                below_cap_profile=below_profiles[dependency_id],
                scalar_policy=(
                    "scalar is diagnostic on below-cap presented text only; capped rows emit "
                    "certificates, never global well-formedness or absence"
                ),
            ))
            specs.append(_relation_spec(
                relation_by_id[layering_id],
                output_mode="positive_local_root_plus_dependent_edge_witness",
                allowed_output=(
                    "positive existence witness backed by an independent root and a valid "
                    "explicit dependent edge",
                ),
                below_cap_profile=below_profiles[layering_id],
                scalar_policy=(
                    "positive finite existence only at the cap; below-cap zero describes the "
                    "presented text and never proves source-wide absence or strategic adequacy"
                ),
            ))
        elif cell_id == R3_CATEGORY:
            relation_id = "statutory_category_surface_coverage"
            specs.append(_relation_spec(
                relation_by_id[relation_id],
                output_mode="positive_category_surface_span_certificates_only",
                allowed_output=(
                    "positive certificate with claim number, category, surface, and exact span",
                ),
                below_cap_profile=below_profiles[relation_id],
                scalar_policy=(
                    "coverage scalar is demoted and must not be emitted; zero is not category "
                    "absence and a positive surface is not legal eligibility"
                ),
            ))
        selected.append({
            "cell_id": cell_id,
            "level": row["level"],
            "selection_rank": row["selection_rank"],
            "construct": row["construct"],
            "maximum_decision_contributing_depth": row["maximum_matching_relation_depth"],
            "ordered_relation_ids": [spec["relation_id"] for spec in specs],
            "relations": specs,
            "selected_for_heldout_pre_reference_execution": True,
            "selection_interpretation": (
                "train-operational relation-local output contract; not whole-criterion "
                "verification, reconstruction, isomorphism, or external truth"
            ),
        })

    static_only = []
    for cell_id in STATIC_ONLY_IDS:
        row = partial[cell_id]
        static_only.append({
            "cell_id": cell_id,
            "level": row["level"],
            "selection_rank": row["selection_rank"],
            "construct": row["construct"],
            "relation_id": "application_section_presence",
            "maximum_decision_contributing_depth": 1,
            "below_cap_relation_profile": section_profile,
            "selected_for_heldout_pre_reference_execution": False,
            "decision": "static_fidelity_retained_but_train_constant_31_of_31",
        })

    return {
        "schema": SCHEMA,
        "status": STATUS,
        "task": TASK,
        "bindings": {
            "compiler_train_source": str(execution_source),
            "compiler_train_artifact_version": "v14",
            "runner_schema": EXECUTION_SCHEMA,
            "program_schema": PROGRAM_SCHEMA,
            "construct_fidelity_source": str(fidelity_source),
            "construct_fidelity_schema": FIDELITY_SCHEMA,
        },
        "selection_basis": (
            "conservative static relation fidelity plus compiler-train v14 relation-local "
            "measurement and finite-witness availability"
        ),
        "channel_boundaries": {
            "input_fields": ["item_key", "ctext"],
            "reference_or_prompt_values_loaded": False,
            "outcomes_loaded": False,
            "prior_art_or_examiner_evidence_loaded": False,
            "external_supervision_loaded": False,
            "heldout_items_or_outputs_loaded": False,
            "models_or_apis_called": False,
            "whole_patent_score_emitted": False,
        },
        "cap_policy": {
            "declared_representation_max_chars": 4000,
            "n_cap_contact_rows": len(cap_rows),
            "n_below_cap_rows": len(below_rows),
            "cap_contact_permitted": [
                "finite positive dependency-edge witnesses",
                "finite local dependency counter-witnesses",
                "positive local layering existence witnesses",
                "positive category surface-and-span certificates",
                "positive functional marker certificates",
                "exact word count of the closed named presented ABSTRACT span",
            ],
            "always_forbidden": [
                "verified absence from the source patent",
                "whole-source claim-set completeness or compliance",
                "whole-criterion verification",
                "legal validity, eligibility, definiteness, or patentability",
            ],
            "below_cap_scalar_scope": (
                "describes only relations in the presented ctext; non-contact with the cap does "
                "not establish whole-source completeness"
            ),
        },
        "summary": {
            "n_compiler_train_rows": len(rows),
            "n_conservative_static_fidelity_cells": len(partial),
            "n_selected_operational_cells": len(selected),
            "n_static_only_constant_cells": len(static_only),
            "selected_cells_by_level": dict(sorted(Counter(
                row["level"] for row in selected
            ).items())),
            "selected_cells_by_maximum_depth": dict(sorted(Counter(
                str(row["maximum_decision_contributing_depth"]) for row in selected
            ).items())),
            "n_whole_construct_cells": 0,
            "prompt_scored_cells": 0,
            "reconstruction_evaluable_cells": 0,
            "isomorphism_evaluable_cells": 0,
        },
        "below_cap_relation_profiles": {
            relation_id: below_profiles[relation_id]
            for relation_id in sorted(below_profiles)
        },
        "finite_evidence_profiles": {
            "dependency_certificates": dependency_certificates,
            "positive_local_layering_witnesses": layering_witnesses,
            "category_surface_span_certificates": category_certificates,
            "functional_marker_certificates": functional_certificates,
        },
        "selected_operational_cells": selected,
        "static_only_cells": static_only,
        "interpretation_limits": [
            "Five selected cells mean five frozen relation-local output contracts, not five reconstructed metrics.",
            "The three section cells remain static partial matches but are excluded operationally because the below-cap train value is constant 31/31.",
            "The category coverage scalar is explicitly demoted; only positive surface-and-span certificates are operational.",
            "Prompt articulability, reference reconstruction, prompt/code isomorphism, and external validity remain unmeasured.",
            "Unselected relations are bounded non-discoveries in this frozen program class, never evidence of tacitness.",
        ],
    }


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen gate: {path}")
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument("--fidelity", type=Path, default=DEFAULT_FIDELITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_patent_train_gate(
        _load(args.execution),
        _load(args.fidelity),
        execution_source=args.execution,
        fidelity_source=args.fidelity,
    )
    _write_new(args.output, payload)
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
