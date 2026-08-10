"""Independent dimension audit for the math symbolic-capability sensitivity.

The deterministic mapper is intentionally permissive.  This module applies a
frozen manual ledger over every retrieved row and separately adjudicates
object, relation, polarity, applicability, and aggregation.  Credit is limited
to explicit rational-algebra equality preservation/nonidentity for a presented
step.  It never upgrades that relation to whole-proof correctness.

The canonical 33-cell static audit is read only to classify accepted mappings
as either newly covered or as an additional formal-symbolic relation on an
already covered cell.  The canonical artifact is not modified.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from methods.metric_seam.hierarchy_math_symbolic_capability_mapper import (
    RELATION_ID,
    SCHEMA as MAP_SCHEMA,
    TASK,
)


SCHEMA = "metric-seam.math-symbolic-capability-construct-fidelity.v1"
CANONICAL_SCHEMA = "metric-seam.math-construct-fidelity-merged.v1"

# Frozen after reviewing the 15 source-only retrievals.  The categories drive
# explicit five-dimension templates below; they do not depend on outputs.
_DECISIONS = {
    # Accepted narrow presented-step relation.
    "TB::math-stackexchange::general::R1::parented_tree::36::1fd33b611482f069f2db": (
        "partial_relation_local",
        "presented_step_validity",
        "The construct explicitly requires fully justified steps and mechanical checkability; "
        "exact rational equality preservation is one narrow formal sub-relation.",
    ),
    "TB::math-stackexchange::general::R2::merged_group::56::fde63c9dd735984587fe": (
        "partial_relation_local",
        "presented_step_validity",
        "Every-step logical correctness includes the narrow case of a presented rational "
        "equality transformation preserving equality.",
    ),
    "TB::math-stackexchange::general::R2::merged_group::67::4a5def4ec1c6928f4cdc": (
        "partial_relation_local",
        "presented_step_validity",
        "The requested correct-inference relation explicitly ranges over every step; a "
        "parseable rational equality step is a bounded member of that object class.",
    ),
    "TB::math-stackexchange::general::R2::merged_group::86::994349c76321692213c7": (
        "partial_relation_local",
        "presented_step_validity",
        "A displayed rational equality transformation is a narrow inference whose exact "
        "validity can be checked, although its warrant is not supplied by the verifier.",
    ),
    "TB::math-stackexchange::general::R3::grandparent::8::2cb22f4124bacb534dec": (
        "partial_relation_local",
        "presented_step_validity",
        "Airtight stepwise reasoning includes rational equality preservation as a narrow "
        "formal relation, not as proof-level completeness.",
    ),
    "TB::math-stackexchange::general::R3::merged_group::10::a7ca222d4664b6e8021d": (
        "partial_relation_local",
        "presented_step_validity",
        "Logical correctness of a proof includes correctness of an explicitly displayed "
        "rational equality step, while coverage and hidden assumptions remain unresolved.",
    ),
    "TB::math-stackexchange::general::R3::merged_group::9::6d40ac86f7e9adf63d2b": (
        "partial_relation_local",
        "presented_step_validity",
        "The correct-implication construct admits exact equality preservation as a narrow "
        "step relation; explicit warrants and goal reductions are residual.",
    ),
    # Retrieved but rejected after dimension review.
    "TB::math-stackexchange::general::R1::parented_tree::81::79313ce4ce283e86ebe7": (
        "relation_mismatch",
        "artifact_verifiability",
        "The construct concerns proof-artifact size, documentation, and reproducibility, "
        "not whether one presented rational equality step preserves equality.",
    ),
    "TB::math-stackexchange::general::R1::merged_tree::99::27a8afde78689275c8e2": (
        "relation_mismatch",
        "definition_not_instance",
        "The cell defines deductive proof in general rather than requesting evaluation of "
        "an explicitly presented algebraic step.",
    ),
    "TB::math-stackexchange::general::R1::parented_tree::277::f8906475181bc68b1585": (
        "relation_mismatch",
        "artifact_verifiability",
        "Code/data availability, error documentation, and corrections are artifact-level "
        "relations not implemented by rational equality checking.",
    ),
    "TB::math-stackexchange::general::R2::grandparent::83::6ec132d68335513e7363": (
        "relation_mismatch",
        "empirical_design",
        "Empirical/causal design validity and model limitations have a different object and "
        "relation from answer-side rational algebra.",
    ),
    "TB::math-stackexchange::general::R2::merged_group::15::b70e6bc584e1ce284c50": (
        "relation_mismatch",
        "community_checkability",
        "Reasonable-effort comprehensibility and community scrutiny are holistic human-use "
        "relations, not exact equality preservation.",
    ),
    "TB::math-stackexchange::general::R2::grandparent::26::7025d317ae10b00048de": (
        "relation_mismatch",
        "visual_epistemic_role",
        "The requested relation concerns whether diagrams or visual analogs warrant claims; "
        "the symbolic verifier does not inspect visuals.",
    ),
    "TB::math-stackexchange::general::R3::grandparent::6::054fe2aa8cf142b7fd45": (
        "relation_mismatch",
        "visual_epistemic_role",
        "Integrated visual reasoning and its epistemic role are outside the rational equality "
        "object and relation.",
    ),
    "TB::math-stackexchange::general::R3::merged_group::14::da47b04ffaa9bf294ae9": (
        "relation_mismatch",
        "modular_structure",
        "Lemma modularity and dependency traceability concern proof organization; exact "
        "equality preservation does not establish that structural relation.",
    ),
}


class SymbolicExpansionAdjudicationError(ValueError):
    """Raised when frozen mapping/adjudication inputs drift."""


def _source_sha256(construct: str, description: str) -> str:
    return hashlib.sha256((construct + "\0" + description).encode("utf-8")).hexdigest()


def _dimension_audit(category: str) -> dict[str, dict[str, str]]:
    if category == "presented_step_validity":
        return {
            "object": {
                "status": "partial_alignment",
                "finding": (
                    "An explicitly displayed rational equality step is a bounded subset of "
                    "the construct's proof-step/inference object."
                ),
            },
            "relation": {
                "status": "aligned_relation_local",
                "finding": (
                    "SymPy/Lark exact identity or nonidentity checks whether that presented "
                    "rational transformation preserves equality."
                ),
            },
            "polarity": {
                "status": "conditional_typed_alignment",
                "finding": (
                    "Identity supports step correctness; nonidentity is formal relation "
                    "evidence but needs external claim-scope adjudication before a defect claim."
                ),
            },
            "applicability": {
                "status": "bounded_partial_alignment",
                "finding": (
                    "Only explicit answer-side equality rows in bounded rational algebra are "
                    "eligible; unsupported mathematics abstains."
                ),
            },
            "aggregation": {
                "status": "mismatch_disclosed",
                "finding": (
                    "Count-only pair receipts do not implement every-step, completeness, "
                    "justification, hidden-assumption, or whole-proof aggregation."
                ),
            },
        }
    findings = {
        "artifact_verifiability": (
            "The construct object is an inspectable/reproducible proof artifact, code, data, "
            "or documentation rather than one algebraic step."
        ),
        "definition_not_instance": (
            "The construct object is a generic definition of proof, not an instance-level "
            "presented equality step."
        ),
        "empirical_design": (
            "The construct object is an empirical/causal study and its modeling assumptions."
        ),
        "community_checkability": (
            "The construct object is a manuscript as encountered by qualified human readers."
        ),
        "visual_epistemic_role": (
            "The construct object is a diagram/visual and its warranting role."
        ),
        "modular_structure": (
            "The construct object is lemma/dependency organization rather than local algebra."
        ),
    }
    object_finding = findings[category]
    return {
        "object": {"status": "mismatch", "finding": object_finding},
        "relation": {
            "status": "mismatch",
            "finding": (
                "Exact rational equality preservation/nonidentity does not implement the "
                "requested relation for this object."
            ),
        },
        "polarity": {
            "status": "not_transferable_after_relation_mismatch",
            "finding": "Identity/nonidentity polarity cannot answer the requested construct.",
        },
        "applicability": {
            "status": "mismatch",
            "finding": "The construct does not explicitly scope itself to rational equality rows.",
        },
        "aggregation": {
            "status": "mismatch",
            "finding": "Count-only pair receipts cannot aggregate the requested construct.",
        },
    }


def _validate_inputs(
    panel: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    canonical: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]], set[str]]:
    if panel.get("schema") != "tacit_breadth_metric_panel/v1":
        raise SymbolicExpansionAdjudicationError("unexpected hierarchy panel schema")
    panel_sha = panel.get("panel_content_sha256")
    if not isinstance(panel_sha, str):
        raise SymbolicExpansionAdjudicationError("panel content identity is missing")
    if retrieval.get("schema") != MAP_SCHEMA or retrieval.get("task") != TASK:
        raise SymbolicExpansionAdjudicationError("unexpected symbolic retrieval artifact")
    if retrieval.get("status") != "static_source_only_candidate_retrieval_complete":
        raise SymbolicExpansionAdjudicationError("symbolic retrieval is incomplete")
    if retrieval.get("panel_content_sha256") != panel_sha:
        raise SymbolicExpansionAdjudicationError("retrieval is bound to another panel")
    if retrieval.get("capability", {}).get("relation_id") != RELATION_ID:
        raise SymbolicExpansionAdjudicationError("retrieval capability relation drifted")
    for field in (
        "programs_imported_or_executed",
        "items_or_articles_loaded",
        "certificate_counts_loaded",
        "prompt_outputs_loaded",
        "reference_values_loaded",
        "outcome_labels_loaded",
        "correlations_or_reconstruction_loaded",
        "models_apis_or_gpus_used",
    ):
        if retrieval.get(field) is not False:
            raise SymbolicExpansionAdjudicationError(
                f"retrieval crossed forbidden boundary: {field}"
            )
    if canonical.get("schema") != CANONICAL_SCHEMA or canonical.get("task") != TASK:
        raise SymbolicExpansionAdjudicationError("unexpected canonical math static audit")
    if canonical.get("status") != "static_construct_fidelity_complete_pre_execution":
        raise SymbolicExpansionAdjudicationError("canonical math static audit is incomplete")
    if canonical.get("panel_content_sha256") != panel_sha:
        raise SymbolicExpansionAdjudicationError("canonical audit is bound to another panel")
    for field in (
        "execution_performed",
        "items_loaded",
        "reference_values_loaded",
        "outcome_labels_loaded",
        "program_outputs_loaded",
        "external_supervision",
    ):
        if canonical.get(field) is not False:
            raise SymbolicExpansionAdjudicationError(
                f"canonical static audit crossed forbidden boundary: {field}"
            )

    panel_cells = {
        str(cell["id"]): cell
        for cell in panel.get("cells", [])
        if cell.get("task") == TASK
    }
    retrieval_rows = {
        str(row.get("cell_id")): row for row in retrieval.get("rows", [])
    }
    canonical_rows = {
        str(row.get("cell_id")): row for row in canonical.get("rows", [])
    }
    if not (
        len(panel_cells)
        == len(retrieval_rows)
        == len(canonical_rows)
        == 90
        and set(panel_cells) == set(retrieval_rows) == set(canonical_rows)
    ):
        raise SymbolicExpansionAdjudicationError(
            "panel/retrieval/canonical identities do not match exactly"
        )
    retrieved_ids: set[str] = set()
    canonical_eligible: set[str] = set()
    for cell_id, cell in panel_cells.items():
        retrieved = retrieval_rows[cell_id]
        existing = canonical_rows[cell_id]
        if (
            retrieved.get("metric_name") != cell.get("construct")
            or retrieved.get("metric_description") != cell.get("description")
            or retrieved.get("metric_source_text_sha256")
            != _source_sha256(cell["construct"], cell["description"])
            or retrieved.get("level") != cell.get("level")
        ):
            raise SymbolicExpansionAdjudicationError(f"{cell_id}: retrieval source drift")
        if retrieved.get("retrieved_candidate") is True:
            retrieved_ids.add(cell_id)
        elif retrieved.get("retrieved_candidate") is not False:
            raise SymbolicExpansionAdjudicationError(f"{cell_id}: invalid retrieval flag")
        eligible = existing.get("eligible_for_relation_local_execution")
        verdict = existing.get("verdict")
        if not isinstance(eligible, bool) or eligible != (verdict in {"partial", "exact"}):
            raise SymbolicExpansionAdjudicationError(
                f"{cell_id}: canonical eligibility drift"
            )
        if eligible:
            canonical_eligible.add(cell_id)
    if retrieved_ids != set(_DECISIONS):
        raise SymbolicExpansionAdjudicationError(
            "source-only retrieval set differs from the frozen independent ledger"
        )
    if len(canonical_eligible) != 33:
        raise SymbolicExpansionAdjudicationError("canonical 33-cell static result drifted")
    if any(row.get("verdict") == "exact" for row in canonical_rows.values()):
        raise SymbolicExpansionAdjudicationError("canonical exact-count assumption drifted")
    return panel_cells, retrieval_rows, canonical_eligible


def build_symbolic_expansion_adjudication(
    panel: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    canonical: Mapping[str, Any],
    *,
    sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    panel_cells, retrieval_rows, canonical_eligible = _validate_inputs(
        panel, retrieval, canonical
    )
    accepted_ids = {
        cell_id
        for cell_id, (verdict, _category, _reason) in _DECISIONS.items()
        if verdict == "partial_relation_local"
    }
    newly_covered = accepted_ids - canonical_eligible
    formalized_existing = accepted_ids & canonical_eligible
    sensitivity_union = canonical_eligible | accepted_ids
    rows: list[dict[str, Any]] = []
    for cell_id, cell in panel_cells.items():
        retrieved = retrieval_rows[cell_id]["retrieved_candidate"]
        if retrieved:
            verdict, category, rationale = _DECISIONS[cell_id]
            dimensions = _dimension_audit(category)
            matched = verdict == "partial_relation_local"
        else:
            verdict = "no_candidate_source_only"
            category = None
            rationale = (
                "The deterministic source-only retrieval rule found no candidate; this is "
                "bounded non-discovery, not evidence of tacitness."
            )
            dimensions = None
            matched = False
        canonical_before = cell_id in canonical_eligible
        if cell_id in newly_covered:
            effect = "newly_covered_in_additive_sensitivity"
        elif cell_id in formalized_existing:
            effect = "existing_cell_adds_formal_symbolic_relation"
        elif retrieved:
            effect = "retrieved_then_rejected_no_coverage_change"
        else:
            effect = "no_symbolic_candidate_no_coverage_change"
        rows.append(
            {
                "cell_id": cell_id,
                "task": TASK,
                "level": cell["level"],
                "metric_name": cell["construct"],
                "metric_description": cell["description"],
                "retrieved_candidate": retrieved,
                "canonical_relation_local_before": canonical_before,
                "adjudication_category": category,
                "dimension_audit": dimensions,
                "verdict": verdict,
                "symbolic_relation_local_static_fidelity": matched,
                "matched_relation_id": RELATION_ID if matched else None,
                "matched_relation_depth": 3 if matched else None,
                "whole_construct_exact": False,
                "sensitivity_effect": effect,
                "rationale": rationale,
            }
        )
    accepted_rows = [row for row in rows if row["symbolic_relation_local_static_fidelity"]]
    new_rows = [row for row in rows if row["cell_id"] in newly_covered]
    existing_rows = [row for row in rows if row["cell_id"] in formalized_existing]
    return {
        "schema": SCHEMA,
        "status": "static_five_dimension_adjudication_complete_pre_execution",
        "task": TASK,
        "design_scope": "additive_manual_capability_expansion_sensitivity",
        "sources": dict(sources or {}),
        "panel_content_sha256": panel["panel_content_sha256"],
        "capability_id": retrieval["capability"]["capability_id"],
        "capability_selection_provenance": retrieval["capability"][
            "selection_provenance"
        ],
        "matched_relation_depth": {
            "depth": 3,
            "meaning": "positive formal relation via computer algebra",
            "isolation_or_test_execution_adds_depth": False,
        },
        "capability_runtime_receipt": retrieval["capability"][
            "isolation_and_test_receipts"
        ],
        "programs_or_items_executed": False,
        "certificate_counts_loaded": False,
        "prompt_outputs_loaded": False,
        "reference_values_loaded": False,
        "outcome_labels_loaded": False,
        "correlations_or_reconstruction_loaded": False,
        "models_apis_or_gpus_used": False,
        "canonical_artifact_modified": False,
        "summary": {
            "n_cells": 90,
            "n_retrieved_candidates": len(_DECISIONS),
            "n_relation_local_static_matches": len(accepted_rows),
            "n_retrieved_relation_mismatches": len(_DECISIONS) - len(accepted_rows),
            "n_newly_covered_cells": len(new_rows),
            "n_existing_cells_adding_formal_symbolic_relation": len(existing_rows),
            "canonical_relation_local_cells_unchanged": len(canonical_eligible),
            "additive_sensitivity_union_cells": len(sensitivity_union),
            "n_whole_construct_exact": 0,
            "accepted_by_level": dict(Counter(row["level"] for row in accepted_rows)),
            "newly_covered_by_level": dict(Counter(row["level"] for row in new_rows)),
        },
        "newly_covered_cells": [
            {"cell_id": row["cell_id"], "level": row["level"], "metric_name": row["metric_name"]}
            for row in new_rows
        ],
        "existing_cells_adding_formal_symbolic_relation": [
            {"cell_id": row["cell_id"], "level": row["level"], "metric_name": row["metric_name"]}
            for row in existing_rows
        ],
        "rows": rows,
        "claim_limits": [
            "The canonical 33/90 static result is unchanged; this is a separate additive sensitivity.",
            "The capability was manually designed previously and is treated as a pipeline seed, not a discovery.",
            "Credit is only rational equality preservation/nonidentity for a presented step.",
            "No whole-proof correctness, completeness, justification, or every-step aggregation is credited.",
            "No verifier, item, certificate, prompt, reference, outcome, correlation, model, API, or GPU was run or loaded.",
            "Negative rows are bounded non-discovery or relation mismatch, never evidence of tacitness.",
        ],
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    payload = build_symbolic_expansion_adjudication(
        _load(args.panel),
        _load(args.retrieval),
        _load(args.canonical),
        sources={
            "panel": str(args.panel),
            "source_only_retrieval": str(args.retrieval),
            "canonical_static_fidelity_read_only": str(args.canonical),
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
