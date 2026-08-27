"""Propose relation-local mappings from the grant hierarchy to deep code.

This is an authoring proposal, not an independent construct-fidelity audit.
Mapped rows remain ineligible for confirmatory execution until a separate
reviewer checks requested relation, implemented relation, polarity,
applicability, aggregation, and depth.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Mapping

from methods.metric_seam.grant_structure_v1 import (
    DISCOVERY_MODE,
    INPUT_REPRESENTATION,
    PROGRAM_ID,
    RELATION_DEPTHS,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/panel_v3.json"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/metric_seam_pilot/hierarchy_r123/grant_structure_static_proposal_v1.json"
)
PROGRAM_SOURCE = ROOT / "methods/metric_seam/grant_structure_v1.py"
LEVELS = ("R1", "R2", "R3")

RELATION_SPECS = {
    "aim_hypothesis_experiment_graph": {
        "implemented_relation": (
            "a hypothesis or explicit prediction is linked within two sentence spans to "
            "an aim, study, experiment, test, or analysis operation"
        ),
        "polarity": "more local hypothesis-to-test links is positive for this subrelation",
        "aggregation": "min(link_count / 3, 1)",
    },
    "budget_sum_consistency": {
        "implemented_relation": (
            "the last checkable stated currency total agrees arithmetically with at least "
            "two preceding itemized currency amounts within twelve nonempty lines"
        ),
        "polarity": "smaller relative arithmetic error is positive",
        "aggregation": "1 - min(relative_error, 1); abstain when not checkable",
    },
    "citation_claim_link": {
        "implemented_relation": (
            "a citation, DOI, or URL occurs in the same sentence as a quantitative or "
            "explicit evidential claim predicate"
        ),
        "polarity": "more document-internal claim-to-citation links is positive",
        "aggregation": "min(link_count / 3, 1)",
    },
    "dissemination_output_channel_graph": {
        "implemented_relation": (
            "a named output is linked in the same or adjacent sentence to a dissemination "
            "channel or audience"
        ),
        "polarity": "more explicit output-to-channel links is positive",
        "aggregation": "min(link_count / 3, 1)",
    },
    "document_outline_structure": {
        "implemented_relation": (
            "the frozen text contains distinct numbered, all-caps, or known grant section headings"
        ),
        "polarity": "more distinct visible headings up to six is positive",
        "aggregation": "min(unique_heading_count / 6, 1)",
    },
    "evaluation_measurement_chain": {
        "implemented_relation": (
            "an outcome, metric, indicator, success criterion, or evaluation predicate is "
            "linked within two sentences to a data/collection/analysis method and at least "
            "one use, reporting, decision, milestone, or quantified-specificity field"
        ),
        "polarity": "more explicit metric-to-method/use chains is positive",
        "aggregation": "min(chain_count / 3, 1)",
    },
    "front_matter_coverage": {
        "implemented_relation": (
            "the first 1,400 characters separately contain problem/need, approach/activity, "
            "outcome/impact, and quantified-specificity cues"
        ),
        "polarity": "more front-matter dimensions present is positive",
        "aggregation": "present_dimensions / 4",
    },
    "partner_role_graph": {
        "implemented_relation": (
            "a partner, collaborator, stakeholder, advisory body, subaward, or consortium "
            "mention is linked in the same or adjacent sentence to a role, commitment, or action"
        ),
        "polarity": "more explicit partner-to-role links is positive",
        "aggregation": "min(link_count / 3, 1)",
    },
    "quantified_need_gap": {
        "implemented_relation": (
            "a quantitative count or percentage occurs in a sentence that explicitly marks "
            "a comparison, gap, shortage, need, underserved group, barrier, or baseline"
        ),
        "polarity": "more quantified need/gap statements is positive",
        "aggregation": "min(statement_count / 3, 1)",
    },
    "resource_use_graph": {
        "implemented_relation": (
            "a facility, equipment, laboratory, center, or infrastructure mention is linked "
            "in the same or adjacent sentence to access, availability, use, or an enabling action"
        ),
        "polarity": "more explicit resource-to-use links is positive",
        "aggregation": "min(link_count / 3, 1)",
    },
    "risk_mitigation_graph": {
        "implemented_relation": (
            "a risk, pitfall, limitation, failure, challenge, or barrier is linked in the same "
            "or adjacent sentence to mitigation, contingency, fallback, adaptation, or an if-plan"
        ),
        "polarity": "more explicit risk-to-mitigation links is positive",
        "aggregation": "min(link_count / 3, 1)",
    },
    "role_responsibility_graph": {
        "implemented_relation": (
            "a named personnel/team role is linked in the same sentence to a responsibility, "
            "leadership, supervision, management, or coordination action"
        ),
        "polarity": "more explicit role-to-responsibility links is positive",
        "aggregation": "min(link_count / 3, 1)",
    },
    "schedule_dependency_graph": {
        "implemented_relation": (
            "a visible time expression is linked within two sentences to an aim, activity, "
            "phase, milestone, deliverable, completion, or ordering predicate"
        ),
        "polarity": "more explicit activity-to-time/order links is positive",
        "aggregation": "min(link_count / 3, 1)",
    },
}


CONSTRUCT_TO_RELATION = {
    # R1
    "Evaluation plan with metrics and methods": "evaluation_measurement_chain",
    "Evidence base and citations quality": "citation_claim_link",
    "Clarity, conciseness, and organization of writing": "document_outline_structure",
    "Approach rigor and study design transparency": "aim_hypothesis_experiment_graph",
    "Progress reporting and outcomes communication": "evaluation_measurement_chain",
    "Hypothesis‑driven aims (avoid descriptive/exploratory framing)": (
        "aim_hypothesis_experiment_graph"
    ),
    "Problem statement and needs assessment quality": "quantified_need_gap",
    "Governance, budgeting, and accountability": "role_responsibility_graph",
    "Budget realism, accuracy, and alignment with scope": "budget_sum_consistency",
    "Staffing and qualifications (people and roles)": "role_responsibility_graph",
    "Milestones and go/no‑go decision points": "schedule_dependency_graph",
    "Partnerships and stakeholder engagement strategy": "partner_role_graph",
    "Implementation management (communication, milestones, monitoring)": (
        "schedule_dependency_graph"
    ),
    "Problem/need statement with documented gap": "quantified_need_gap",
    "Use credible evidence to document need and feasibility": "citation_claim_link",
    "Expected outcomes and long‑term impacts": "evaluation_measurement_chain",
    # R2
    "Feasibility evidence and risk robustness": "risk_mitigation_graph",
    "Investigator qualifications, roles, and mentoring": "role_responsibility_graph",
    "Risk identification, mitigation, and contingency planning": "risk_mitigation_graph",
    "Dissemination for uptake and impact": "dissemination_output_channel_graph",
    "Team roles, qualifications, and coverage": "role_responsibility_graph",
    "Outcomes and broader impact articulation": "evaluation_measurement_chain",
    "Front‑matter strength and frontloading for skim‑readers": "front_matter_coverage",
    "Budget quality, realism, alignment, and compliance": "budget_sum_consistency",
    "Dissemination and audience engagement plan": "dissemination_output_channel_graph",
    "Front‑matter impact communication": "front_matter_coverage",
    "Positioning within prior work and alternatives": "citation_claim_link",
    "Specific Aims page: problem, gap, hypotheses, and aim design": (
        "aim_hypothesis_experiment_graph"
    ),
    "Methodology and work plan specificity and justification (evaluability)": (
        "aim_hypothesis_experiment_graph"
    ),
    "Timeline and milestones: clarity and feasibility": "schedule_dependency_graph",
    "Expected outcomes specified": "evaluation_measurement_chain",
    "Outcomes, impact articulation, and success tracking": "evaluation_measurement_chain",
    "Investigator and team merit": "role_responsibility_graph",
    "Work plan structure, sequencing, timeline, and milestones": "schedule_dependency_graph",
    # R3
    "Objectives, metrics, and performance management": "evaluation_measurement_chain",
    "Reviewer‑oriented usability and document/visual design": "document_outline_structure",
    "Problem/needs statement quality, evidence, and urgency": "quantified_need_gap",
    "Writing clarity, organization, and concision (grantsmanship)": (
        "document_outline_structure"
    ),
    "Budget quality, reasonableness, and compliance": "budget_sum_consistency",
    "Partnerships and external support": "partner_role_graph",
    "Budget quality, realism, completeness, alignment, and compliance": (
        "budget_sum_consistency"
    ),
    "Research approach and workplan rigor, specificity, and feasibility": (
        "aim_hypothesis_experiment_graph"
    ),
    "Institutional environment and resources": "resource_use_graph",
    "Evidence standards, citation integrity, and limitation transparency": (
        "citation_claim_link"
    ),
    "Collaboration and partnership strategy and plan": "partner_role_graph",
    "Dissemination, translation/scale, sustainability, and open sharing": (
        "dissemination_output_channel_graph"
    ),
    "Team and organizational capacity, roles, and governance": "role_responsibility_graph",
    "Facilities, environment, and equipment adequacy and justification": "resource_use_graph",
    "Front‑matter effectiveness and first‑impression skimmability": "front_matter_coverage",
    "Reviewer‑friendly structure and skimmability": "document_outline_structure",
    "Problem, gap, significance, and outcomes/impact framing": "quantified_need_gap",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(rows: list[dict]) -> dict:
    proposed = [row for row in rows if row["proposal_status"] == "mapped_pending_audit"]
    return {
        "panel_cells": len(rows),
        "mapped_pending_independent_audit": len(proposed),
        "bounded_non_discovery_in_frozen_program_class": len(rows) - len(proposed),
        "mapped_by_level": dict(sorted(Counter(row["level"] for row in proposed).items())),
        "mapped_by_relation": dict(
            sorted(Counter(row["implemented_relation_id"] for row in proposed).items())
        ),
        "mapped_by_proposed_depth": dict(
            sorted(Counter(str(row["proposed_depth"]) for row in proposed).items())
        ),
        "whole_construct_exact": 0,
        "eligible_for_execution_before_independent_audit": 0,
    }


def build(panel: Mapping) -> dict:
    cells = [cell for cell in panel.get("cells", []) if cell.get("task") == "grant-funding"]
    if len(cells) != 90 or Counter(cell.get("level") for cell in cells) != Counter(
        {"R1": 30, "R2": 30, "R3": 30}
    ):
        raise ValueError("expected the balanced 90-cell grant-funding panel")
    panel_constructs = {str(cell["construct"]) for cell in cells}
    unknown = set(CONSTRUCT_TO_RELATION) - panel_constructs
    if unknown:
        raise ValueError(f"grant mapping names drifted from panel: {sorted(unknown)}")
    rows = []
    for cell in cells:
        construct = str(cell["construct"])
        relation = CONSTRUCT_TO_RELATION.get(construct)
        base = {
            "cell_id": str(cell["id"]),
            "metric_id": str(cell["metric_id"]),
            "task": "grant-funding",
            "level": str(cell["level"]),
            "construct": construct,
            "requested_construct": str(cell["description"]),
            "program_id": PROGRAM_ID if relation else None,
            "program_source": "methods/metric_seam/grant_structure_v1.py" if relation else None,
            "program_provenance": DISCOVERY_MODE if relation else None,
            "input_representation": INPUT_REPRESENTATION,
            "whole_construct_exact": False,
            "eligible_for_execution": False,
            "independent_construct_audit_complete": False,
        }
        if relation:
            spec = RELATION_SPECS[relation]
            base.update(
                {
                    "proposal_status": "mapped_pending_audit",
                    "verdict": "proposed_partial_relation_local_not_yet_audited",
                    "implemented_relation_id": relation,
                    "implemented_relation": spec["implemented_relation"],
                    "polarity": spec["polarity"],
                    "aggregation": spec["aggregation"],
                    "proposed_depth": RELATION_DEPTHS[relation],
                    "channel": "pure_code_exact_ctext",
                    "scope_limit": (
                        "The program measures only the named document-internal subrelation; "
                        "it does not score the whole construct or read solicitation rules."
                    ),
                }
            )
        else:
            base.update(
                {
                    "proposal_status": "bounded_non_discovery",
                    "verdict": "no_candidate_bounded_non_discovery",
                    "implemented_relation_id": None,
                    "implemented_relation": None,
                    "polarity": None,
                    "aggregation": None,
                    "proposed_depth": None,
                    "channel": None,
                    "scope_limit": (
                        "No relation-matched witness was found in the frozen thirteen-relation "
                        "program class over the 4,000-character exact-ctext representation."
                    ),
                }
            )
        rows.append(base)
    summary = _summary(rows)
    if summary != {
        "panel_cells": 90,
        "mapped_pending_independent_audit": 52,
        "bounded_non_discovery_in_frozen_program_class": 38,
        "mapped_by_level": {"R1": 16, "R2": 18, "R3": 18},
        "mapped_by_relation": {
            "aim_hypothesis_experiment_graph": 5,
            "budget_sum_consistency": 4,
            "citation_claim_link": 4,
            "dissemination_output_channel_graph": 3,
            "document_outline_structure": 4,
            "evaluation_measurement_chain": 7,
            "front_matter_coverage": 3,
            "partner_role_graph": 3,
            "quantified_need_gap": 4,
            "resource_use_graph": 2,
            "risk_mitigation_graph": 3,
            "role_responsibility_graph": 6,
            "schedule_dependency_graph": 4,
        },
        "mapped_by_proposed_depth": {"1": 7, "2": 41, "3": 4},
        "whole_construct_exact": 0,
        "eligible_for_execution_before_independent_audit": 0,
    }:
        raise ValueError(f"grant proposal counts drifted: {summary}")
    return {
        "schema": "metric-seam.grant-structure-static-proposal.v1",
        "status": "author_proposal_complete_pending_independent_construct_audit",
        "panel_content_sha256": panel.get("panel_content_sha256"),
        "program": {
            "program_id": PROGRAM_ID,
            "source": "methods/metric_seam/grant_structure_v1.py",
            "source_sha256": _sha256(PROGRAM_SOURCE),
            "discovery_mode": DISCOVERY_MODE,
            "relations": RELATION_SPECS,
        },
        "blindness": {
            "outcome_labels_used": False,
            "reference_values_used": False,
            "heldout_items_or_outputs_used": False,
            "external_supervised_anchor_used": False,
            "model_or_api_used": False,
            "accelerator_used": False,
        },
        "claim_boundary": {
            "articulability_measured": False,
            "code_verifiability_established": False,
            "construct_fidelity_established": False,
            "reconstruction_measured": False,
            "isomorphism_measured": False,
            "codability_measured": False,
            "negative_result_policy": (
                "Unmapped cells are bounded non-discovery in this frozen program class, "
                "representation, and manual authoring budget; never evidence of tacitness."
            ),
        },
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    payload = build(panel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
