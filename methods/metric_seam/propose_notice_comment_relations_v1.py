"""Freeze a conservative source-only notice/comment relation proposal."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Mapping

from methods.metric_seam.notice_comment_relations_v1 import (
    DISCOVERY_MODE,
    INPUT_REPRESENTATION,
    PARSER_MODEL,
    PROGRAM_ID,
    RELATION_DEPTHS,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
DEFAULT_PANEL = BASE / "panel_v3.json"
DEFAULT_OUTPUT = BASE / "notice_comment_relations_static_proposal_v1.json"
PROGRAM_SOURCE = ROOT / "methods/metric_seam/notice_comment_relations_v1.py"

RELATION_CONTRACTS = {
    "actionable_target_dependency": (
        "a directive/modal action is dependency-linked to an explicit target"
    ),
    "burden_breakdown_relation": (
        "a sentence contains both a burden-breakdown field and a numeric quantity"
    ),
    "causal_support_action_link": (
        "a directive/action is locally linked to an explicit causal-support marker"
    ),
    "corrective_target_dependency": (
        "an amend/clarify/correct/remove/revise/withdraw action is dependency-linked to a target"
    ),
    "cost_comparison_relation": (
        "a sentence jointly instantiates at least two of cost, comparison, and quantity structures"
    ),
    "distributional_group_impact_link": (
        "a named affected group and an impact predicate occur in the same parsed sentence"
    ),
    "identity_authenticity_action_link": (
        "a dependency-linked action/target is explicitly scoped to identity, consent, fabrication, "
        "impersonation, sponsorship, or third-party authenticity"
    ),
    "legal_authority_action_link": (
        "a USC/CFR/Federal Register/section/Executive Order span is locally linked to a directive"
    ),
    "pinpoint_provision_action_link": (
        "a section/subsection/paragraph/page/CFR pinpoint is locally linked to a directive"
    ),
    "privacy_restriction_action_link": (
        "a dependency-linked action/target is explicitly scoped to privacy, PII, confidentiality, "
        "sensitive/restricted content, or copyright"
    ),
    "quantified_action_link": (
        "a non-citation numeric quantity is locally linked to a directive/action"
    ),
    "supported_actionable_target_graph": (
        "an action-target dependency is composed with a local legal, pinpoint, quantitative, or "
        "causal-support node"
    ),
    "time_value_relation": (
        "a sentence jointly instantiates a time-value concept and a numeric quantity"
    ),
    "uncertainty_bound_relation": (
        "a sentence jointly instantiates an uncertainty/bounds concept and a numeric quantity"
    ),
}

# Multiple relations on one cell are allowed only when they name distinct
# subrelations.  The whole construct is never credited.
CONSTRUCT_TO_RELATIONS = {
    # R1: only relations visible in a short submitted comment.
    "Treatment and reporting of uncertainty (expected values, distributions, bounds, sensitivity)": [
        "uncertainty_bound_relation"
    ],
    "Privacy and restricted‑content guidance for public filings/comments": [
        "privacy_restriction_action_link"
    ],
    "Critically evaluate projected savings from competitions": ["cost_comparison_relation"],
    "Accuracy and corrective practices in rulemaking documents": [
        "corrective_target_dependency"
    ],
    "PRA burden estimate—completeness and breakdown": ["burden_breakdown_relation"],
    "Publish corrections to Federal Register notices and ensure CRA‑compliant effective dates": [
        "corrective_target_dependency"
    ],
    "Prohibit fabrication of consumer responses/enrollments by lead generators": [
        "identity_authenticity_action_link"
    ],
    "In‑house versus contract‑out alternatives and cost comparison": [
        "cost_comparison_relation"
    ],
    # R2: comment-specific argument/evidence relations plus narrow analytic structures.
    "Distributional analysis and reporting": ["distributional_group_impact_link"],
    "Public comment campaign transparency and evidence integrity": [
        "identity_authenticity_action_link"
    ],
    "Agency legal authority and alignment with statutes and policy": [
        "legal_authority_action_link"
    ],
    "Comment precision: pinpoint citations and verifiable sourcing": [
        "pinpoint_provision_action_link"
    ],
    "Paperwork Reduction Act implementation and notices": ["burden_breakdown_relation"],
    "PRA ICR burden estimation—need, breakdown, and methodology": [
        "burden_breakdown_relation"
    ],
    "Use incremental comparisons and sound decision rules": ["cost_comparison_relation"],
    "Public comment process—notice practices, scope/reopeners, and integrity safeguards": [
        "identity_authenticity_action_link"
    ],
    "Comment quality, content, and responsiveness": ["supported_actionable_target_graph"],
    "Public comment integrity, privacy, and e‑filing constraints": [
        "privacy_restriction_action_link"
    ],
    "Commercial cost figures based on firm, solicited bids": ["cost_comparison_relation"],
    # R3.
    "Comment quality—specific, evidenced, and actionable": [
        "supported_actionable_target_graph"
    ],
    "Public comment submissions—identity/authenticity, formatting, and content/privacy limits": [
        "identity_authenticity_action_link",
        "privacy_restriction_action_link",
    ],
    "Comment legal grounding—statutes/regulations and precise citations": [
        "legal_authority_action_link"
    ],
    "A‑76 cost comparisons—completeness, rigor, and documentation": [
        "cost_comparison_relation"
    ],
    "Public comment integrity, identity, and campaign transparency": [
        "identity_authenticity_action_link"
    ],
    "Distributional and environmental justice analysis": [
        "distributional_group_impact_link"
    ],
    "Comment precision—pinpoint provisions and verifiable sources": [
        "pinpoint_provision_action_link"
    ],
    "PRA ICR burden estimation—identification, breakdown, and methods": [
        "burden_breakdown_relation"
    ],
    "Comment quality, targeting, and legal grounding": [
        "supported_actionable_target_graph",
        "legal_authority_action_link",
    ],
    "Time value treatment—discounting and inflation assumptions": ["time_value_relation"],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(panel: Mapping) -> dict:
    cells = [cell for cell in panel.get("cells", []) if cell.get("task") == "notice-and-comment"]
    if len(cells) != 90 or Counter(cell["level"] for cell in cells) != Counter(
        {"R1": 30, "R2": 30, "R3": 30}
    ):
        raise ValueError("expected balanced 90-cell notice-and-comment panel")
    panel_names = {str(cell["construct"]) for cell in cells}
    unknown = set(CONSTRUCT_TO_RELATIONS) - panel_names
    if unknown:
        raise ValueError(f"notice/comment mapping names drifted: {sorted(unknown)}")
    rows = []
    for cell in cells:
        construct = str(cell["construct"])
        relation_ids = CONSTRUCT_TO_RELATIONS.get(construct, [])
        mappings = [
            {
                "implemented_relation_id": relation,
                "implemented_relation": RELATION_CONTRACTS[relation],
                "proposed_depth": RELATION_DEPTHS[relation],
                "channel": "pure_code_local_spacy_dependency_graph",
                "whole_construct_exact": False,
            }
            for relation in relation_ids
        ]
        rows.append(
            {
                "cell_id": str(cell["id"]),
                "metric_id": str(cell["metric_id"]),
                "task": "notice-and-comment",
                "level": str(cell["level"]),
                "construct": construct,
                "requested_construct": str(cell["description"]),
                "proposal_status": (
                    "mapped_pending_independent_audit"
                    if mappings
                    else "bounded_non_discovery"
                ),
                "verdict": (
                    "proposed_partial_relation_local_not_yet_audited"
                    if mappings
                    else "no_candidate_bounded_non_discovery"
                ),
                "relation_mappings": mappings,
                "eligible_for_execution": False,
                "independent_construct_audit_complete": False,
                "scope_limit": (
                    "Only named relations in the short presented comment are proposed. Full rule, "
                    "agency procedure, external authority correctness, and whole-construct quality "
                    "are unavailable."
                    if mappings
                    else "No witness found in the frozen local comment-graph program class."
                ),
            }
        )
    mapped = [row for row in rows if row["relation_mappings"]]
    mappings = [mapping for row in mapped for mapping in row["relation_mappings"]]
    summary = {
        "panel_cells": 90,
        "mapped_cells_pending_independent_audit": len(mapped),
        "relation_mappings_pending_independent_audit": len(mappings),
        "mapped_cells_by_level": dict(sorted(Counter(row["level"] for row in mapped).items())),
        "mapping_depth_counts": dict(
            sorted(Counter(str(mapping["proposed_depth"]) for mapping in mappings).items())
        ),
        "bounded_non_discovery_cells": 90 - len(mapped),
        "whole_construct_exact": 0,
        "execution_eligible_before_independent_audit": 0,
    }
    expected = {
        "panel_cells": 90,
        "mapped_cells_pending_independent_audit": 29,
        "relation_mappings_pending_independent_audit": 31,
        "mapped_cells_by_level": {"R1": 8, "R2": 11, "R3": 10},
        "mapping_depth_counts": {"2": 28, "3": 3},
        "bounded_non_discovery_cells": 61,
        "whole_construct_exact": 0,
        "execution_eligible_before_independent_audit": 0,
    }
    if summary != expected:
        raise ValueError(f"notice/comment proposal counts drifted: {summary}")
    return {
        "schema": "metric-seam.notice-comment-static-proposal.v1",
        "status": "author_proposal_complete_pending_independent_construct_audit",
        "panel_content_sha256": panel.get("panel_content_sha256"),
        "program": {
            "program_id": PROGRAM_ID,
            "source": "methods/metric_seam/notice_comment_relations_v1.py",
            "source_sha256": _sha256(PROGRAM_SOURCE),
            "parser_model": PARSER_MODEL,
            "discovery_mode": DISCOVERY_MODE,
            "input_representation": INPUT_REPRESENTATION,
        },
        "blindness": {
            "outcomes_used": False,
            "reference_scores_used": False,
            "heldout_items_or_outputs_used": False,
            "external_authority_or_docket_loaded": False,
            "remote_model_or_api_used": False,
            "accelerator_used": False,
        },
        "representation_finding": {
            "compiler_train_items": 150,
            "compiler_train_median_characters": 110.5,
            "compiler_train_max_characters": 293,
            "implication": (
                "Most full-rule, agency-process, external-authority-correctness, and document-"
                "compliance constructs are unavailable from the short comment representation."
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

