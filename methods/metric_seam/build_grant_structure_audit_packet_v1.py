"""Freeze the grant structure proposal for an independent source audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
DEFAULT_PROPOSAL = BASE / "grant_structure_static_proposal_v1.json"
DEFAULT_OUTPUT = BASE / "grant_structure_independent_audit_packet_v1.json"
PROGRAM_SOURCE = ROOT / "methods/metric_seam/grant_structure_v1.py"
PROGRAM_TEST = ROOT / "methods/metric_seam/test_grant_structure_v1.py"

AUTHOR_WITHDRAWALS = {
    "Internal consistency across application components": (
        "Internal itemized-budget arithmetic is not cross-component consistency."
    ),
    "Cross‑document consistency and alignment": (
        "Internal itemized-budget arithmetic is not cross-document consistency."
    ),
    "Methodology specificity and justification": (
        "A hypothesis-to-test link does not establish procedural specificity or justification."
    ),
    "Preliminary data to establish feasibility (when allowed)": (
        "A generic citation-to-claim link does not identify applicant preliminary data."
    ),
    "Preliminary evidence appropriateness and feasibility (per FOA)": (
        "A generic citation-to-claim link does not identify preliminary evidence or FOA fit."
    ),
    "External letters and institutional support": (
        "A partner-to-role link does not establish that a letter or institutional commitment exists."
    ),
    "External endorsements and documentation support": (
        "A partner-to-role link does not establish an endorsement or supporting document."
    ),
    "Inclusive design and stakeholder/patient engagement": (
        "The generic partner graph does not distinguish patients, inclusion, or participatory design."
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(proposal: Mapping) -> dict:
    if (
        proposal.get("schema") != "metric-seam.grant-structure-static-proposal.v1"
        or proposal.get("status")
        != "author_proposal_complete_pending_independent_construct_audit"
        or proposal.get("summary", {}).get("mapped_pending_independent_audit") != 52
        or proposal.get("summary", {}).get("eligible_for_execution_before_independent_audit")
        != 0
    ):
        raise ValueError("grant proposal is not the expected frozen pre-audit version")
    program = proposal.get("program", {})
    if program.get("source_sha256") != _sha256(PROGRAM_SOURCE):
        raise ValueError("grant program source changed after proposal freeze")
    proposed = [
        row for row in proposal["rows"] if row["proposal_status"] == "mapped_pending_audit"
    ]
    bounded = [
        row for row in proposal["rows"] if row["proposal_status"] == "bounded_non_discovery"
    ]
    panel_names = {row["construct"] for row in proposal["rows"]}
    if len(proposed) != 52 or len(bounded) != 38 or not set(AUTHOR_WITHDRAWALS) <= panel_names:
        raise ValueError("grant proposal row partition drifted")
    review_rows = []
    for row in proposed:
        review_rows.append(
            {
                "cell_id": row["cell_id"],
                "metric_id": row["metric_id"],
                "level": row["level"],
                "construct": row["construct"],
                "requested_construct": row["requested_construct"],
                "implemented_relation_id": row["implemented_relation_id"],
                "implemented_relation": row["implemented_relation"],
                "proposed_depth": row["proposed_depth"],
                "polarity": row["polarity"],
                "aggregation": row["aggregation"],
                "scope_limit": row["scope_limit"],
                "review_required": {
                    "object_match": None,
                    "relation_match": None,
                    "polarity_match": None,
                    "applicability_match": None,
                    "aggregation_match": None,
                    "depth_match": None,
                    "audited_depth": None,
                    "verdict": None,
                    "reason": None,
                },
            }
        )
    return {
        "schema": "metric-seam.grant-structure-independent-audit-packet.v1",
        "status": "frozen_for_independent_source_audit",
        "panel_content_sha256": proposal["panel_content_sha256"],
        "source_freeze": {
            "program_source": "methods/metric_seam/grant_structure_v1.py",
            "program_source_sha256": _sha256(PROGRAM_SOURCE),
            "counterexample_tests": "methods/metric_seam/test_grant_structure_v1.py",
            "counterexample_tests_sha256": _sha256(PROGRAM_TEST),
            "proposal": (
                "outputs/metric_seam_pilot/hierarchy_r123/"
                "grant_structure_static_proposal_v1.json"
            ),
        },
        "review_protocol": {
            "independence": (
                "Reviewer must not be the program/proposal author and must issue row-level "
                "decisions from source contracts before held-out execution."
            ),
            "accepted_verdicts": [
                "partial_relation_local",
                "mismatch",
            ],
            "acceptance_rule": (
                "partial_relation_local requires object, relation, polarity, applicability, "
                "aggregation, and depth match; otherwise mismatch"
            ),
            "depth_vocabulary": {
                "1": "parsed document structure",
                "2": "cross-span or cross-section relation checking",
                "3": "formal solver or evidence-graph execution",
                "4": "environment or world execution",
            },
            "whole_construct_credit_allowed": False,
            "train_output_required_for_source_audit": False,
            "heldout_access_allowed": False,
            "reference_or_outcome_access_allowed": False,
        },
        "author_withdrawals_before_freeze": [
            {"construct": construct, "reason": reason}
            for construct, reason in AUTHOR_WITHDRAWALS.items()
        ],
        "summary": {
            "panel_cells": 90,
            "rows_requiring_independent_decision": len(review_rows),
            "author_bounded_non_discovery_rows": len(bounded),
            "author_withdrawals_before_freeze": len(AUTHOR_WITHDRAWALS),
            "independent_decisions_present": 0,
            "execution_eligible_rows": 0,
        },
        "rows": review_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, default=DEFAULT_PROPOSAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
    payload = build(proposal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

