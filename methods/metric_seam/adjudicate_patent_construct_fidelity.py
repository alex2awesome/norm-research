"""Adjudicate relation-local fidelity for the six retrieved patent seeds.

The decisions in this file are a versioned static source audit.  They bind to
the source-only seed inventory and distinguish the depth of the *matching
sub-relation* from the maximum depth of the surrounding hybrid program.

No item, outcome, judge score, correlation, or reconstruction result is read.
The historical prior-art operation remains explicitly oracle conditioned and
model assisted.  Passing this audit does not turn it into pure code or
autonomous unsupervised discovery.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA = "metric-seam.hierarchy-patent-construct-fidelity.v1"
SEED_SCHEMA = "metric-seam.hierarchy-patent-seed-map.v1"
TASK = "patents"


_DECISIONS = {
    "TB::patents::specific::R1::merged_tree::112::2aceb96b72aeb390d039": {
        "aspect_id": "a34",
        "verdict": "partial_relation_local",
        "matched_subrelations": [
            {
                "relation": "claim-level anticipation exposure from retrieved single references",
                "channels": ["external_evidence_operation", "local_code_aggregation"],
                "effective_code_depth": 3,
                "polarity": "more disclosure lowers the novelty score",
            },
            {
                "relation": "document mentions novelty, prior art, or disclosure bars",
                "channels": ["local_regex_code"],
                "effective_code_depth": 1,
                "polarity": "engagement markers weakly raise the score",
            },
        ],
        "unimplemented_or_weak_relations": [
            "effective filing-date comparison",
            "public use/on-sale event chronology",
            "enabling-disclosure legal determination beyond the stored relation verdict",
        ],
        "aggregation_assessment": (
            "claim-level evidence is collapsed to application-level fractions; the output is a "
            "proxy for a novelty sub-relation, not a statutory novelty determination"
        ),
    },
    "TB::patents::specific::R1::merged_tree::136::767144d5db0c84220957": {
        "aspect_id": "a35",
        "verdict": "partial_relation_local",
        "matched_subrelations": [
            {
                "relation": "novelty exposure from claim-reference disclosure records",
                "channels": ["external_evidence_operation", "local_code_aggregation"],
                "effective_code_depth": 3,
                "polarity": "more disclosed claims lower the score",
            },
            {
                "relation": "inventive-step risk from distributed reference coverage",
                "channels": ["external_evidence_operation", "local_code_aggregation"],
                "effective_code_depth": 3,
                "polarity": "broader and more uneven multi-reference coverage lowers the score",
            },
            {
                "relation": "industrial-use language and a grounded stated practical use",
                "channels": ["local_regex_code", "prompt_field"],
                "effective_code_depth": 1,
                "polarity": "concrete use markers raise the score",
            },
        ],
        "unimplemented_or_weak_relations": [
            "motivation-to-combine and POSITA reasoning",
            "operative/specific/substantial/credible utility",
            "jurisdiction-specific patentability rules and temporal cutoffs",
        ],
        "aggregation_assessment": (
            "the program touches all three named pillars but blends coarse proxies and a prompt "
            "field; this is broad partial coverage, never exact whole-construct fidelity"
        ),
    },
    "TB::patents::specific::R2::merged_group::27::77fd68cb0e268346f89c": {
        "aspect_id": "a34",
        "verdict": "partial_relation_local",
        "matched_subrelations": [
            {
                "relation": "claim-level anticipation exposure from retrieved references",
                "channels": ["external_evidence_operation", "local_code_aggregation"],
                "effective_code_depth": 3,
                "polarity": "more disclosure lowers the novelty score",
            },
            {
                "relation": "public-disclosure/on-sale/grace-period language is present",
                "channels": ["local_regex_code"],
                "effective_code_depth": 1,
                "polarity": "mention of the issue weakly raises the score",
            },
        ],
        "unimplemented_or_weak_relations": [
            "event dates relative to the filing date",
            "jurisdiction-specific grace-period logic",
            "design-specific identity or obviousness analysis",
        ],
        "aggregation_assessment": (
            "the novelty evidence relation is substantial, but disclosure bars are mention-only "
            "and lack temporal predicates"
        ),
    },
    "TB::patents::specific::R3::grandparent::15::847b03c786373669ff69": {
        "aspect_id": "a60",
        "verdict": "partial_relation_local",
        "matched_subrelations": [
            {
                "relation": "prior-art differentiation talk agrees with aggregate disclosure gaps",
                "channels": [
                    "external_evidence_operation",
                    "local_regex_code",
                    "local_code_aggregation",
                ],
                "effective_code_depth": 3,
                "polarity": "supported differentiation raises the score; contradicted talk lowers it",
            },
            {
                "relation": "KSR-resistance and harmful-admission markers",
                "channels": ["local_regex_code"],
                "effective_code_depth": 1,
                "polarity": "argument markers raise and admissions lower the score",
            },
        ],
        "unimplemented_or_weak_relations": [
            "element-by-element chart completeness",
            "pinpoint-citation accuracy",
            "the score uses aggregate gaps rather than the stored per-element mappings",
            "accused-product infringement mapping",
        ],
        "aggregation_assessment": (
            "the capability payload contains claim/reference records, but the scoring program "
            "collapses them; it checks differentiation substance only coarsely"
        ),
    },
    "TB::patents::specific::R3::grandparent::18::31b5e510aab1b9d94ba2": {
        "aspect_id": "a35",
        "verdict": "partial_relation_local",
        "matched_subrelations": [
            {
                "relation": "industrial-use language and a grounded stated practical use",
                "channels": ["local_regex_code", "prompt_field"],
                "effective_code_depth": 1,
                "polarity": "concrete use markers raise the score",
            }
        ],
        "unimplemented_or_weak_relations": [
            "operability",
            "specific, substantial, and credible utility",
            "jurisdiction-specific industrial-applicability doctrine",
            "the program's dominant novelty/inventive-step branches are unrelated contamination",
        ],
        "aggregation_assessment": (
            "only a shallow component matches; the depth-3 prior-art branch must not be credited "
            "to utility and the whole mixed output is not a clean utility witness"
        ),
    },
    "TB::patents::specific::R3::grandparent::7::5c8c71924a36750e3c31": {
        "aspect_id": "a34",
        "verdict": "partial_relation_local",
        "matched_subrelations": [
            {
                "relation": "claim-level anticipation exposure from retrieved references",
                "channels": ["external_evidence_operation", "local_code_aggregation"],
                "effective_code_depth": 3,
                "polarity": "more disclosure lowers the novelty score",
            },
            {
                "relation": "disclosure/on-sale/grace-period issue language is present",
                "channels": ["local_regex_code"],
                "effective_code_depth": 1,
                "polarity": "mention of the issue weakly raises the score",
            },
        ],
        "unimplemented_or_weak_relations": [
            "filing and disclosure date extraction",
            "jurisdiction-specific grace-period calculation",
            "inherent-disclosure reasoning",
            "exception applicability",
        ],
        "aggregation_assessment": (
            "anticipation is relation-matched, but the temporal/grace-period half is surface-only"
        ),
    },
}


def _audit_row(seed_row: Mapping) -> dict:
    cell_id = str(seed_row["cell_id"])
    seed = seed_row.get("selected_seed")
    if seed is None:
        return {
            "cell_id": cell_id,
            "task": TASK,
            "level": str(seed_row["level"]),
            "metric_name": str(seed_row["metric_name"]),
            "candidate_aspect_id": None,
            "verdict": "no_candidate",
            "matched_subrelations": [],
            "unimplemented_or_weak_relations": [],
            "eligible_relation_local_depths": [],
            "exact_whole_construct_fidelity": False,
            "pure_code_witness": False,
        }
    if cell_id not in _DECISIONS:
        raise ValueError(f"retrieved patent seed lacks a frozen adjudication: {cell_id}")
    decision = _DECISIONS[cell_id]
    aspect_id = str(seed["aspect_id"])
    if aspect_id != decision["aspect_id"]:
        raise ValueError(
            f"adjudication/seed mismatch for {cell_id}: {decision['aspect_id']} != {aspect_id}"
        )
    relations = [dict(relation) for relation in decision["matched_subrelations"]]
    depths = sorted({int(relation["effective_code_depth"]) for relation in relations})
    return {
        "cell_id": cell_id,
        "task": TASK,
        "level": str(seed_row["level"]),
        "metric_name": str(seed_row["metric_name"]),
        "metric_description": str(seed_row["metric_description"]),
        "candidate_aspect_id": aspect_id,
        "candidate_source_path": str(seed["source_path"]),
        "verdict": str(decision["verdict"]),
        "matched_subrelations": relations,
        "unimplemented_or_weak_relations": list(decision["unimplemented_or_weak_relations"]),
        "aggregation_assessment": str(decision["aggregation_assessment"]),
        "eligible_relation_local_depths": depths,
        "maximum_matching_relation_depth": max(depths),
        "surrounding_program_depth": int(seed["depth_provenance"]["derived_program_depth"]),
        "exact_whole_construct_fidelity": False,
        "pure_code_witness": False,
        "autonomous_unsupervised_discovery": False,
        "eligible_for_retrospective_oracle_conditioned_replay": True,
        "interpretation": (
            "static partial relation match in a manual hybrid; evidence-channel provenance and "
            "missing relations remain operative"
        ),
    }


def build_audit(seed_map: Mapping) -> dict:
    if seed_map.get("schema") != SEED_SCHEMA:
        raise ValueError(f"expected {SEED_SCHEMA}")
    if seed_map.get("task") != TASK or seed_map.get("n_cells") != 90:
        raise ValueError("expected the 90-cell patent seed inventory")
    rows = [_audit_row(row) for row in seed_map.get("rows", [])]
    if len(rows) != 90 or len({row["cell_id"] for row in rows}) != 90:
        raise ValueError("patent fidelity audit requires 90 unique cells")
    retrieved_ids = {
        row["cell_id"] for row in seed_map["rows"] if row.get("selected_seed") is not None
    }
    if retrieved_ids != set(_DECISIONS):
        missing = sorted(retrieved_ids - set(_DECISIONS))
        stale = sorted(set(_DECISIONS) - retrieved_ids)
        raise ValueError(f"frozen adjudication coverage mismatch; missing={missing}, stale={stale}")

    verdicts = Counter(row["verdict"] for row in rows)
    eligible = [row for row in rows if row["verdict"] == "partial_relation_local"]
    by_level = {
        level: {
            "n_cells": sum(row["level"] == level for row in rows),
            "n_retrieved": sum(
                row["level"] == level and row["candidate_aspect_id"] is not None for row in rows
            ),
            "n_partial_relation_local": sum(
                row["level"] == level and row["verdict"] == "partial_relation_local"
                for row in rows
            ),
        }
        for level in ("R1", "R2", "R3")
    }
    depth_counts = Counter(
        str(row["maximum_matching_relation_depth"]) for row in eligible
    )
    return {
        "schema": SCHEMA,
        "status": "static-relation-local-adjudication-complete",
        "task": TASK,
        "n_cells": len(rows),
        "source_seed_schema": seed_map["schema"],
        "source_panel_content_sha256": seed_map.get("panel_content_sha256"),
        "design_scope": "static_source_only_manual_construct_fidelity_adjudication",
        "forbidden_inputs": list(seed_map.get("forbidden_inputs", [])),
        "execution_performed": False,
        "items_loaded": False,
        "reference_values_loaded": False,
        "outcome_labels_loaded": False,
        "program_outputs_loaded": False,
        "external_supervision_loaded_for_this_audit": False,
        "fidelity_rule": (
            "credit only an executable sub-relation that has the criterion's object, relation, "
            "direction, applicability, and aggregation at least partially aligned; report the "
            "matching sub-relation's depth rather than the surrounding program's maximum depth"
        ),
        "provenance_rule": (
            "manual hybrid authorship, examiner/oracle candidate injection, precomputed reading-"
            "model relation labels, and duplicate extraction rows remain attached to every pass"
        ),
        "summary": {
            "verdict_counts": dict(sorted(verdicts.items())),
            "by_level": by_level,
            "n_retrieved": sum(row["candidate_aspect_id"] is not None for row in rows),
            "n_partial_relation_local": len(eligible),
            "n_exact_whole_construct": 0,
            "n_pure_code_witnesses": 0,
            "n_autonomous_unsupervised_discoveries": 0,
            "maximum_matching_relation_depth_counts": dict(sorted(depth_counts.items())),
            "n_with_depth3_matching_relation": sum(
                3 in row["eligible_relation_local_depths"] for row in eligible
            ),
            "n_with_only_depth1_matching_relation": sum(
                row["eligible_relation_local_depths"] == [1] for row in eligible
            ),
        },
        "interpretation_limits": [
            "retrieval plus static fidelity is not execution or reconstruction",
            "partial relation-local fidelity is not whole-criterion codability",
            "program depth cannot be transferred to an unrelated matched sub-relation",
            "the historical evidence operation is not autonomous or pure code",
            "failure to retrieve is bounded non-discovery, never evidence of tacitness",
        ],
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    seed_map = json.loads(args.seed_map.read_text(encoding="utf-8"))
    result = build_audit(seed_map)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
