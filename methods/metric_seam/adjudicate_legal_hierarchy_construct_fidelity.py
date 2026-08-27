"""Independently audit legal hierarchy candidates at the sub-relation boundary.

The adjudication is independent of retrieval score and runtime variability: it
compares the target criterion text with the projection's declared implemented
relation and exclusions.  It does not load items, program outputs, prompt
outputs, references, judge scores, or outcomes.

``partial_relation_local`` means only the named sub-relation is executable.
It never means the whole criterion is code-verifiable, articulable,
reconstructed, or isomorphic.  All negative rows are bounded non-discovery in
this frozen program class, not evidence of tacitness.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from methods.metric_seam.hierarchy_legal_seed_mapper import SCHEMA as SEED_SCHEMA
from methods.metric_seam.legal_hierarchy_projection import RELATION_BY_ID, RELATIONS


SCHEMA = "metric-seam.hierarchy-legal-construct-fidelity.v1"
TASK = "legal-outcome-prediction"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
DEFAULT_SEED_MAP = DEFAULT_BASE / "legal_capability_seed_map_v1.json"
DEFAULT_OUTPUT = DEFAULT_BASE / "legal_construct_fidelity_v1.json"


# Frozen source-text adjudication.  Keys are (level, selection_rank), so this
# cannot silently remap when a panel cell changes identity or ordering.
ACCEPTED_RELATIONS: dict[tuple[str, int], tuple[str, ...]] = {
    # R1: concrete/local writing prescriptions.
    ("R1", 0): ("plain_language_surface",),
    ("R1", 1): ("sentence_clarity_parse",),
    ("R1", 2): ("discourse_cohesion_graph",),
    ("R1", 7): ("concrete_fact_anchors",),
    ("R1", 9): ("discourse_cohesion_graph",),
    ("R1", 11): ("concrete_fact_anchors",),
    ("R1", 12): ("plain_language_surface",),
    ("R1", 15): ("concrete_fact_anchors", "discourse_cohesion_graph"),
    ("R1", 18): ("counterposition_structure",),
    ("R1", 19): ("heading_roadmap_structure",),
    ("R1", 20): ("question_frame_structure",),
    ("R1", 23): ("numeric_consistency_check", "citation_format_structure"),
    ("R1", 24): ("frontloaded_disposition_structure",),
    ("R1", 25): ("negation_stack_parse",),
    ("R1", 27): ("tone_restraint_surface",),
    ("R1", 29): ("question_frame_structure", "frontloaded_disposition_structure"),
    # R2: grouped stylistic and structural prescriptions.
    ("R2", 3): ("active_voice_parse",),
    ("R2", 5): ("frontloaded_disposition_structure",),
    ("R2", 6): ("citation_format_structure",),
    ("R2", 7): ("plain_language_surface", "definition_use_graph"),
    ("R2", 9): ("plain_language_surface",),
    ("R2", 10): ("plain_language_surface", "sentence_clarity_parse"),
    ("R2", 11): ("heading_roadmap_structure", "frontloaded_disposition_structure"),
    ("R2", 13): ("plain_language_surface",),
    ("R2", 14): ("inclusive_language_surface",),
    ("R2", 15): ("tone_restraint_surface",),
    ("R2", 21): ("citation_format_structure",),
    ("R2", 22): ("numeric_consistency_check", "definition_use_graph"),
    ("R2", 23): ("frontloaded_disposition_structure",),
    ("R2", 24): ("plain_language_surface",),
    ("R2", 25): ("counterposition_structure",),
    ("R2", 27): ("counterposition_structure",),
    ("R2", 28): ("plain_language_surface", "sentence_clarity_parse"),
    # R3: higher-level groups, accepted only for their explicit local clauses.
    ("R3", 0): ("discourse_cohesion_graph",),
    ("R3", 1): ("plain_language_surface",),
    ("R3", 2): ("plain_language_surface", "concrete_fact_anchors"),
    ("R3", 5): ("question_frame_structure", "frontloaded_disposition_structure"),
    ("R3", 6): ("sentence_clarity_parse", "numeric_consistency_check"),
    ("R3", 7): ("citation_format_structure",),
    ("R3", 8): ("paragraph_cohesion_graph",),
    ("R3", 9): ("discourse_cohesion_graph", "heading_roadmap_structure"),
    ("R3", 10): ("heading_roadmap_structure",),
    ("R3", 11): ("plain_language_surface", "tone_restraint_surface"),
    ("R3", 13): ("frontloaded_disposition_structure",),
    ("R3", 14): ("heading_roadmap_structure",),
    ("R3", 15): ("frontloaded_disposition_structure",),
    ("R3", 18): ("frontloaded_disposition_structure",),
    ("R3", 19): ("discourse_cohesion_graph",),
    ("R3", 20): ("deadline_remedy_consequence_structure", "tone_restraint_surface"),
    ("R3", 21): ("active_voice_parse",),
    ("R3", 22): ("sentence_clarity_parse",),
    ("R3", 23): ("definition_use_graph",),
    ("R3", 24): ("concrete_fact_anchors",),
    ("R3", 25): ("temporal_order_graph", "discourse_cohesion_graph"),
    ("R3", 26): ("question_frame_structure",),
    ("R3", 28): ("tone_restraint_surface",),
}


REQUESTED_SUBRELATION = {
    "plain_language_surface": "plain/familiar/direct wording; avoidance of legalese, needless affectation, or wordiness",
    "sentence_clarity_parse": "short, clear sentences with controlled embedding and readable mechanics",
    "active_voice_parse": "strong active verbs, actor-forward clauses, and avoidance of nominalizations/buried verbs",
    "negation_stack_parse": "avoid multiple negatives within a sentence or clause",
    "concrete_fact_anchors": "concrete details, dates, names, quantities, or examples rather than ungrounded generalities",
    "temporal_order_graph": "chronological or explicitly signaled temporal ordering of facts",
    "numeric_consistency_check": "internal consistency/correctness of reported numbers, dates, and details",
    "definition_use_graph": "introduce and consistently use defined terms/acronyms",
    "citation_format_structure": "structured legal citations and disciplined explanatory citation presentation",
    "quote_attribution_parse": "attributed quotation structure",
    "discourse_cohesion_graph": "coherent reinforcement and local/macro flow across the document",
    "paragraph_cohesion_graph": "single-topic paragraph cohesion linked to an initial orienting sentence",
    "frontloaded_disposition_structure": "position the issue, result, strongest point, or requested relief in the prescribed opening/coda zone",
    "counterposition_structure": "explicitly identify opposing positions and connect them to contrast/refutation structure",
    "tone_restraint_surface": "avoid incendiary, ad-hominem, hyperbolic, or unrestrained wording",
    "heading_roadmap_structure": "headings, nested signposts, and roadmap cues that expose organization",
    "question_frame_structure": "early, succinct question/issue framing, including yes/no form where requested",
    "inclusive_language_surface": "bias-free and gender-neutral surface language",
    "deadline_remedy_consequence_structure": "state a demanded remedy, a firm deadline, and consequences",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_seed_map(seed_map: Mapping[str, Any]) -> None:
    if seed_map.get("schema") != SEED_SCHEMA or seed_map.get("task") != TASK:
        raise ValueError("unexpected legal seed map")
    separation = seed_map.get("separation", {})
    for field in (
        "prompt_articulability_measured",
        "code_verifiability_measured",
        "reconstruction_measured",
        "isomorphism_measured",
        "outcomes_or_reference_values_loaded",
        "items_or_heldout_identifiers_loaded",
        "external_supervision_used",
    ):
        if separation.get(field) is not False:
            raise ValueError(f"seed map violates pre-execution separation field {field}")
    rows = seed_map.get("rows")
    if not isinstance(rows, list) or len(rows) != 90:
        raise ValueError("legal seed map must contain 90 rows")
    counts = Counter(row.get("level") for row in rows)
    if counts != Counter({"R1": 30, "R2": 30, "R3": 30}):
        raise ValueError("legal seed map level counts drifted")


def build_fidelity_audit(seed_map: Mapping[str, Any], *, seed_source: Path) -> dict[str, Any]:
    _validate_seed_map(seed_map)
    rows = []
    for source_row in seed_map["rows"]:
        key = (source_row["level"], source_row["selection_rank"])
        accepted = ACCEPTED_RELATIONS.get(key, ())
        candidates = {candidate["relation_id"]: candidate for candidate in source_row["candidates"]}
        missing_retrieval = sorted(set(accepted) - set(candidates))
        if missing_retrieval:
            raise ValueError(f"{key}: accepted relation was not source-retrieved: {missing_retrieval}")
        matches = []
        for relation_id in accepted:
            relation = RELATION_BY_ID[relation_id]
            matches.append(
                {
                    "relation_id": relation_id,
                    "requested_subrelation": REQUESTED_SUBRELATION[relation_id],
                    "implemented_relation": relation["implemented_relation"],
                    "explicit_exclusions": list(relation["exclusions"]),
                    "effective_code_depth": relation["effective_code_depth"],
                    "historical_seed_ids": list(relation["historical_seed_ids"]),
                    "construct_fidelity": "partial_relation_local",
                    "whole_construct_fidelity": False,
                    "execution_eligibility": "relation_local_only",
                }
            )
        rejected = []
        for relation_id, candidate in candidates.items():
            if relation_id in accepted:
                continue
            rejected.append(
                {
                    "relation_id": relation_id,
                    "retrieved_on": candidate["matched_source_phrases"],
                    "verdict": "candidate_overlap_not_accepted",
                    "reason": "source phrase overlap does not establish the projection's narrower implemented relation as a faithful part of this criterion",
                }
            )
        if matches:
            verdict = "partial_relation_local"
            negative_scope = None
        elif candidates:
            verdict = "candidate_mismatch_or_incomplete"
            negative_scope = "bounded non-discovery after source-level relation audit"
        else:
            verdict = "no_candidate_bounded_non_discovery"
            negative_scope = "bounded non-discovery in the frozen historical-plus-additive program class"
        rows.append(
            {
                "cell_id": source_row["cell_id"],
                "task": TASK,
                "level": source_row["level"],
                "selection_rank": source_row["selection_rank"],
                "construct": source_row["construct"],
                "description": source_row["description"],
                "verdict": verdict,
                "matched_relations": matches,
                "rejected_candidates": rejected,
                "maximum_matching_relation_depth": max((row["effective_code_depth"] for row in matches), default=None),
                "exact_whole_construct_fidelity": False,
                "negative_scope": negative_scope,
            }
        )
    counts = Counter(row["verdict"] for row in rows)
    by_level = {}
    for level in ("R1", "R2", "R3"):
        subset = [row for row in rows if row["level"] == level]
        by_level[level] = {
            "n_cells": len(subset),
            "n_partial_relation_local": sum(row["verdict"] == "partial_relation_local" for row in subset),
            "n_candidate_mismatch_or_incomplete": sum(row["verdict"] == "candidate_mismatch_or_incomplete" for row in subset),
            "n_no_candidate_bounded_non_discovery": sum(row["verdict"] == "no_candidate_bounded_non_discovery" for row in subset),
        }
    relation_counts = Counter(match["relation_id"] for row in rows for match in row["matched_relations"])
    depth_counts = Counter(match["effective_code_depth"] for row in rows for match in row["matched_relations"])
    return {
        "schema": SCHEMA,
        "status": "static-source-fidelity-audit-complete-before-execution",
        "task": TASK,
        "source_seed_map": {
            "path": str(seed_source.resolve().relative_to(ROOT.resolve())),
            "sha256": _sha256(seed_source),
            "schema": SEED_SCHEMA,
        },
        "audit_design": {
            "independent_of_retrieval_score": True,
            "requested_vs_implemented_subrelation_compared": True,
            "program_source_executed": False,
            "items_loaded": False,
            "prompt_outputs_loaded": False,
            "reference_or_outcome_values_loaded": False,
            "external_supervision_used": False,
            "historical_600_character_constructs_modified": False,
            "historical_programs_modified": False,
            "negative_result_interpretation": "bounded non-discovery only; never tacitness",
        },
        "depth_vocabulary": {
            "1": "surface text, finite lexicon, or local regular-language measurement",
            "2": "dependency/entity parse, graph aggregation, date arithmetic, or finite consistency check",
            "3": "retrieval or external-evidence pipeline (not used)",
            "4": "sandboxed execution, tests, or formal proof checking (not used)",
        },
        "summary": {
            "n_cells": len(rows),
            "by_level": by_level,
            "verdict_counts": dict(sorted(counts.items())),
            "n_partial_relation_local_cells": counts["partial_relation_local"],
            "n_exact_whole_construct_cells": 0,
            "n_relation_mappings": sum(relation_counts.values()),
            "relation_mapping_counts": dict(sorted(relation_counts.items())),
            "relation_mapping_depth_counts": {str(key): value for key, value in sorted(depth_counts.items())},
            "relation_catalog_size": len(RELATIONS),
            "not_codability_or_isomorphism": True,
        },
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-map", type=Path, default=DEFAULT_SEED_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    seed_map = json.loads(args.seed_map.read_text(encoding="utf-8"))
    payload = build_fidelity_audit(seed_map, seed_source=args.seed_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
