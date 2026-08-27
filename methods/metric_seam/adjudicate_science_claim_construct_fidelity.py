"""Static relation-fidelity audit for the full-article science claim seed map.

The audit is deliberately separate from candidate retrieval.  It credits only the
existing verifier's narrow executable relation: selected numeric/comparative abstract
claims can be supported or contradicted by distinct full-paper body sentences under a
document-local retrieval and exact matching contract.  It does not infer scientific
truth, evidence reliability, causal validity, generalizability, writing quality, or a
whole peer-review judgement.

No article, item, outcome, historical certificate, program output, prompt output, or
reconstruction result is read by this module.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA = "metric-seam.hierarchy-science-claim-construct-fidelity.v1"
SEED_SCHEMA = "metric-seam.hierarchy-science-claim-seed-map.v1"
TASK = "peer-review"
CAPABILITY_ID = "science_claims_v2_relation_strict_full_article"


def _matched_decision(
    *,
    object_assessment: str,
    relation_assessment: str,
    unimplemented: list[str],
    aggregation_assessment: str,
) -> dict:
    return {
        "verdict": "partial_relation_local",
        "object_assessment": object_assessment,
        "relation_assessment": relation_assessment,
        "polarity_assessment": (
            "aligned for the narrow relation: directed comparative role reversal can produce "
            "contradicted, aligned numeric/comparative relations can produce supported, and "
            "missing/nonassertive/insufficient relations abstain or remain uncertified"
        ),
        "applicability_assessment": (
            "aligned only when an article has an abstract, a distinct full-paper body, and an "
            "extractable result-bearing numeric/comparative relation; explicit abstention is "
            "part of the capability rather than a negative score"
        ),
        "aggregation_assessment": aggregation_assessment,
        "matched_subrelations": [
            {
                "relation": (
                    "document-internal support or contradiction of a result-bearing "
                    "numeric/comparative abstract claim by a distinct full-paper body sentence"
                ),
                "channels": ["pure_code"],
                "capability_chain": [
                    "abstract claim parser",
                    "document-local BM25 body-sentence retrieval",
                    "numeric/unit/entity/direction or comparative-role predicate",
                    "exact maximum-weight one-to-one matching",
                ],
                "effective_code_depth": 3,
                "polarity": "supported / contradicted / uncertified-or-abstain",
            }
        ],
        "unimplemented_or_weak_relations": unimplemented,
    }


def _mismatch_decision(
    *,
    object_assessment: str,
    relation_assessment: str,
    applicability_assessment: str,
    aggregation_assessment: str,
    unimplemented: list[str],
) -> dict:
    return {
        "verdict": "relation_mismatch",
        "object_assessment": object_assessment,
        "relation_assessment": relation_assessment,
        "polarity_assessment": (
            "the verifier's supported/contradicted polarity is not the requested criterion "
            "polarity for this object and therefore receives no relation-local credit"
        ),
        "applicability_assessment": applicability_assessment,
        "aggregation_assessment": aggregation_assessment,
        "matched_subrelations": [],
        "unimplemented_or_weak_relations": unimplemented,
    }


_DECISIONS = {
    "TB::peer-review::general::R1::parented_tree::91::5d3565fab492bade448e": _matched_decision(
        object_assessment=(
            "partial object match: the executable object is a selected result-bearing abstract "
            "claim paired with a distinct sentence from that article's full-paper body"
        ),
        relation_assessment=(
            "partial relation match: exact numeric/comparative consistency is one narrow form "
            "of claim-to-presented-evidence alignment, but does not establish adequacy, "
            "reliability, evidence level, or appropriately calibrated tone"
        ),
        aggregation_assessment=(
            "up to five selected abstract claims are matched one-to-one to body sentences and "
            "summarized as document status; unselected claims and whole-paper tone are outside "
            "the aggregation frame"
        ),
        unimplemented=[
            "adequacy and reliability of the underlying evidence",
            "evidence hierarchy or level-of-evidence classification",
            "tone calibration, hedging, and overclaim assessment",
            "coverage of all manuscript claims",
        ],
    ),
    "TB::peer-review::general::R1::parented_tree::69::bb908d589408a9fe234e": _matched_decision(
        object_assessment=(
            "partial object match: the verifier reads the abstract and full-paper body, but "
            "does not read or evaluate the title as a separate object"
        ),
        relation_assessment=(
            "partial relation match: consistency of numeric/comparative abstract findings with "
            "distinct body sentences is a narrow executable component of abstract accuracy"
        ),
        aggregation_assessment=(
            "the pipeline selects at most five relation-rich abstract sentences rather than "
            "assessing complete abstract coverage, title quality, or venue relevance"
        ),
        unimplemented=[
            "title clarity and title-to-article fidelity",
            "complete abstract coverage of scope, methods, contribution, and limitations",
            "venue or domain relevance",
            "readability and communicative clarity",
        ],
    ),
    "TB::peer-review::general::R2::merged_group::43::1af77346eecd6dfa1f19": _matched_decision(
        object_assessment=(
            "partial object match: selected result-bearing abstract claims and distinct body "
            "sentences are both manuscript-internal objects named by the criterion's broad "
            "claim/evidence relation"
        ),
        relation_assessment=(
            "partial relation match: the strict verifier checks numeric/comparative relation "
            "agreement and reversal, but not whether study design warrants causality or whether "
            "the body sentence itself is strong evidence"
        ),
        aggregation_assessment=(
            "exact one-to-one matching prevents reuse of a body sentence across selected claims, "
            "but does not aggregate confounders, alternatives, causal design, or epistemic tone"
        ),
        unimplemented=[
            "causal versus correlational claim identification",
            "confounder and alternative-explanation analysis",
            "evidence-strength and epistemic-humility judgement",
            "whole-manuscript claim coverage",
        ],
    ),
    "TB::peer-review::general::R2::grandparent::23::b9e2d7adb550864a17f7": _matched_decision(
        object_assessment=(
            "partial object match: selected abstract result claims and their retrieved body "
            "relations are a narrow subset of the criterion's claims, methods, and baselines"
        ),
        relation_assessment=(
            "partial relation match: document-internal numeric/comparative support is relevant "
            "to claim calibration, while analytical correctness and causal warrant remain absent"
        ),
        aggregation_assessment=(
            "the document status aggregates at most five one-to-one relation matches; it does "
            "not combine mathematical correctness, baseline quality, or causal caution"
        ),
        unimplemented=[
            "analytical and mathematical correctness",
            "baseline strength and appropriateness",
            "causal inference validity",
            "whether a repeated body assertion constitutes reliable evidence",
        ],
    ),
    "TB::peer-review::general::R3::merged_group::7::ef507ad244604eac2ccb": _mismatch_decision(
        object_assessment=(
            "object mismatch: the criterion asks about references and citation practices, while "
            "the verifier pairs abstract prose with body prose and has no citation object"
        ),
        relation_assessment=(
            "relation mismatch: body-sentence consistency cannot establish reference accuracy, "
            "coverage, fair representation, or whether citations support claims"
        ),
        applicability_assessment=(
            "the verifier does not parse a bibliography, citation spans, cited sources, or "
            "citation-to-claim links, so its applicability contract cannot reach this construct"
        ),
        aggregation_assessment=(
            "one-to-one abstract/body matching is unrelated to bibliography coverage or "
            "manipulation/self-citation aggregation"
        ),
        unimplemented=[
            "reference currency, relevance, and accuracy",
            "citation coverage and citation-to-claim entailment",
            "fair representation and phantom-reference detection",
            "self-citation and citation-manipulation assessment",
        ],
    ),
    "TB::peer-review::general::R3::grandparent::13::a05f17758c47ae9a2359": _mismatch_decision(
        object_assessment=(
            "adjacent object only: an abstract causal/generalization sentence may be selected, "
            "but the verifier does not type a claim as causal, transport, or external-validity"
        ),
        relation_assessment=(
            "relation mismatch: repeating a numeric/comparative relation in a body sentence does "
            "not establish causal identification, extrapolation validity, or transportability"
        ),
        applicability_assessment=(
            "the executable applicability gate is presence of a relation-rich abstract claim, "
            "not presence of a causal design or a target-population generalization analysis"
        ),
        aggregation_assessment=(
            "claim/body matching has no aggregation over design assumptions, populations, "
            "settings, interventions, or identification threats"
        ),
        unimplemented=[
            "causal-claim and generalization-claim typing",
            "study-design and identification-assumption checks",
            "population/setting transportability analysis",
            "calibration of causal or external-validity language to design strength",
        ],
    ),
    "TB::peer-review::general::R3::merged_group::3::f1b521e38ed547fd635e": _matched_decision(
        object_assessment=(
            "partial object match: selected abstract result claims and distinct full-paper body "
            "sentences instantiate a narrow claim/presented-evidence pair"
        ),
        relation_assessment=(
            "partial relation match: executable numeric/comparative support or contradiction is "
            "one component of alignment, but causal caution, confounders, alternatives, and "
            "evidence strength are not represented"
        ),
        aggregation_assessment=(
            "one-to-one matching is faithful for selected relations and prevents evidence reuse; "
            "the final document status is not a holistic claim-evidence score"
        ),
        unimplemented=[
            "causal and ecological-fallacy detection",
            "confounder and alternative-explanation assessment",
            "epistemic humility and tone calibration",
            "evidence reliability and complete claim coverage",
        ],
    ),
    "TB::peer-review::general::R3::merged_group::2::2e0e2346ca9cc79f8986": _mismatch_decision(
        object_assessment=(
            "object mismatch: the verifier extracts claims only from the abstract, whereas this "
            "criterion specifically evaluates discussion and conclusion interpretation"
        ),
        relation_assessment=(
            "relation mismatch: abstract/body consistency does not evaluate contextualized "
            "interpretation, alternatives, speculation, limitations, or implications"
        ),
        applicability_assessment=(
            "the source contract has no discussion/conclusion section detector and no "
            "interpretive-claim applicability gate"
        ),
        aggregation_assessment=(
            "selected abstract relations cannot be transferred to a whole discussion/conclusion "
            "aggregation merely because both mention data support"
        ),
        unimplemented=[
            "discussion and conclusion section identification",
            "interpretation versus speculation distinction",
            "alternatives, limitations, and prior-work contextualization",
            "practical and scientific implication quality",
        ],
    ),
    "TB::peer-review::general::R3::grandparent::5::f54f42e2e95c866f95a0": _matched_decision(
        object_assessment=(
            "partial object match: the abstract and full-paper body are present, but title and "
            "plain-language summary objects are not separately represented"
        ),
        relation_assessment=(
            "partial relation match: strict body agreement for selected numeric/comparative "
            "abstract findings is a narrow component of front-matter fidelity"
        ),
        aggregation_assessment=(
            "at most five relation-rich abstract claims are covered; there is no aggregation over "
            "scope, design, all key findings, all conclusions, or multiple audience versions"
        ),
        unimplemented=[
            "title fidelity and clarity",
            "plain-language summary fidelity and audience tailoring",
            "complete scope/design/finding/conclusion coverage",
            "hype, accessibility, and communicative clarity",
        ],
    ),
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
            "candidate_capability_id": None,
            "verdict": "no_candidate",
            "matched_subrelations": [],
            "unimplemented_or_weak_relations": [],
            "eligible_relation_local_depths": [],
            "maximum_matching_relation_depth": None,
            "exact_whole_construct_fidelity": False,
            "execution_witness_established": False,
        }
    if cell_id not in _DECISIONS:
        raise ValueError(f"retrieved science seed lacks a frozen adjudication: {cell_id}")
    if seed.get("capability_id") != CAPABILITY_ID:
        raise ValueError(
            f"unexpected capability for {cell_id}: {seed.get('capability_id')}"
        )
    decision = _DECISIONS[cell_id]
    relations = [dict(relation) for relation in decision["matched_subrelations"]]
    depths = sorted({int(relation["effective_code_depth"]) for relation in relations})
    return {
        "cell_id": cell_id,
        "task": TASK,
        "level": str(seed_row["level"]),
        "metric_name": str(seed_row["metric_name"]),
        "metric_description": str(seed_row["metric_description"]),
        "candidate_capability_id": CAPABILITY_ID,
        "candidate_source_paths": list(seed["source_paths"]),
        "verdict": str(decision["verdict"]),
        "object_assessment": str(decision["object_assessment"]),
        "relation_assessment": str(decision["relation_assessment"]),
        "polarity_assessment": str(decision["polarity_assessment"]),
        "applicability_assessment": str(decision["applicability_assessment"]),
        "aggregation_assessment": str(decision["aggregation_assessment"]),
        "matched_subrelations": relations,
        "unimplemented_or_weak_relations": list(
            decision["unimplemented_or_weak_relations"]
        ),
        "eligible_relation_local_depths": depths,
        "maximum_matching_relation_depth": max(depths) if depths else None,
        "surrounding_relation_chain_depth": int(
            seed["maximum_relation_chain_depth"]
        ),
        "static_pure_code_capability": True,
        "exact_whole_construct_fidelity": False,
        "execution_witness_established": False,
        "external_scientific_truth_established": False,
        "automatic_discovery": False,
        "eligible_for_later_relation_local_execution": (
            decision["verdict"] == "partial_relation_local"
        ),
        "interpretation": (
            "static relation-local source audit of a manually designed pure-code capability; "
            "no article was run and no scientific-truth or whole-review claim follows"
        ),
    }


def build_audit(seed_map: Mapping) -> dict:
    if seed_map.get("schema") != SEED_SCHEMA:
        raise ValueError(f"expected {SEED_SCHEMA}")
    if seed_map.get("task") != TASK or seed_map.get("n_cells") != 90:
        raise ValueError("expected the 90-cell peer-review science seed inventory")
    rows = [_audit_row(row) for row in seed_map.get("rows", [])]
    if len(rows) != 90 or len({row["cell_id"] for row in rows}) != 90:
        raise ValueError("science fidelity audit requires 90 unique cells")
    retrieved_ids = {
        row["cell_id"]
        for row in seed_map["rows"]
        if row.get("selected_seed") is not None
    }
    if retrieved_ids != set(_DECISIONS):
        missing = sorted(retrieved_ids - set(_DECISIONS))
        stale = sorted(set(_DECISIONS) - retrieved_ids)
        raise ValueError(
            f"frozen adjudication coverage mismatch; missing={missing}, stale={stale}"
        )
    verdicts = Counter(row["verdict"] for row in rows)
    eligible = [row for row in rows if row["verdict"] == "partial_relation_local"]
    mismatches = [row for row in rows if row["verdict"] == "relation_mismatch"]
    by_level = {
        level: {
            "n_cells": sum(row["level"] == level for row in rows),
            "n_retrieved": sum(
                row["level"] == level
                and row["candidate_capability_id"] is not None
                for row in rows
            ),
            "n_partial_relation_local": sum(
                row["level"] == level
                and row["verdict"] == "partial_relation_local"
                for row in rows
            ),
            "n_relation_mismatch": sum(
                row["level"] == level and row["verdict"] == "relation_mismatch"
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
        "status": "static-relation-local-adjudication-complete-pre-execution",
        "task": TASK,
        "n_cells": len(rows),
        "source_seed_schema": seed_map["schema"],
        "source_panel_content_sha256": seed_map.get("panel_content_sha256"),
        "design_scope": "static_source_only_manual_construct_fidelity_adjudication",
        "forbidden_inputs": list(seed_map.get("forbidden_inputs", [])),
        "execution_performed": False,
        "articles_or_items_loaded": False,
        "reference_values_loaded": False,
        "outcome_labels_loaded": False,
        "historical_certificates_or_program_outputs_loaded": False,
        "prompt_or_reconstruction_outputs_loaded": False,
        "external_supervision_loaded_for_this_audit": False,
        "fidelity_rule": (
            "credit only an executable sub-relation whose object, relation, polarity, "
            "applicability, and aggregation align at least partially; report the depth of that "
            "matched relation chain rather than surrounding source size"
        ),
        "provenance_rule": (
            "the verifier remains a manually designed full-article pure-code pipeline seed; "
            "static mapping is not automatic discovery, execution, reconstruction, external "
            "scientific truth, or a whole peer-review judgement"
        ),
        "summary": {
            "verdict_counts": dict(sorted(verdicts.items())),
            "by_level": by_level,
            "n_retrieved": len(eligible) + len(mismatches),
            "n_partial_relation_local": len(eligible),
            "n_relation_mismatch": len(mismatches),
            "n_exact_whole_construct": 0,
            "n_mappings_to_static_pure_code_capability": len(eligible),
            "n_execution_witnesses": 0,
            "n_external_scientific_truth_claims": 0,
            "n_automatic_discoveries": 0,
            "maximum_matching_relation_depth_counts": dict(sorted(depth_counts.items())),
            "n_with_depth3_matching_relation": sum(
                3 in row["eligible_relation_local_depths"] for row in eligible
            ),
        },
        "interpretation_limits": [
            "candidate retrieval plus static fidelity is not execution or reconstruction",
            "partial relation-local fidelity is not whole-criterion codability",
            "document-internal consistency is not external scientific truth",
            "a body assertion is not automatically reliable scientific evidence",
            "depth belongs only to the matched relation chain, not to adjacent constructs",
            "failure to retrieve or match is bounded non-discovery, never evidence of tacitness",
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
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
