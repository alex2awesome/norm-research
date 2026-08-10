from __future__ import annotations

from methods.metric_seam.hierarchy_science_exact_ctext_prompt_batch import (
    DEFAULT_ITEMS,
    _parse_ctext,
    load_items,
)
from methods.metric_seam.science_claims_v2 import core as v2
from methods.metric_seam.science_claims_v2 import core_relation_strict as strict
from methods.metric_seam.science_claims_v2.numeric_comparative_projection_v1 import (
    DECISIONS,
    RELATIONS,
    build_projection,
    project_document,
    select_numeric_comparative_claims,
)


def test_projection_filters_relation_class_before_top_five_ranking():
    abstract = " ".join(
        ["We prove convergence of the estimator."]
        + [
            f"Accuracy improves to {90 + index}% on task {chr(65 + index)}."
            for index in range(6)
        ]
    )
    with strict._strict_bindings():  # type: ignore[attr-defined]
        archived_global = v2.extract_claims(abstract)
        projected = select_numeric_comparative_claims(abstract)
    assert len(archived_global) == len(projected) == 5
    assert archived_global[0].relation == "theoretical"
    assert {claim.relation for claim in projected} == {"numeric"}
    assert projected[-1].sentence.text == "Accuracy improves to 94% on task E."


def test_projection_uses_only_exact_reconstruction_decision_vocabulary():
    supported = project_document(
        "supported",
        "We show a 28% improvement in robustness on the benchmark.",
        "Table 2 shows a 28% improvement in robustness on the benchmark.",
    )
    contradicted = project_document(
        "contradicted",
        "We show that our method outperforms BERT.",
        "Table 2 shows that BERT outperforms our method.",
    )
    insufficient = project_document(
        "insufficient",
        "We report 91% accuracy on the benchmark.",
        "The appendix discusses implementation details without a result value.",
    )
    assert supported["decision_counts"] == {"supported": 1}
    assert contradicted["decision_counts"] == {"contradicted": 1}
    assert insufficient["decision_counts"] == {"insufficient": 1}
    assert {
        selection["decision"]
        for result in (supported, contradicted, insufficient)
        for selection in result["selections"]
    } <= DECISIONS
    assert all(
        selection["claim"]["relation"] in RELATIONS
        for result in (supported, contradicted, insufficient)
        for selection in result["selections"]
    )


def test_real_300_item_projection_is_deterministic_and_has_no_evidence_link():
    _manifest, items = load_items(DEFAULT_ITEMS)
    first = build_projection(items, parse_ctext=_parse_ctext)
    second = build_projection(items, parse_ctext=_parse_ctext)
    assert first == second
    assert first["summary"] == {
        "items": 300,
        "selected_claims": 158,
        "decision_counts": {"insufficient": 141, "supported": 17},
        "evidence_link_decisions": 0,
    }
    assert first["selection_contract"] == {
        "candidate_relations": ["comparative", "numeric"],
        "relation_classification_priority": [
            "comparative",
            "theoretical",
            "numeric",
            "empirical",
            "qualitative",
        ],
        "filter_before_top_five_ranking": True,
        "candidate_score": {
            "explicit_claim_cue": 2,
            "directed_comparison": 2,
            "parsed_quantity": 1,
            "result_or_theory_marker": 1,
            "minimum": 2,
        },
        "ranking": "selection_score descending, sentence index ascending",
        "output_order": "source sentence order",
        "limit": 5,
    }
    assert first["decision_contract"] == {
        "reconstruction_target": [
            "contradicted",
            "insufficient",
            "supported",
        ],
        "evidence_link_in_reconstruction_target": False,
        "unmatched_selected_claim": "insufficient",
    }
    assert not {
        selection["decision"]
        for row in first["rows"]
        for selection in row["selections"]
    } - DECISIONS
