from __future__ import annotations

import json
from pathlib import Path

from methods.metric_seam.science_claims_v2.core import (
    extract_quantities,
    metamorphic_self_check,
    quantity_equal,
    segment_sentences,
    verify_document,
)
from methods.metric_seam.science_claims_v2.evaluate import load_unlabelled


ABSTRACT = "We show that our method outperforms BERT by 12.5% on image accuracy."
EVIDENCE = (
    "We evaluate both systems on the held-out image benchmark. "
    "Table 2 shows that our method outperforms BERT by 12.5% on image accuracy."
)


def test_sentence_segmentation_protects_decimals() -> None:
    sentences = segment_sentences("Accuracy is 12.5%. We report Fig. 2 in the appendix.")
    assert [s.text for s in sentences] == [
        "Accuracy is 12.5%.",
        "We report Fig. 2 in the appendix.",
    ]


def test_unit_normalization() -> None:
    ms = extract_quantities("latency is 500 ms")[0]
    sec = extract_quantities("latency is 0.5 seconds")[0]
    pct_word = extract_quantities("gain is 12.5 percent")[0]
    pct_symbol = extract_quantities("gain is 12.5%")[0]
    assert quantity_equal(ms, sec)
    assert quantity_equal(pct_word, pct_symbol)


def test_model_versions_dimensions_and_table_indices_are_not_quantities() -> None:
    assert extract_quantities("GPT-4 uses 3D inputs; see Table 2 for L_2 regularization") == ()
    lower_bound = extract_quantities("recovery exceeds 60+%")[0]
    assert lower_bound.unit == "percent" and lower_bound.value == 0.6


def test_enumeration_indices_are_not_quantitative_obligations() -> None:
    quantities = extract_quantities(
        "Our contributions are (1) retrieval, (2) matching, and (3) certificates with 64% coverage."
    )
    assert len(quantities) == 1
    assert quantities[0].unit == "percent" and quantities[0].value == 0.64


def test_numeric_comparative_support_certificate() -> None:
    result = verify_document("p", ABSTRACT, EVIDENCE)
    assert result["status"] == "supported"
    cert = next(c for c in result["certificates"] if c["decision"] == "supported")
    assert cert["checks"]["quantity_matches"] == cert["checks"]["quantity_required"] == 1
    assert cert["checks"]["relation_state"] == "aligned"


def test_remove_evidence_number_destroys_support_certificate() -> None:
    original = verify_document("p", ABSTRACT, EVIDENCE)
    mutated = verify_document("p", ABSTRACT, EVIDENCE.replace("12.5%", "the reported margin"))
    assert original["status"] == "supported"
    assert mutated["status"] != "supported"
    assert not any(c["decision"] == "supported" for c in mutated["certificates"])


def test_swap_comparison_direction_yields_contradiction() -> None:
    body = (
        "We evaluate both systems on the held-out image benchmark. "
        "Table 2 shows that BERT outperforms our method by 12.5% on image accuracy."
    )
    result = verify_document("p", ABSTRACT, body)
    assert result["status"] == "contradicted"
    assert any(c["reason"] == "reversed_roles" for c in result["certificates"])


def test_negated_claim_and_reversed_roles_are_signed_support() -> None:
    abstract = "We show that GSL does not outperform baseline GNNs under matched tuning."
    body = (
        "We use the same hyperparameter search for all systems. "
        "Table 3 shows that baseline GNNs outperform GSL under matched tuning."
    )
    result = verify_document("p", abstract, body)
    assert result["status"] == "supported"
    assert any(c["checks"]["relation_state"] == "aligned_reversed"
               for c in result["certificates"] if c["decision"] == "supported")


def test_swap_baseline_cannot_certify_original_claim() -> None:
    body = (
        "We evaluate systems on the held-out image benchmark. "
        "Table 2 shows that our method outperforms RoBERTa by 12.5% on image accuracy."
    )
    result = verify_document("p", ABSTRACT, body)
    assert result["status"] != "supported"
    assert not any(c["decision"] == "supported" for c in result["certificates"])


def test_abstract_only_abstains_honestly() -> None:
    result = verify_document("p", ABSTRACT, ABSTRACT)
    assert result["status"] == "abstain"
    assert result["reason"] == "abstract_only_no_independent_evidence"


def test_marker_match_is_evidence_link_not_semantic_support() -> None:
    abstract = "We introduce WidgetNet, a robust model for image classification."
    body = "Figure 2 shows robust image classification results for WidgetNet on the evaluation set."
    result = verify_document("p", abstract, body)
    assert result["status"] == "evidence_link"
    assert result["certificate_count"] == 0
    assert result["evidence_link_count"] == 1
    assert result["evidence_links"][0]["witness_kind"] == "evidence_link"


def test_loader_ignores_labels(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps({"paper_id": "p", "y": 0, "abstract": ABSTRACT, "body": EVIDENCE}) + "\n")
    second.write_text(json.dumps({"paper_id": "p", "y": 1, "abstract": ABSTRACT, "body": EVIDENCE}) + "\n")
    assert list(load_unlabelled(first)) == list(load_unlabelled(second))
    assert "y" not in list(load_unlabelled(first))[0]


def test_articulability_counterpart_is_frozen_but_not_run() -> None:
    path = Path(__file__).with_name("articulability_prompt.json")
    spec = json.loads(path.read_text())
    assert spec["input_allowlist"] == ["paper_id", "abstract", "body"]
    assert spec["status"] == "frozen_not_run"
    assert spec["run_result"] == "unavailable_not_run"
    assert set(spec["decision_semantics"]) == {
        "supported", "contradicted", "evidence_link", "insufficient", "abstain"
    }


def test_metamorphic_self_check() -> None:
    assert all(metamorphic_self_check().values())
