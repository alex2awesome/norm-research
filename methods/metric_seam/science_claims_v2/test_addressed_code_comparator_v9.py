from __future__ import annotations

from methods.metric_seam.science_claims_v2 import addressed_code_comparator_v9 as v9
from methods.metric_seam.science_claims_v2 import addressed_pipeline


def _source_map(abstract: str, body: str, paper_id: str = "p") -> dict:
    return addressed_pipeline.build_source_map(
        {"paper_id": paper_id, "abstract": abstract, "body": body}
    )


def test_addressed_strict_positive_numeric_control() -> None:
    result = v9.verify_addressed_document(
        "p",
        _source_map(
            "We show a 28% improvement in robustness on the benchmark.",
            "Table 2 shows a 28% improvement in robustness on the benchmark.",
        ),
    )
    assert result["status"] == "supported"


def test_addressed_strict_rejects_percentage_metric_collision() -> None:
    result = v9.verify_addressed_document(
        "p",
        _source_map(
            "We show that VQMoE achieves a 28% improvement in robustness compared to SMoE.",
            "VQMoE saves 28% of computational resources compared to SMoE.",
        ),
    )
    assert result["status"] != "supported"


def test_addressed_strict_rejects_large_entity_collision() -> None:
    result = v9.verify_addressed_document(
        "p",
        _source_map(
            "Experiments with 1000 nodes demonstrate nearly optimal solutions.",
            "After 1000 rounds of iteration, the method is nearly optimal.",
        ),
    )
    assert result["status"] != "supported"


def test_addressed_strict_rejects_question_as_comparison_evidence() -> None:
    result = v9.verify_addressed_document(
        "p",
        _source_map(
            "We show that OTDF outperforms prior strong baselines.",
            "Can OTDF beat prior strong baselines across varied shifts?",
        ),
    )
    assert result["status"] != "supported"


def test_addressed_strict_does_not_parse_codec_as_quantity() -> None:
    result = v9.verify_addressed_document(
        "p",
        _source_map(
            "We model videos directly using the AVC/H.264 codec.",
            "The AVC/H.264 codec is modeled by an autoregressive transformer.",
        ),
    )
    assert not result["certificates"]


def test_addressed_uses_same_strict_quantity_uniqueness() -> None:
    result = v9.verify_addressed_document(
        "p",
        _source_map(
            "Accuracy gains reach 90.9% in one setting and 91.3% in another.",
            "Table 2 reports one accuracy gain of 91.3% in the evaluated setting.",
        ),
    )
    assert result["status"] != "supported"
