from __future__ import annotations

import math

from methods.metric_seam.science_claims_v2.core_corrected import (
    extract_quantities,
    metamorphic_self_check,
    quantity_anchor_terms,
    verify_document,
)


def test_audited_suffixes_are_complete_normalized_tokens() -> None:
    expected = {
        "100k examples": (100_000.0, "unitless"),
        "33B parameters": (33_000_000_000.0, "unitless"),
        "6.7B parameters": (6_700_000_000.0, "unitless"),
        "1.5B parameters": (1_500_000_000.0, "unitless"),
        "30nm process": (30e-9, "meter"),
    }
    for text, (value, unit) in expected.items():
        quantities = extract_quantities(text)
        assert len(quantities) == 1, text
        assert math.isclose(quantities[0].value, value, rel_tol=1e-12, abs_tol=1e-18)
        assert quantities[0].unit == unit
        assert quantities[0].raw.strip() == text.split()[0]


def test_unrecognized_suffix_never_backtracks_to_digit_prefix() -> None:
    assert extract_quantities("uses 100quux examples") == ()
    assert extract_quantities("uses 6.7widgets") == ()
    assert extract_quantities("the BCI 2b dataset") == ()
    assert extract_quantities("the Human3.6M dataset") == ()


def test_process_and_document_indices_are_not_quantities() -> None:
    for text in (
        "Stage 1 focuses on proposals.",
        "Step 2 performs matching.",
        "Phase 3 evaluates the result.",
        "Section 4.2 gives results.",
        "Table 1 reports accuracy.",
        "Algorithm 2 converges.",
    ):
        assert extract_quantities(text) == (), text


def test_named_versions_and_slash_dataset_identifiers_are_not_quantities() -> None:
    for text in (
        "the Lean 4 compiler",
        "DALL·E 3 and GPT-4V",
        "the Habitat 3.0 platform",
        "CIFAR-10/100 and ImageNet",
        "ViT-B/16 on CIFAR",
        "AlpacaEval 2.0 and MT-Bench",
    ):
        assert extract_quantities(text) == (), text


def test_p3eft_superscript_is_not_a_quantity() -> None:
    assert extract_quantities("We propose P$^3$EFT.") == ()
    assert extract_quantities("We propose P^{3}EFT.") == ()


def test_small_count_anchor_is_local_entity_head() -> None:
    adapters = extract_quantities("We use 28 existing LoRA adapters through MeteoRA.")[0]
    tasks = extract_quantities("We evaluate across 28 tasks.")[0]
    assert "adapter" in quantity_anchor_terms(
        "We use 28 existing LoRA adapters through MeteoRA.", adapters
    )
    assert quantity_anchor_terms("We evaluate across 28 tasks.", tasks) == ("task",)


def test_28_adapters_cannot_certify_against_28_tasks() -> None:
    result = verify_document(
        "p",
        "Our evaluation uses 28 existing LoRA adapters and demonstrates robust performance.",
        "Table 2 demonstrates robust performance across the selected 28 tasks.",
    )
    assert result["status"] != "supported"
    assert not result["certificates"]


def test_28_adapters_can_certify_against_28_adapters() -> None:
    result = verify_document(
        "p",
        "Our evaluation uses 28 existing LoRA adapters and demonstrates robust performance.",
        "Table 2 demonstrates robust performance across all 28 LoRA adapters.",
    )
    assert result["status"] == "supported"
    assert result["certificates"]


def test_replace_quantity_entity_invalidates_certificate() -> None:
    abstract = "We show robust performance across 20 optimization algorithms."
    supported = verify_document(
        "p", abstract, "Table 2 shows robust performance across 20 optimization algorithms."
    )
    mutated = verify_document(
        "p", abstract, "Table 2 shows robust performance across 20 image datasets."
    )
    assert supported["status"] == "supported"
    assert mutated["status"] != "supported"


def test_p3eft_cannot_certify_against_three_following_measures() -> None:
    result = verify_document(
        "p",
        "Using this analysis, we propose P$^3$EFT and demonstrate lower performance overhead.",
        "For an unrelated baseline, we evaluated the 3 following measures of privacy leakage.",
    )
    assert result["status"] != "supported"


def test_corrected_metamorphic_suite() -> None:
    checks = metamorphic_self_check()
    assert len(checks) == 10
    assert all(checks.values())
