from __future__ import annotations

from methods.metric_seam.press_release_relations_v1 import (
    RELATION_SPECS,
    analyze_press_release_ctext,
)


RICH_TEXT = """ACME Announces Community Research Event

SAN JOSE, April 30, 2026 — ACME will host a research summit in San Jose on May 2, 2026.
“The study may improve access for families,” said Jane Doe, chief scientist at ACME.
According to the North Institute study, 20 out of 100 participants (20%) completed
the program, compared with 10 participants last year. Register for the event at
https://example.org/register.

ABOUT ACME
For more information, visit www.acme.com or contact media@acme.com.
"""


def test_relation_registry_and_result_are_exhaustive():
    result = analyze_press_release_ctext(RICH_TEXT)
    assert set(result["relations"]) == set(RELATION_SPECS)
    assert result["input_contract"] == {
        "field": "ctext",
        "exact_input_chars": len(RICH_TEXT),
        "corpus_state_loaded": False,
        "external_resources_loaded": False,
        "network_access_used": False,
        "outcomes_or_references_loaded": False,
    }
    assert all(
        relation["absence_certificate"] is False
        for relation in result["relations"].values()
    )


def test_deep_relations_emit_local_witnesses_without_truth_claims():
    relations = analyze_press_release_ctext(RICH_TEXT)["relations"]
    expected_witnessed = {
        "attribution_claim_binding",
        "entity_evidence_graph",
        "claim_evidence_alignment",
        "url_role_clause_binding",
        "cta_resource_binding",
        "boilerplate_contact_structure",
        "event_logistics_binding",
        "significance_comparison_binding",
        "uncertainty_claim_scope_binding",
    }
    for relation_id in expected_witnessed:
        assert relations[relation_id]["status"] in {"measured", "witnessed"}
        assert relations[relation_id]["witness_count"] > 0
    assert relations["attribution_claim_binding"]["realized_depth"] == 3
    assert relations["claim_evidence_alignment"]["realized_depth"] in {2, 3}
    assert "factual correctness" in relations["claim_evidence_alignment"][
        "does_not_establish"
    ]
    assert relations["url_role_clause_binding"]["summary"][
        "network_resolution_attempted"
    ] is False


def test_claim_evidence_d3_requires_a_cross_sentence_retrieval_edge():
    text = (
        "The Boston program expects to improve access for families. "
        "A Boston survey counted 50 participating families in 2025."
    )
    relation = analyze_press_release_ctext(text)["relations"][
        "claim_evidence_alignment"
    ]
    assert relation["program_relation_depth_ceiling"] == 3
    assert relation["matched_relation_depth"] == 3
    assert relation["summary"]["retrieval_edges"] == 1


def test_quantity_arithmetic_is_rederived_and_not_just_detected():
    relation = analyze_press_release_ctext(RICH_TEXT)["relations"][
        "date_quantity_internal_consistency"
    ]
    assert relation["summary"]["rederived_arithmetic_relations"] == 1
    assert relation["summary"]["arithmetic_consistent"] == 1
    assert relation["realized_depth"] == 3
    arithmetic = [
        row for row in relation["witnesses"] if row["kind"] == "arithmetic_recomputation"
    ]
    assert arithmetic[0]["computed_pct"] == 20.0
    assert arithmetic[0]["stated_pct"] == 20.0


def test_invalid_calendar_surface_is_retained_as_counter_witness():
    text = "ACME schedules the event for April 31, 2026 in Boston."
    relation = analyze_press_release_ctext(text)["relations"][
        "date_quantity_internal_consistency"
    ]
    assert relation["summary"]["invalid_calendar_date_surfaces"] == 1
    invalid = [
        row for row in relation["witnesses"] if row["kind"] == "invalid_calendar_surface"
    ]
    assert invalid[0]["surface"] == "April 31, 2026"
    assert invalid[0]["valid_calendar_date"] is False


def test_calendar_month_may_is_not_an_uncertainty_operator():
    text = "The tour visits Boston on May 12, 2026. It may later visit Providence."
    relation = analyze_press_release_ctext(text)["relations"][
        "uncertainty_claim_scope_binding"
    ]
    cues = [
        binding["cue"]
        for witness in relation["witnesses"]
        for binding in witness["bindings"]
    ]
    assert cues == ["may"]


def test_commitment_action_and_opening_locality_require_dependency_bindings():
    text = (
        "ACME will publish a security review in Boston on May 12, 2026. "
        "The Boston office will then update its public safeguards page."
    )
    relations = analyze_press_release_ctext(text)["relations"]
    commitment = relations["commitment_action_binding"]
    locality = relations["opening_locality_binding"]
    assert commitment["status"] == "witnessed"
    assert commitment["matched_relation_depth"] == 3
    assert commitment["witnesses"][0]["action_lemma"] == "publish"
    assert (
        commitment["witnesses"][0]["action_class"]
        == "public_reporting_commitment"
    )
    assert commitment["does_not_establish"].startswith("that the action occurred")
    assert locality["status"] == "witnessed"
    assert locality["matched_relation_depth"] == 3
    assert any(row["place"] == "Boston" for row in locality["witnesses"])


def test_nonconcrete_future_copula_does_not_create_commitment_witness():
    relation = analyze_press_release_ctext("ACME will be ready.")["relations"][
        "commitment_action_binding"
    ]
    assert relation["status"] == "relation_not_instantiated"
    assert relation["matched_relation_depth"] is None


def test_negated_reporting_action_is_not_a_commitment_witness():
    relation = analyze_press_release_ctext(
        "ACME will not update the public security review."
    )["relations"]["commitment_action_binding"]
    assert relation["status"] == "relation_not_instantiated"


def test_attribution_scopes_claim_language_to_exact_quote():
    text = (
        'ACME released a routine update. “This is a groundbreaking result,” '
        "said Jane Doe, chief scientist at ACME."
    )
    relation = analyze_press_release_ctext(text)["relations"][
        "attribution_scoped_claim_language"
    ]
    assert relation["witness_count"] == 1
    assert relation["witnesses"][0]["cue"].casefold() == "groundbreaking"
    assert relation["witnesses"][0]["scope"] in {
        "dependency_bound_speaker",
        "quoted_bound_speaker",
    }
    assert relation["does_not_establish"].startswith("sensationalism")


def test_no_witness_is_not_an_absence_certificate():
    relations = analyze_press_release_ctext("ACME issued a short notice.")["relations"]
    uninstantiated = [
        relation
        for relation in relations.values()
        if relation["status"] in {
            "relation_not_instantiated",
            "representation_not_instantiated",
        }
    ]
    assert uninstantiated
    assert all(row["absence_certificate"] is False for row in uninstantiated)


def test_scannability_refuses_to_reconstruct_discarded_layout():
    flat = analyze_press_release_ctext("Headline. Body sentence. Footer sentence.")[
        "relations"
    ]["section_scannability_structure"]
    structured = analyze_press_release_ctext(
        "HEADLINE\n\nBody sentence.\n- First fact\n"
    )["relations"]["section_scannability_structure"]
    assert flat["status"] == "representation_not_instantiated"
    assert structured["status"] == "measured"
    assert structured["summary"]["heading_like_lines"] == 1
    assert structured["summary"]["list_item_lines"] == 1
    assert structured["summary"]["source_layout_recoverable"] is False
