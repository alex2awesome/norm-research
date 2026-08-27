from __future__ import annotations

from methods.metric_seam.patent_claim_graph_additive_v1 import (
    RELATIONS,
    analyze_patent_claim_graph,
    score,
)


def _claims(body: str) -> str:
    return f"ABSTRACT:\nA bounded test document.\n\nCLAIMS:\n{body}"


def test_relation_catalog_is_pure_code_deep_and_nonaggregating() -> None:
    assert {row["relation_id"] for row in RELATIONS} == {
        "claim_status_and_local_listing_witnesses",
        "two_part_or_jepson_structure",
        "markush_closed_group_structure",
        "bounded_antecedent_term_reference_graph",
        "numeric_constraint_definition_graph",
        "formula_variable_definition_alignment",
    }
    assert {row["channel"] for row in RELATIONS} == {"code"}
    assert {row["depth"] for row in RELATIONS} == {2, 3}
    assert score("anything") is None


def test_claim_status_is_a_finite_marker_not_global_compliance() -> None:
    result = analyze_patent_claim_graph(
        _claims(
            "1 . (currently amended) A system comprising a sensor.\n\n"
            "2 . (canceled)\n"
        )
    )
    nodes = result["graphs"]["status_listing"]["nodes"]
    assert [(row["claim"], row["status"]) for row in nodes] == [
        (1, "currently amended"),
        (2, "cancelled"),
    ]
    assert result["relation_values"][
        "claim_status_and_local_listing_witnesses"
    ]["value"] is None
    assert result["scope"]["legal_compliance_or_definiteness_established"] is False


def test_duplicate_ordinal_is_local_counter_witness_and_does_not_break_graph() -> None:
    result = analyze_patent_claim_graph(
        _claims(
            "1 . A system comprising a sensor.\n\n"
            "1 . The system comprising a controller.\n"
        )
    )
    assert result["graphs"]["status_listing"]["counter_witnesses"] == [
        {
            "claim": 1,
            "reason": "duplicate_presented_ordinal",
            "positions": [0, 1],
        }
    ]
    ids = [
        row["mention_id"]
        for row in result["graphs"]["term_reference"]["nodes"]
    ]
    assert len(ids) == len(set(ids))


def test_unknown_leading_parenthetical_is_not_invented_as_claim_status() -> None:
    result = analyze_patent_claim_graph(
        _claims("1 . (with a polymer layer) A device comprising a sensor.\n")
    )
    assert result["graphs"]["status_listing"]["nodes"] == []


def test_jepson_and_epc_boundaries_are_limited_to_independent_claims() -> None:
    result = analyze_patent_claim_graph(
        _claims(
            "1 . In a pump having a housing, the improvement comprising a ceramic rotor.\n\n"
            "2 . A valve having an inlet, characterized in that the inlet is tapered.\n\n"
            "3 . The pump of claim 1, characterized in that the rotor is coated.\n"
        )
    )
    nodes = result["graphs"]["two_part_jepson"]["nodes"]
    assert [(row["claim"], row["boundary_kind"]) for row in nodes] == [
        (1, "jepson_improvement"),
        (2, "epc_characterising_boundary"),
    ]
    assert all(row["preamble_chars"] > 0 for row in nodes)
    assert all(row["characterising_chars"] > 0 for row in nodes)


def test_markush_requires_closed_group_opener_and_enumerated_tail() -> None:
    positive = analyze_patent_claim_graph(
        _claims(
            "1 . A composition comprising a metal selected from the group consisting of "
            "copper, zinc, and mixtures thereof.\n"
        )
    )
    node = positive["graphs"]["markush"]["nodes"][0]
    assert node["presented_alternative_count_lower_bound"] >= 2
    assert node["explicit_mixture_or_combination_qualifier"].endswith(
        "mixtures thereof"
    )

    generic = analyze_patent_claim_graph(
        _claims("1 . A composition with a metal selected from copper and zinc.\n")
    )
    truncated = analyze_patent_claim_graph(
        _claims("1 . A composition selected from the group consisting of copper.\n")
    )
    assert generic["graphs"]["markush"]["nodes"] == []
    assert truncated["graphs"]["markush"]["nodes"] == []


def test_term_graph_resolves_same_claim_and_explicit_ancestor_only() -> None:
    result = analyze_patent_claim_graph(
        _claims(
            "1 . A system comprising a pressure sensor and a controller, the controller "
            "being coupled to the pressure sensor.\n\n"
            "2 . The system of claim 1, wherein the pressure sensor emits a signal.\n\n"
            "3 . A method comprising receiving the signal.\n"
        )
    )
    edges = result["graphs"]["term_reference"]["edges"]
    by_surface = {row["reference_surface"].casefold(): row for row in edges}
    assert by_surface["the controller"]["status"] == "resolved_exact"
    assert by_surface["the pressure sensor"]["status"] == "resolved_exact"
    assert by_surface["the system"]["status"] == "resolved_exact"
    # Claim 3 does not reference claim 1 or 2, so their introductions cannot
    # satisfy its definite reference.
    assert by_surface["the signal"]["claim"] == 3
    assert by_surface["the signal"]["status"] == "unresolved"


def test_term_graph_surfaces_ambiguity_instead_of_arbitrary_resolution() -> None:
    result = analyze_patent_claim_graph(
        _claims(
            "1 . A system comprising a first sensor and a second sensor, wherein the sensor "
            "emits a signal.\n"
        )
    )
    edge = next(
        row
        for row in result["graphs"]["term_reference"]["edges"]
        if row["reference_surface"].casefold() == "the sensor"
    )
    assert edge["status"] == "ambiguous"
    assert len(edge["candidate_introduction_ids"]) == 2


def test_term_graph_does_not_promote_unique_head_only_overlap() -> None:
    result = analyze_patent_claim_graph(
        _claims(
            "1 . A delivery device comprising an outer shaft, wherein the delivery shaft "
            "is movable.\n"
        )
    )
    edge = next(
        row
        for row in result["graphs"]["term_reference"]["edges"]
        if row["reference_surface"].casefold() == "the delivery shaft"
    )
    assert edge["status"] == "head_only_near_match"
    certificate = next(
        row
        for row in result["certificates"]
        if row.get("reference_surface", "").casefold() == "the delivery shaft"
    )
    assert certificate["kind"] == "bounded_counter_witness"


def test_numeric_constraint_graph_links_only_explicit_measurement_definition() -> None:
    result = analyze_patent_claim_graph(
        _claims(
            "1 . A composition wherein a particle size is measured using laser diffraction, "
            "the particle size being at least 10 nm.\n"
        )
    )
    graph = result["graphs"]["numeric_constraint_definition"]
    assert len(graph["constraint_nodes"]) == 1
    assert graph["constraint_nodes"][0]["surface"] == "at least 10 nm"
    assert len(graph["definition_nodes"]) == 1
    assert len(graph["links"]) == 1

    unlinked = analyze_patent_claim_graph(
        _claims("1 . A composition having a particle size of at least 10 nm.\n")
    )
    unlinked_graph = unlinked["graphs"]["numeric_constraint_definition"]
    assert len(unlinked_graph["constraint_nodes"]) == 1
    assert unlinked_graph["links"] == []
    assert unlinked["relation_values"]["numeric_constraint_definition_graph"][
        "value"
    ] is None


def test_claim_and_figure_ordinals_are_not_numeric_constraint_nodes() -> None:
    result = analyze_patent_claim_graph(
        _claims(
            "1 . A system shown in FIG. 2.\n\n"
            "2 . The system of claim 1 comprising a sensor.\n"
        )
    )
    assert result["graphs"]["numeric_constraint_definition"][
        "constraint_nodes"
    ] == []


def test_formula_variable_graph_executes_over_dependency_ancestors() -> None:
    result = analyze_patent_claim_graph(
        _claims(
            "1 . A compound wherein m is an integer and x is a coefficient.\n\n"
            "2 . The compound of claim 1, wherein m=0 and x=2.\n\n"
            "3 . A compound wherein m=4.\n"
        )
    )
    graph = result["graphs"]["formula_variable_definition"]
    assert [(row["claim"], row["symbol"], row["value"]) for row in graph["assignment_nodes"]] == [
        (2, "m", "0"),
        (2, "x", "2"),
        (3, "m", "4"),
    ]
    assert len(graph["links"]) == 2
    linked_assignments = {row["assignment_id"] for row in graph["links"]}
    assert linked_assignments == {"c2:a1", "c2:a2"}
    assert graph["conflicts"] == []


def test_formula_graph_emits_dependency_path_equality_contradiction() -> None:
    result = analyze_patent_claim_graph(
        _claims(
            "1 . A compound wherein m is an integer and m=0.\n\n"
            "2 . The compound of claim 1, wherein m=1.\n\n"
            "3 . A compound wherein m=1.\n"
        )
    )
    conflicts = result["graphs"]["formula_variable_definition"]["conflicts"]
    assert conflicts == [
        {
            "claim": 2,
            "symbol": "m",
            "incompatible_values": ["0", "1"],
            "assignment_ids": ["c1:a1", "c2:a2"],
        }
    ]


def test_missing_claims_abstains_and_exact_input_length_is_recorded() -> None:
    ctext = "ABSTRACT:\nOnly an abstract."
    result = analyze_patent_claim_graph(ctext)
    assert result["presented_character_count"] == len(ctext)
    assert result["claim_count"] == 0
    assert {row["reason"] for row in result["abstentions"]} >= {
        "named_claims_section_absent",
        "no_finite_relation_witness",
    }
    assert result["scope"]["outcome_or_reference_values_used"] is False
    assert result["scope"]["reconstruction_or_isomorphism_measured"] is False
