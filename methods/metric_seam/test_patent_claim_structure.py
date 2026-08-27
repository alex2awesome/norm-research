from __future__ import annotations

from methods.metric_seam.patent_claim_structure import (
    RELATIONS,
    analyze_patent_ctext,
    parse_claims,
    score,
    split_named_sections,
)


SAMPLE = """ABSTRACT:
A compact valve controller for a fluid circuit.

CLAIMS:
1. A system comprising a controller configured to open a valve at 100 kHz.
2. The system of claim 1, wherein the controller has a pressure sensor.
3. The system according to claims 1-2, wherein the pressure sensor operates from 10 to 20 kHz.
"""


def test_parses_named_sections_claims_and_cross_claim_graph() -> None:
    sections, claims = parse_claims(SAMPLE)
    assert set(sections) == {"ABSTRACT", "CLAIMS"}
    assert [claim.number for claim in claims] == [1, 2, 3]
    assert claims[0].explicit_dependencies == ()
    assert claims[1].explicit_dependencies == (1,)
    assert claims[2].explicit_dependencies == (1, 2)

    result = analyze_patent_ctext(SAMPLE)
    dependency = result["relation_values"]["claim_dependency_well_formedness"]
    assert dependency["value"] == 1.0
    assert dependency["support"]["valid_edges"] == 3
    assert result["graph"]["cycle_detected"] is False
    assert result["relation_values"]["claim_set_layering"]["value"] == 1.0


def test_invalid_forward_and_self_references_are_counter_witnesses() -> None:
    text = """CLAIMS:
1. A method according to claim 2, comprising sorting values.
2. The method of claim 2, comprising returning the values.
"""
    result = analyze_patent_ctext(text)
    invalid = result["graph"]["invalid_edges"]
    assert invalid == [
        {
            "child": 1,
            "parent": 2,
            "reasons": [
                "reference_is_not_to_an_earlier_claim",
                "referenced_claim_number_is_not_lower",
            ],
        },
        {
            "child": 2,
            "parent": 2,
            "reasons": [
                "reference_is_not_to_an_earlier_claim",
                "referenced_claim_number_is_not_lower",
            ],
        },
    ]
    assert result["relation_values"]["claim_dependency_well_formedness"]["value"] == 0.0
    assert result["relation_values"]["claim_set_layering"]["value"] == 0.0
    assert result["graph"]["cycle_detected"] is True


def test_contiguity_preserves_presented_order_instead_of_sorting_claims() -> None:
    text = """CLAIMS:
2. A system comprising a sensor.
1. A method comprising sampling the sensor.
"""
    result = analyze_patent_ctext(text)
    assert [row["number"] for row in result["claims"]] == [2, 1]
    contiguity = result["relation_values"]["claim_number_contiguity"]
    assert contiguity["value"] == 0.0
    assert contiguity["support"]["presented_in_increasing_order"] is False


def test_dependency_parser_supports_through_in_opening_incorporation_clause() -> None:
    text = """CLAIMS:
1. A system comprising a sensor.
2. A device comprising a processor.
3. A method comprising sampling.
4. The system according to claims 1 through 3, wherein the controller stores data.
"""
    result = analyze_patent_ctext(text)
    assert result["claims"][3]["explicit_dependencies"] == [1, 2, 3]
    assert result["relation_values"]["claim_dependency_well_formedness"]["value"] == 1.0


def test_substantive_body_claim_phrases_do_not_manufacture_dependency_edges() -> None:
    text = """CLAIMS:
1. A system comprising a database.
2. A method comprising storing a database of claims 1 and 2; and receiving data from claim 1.
"""
    result = analyze_patent_ctext(text)
    assert result["claims"][1]["explicit_dependencies"] == []
    assert result["relation_values"]["claim_dependency_well_formedness"]["value"] is None


def test_dependency_requires_lower_number_even_when_parent_is_presented_first() -> None:
    text = """CLAIMS:
5. A system comprising a database.
3. The system of claim 5, wherein the database stores data.
"""
    result = analyze_patent_ctext(text)
    assert result["relation_values"]["claim_dependency_well_formedness"]["value"] == 0.0
    assert result["graph"]["invalid_edges"] == [
        {
            "child": 3,
            "parent": 5,
            "reasons": ["referenced_claim_number_is_not_lower"],
        }
    ]


def test_repeated_claim_word_in_reference_list_is_parsed() -> None:
    text = """CLAIMS:
1. A system comprising a sensor.
2. A device comprising a processor.
3. The system according to claim 1 or claim 2, wherein the sensor stores data.
"""
    result = analyze_patent_ctext(text)
    assert result["claims"][2]["explicit_dependencies"] == [1, 2]


def test_comma_inside_opening_clause_does_not_hide_dependency() -> None:
    text = """CLAIMS:
1. A system comprising a sensor.
2. The system, according to claim 1, wherein the sensor stores data.
"""
    result = analyze_patent_ctext(text)
    assert result["claims"][1]["explicit_dependencies"] == [1]
    assert result["claims"][1]["independent"] is False


def test_common_dependency_idioms_are_parsed_without_promoting_false_roots() -> None:
    text = """CLAIMS:
1. A system comprising a sensor.
2. A method in accordance with claim 1, wherein the sensor stores data.
3. The system of the claim 1, wherein the sensor transmits data.
4. The system accordingly to claim 1, wherein the sensor filters data.
"""
    result = analyze_patent_ctext(text)
    assert [row["explicit_dependencies"] for row in result["claims"][1:]] == [
        [1],
        [1],
        [1],
    ]
    assert all(row["independent"] is False for row in result["claims"][1:])


def test_dependency_marker_without_ordinal_abstains_and_is_not_a_root() -> None:
    text = """CLAIMS:
1. A system comprising a sensor.
2. The system as claimed in claim
"""
    result = analyze_patent_ctext(text)
    second = result["claims"][1]
    assert second["dependency_parse_issues"] == [
        {
            "surface": "as claimed in claim",
            "reason": "missing_dependency_ordinal",
        }
    ]
    assert second["independent"] is False
    assert second["statutory_category_surface"] is None
    assert result["relation_values"]["claim_dependency_well_formedness"]["value"] is None
    assert result["relation_values"]["claim_set_layering"]["value"] is None
    assert "dependency_marker_missing_ordinal" in {
        row["reason"] for row in result["abstentions"]
    }


def test_truncated_dependency_marker_at_boundary_is_not_promoted_to_root() -> None:
    text = """CLAIMS:
1. A system comprising a sensor.
9. The switching device according to clai
"""
    result = analyze_patent_ctext(text)
    ninth = result["claims"][1]
    assert ninth["dependency_parse_issues"] == [
        {
            "surface": "according to clai",
            "reason": "truncated_dependency_marker",
        }
    ]
    assert ninth["independent"] is False
    assert ninth["statutory_category_surface"] is None
    assert "dependency_marker_truncated_at_presented_boundary" in {
        row["reason"] for row in result["abstentions"]
    }


def test_dependency_parser_rejects_descending_and_abstains_on_wide_ranges() -> None:
    descending = analyze_patent_ctext(
        """CLAIMS:
1. A system comprising a sensor.
4. The system according to claims 3 through 1, comprising a controller.
"""
    )
    dependency = descending["relation_values"]["claim_dependency_well_formedness"]
    assert dependency["value"] == 0.0
    assert dependency["support"]["dependency_parse_issues"] == [
        {
            "claim": 4,
            "surface": "3 through 1",
            "reason": "descending_dependency_range",
        }
    ]

    wide = analyze_patent_ctext(
        """CLAIMS:
1. A system comprising a sensor.
200. The system according to claims 1 through 150, comprising a controller.
"""
    )
    assert wide["relation_values"]["claim_dependency_well_formedness"]["value"] is None
    assert "dependency_range_exceeds_bounded_expansion" in {
        row["reason"] for row in wide["abstentions"]
    }


def test_insurance_claim_counts_are_not_parsed_as_patent_dependencies() -> None:
    text = """CLAIMS:
1. A system for processing insurance claims 1 and 2 under a policy.
"""
    result = analyze_patent_ctext(text)
    assert result["claims"][0]["explicit_dependencies"] == []
    assert result["relation_values"]["claim_dependency_well_formedness"]["value"] is None


def test_missing_claim_section_abstains_instead_of_scoring_absence() -> None:
    result = analyze_patent_ctext("ABSTRACT:\nA useful widget.")
    assert result["claims"] == []
    assert result["relation_values"]["claim_number_contiguity"]["value"] is None
    assert {row["reason"] for row in result["abstentions"]} >= {
        "named_claims_section_absent",
        "no_explicit_claim_reference",
    }


def test_complete_numeric_suffix_and_range_are_preserved() -> None:
    text = """CLAIMS:
1. A method operating at 100 kHz and between 10 and 20 kHz.
"""
    result = analyze_patent_ctext(text)
    claim = result["claims"][0]
    assert "100 kHz" in claim["numeric_tokens"]
    assert claim["numeric_ranges"] == [
        {
            "surface": "between 10 and 20 kHz",
            "left": "10",
            "right": "20",
            "connector": "and",
            "unit": "khz",
        }
    ]
    numerical = result["relation_values"]["numerical_limitation_incidence"]
    assert numerical["value"] == 1.0
    assert numerical["support"]["claims_with_number"] == [1]


def test_unitless_between_range_is_not_silently_dropped() -> None:
    text = """CLAIMS:
1. A method using a value between 10 and 20.
"""
    result = analyze_patent_ctext(text)
    claim = result["claims"][0]
    assert claim["numeric_ranges"] == [
        {
            "surface": "between 10 and 20",
            "left": "10",
            "right": "20",
            "connector": "and",
            "unit": None,
        }
    ]
    assert result["relation_values"]["numerical_limitation_incidence"]["value"] == 1.0


def test_numeric_parser_handles_punctuation_signs_commas_and_units() -> None:
    text = """CLAIMS:
1. A method operating at least -5 degrees and at 20 kHz.
2. The method of claim 1, operating between 1,000 and 2,000 kg.
3. The method of claim 1, operating from 10 to 20 meters.
"""
    result = analyze_patent_ctext(text)
    assert "-5 degrees" in result["claims"][0]["numeric_tokens"]
    assert "20 kHz" in result["claims"][0]["numeric_tokens"]
    assert result["claims"][1]["numeric_ranges"] == [
        {
            "surface": "between 1,000 and 2,000 kg",
            "left": "1,000",
            "right": "2,000",
            "connector": "and",
            "unit": "kg",
        }
    ]
    assert result["claims"][2]["numeric_ranges"] == [
        {
            "surface": "from 10 to 20 meters",
            "left": "10",
            "right": "20",
            "connector": "to",
            "unit": "meters",
        }
    ]


def test_figure_decimal_and_figure_range_are_not_numeric_limitations() -> None:
    text = """CLAIMS:
1. A system depicted in FIG. 2.1 and FIGS. 1-3.
"""
    result = analyze_patent_ctext(text)
    assert result["claims"][0]["numeric_tokens"] == []
    assert result["claims"][0]["numeric_ranges"] == []


def test_bare_claim_references_and_drawing_numerals_are_not_measurements() -> None:
    text = """CLAIMS:
1. A system comprising a sensor 3142.
2. The system of claim 1, wherein FIG. 2 depicts the sensor.
"""
    result = analyze_patent_ctext(text)
    assert result["claims"][0]["numeric_tokens"] == []
    assert result["claims"][1]["numeric_tokens"] == []


def test_claim_reference_ranges_are_not_numeric_limitation_ranges() -> None:
    text = """CLAIMS:
1. A system comprising a sensor.
2. The system of any one of claims 1 - 6, operating between 10 and 20 kHz.
"""
    result = analyze_patent_ctext(text)
    ranges = result["claims"][1]["numeric_ranges"]
    assert [row["surface"] for row in ranges] == ["between 10 and 20 kHz"]


def test_antecedent_tracker_inherits_introduction_from_explicit_parent() -> None:
    text = """CLAIMS:
1. A system comprising a controller and a sensor.
2. The system of claim 1, wherein the sensor communicates with the controller.
"""
    result = analyze_patent_ctext(text)
    second = result["claims"][1]
    assert not ({"sensor", "controller"} & set(second["possible_missing_antecedent_heads"]))


def test_open_dependency_abstains_from_explicit_graph_layering() -> None:
    text = """CLAIMS:
1. A system comprising a controller.
2. The system according to any preceding claim, wherein the controller stores data.
"""
    result = analyze_patent_ctext(text)
    layering = result["relation_values"]["claim_set_layering"]
    assert layering["value"] is None
    assert layering["support"]["open_dependency_claims"] == [2]
    assert {
        row["reason"]
        for row in result["abstentions"]
        if row["relation"] == "claim_set_layering"
    } == {"open_dependency_not_explicitly_enumerable"}


def test_category_and_function_markers_are_presence_witnesses_only() -> None:
    result = analyze_patent_ctext(SAMPLE)
    first = result["claims"][0]
    assert first["statutory_category_surface"] == "machine_or_apparatus"
    assert first["statutory_category_witness"] == {
        "surface": "system",
        "span": [2, 8],
    }
    assert first["functional_markers"] == ["configured to"]
    assert result["scope"]["legal_validity_or_patentability_established"] is False
    assert all(relation["channel"] == "code" for relation in RELATIONS)
    assert score(SAMPLE) is None


def test_category_is_taken_from_claim_preamble_not_later_method_language() -> None:
    text = """CLAIMS:
1. A system comprising a processor configured to perform a method for sorting records.
"""
    result = analyze_patent_ctext(text)
    assert result["claims"][0]["statutory_category_surface"] == "machine_or_apparatus"


def test_earliest_category_surface_wins_inside_the_preamble() -> None:
    text = """CLAIMS:
1. A system for performing a method of sorting records.
2. A method implemented by a system for sorting records.
3. A composition for use in a method of treatment.
"""
    result = analyze_patent_ctext(text)
    assert [row["statutory_category_surface"] for row in result["claims"]] == [
        "machine_or_apparatus",
        "process",
        "composition",
    ]


def test_compound_claimed_object_uses_rightmost_category_head() -> None:
    text = """CLAIMS:
1. A semiconductor device manufacturing method, comprising forming a layer.
2. A method implemented by a system for sorting records.
"""
    result = analyze_patent_ctext(text)
    assert [row["statutory_category_surface"] for row in result["claims"]] == [
        "process",
        "process",
    ]


def test_use_context_location_does_not_override_method_head() -> None:
    text = """CLAIMS:
1. A broadcast providing method implemented at an electronic device comprising sending data.
"""
    result = analyze_patent_ctext(text)
    claim = result["claims"][0]
    assert claim["statutory_category_surface"] == "process"
    assert claim["statutory_category_witness"] == {
        "surface": "method",
        "span": [22, 28],
    }
    certificate = next(
        row
        for row in result["certificates"]
        if row["relation"] == "statutory_category_surface_coverage"
    )
    assert certificate["surface"] == "method"
    assert certificate["span"] == [22, 28]


def test_codec_and_version_identifiers_are_not_numeric_limitations() -> None:
    text = """CLAIMS:
1. A decoder implementing H.264 codec version 2.0.
"""
    result = analyze_patent_ctext(text)
    assert result["claims"][0]["numeric_tokens"] == []


def test_duplicate_headings_preserve_presented_bytes() -> None:
    sections = split_named_sections("ABSTRACT:\nfirst\nABSTRACT:\nsecond\nCLAIMS:\n1. A device.")
    assert sections["ABSTRACT"] == "first\nsecond"


def test_canceled_range_is_preserved_as_presented_claim_nodes() -> None:
    text = """CLAIMS:
1 - 12 . (canceled)
13. A method comprising culturing a cell.
14. The method of claim 1, further comprising imaging the cell.
"""
    result = analyze_patent_ctext(text)
    assert [row["number"] for row in result["claims"]] == list(range(1, 15))
    assert all(row["canceled"] for row in result["claims"][:12])
    assert result["relation_values"]["claim_number_contiguity"]["value"] == 1.0
    assert result["graph"]["invalid_edges"] == [
        {
            "child": 14,
            "parent": 1,
            "reasons": ["referenced_claim_is_canceled_in_presented_text"],
        }
    ]
