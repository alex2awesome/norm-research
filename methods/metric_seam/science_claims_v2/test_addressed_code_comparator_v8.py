from __future__ import annotations

import ast
import copy
import inspect
import json
import textwrap
from pathlib import Path

import pytest

from methods.metric_seam.science_claims_v2 import addressed_pipeline as addressed
from methods.metric_seam.science_claims_v2 import addressed_code_comparator_v8 as code


def source_map(abstract: str, body: str, *, paper_id: str = "p") -> dict:
    return addressed.build_source_map(
        {"paper_id": paper_id, "abstract": abstract, "body": body}
    )


def test_metamorphic_suite_covers_required_relation_failures() -> None:
    checks = code.metamorphic_self_check()
    assert len(checks) == 10
    assert all(checks.values())
    for required in (
        "number_suffix_mutation_invalidates",
        "quantity_entity_mutation_invalidates",
        "comparison_direction_is_contradiction",
        "comparison_role_swap_is_contradiction",
        "exact_claim_address_preserved",
        "exact_evidence_address_preserved",
        "repeated_abstract_is_excluded",
        "missing_body_abstains",
    ):
        assert checks[required]


def test_number_suffix_is_parsed_as_a_complete_token() -> None:
    claim = "We report benchmark results on 100k examples."
    supported = code.verify_addressed_document(
        "p", source_map(claim, "Table 2 reports benchmark results on 100k examples.")
    )
    mutated = code.verify_addressed_document(
        "p", source_map(claim, "Table 2 reports benchmark results on 10 examples.")
    )
    assert supported["status"] == "supported"
    assert supported["certificates"][0]["checks"]["quantity_matches"] == 1
    assert mutated["status"] != "supported"
    assert not mutated["certificates"]


def test_same_value_different_entity_does_not_certify() -> None:
    claim = "We show robust performance across 28 LoRA adapters."
    correct = code.verify_addressed_document(
        "p",
        source_map(claim, "Table 2 shows robust performance across 28 LoRA adapters."),
    )
    wrong = code.verify_addressed_document(
        "p",
        source_map(claim, "Table 2 shows robust performance across 28 image tasks."),
    )
    assert correct["status"] == "supported"
    assert wrong["status"] != "supported"
    assert wrong["matches"][0]["reason"] == "quantity_entity_binding_failed"


def test_passive_direction_change_is_detected_before_active_participle() -> None:
    claim = "We show that our method outperforms BERT."
    result = code.verify_addressed_document(
        "p",
        source_map(claim, "Table 2 shows that our method is outperformed by BERT."),
    )
    certificate = result["certificates"][0]
    assert result["status"] == "contradicted"
    assert certificate["checks"]["relation_state"] == "direction_mismatch"
    assert certificate["evidence"]["comparison"]["polarity"] == -1


def test_swapped_roles_are_detected_separately_from_direction() -> None:
    claim = "We show that our method outperforms BERT."
    result = code.verify_addressed_document(
        "p",
        source_map(claim, "Table 2 shows that BERT outperforms our method."),
    )
    assert result["status"] == "contradicted"
    assert result["certificates"][0]["checks"]["relation_state"] == "reversed_roles"


def test_semantically_equivalent_passive_with_reversed_roles_supports() -> None:
    claim = "We show that our method outperforms BERT."
    result = code.verify_addressed_document(
        "p",
        source_map(claim, "Table 2 shows that BERT is outperformed by our method."),
    )
    assert result["status"] == "supported"
    assert result["certificates"][0]["checks"]["relation_state"] == "aligned_reversed"


def test_baseline_mismatch_is_not_a_relation_certificate() -> None:
    claim = "We show that our method outperforms BERT."
    result = code.verify_addressed_document(
        "p",
        source_map(claim, "Table 2 shows that our method outperforms RoBERTa."),
    )
    assert result["status"] == "insufficient"
    assert not result["certificates"]
    assert result["matches"][0]["checks"]["relation_state"] == "baseline_mismatch"


def test_result_preserves_exact_selected_addresses_without_source_prose_copy() -> None:
    abstract = (
        "This sentence gives background only. "
        "We report a 12 percent improvement on the benchmark."
    )
    body = (
        "This is setup. "
        "Table 2 reports a 12 percent improvement on the benchmark."
    )
    mapping = source_map(abstract, body)
    result = code.verify_addressed_document("p", mapping)
    certificate = result["certificates"][0]
    assert certificate["claim"]["source_address"] == code._source_address(
        mapping["abstract"][1]
    )
    assert certificate["evidence"]["source_address"] == code._source_address(
        mapping["body"][1]
    )
    assert result["selected_claims"] == [certificate["claim"]]
    assert certificate["reconstruction_key"] == {
        "claim_sentence_id": "A0002",
        "evidence_sentence_id": "B0002",
        "relation": "numeric",
    }
    assert "text" not in certificate["claim"]["source_address"]
    assert "text" not in certificate["evidence"]["source_address"]


def test_source_map_text_hash_and_offsets_are_enforced() -> None:
    mapping = source_map(
        "We report a 12 percent improvement.",
        "Table 2 reports a 12 percent improvement.",
    )
    bad_text = copy.deepcopy(mapping)
    bad_text["body"][0]["text"] += " Mutation"
    with pytest.raises(ValueError, match="offset length mismatch|text hash mismatch"):
        code.verify_addressed_document("p", bad_text)

    bad_offset = copy.deepcopy(mapping)
    bad_offset["body"][0]["start"] += 1
    with pytest.raises(ValueError, match="offset length mismatch"):
        code.verify_addressed_document("p", bad_offset)

    bad_id = copy.deepcopy(mapping)
    bad_id["body"][0]["sentence_id"] = "B9999"
    with pytest.raises(ValueError, match="noncanonical address sequence"):
        code.verify_addressed_document("p", bad_id)


def test_exact_repeated_abstract_address_is_not_independent_evidence() -> None:
    claim = "We report a 12 percent improvement on the benchmark."
    mapping = source_map(claim, claim)
    result = code.verify_addressed_document("p", mapping)
    assert result["status"] == "abstain"
    assert result["reason"] == "abstract_only_no_independent_addressed_evidence"
    assert result["coverage"]["repeated_abstract_addresses_excluded"] == 1
    assert result["excluded_repeated_abstract_addresses"] == [
        code._source_address(mapping["body"][0])
    ]


def test_missing_body_is_a_deterministic_structural_abstention() -> None:
    result = code.verify_addressed_document(
        "p", source_map("We report a 12 percent improvement.", "")
    )
    assert result["status"] == "abstain"
    assert result["reason"] == "missing_fullpaper_body_addresses"
    assert result["coverage"]["body_addresses"] == 0
    assert result["certificate_count"] == 0


def test_exact_matching_does_not_reuse_one_body_address() -> None:
    abstract = (
        "We report results on 10 cats. "
        "We report results on 20 dogs."
    )
    body = "Table 2 reports results on 10 cats and 20 dogs."
    result = code.verify_addressed_document("p", source_map(abstract, body))
    assert result["claim_count"] == 2
    assert result["graph"]["evidence_nodes"] == 1
    assert result["graph"]["matched_edges"] == 1
    assert result["coverage"]["matched_claim_addresses"] == 1
    assert len(result["selected_claims"]) == 2
    assert len({
        claim["source_address"]["sentence_id"] for claim in result["selected_claims"]
    }) == 2


def test_request_projection_binds_exact_source_map_without_paper_input() -> None:
    mapping = source_map(
        "We report a 12 percent improvement.",
        "Table 2 reports a 12 percent improvement.",
    )
    request = {
        "source_index": 7,
        "paper_id": "p",
        "request_id": "request-7",
        "request_sha256": "request-sha",
        "source_map": mapping,
        "source_map_sha256": addressed.hash_value(mapping),
        "paper_input": {"forbidden_if_read": object()},
        "prompt_result": {"forbidden_if_read": object()},
    }
    snapshot = {
        "manifest_sha256": "manifest-sha",
        "requests_sha256": "requests-sha",
        "implementation_bindings": {
            "exact_address_comparator": {"sha256": "comparator-sha"}
        },
    }
    row = code._request_result(request, bundle_snapshot=snapshot)
    assert row["source_index"] == 7
    assert row["source_map_sha256"] == addressed.hash_value(mapping)
    assert row["result"]["status"] == "supported"
    material = {key: value for key, value in row.items() if key != "row_sha256"}
    assert row["row_sha256"] == addressed.hash_value(material)


def test_request_boundary_ast_indexes_only_declared_fields() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(code._request_result)))
    indexed: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "request"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            indexed.add(node.slice.value)
    assert indexed == {
        "source_index",
        "paper_id",
        "request_id",
        "request_sha256",
        "source_map",
        "source_map_sha256",
    }
    assert indexed.isdisjoint({"paper_input", "response", "y", "label", "acceptance"})


def test_provenance_does_not_claim_automatic_discovery() -> None:
    provenance = code.METHOD_PROVENANCE
    assert provenance["automatically_discovered"] is False
    assert provenance["current_pipeline_selection"] == "selected_retrospective_seed"
    assert "manually_constructed_mock" in provenance[
        "historical_deep_decomposition_origin"
    ]


def test_manifest_records_conditional_representation_and_label_boundary(
    tmp_path: Path,
) -> None:
    mapping = source_map(
        "We report a 12 percent improvement.",
        "Table 2 reports a 12 percent improvement.",
    )
    request = {
        "source_index": 0,
        "paper_id": "p",
        "request_id": "request-0",
        "request_sha256": "request-sha",
        "source_map": mapping,
        "source_map_sha256": addressed.hash_value(mapping),
    }
    snapshot = {
        "manifest_sha256": "manifest-sha",
        "requests_sha256": "requests-sha",
        "structural_abstentions_sha256": "abstentions-sha",
        "request_index_sha256": "request-index-sha",
        "abstention_index_sha256": "abstention-index-sha",
        "implementation_bindings": {
            "exact_address_comparator": {"sha256": "comparator-sha"}
        },
    }
    row = code._request_result(request, bundle_snapshot=snapshot)
    results_path = tmp_path / code.RESULTS_NAME
    results_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    manifest = code.build_manifest(
        bundle=tmp_path / "source",
        source_manifest={
            "schema_version": "v8-fixture",
            "input": {"source_file_sha256": "a" * 64},
            "historical_code_comparator": {
                "input_source_path": "fixture.jsonl",
                "input_source_sha256": "a" * 64,
            },
        },
        bundle_snapshot=snapshot,
        rows=[row],
        results_path=results_path,
        metamorphic_checks={"fixture": True},
    )
    representation = manifest["representation_contract"]
    assert representation["same_visible_A_B_ids_and_exact_span_texts_as_prompt"] is True
    assert representation["exact_offsets_preserved_from_bound_request"] is True
    assert representation["offsets_rendered_to_model"] is False
    assert representation["serialized_prompt_bytes_identical"] is False
    boundary = manifest["input_boundary"]
    assert boundary["comparator_indexes_paper_input"] is False
    assert boundary["comparator_indexes_prompt_output"] is False
    assert boundary["comparator_indexes_acceptance_or_judgment_fields"] is False
    assert manifest["interpretation_guard"][
        "agreement_alone_licenses_isomorphism"
    ] is False
    program = manifest["executable_program"]
    assert program["retrieval_scope"] == "one_presented_paper_body_only"
    assert program["matching"] == "exact_max_weight_bipartite"
    assert "local_quantity_entity_binding" in program["stages"]
    assert "directed_comparison_entity_role_and_polarity_parsing" in program["stages"]
    assert manifest["source_bundle"]["historical_comparator_input_identity"][
        "machine_equal"
    ] is True


def test_historical_comparator_input_identity_fails_closed() -> None:
    valid = {
        "input": {"source_file_sha256": "a" * 64},
        "historical_code_comparator": {
            "input_source_path": "source.jsonl",
            "input_source_sha256": "a" * 64,
        },
    }
    code._assert_historical_comparator_input_identity(valid)

    missing = copy.deepcopy(valid)
    del missing["historical_code_comparator"]["input_source_sha256"]
    with pytest.raises(ValueError, match="incomplete"):
        code._assert_historical_comparator_input_identity(missing)

    mismatched = copy.deepcopy(valid)
    mismatched["historical_code_comparator"]["input_source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="differs"):
        code._assert_historical_comparator_input_identity(mismatched)
