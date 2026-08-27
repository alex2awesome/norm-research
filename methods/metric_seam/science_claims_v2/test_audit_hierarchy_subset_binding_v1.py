from __future__ import annotations

import json

from methods.metric_seam.science_claims_v2 import (
    audit_hierarchy_subset_binding_v1 as binding,
)


def test_current_hierarchy_subset_is_bound_pre_prompt_and_exploratory() -> None:
    result = binding.audit()

    assert result["status"] == "cpu_only_subset_binding_complete_pre_prompt"
    assert result["representation_contract"] == {
        "same_evidence_content": True,
        "same_source_address_inventory_for_v8_prompt_and_v9_code": True,
        "same_input_representation": False,
        "exact_hierarchy_ctext_rendered_to_prompt": False,
        "full_isomorphism_licensed": False,
        "transport_class": "same_evidence_source_addressed_not_exact_ctext",
        "reason": (
            "The prompt sees addressed JSONL with IDs, JSON escaping, and omitted "
            "inter-sentence whitespace; historical code sees continuous abstract/body."
        ),
    }
    assert result["prompt_plane"] == {
        "selected_items": 300,
        "distinct_prepared_unscored_request_records": 235,
        "structural_abstentions_without_remote_call": 65,
        "prompt_responses": 0,
        "prompt_articulability_measured": False,
        "prompt_code_reconstruction_measured": False,
        "planned_stateless_passes": 2,
        "planned_two_pass_prompt_jobs_if_executed": 470,
        "two_pass_jobs_materialized_as_separate_requests": False,
        "six_relation_mappings_share_one_result_vector": True,
    }
    assert result["split_summaries"]["compiler_train"]["prompt_transport"] == {
        "compiled_unscored_request": 124,
        "structural_abstention_no_remote_call": 26,
    }
    assert result["split_summaries"]["sealed_heldout"]["prompt_transport"] == {
        "compiled_unscored_request": 111,
        "structural_abstention_no_remote_call": 39,
    }
    assert all(
        row["agreement"]["status_and_output_counts_exact"]
        for row in result["crosswalk"]
    )
    assert result["combined_summary"]["v9_hierarchy_item_field_agreement"] == {
        "agree": 300,
        "total": 300,
        "fields": [
            "verifier_status",
            "claim_count",
            "certificate_count",
            "evidence_link_count",
            "decision_counts",
        ],
    }
    assert result["combined_summary"]["reason_label_agreement"] == {
        "exact": 235,
        "representation_specific": 65,
    }
    assert result["combined_summary"][
        "v9_hierarchy_aggregate_exact_for_both_splits"
    ] is True
    assert result["temporal_disposition"][
        "fresh_split_required_for_confirmatory_prompt_code_claim"
    ] is True
    assert result["execution_policy"]["models_or_apis_called"] is False
    assert result["execution_policy"]["accelerators_used"] is False
    assert len(result["crosswalk"]) == 300
    assert len({row["item_key"] for row in result["crosswalk"]}) == 300
    assert len({row["source_index"] for row in result["crosswalk"]}) == 300

    archived = json.loads(binding.DEFAULT_OUT.read_text(encoding="utf-8"))
    assert archived == result
