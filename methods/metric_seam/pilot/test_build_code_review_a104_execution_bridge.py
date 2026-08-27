from __future__ import annotations

from methods.metric_seam.pilot import build_code_review_a104_execution_bridge as bridge


def test_projection_is_invariant_to_accept_reject_fields() -> None:
    base = {
        "datapoint_id": "d",
        "repo": "owner/repo",
        "pr_number": 7,
        "ctext": "diff",
        "judgement": 0,
        "other_outcome": "x",
    }
    changed = {**base, "judgement": 1, "other_outcome": "y"}
    assert bridge.project_items([base]) == bridge.project_items([changed])


def test_current_a104_execution_bridge_is_sparse_and_non_isomorphic() -> None:
    result = bridge.build()

    assert result["status"] == "stored_execution_telemetry_join_complete"
    assert result["summary"]["active_items"] == 250
    assert result["summary"]["exact_repository_pr_overlap"] == 32
    assert result["summary"]["finite_execution_certificates"] == 1
    assert result["summary"]["overlap_rate"] == 32 / 250
    assert result["summary"]["finite_certificate_rate_conditional_overlap"] == 1 / 32
    assert result["summary"]["finite_certificate_rate_over_active_items"] == 1 / 250
    assert result["summary"]["by_split"]["train"]["rows"] == 18
    assert result["summary"]["by_split"]["heldout"][
        "finite_execution_certificates"
    ] == 1
    assert result["representation_boundary"]["same_input_representation"] is False
    assert result["axis_status"]["reconstruction_agreement"] == (
        "not_estimated_by_this_bridge"
    )
    assert result["execution_provenance"]["repositories_or_tests_executed_in_this_bridge"] is False
    assert result["execution_provenance"]["models_or_apis_called"] is False
    assert result["execution_provenance"]["accelerators_used"] is False
