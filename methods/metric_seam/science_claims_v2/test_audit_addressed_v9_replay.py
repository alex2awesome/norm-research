from __future__ import annotations

from methods.metric_seam.science_claims_v2 import audit_addressed_v9_replay as replay


def test_current_addressed_v9_replays_byte_exact_and_stays_pre_prompt() -> None:
    result = replay.audit()

    assert result["status"] == "byte_exact_cpu_replay_complete"
    assert result["replay"]["byte_exact_all_outputs"] is True
    assert result["replay"]["records"] == 2400
    assert result["replay"]["strong_relation_witnesses"] == 100
    assert result["replay"]["strong_whitespace_normalized_witnesses_shared"] == 100
    assert result["prompt_plane"] == {
        "compiled_unscored_jobs": 1957,
        "structural_abstentions_without_remote_call": 443,
        "prompt_responses_in_current_v8_bundle": 0,
        "prompt_articulability_measured": False,
        "semantic_prompt_code_comparison_measured": False,
    }
    assert result["channel_contract"]["models_or_apis_called"] is False
    assert result["channel_contract"]["accelerators_used"] is False
    assert result["temporal_disposition"][
        "fresh_split_required_for_confirmatory_prompt_code_claim"
    ] is True
