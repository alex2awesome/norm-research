from __future__ import annotations

import json

import pytest

from methods.metric_seam.pilot import a407_matched_prompt_payload_v2 as payload
from methods.metric_seam.pilot import prepare_a407_matched_prompt_arms_v2 as matched


def _requests() -> tuple[dict, dict, dict, dict]:
    spec = matched._load_spec()
    model = json.loads(matched.MODEL_SPEC_PATH.read_text(encoding="utf-8"))
    item_key = "heldout_0001"
    ctext = "diff --git a/a.py b/a.py\n+retry_count = 2"
    facts = {
        "schema": "metric-seam.code-scope-declaration-use-graph.v3",
        "files": [],
    }
    raw = matched._request(
        arm="raw_prompt",
        ordinal=1,
        item_key=item_key,
        ctext=ctext,
        facts=None,
        spec=spec,
        model=model,
    )
    hybrid = matched._request(
        arm="hybrid",
        ordinal=1,
        item_key=item_key,
        ctext=ctext,
        facts=facts,
        spec=spec,
        model=model,
    )
    return raw, hybrid, spec, model


def test_final_payload_isolates_fact_augmentation() -> None:
    raw, hybrid, spec, model = _requests()
    raw_payload = payload.api_payload_for_request(raw, model, spec)
    hybrid_payload = payload.api_payload_for_request(hybrid, model, spec)
    assert raw_payload["messages"][0] == hybrid_payload["messages"][0]
    system = raw_payload["messages"][0]["content"]
    assert system.count(spec["construct"]) == 1
    assert all(system.count(definition) == 1 for definition in spec["relation_semantics"].values())
    assert raw_payload["response_format"] == hybrid_payload["response_format"]
    schema = raw_payload["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["relation_scores"]["required"] == list(
        matched.RELATIONS
    )
    raw_input = json.loads(raw_payload["messages"][1]["content"].split("\n", 1)[1])
    hybrid_input = json.loads(
        hybrid_payload["messages"][1]["content"].split("\n", 1)[1]
    )
    assert raw_input["ctext"] == hybrid_input["ctext"]
    assert raw_input["item_key"] == hybrid_input["item_key"]
    assert raw_input["codescope_v3_facts"] is None
    assert isinstance(hybrid_input["codescope_v3_facts"], dict)


def test_full_preparation_replays_final_payload_contract() -> None:
    raw, hybrid, spec, model = payload.load_prepared()
    assert len(raw) == len(hybrid) == 100
    assert all(
        payload.api_payload_for_request(left, model, spec)["response_format"]
        == payload.api_payload_for_request(right, model, spec)["response_format"]
        for left, right in zip(raw, hybrid)
    )


def test_response_validation_preserves_abstentions() -> None:
    raw, _hybrid, spec, _model = _requests()
    valid = {
        "item_key": "heldout_0001",
        "abstained": False,
        "abstention_reason": "none",
        "declared_holistic_score": 0.7,
        "relation_scores": {relation: 0.7 for relation in matched.RELATIONS},
        "relation_abstentions": {
            relation: False for relation in matched.RELATIONS
        },
    }
    assert payload.validate_response(valid, request=raw, spec=spec) == valid
    invalid = dict(valid)
    invalid["relation_scores"] = dict(valid["relation_scores"])
    invalid["relation_scores"]["semantic_context_fit"] = None
    with pytest.raises(ValueError, match="pairing"):
        payload.validate_response(invalid, request=raw, spec=spec)
