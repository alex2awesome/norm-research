from __future__ import annotations

import json

from methods.metric_seam.pilot import prepare_a407_matched_prompt_arms_v2 as matched


def _source_rows() -> tuple[list[dict], list[dict]]:
    raw: list[dict] = []
    hybrid: list[dict] = []
    for ordinal in (1, 2):
        item = {"ctext": f"diff {ordinal}", "item_key": f"heldout_{ordinal:04d}"}
        common = {
            "heldout_ordinal": ordinal,
            "item_key": item["item_key"],
            "input": item,
        }
        raw.append(dict(common))
        hybrid.append(
            {
                **common,
                "codescope_v3_facts": {
                    "schema": "metric-seam.code-scope-declaration-use-graph.v3",
                    "files": [],
                },
            }
        )
    return raw, hybrid


def test_matched_arms_change_only_the_registered_fact_intervention() -> None:
    spec = matched._load_spec()
    model = json.loads(matched.MODEL_SPEC_PATH.read_text(encoding="utf-8"))
    raw_source, hybrid_source = _source_rows()
    raw, hybrid = matched.build_matched_arms(
        raw_source, hybrid_source, spec=spec, model=model
    )
    assert len(raw) == len(hybrid) == 2
    for left, right in zip(raw, hybrid):
        assert left["system_prompt"] == right["system_prompt"]
        assert spec["construct"] in left["system_prompt"]
        assert "RELATION_DEFINITIONS_JSON" in left["system_prompt"]
        assert all(
            definition in left["system_prompt"]
            for definition in spec["relation_semantics"].values()
        )
        assert left["response_relations"] == right["response_relations"]
        assert left["ctext_sha256"] == right["ctext_sha256"]
        assert left["codescope_v3_facts_present"] is False
        assert right["codescope_v3_facts_present"] is True
        left_input = json.loads(left["user_prompt"].split("\n", 1)[1])
        right_input = json.loads(right["user_prompt"].split("\n", 1)[1])
        assert set(left_input) == set(right_input) == {
            "codescope_v3_facts",
            "ctext",
            "item_key",
        }
        assert left_input["ctext"] == right_input["ctext"]
        assert left_input["item_key"] == right_input["item_key"]
        assert left_input["codescope_v3_facts"] is None
        assert isinstance(right_input["codescope_v3_facts"], dict)


def test_spec_uses_all_six_relations_with_one_output_contract() -> None:
    spec = matched._load_spec()
    assert tuple(spec["relation_semantics"]) == matched.RELATIONS
    assert tuple(spec["output_contract"]["relation_scores"]) == matched.RELATIONS
    assert tuple(spec["output_contract"]["relation_abstentions"]) == matched.RELATIONS
    assert spec["experimental_contrast"][
        "all_other_system_prompt_and_output_contract_content_identical"
    ] is True
