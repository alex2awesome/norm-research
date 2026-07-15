from __future__ import annotations

from collections import defaultdict
import json

import pytest

from methods.metric_seam.family_scale.shared_occasion_compiler import (
    BUNDLE_SCHEMA,
    BlindDiscoveryArm,
    ContractError,
    Occasion,
    PromptChannel,
    Relation,
    compile_shared_occasion_bundle,
    parse_discovery_response,
    parse_relation_response,
)


OCCASIONS = [
    Occasion(f"o{index:02d}", {"text": f"payload {index}", "ordinal": index})
    for index in range(20)
]
RELATIONS = [
    Relation(f"r{index}", f"relation {index}", f"Assess relation number {index}.")
    for index in range(7)
]
CHANNELS = [
    PromptChannel("articulated", "Use only the articulated relation."),
    PromptChannel("implementation_disclosed", "Use the disclosed implementation contract."),
]


def _compile(**overrides):
    arguments = {
        "occasions": OCCASIONS,
        "relations": RELATIONS,
        "channels": CHANNELS,
        "model": "model-pinned",
        "randomization_seed": "family-seed-v1",
    }
    arguments.update(overrides)
    return compile_shared_occasion_bundle(**arguments)


def test_deterministic_shared_occasion_plan_two_passes_and_ten_percent_calibration() -> None:
    one = _compile()
    again = _compile()
    assert one == again
    assert one["schema"] == BUNDLE_SCHEMA
    manifest = one["manifest"]
    assert manifest["passes"] == [1, 2]
    assert manifest["calibration"]["target_percent"] == 10
    assert manifest["calibration"]["occasion_count"] == 2
    assert len(manifest["manifest_sha256"]) == 64

    requests = one["relation_conditioned_requests"]
    projections = defaultdict(set)
    plans = defaultdict(set)
    pass_counts = defaultdict(set)
    calibration_ids = set(manifest["calibration"]["occasion_ids"])
    for request in requests:
        occasion = request["occasion"]
        key = occasion["occasion_id"]
        projections[key].add(json.dumps(occasion, sort_keys=True))
        plan_key = (key, request["batch_index"])
        plans[plan_key].add(tuple(row["relation_id"] for row in request["relations"]))
        pass_counts[(plan_key, request["channel_id"])].add(request["pass_index"])
        size = len(request["relations"])
        if key in calibration_ids:
            assert request["calibration_unbatched"] is True
            assert size == 1
        else:
            assert request["calibration_unbatched"] is False
            assert 2 <= size <= 3

    assert all(len(values) == 1 for values in projections.values())
    assert all(len(values) == 1 for values in plans.values())
    assert all(values == {1, 2} for values in pass_counts.values())


def test_cooccurrence_is_randomized_by_occasion_but_channel_invariant() -> None:
    bundle = _compile()
    calibration = set(bundle["manifest"]["calibration"]["occasion_ids"])
    first_batches = {
        tuple(row["relation_id"] for row in request["relations"])
        for request in bundle["relation_conditioned_requests"]
        if request["occasion"]["occasion_id"] not in calibration
        and request["batch_index"] == 0
    }
    assert len(first_batches) > 1
    changed_seed = _compile(randomization_seed="family-seed-v2")
    assert (
        bundle["manifest"]["batch_plan_sha256"]
        != changed_seed["manifest"]["batch_plan_sha256"]
    )


def test_manifest_hashes_bind_payload_relation_channel_and_requests() -> None:
    base = _compile()["manifest"]
    payload_change = _compile(
        occasions=[Occasion("o00", {"text": "changed"}), *OCCASIONS[1:]]
    )["manifest"]
    relation_change = _compile(
        relations=[Relation("r0", "changed", "Changed relation."), *RELATIONS[1:]]
    )["manifest"]
    channel_change = _compile(
        channels=[PromptChannel("articulated", "Changed instruction."), CHANNELS[1]]
    )["manifest"]
    assert base["occasion_set_sha256"] != payload_change["occasion_set_sha256"]
    assert base["relation_set_sha256"] != relation_change["relation_set_sha256"]
    assert base["channel_set_sha256"] != channel_change["channel_set_sha256"]
    assert len({
        base["manifest_sha256"],
        payload_change["manifest_sha256"],
        relation_change["manifest_sha256"],
        channel_change["manifest_sha256"],
    }) == 4


def test_relation_parser_accepts_exact_flat_rows_and_fences() -> None:
    request = _compile()["relation_conditioned_requests"][0]
    rows = [
        {
            "occasion_id": request["occasion"]["occasion_id"],
            "relation_id": relation["relation_id"],
            "applies": True,
            "violated": False,
            "witness": "the payload contains the occasion",
        }
        for relation in request["relations"]
    ]
    parsed = parse_relation_response(
        f"```json\n{json.dumps(rows)}\n```", request=request
    )
    assert parsed.parse_mode == "fence_unwrapped"
    assert len(parsed.rows) == len(request["relations"])


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "wrong_relation",
        "invalid_state",
        "float",
    ],
)
def test_relation_parser_rejects_contract_drift(mutation: str) -> None:
    request = _compile()["relation_conditioned_requests"][0]
    rows = [
        {
            "occasion_id": request["occasion"]["occasion_id"],
            "relation_id": relation["relation_id"],
            "applies": False,
            "violated": False,
            "witness": None,
        }
        for relation in request["relations"]
    ]
    if mutation == "extra":
        rows[0]["confidence"] = 1
    elif mutation == "wrong_relation":
        rows[0]["relation_id"] = "wrong"
    elif mutation == "invalid_state":
        rows[0]["violated"] = True
    else:
        rows[0]["confidence"] = 0.5
    with pytest.raises(ContractError):
        parse_relation_response(json.dumps(rows), request=request)


def test_blind_discovery_is_explicitly_separate_relation_blind_and_two_pass() -> None:
    bundle = _compile(
        blind_discovery=BlindDiscoveryArm(
            "blind_discovery", "Discover executable relation types from the occasion."
        )
    )
    assert bundle["manifest"]["blind_discovery"] == {
        "enabled": True,
        "separate_from_relation_conditioned": True,
        "channel_id": "blind_discovery",
    }
    discovery = bundle["blind_discovery_requests"]
    assert len(discovery) == 2 * len(OCCASIONS)
    assert {row["pass_index"] for row in discovery} == {1, 2}
    relation_phrases = {
        relation.name for relation in RELATIONS
    } | {relation.description for relation in RELATIONS}
    assert all(
        all(phrase not in request["user_prompt"] for phrase in relation_phrases)
        for request in discovery
    )
    assert all(request["arm"] == "blind_discovery" for request in discovery)


def test_discovery_parser_is_flat_strict_and_fence_resilient() -> None:
    request = _compile(
        blind_discovery=BlindDiscoveryArm("blind_discovery", "Discover relations.")
    )["blind_discovery_requests"][0]
    occasion_id = request["occasion"]["occasion_id"]
    raw = json.dumps(
        [
            {
                "occasion_id": occasion_id,
                "candidate_id": "c1",
                "witness_kind": "dated disclosure record",
                "relation": "bind disclosure content to public availability date",
            }
        ]
    )
    parsed = parse_discovery_response(f"```\n{raw}\n```", request=request)
    assert parsed.parse_mode == "fence_unwrapped"
    assert parsed.rows[0]["candidate_id"] == "c1"

    duplicate = json.dumps([json.loads(raw)[0], json.loads(raw)[0]])
    with pytest.raises(ContractError, match="unique"):
        parse_discovery_response(duplicate, request=request)


def test_compiler_rejects_single_relation_and_discovery_channel_collision() -> None:
    with pytest.raises(ContractError, match="at least two"):
        _compile(relations=RELATIONS[:1])
    with pytest.raises(ContractError, match="separate"):
        _compile(
            blind_discovery=BlindDiscoveryArm(
                CHANNELS[0].channel_id, "Discover relations."
            )
        )
