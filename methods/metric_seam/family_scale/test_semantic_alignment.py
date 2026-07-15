from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from methods.metric_seam.family_scale.decomposition_stability import SCHEMA
from methods.metric_seam.family_scale.semantic_alignment import (
    AlignmentError,
    compile_alignment_request,
    main,
    parse_alignment_response,
    run_alignment_call,
)


EXACT = {
    "op_class": "computation",
    "witness_kind": "critical date comparison",
    "relation": "determine whether the disclosure precedes the critical date",
}
A_SEMANTIC = {
    "op_class": "computation",
    "witness_kind": "single disclosure limitation map",
    "relation": "test whether one disclosure contains every claim limitation",
}
B_SEMANTIC = {
    "op_class": "computation",
    "witness_kind": "feature coverage map",
    "relation": "compare all required limitations against one prior reference",
}
A_ONLY = {
    "op_class": "evidence",
    "witness_kind": "claim limitation set",
    "relation": "extract every required claim limitation",
}
B_ONLY = {
    "op_class": "evidence",
    "witness_kind": "dated disclosure record",
    "relation": "bind a disclosure to its public availability date",
}
C_ONLY = {
    "op_class": "individuation",
    "witness_kind": "single disclosure unit",
    "relation": "keep each prior disclosure separate",
}


def _submission() -> dict:
    return {
        "schema": SCHEMA,
        "metric": {
            "name": "Novelty and public-disclosure bars",
            "text": "A claimed invention must be new over one enabling disclosure.",
        },
        "fleets": [
            {"fleet_id": "fleet alpha", "relations": [EXACT, A_SEMANTIC, A_ONLY]},
            {"fleet_id": "fleet beta", "relations": [EXACT, B_SEMANTIC, B_ONLY]},
            {"fleet_id": "fleet gamma", "relations": [EXACT, C_ONLY]},
        ],
    }


def _unit(request: dict, fleet_id: str, relation: dict) -> str:
    return next(
        row["unit_id"]
        for row in request["private_unit_map"]
        if row["fleet_id"] == fleet_id and row["relation"] == relation
    )


def _valid_response(request: dict) -> str:
    exact = [_unit(request, fleet, EXACT) for fleet in ("fleet alpha", "fleet beta", "fleet gamma")]
    semantic = [
        _unit(request, "fleet alpha", A_SEMANTIC),
        _unit(request, "fleet beta", B_SEMANTIC),
    ]
    clustered = set(exact + semantic)
    unmatched = [
        row["unit_id"] for row in request["private_unit_map"]
        if row["unit_id"] not in clustered
    ]
    return json.dumps(
        {
            "clusters": [
                {"cluster_id": "c_exact", "unit_ids": exact},
                {"cluster_id": "c_semantic", "unit_ids": semantic},
            ],
            "unmatched_unit_ids": unmatched,
        }
    )


def test_compile_is_deterministic_blinded_and_seed_randomized() -> None:
    first = compile_alignment_request(_submission(), randomization_seed="seed-1")
    again = compile_alignment_request(_submission(), randomization_seed="seed-1")
    changed = compile_alignment_request(_submission(), randomization_seed="seed-2")
    assert first == again
    assert first["request_sha256"] != changed["request_sha256"]
    assert {row["unit_id"] for row in first["public_relation_units"]} != {
        row["unit_id"] for row in changed["public_relation_units"]
    }
    assert "fleet alpha" not in first["user_prompt"]
    assert "fleet beta" not in first["user_prompt"]
    assert "fleet gamma" not in first["user_prompt"]
    assert first["status"] == "compiled_for_exactly_one_alignment_call"
    assert len(first["public_relation_units"]) == 8


def test_parser_accepts_fence_and_enforces_partition_and_one_per_fleet() -> None:
    request = compile_alignment_request(_submission(), randomization_seed="seed")
    raw = _valid_response(request)
    parsed = parse_alignment_response(f"```json\n{raw}\n```", request=request)
    assert parsed.parse_mode == "fence_unwrapped"
    assert len(parsed.clusters) == 2
    assert len(parsed.unmatched_unit_ids) == 3

    value = json.loads(raw)
    same_fleet = [
        _unit(request, "fleet alpha", EXACT),
        _unit(request, "fleet alpha", A_ONLY),
    ]
    value["clusters"][0]["unit_ids"] = same_fleet
    all_clustered = {
        unit_id
        for cluster in value["clusters"]
        for unit_id in cluster["unit_ids"]
    }
    value["unmatched_unit_ids"] = [
        row["unit_id"] for row in request["private_unit_map"]
        if row["unit_id"] not in all_clustered
    ]
    with pytest.raises(AlignmentError, match="at most one"):
        parse_alignment_response(json.dumps(value), request=request)


def test_parser_retains_lexical_exact_as_a_required_lower_bound() -> None:
    request = compile_alignment_request(_submission(), randomization_seed="seed")
    value = json.loads(_valid_response(request))
    exact_ids = set(value["clusters"][0]["unit_ids"])
    value["clusters"] = value["clusters"][1:]
    value["unmatched_unit_ids"].extend(exact_ids)
    with pytest.raises(AlignmentError, match="lexical-exact"):
        parse_alignment_response(json.dumps(value), request=request)


def test_report_has_pairwise_semantic_precision_recall_jaccard_and_unmatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = compile_alignment_request(_submission(), randomization_seed="seed")
    response = _valid_response(request)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, response, "")

    monkeypatch.setattr(
        "methods.metric_seam.family_scale.semantic_alignment.subprocess.run", fake_run
    )
    report = run_alignment_call(request)
    assert len(calls) == 1
    assert calls[0][0][calls[0][0].index("--model") + 1] == "sonnet"
    assert "-p" in calls[0][0]
    assert report["alignment_call_count"] == 1
    assert len(report["unmatched_units"]) == 3
    alpha_beta = next(
        pair for pair in report["pairwise"]
        if pair["fleet_a"] == "fleet alpha" and pair["fleet_b"] == "fleet beta"
    )
    assert alpha_beta["semantic_match_count"] == 2
    assert alpha_beta["semantic_precision_a_to_b"]["numerator"] == 2
    assert alpha_beta["semantic_precision_a_to_b"]["denominator"] == 3
    assert alpha_beta["semantic_recall_a_to_b"]["numerator"] == 2
    assert alpha_beta["semantic_recall_a_to_b"]["denominator"] == 3
    assert alpha_beta["semantic_jaccard"]["numerator"] == 2
    assert alpha_beta["semantic_jaccard"]["denominator"] == 4
    assert alpha_beta["lexical_exact_lower_bound"]["match_count"] == 1
    assert alpha_beta["semantic_gain_over_lexical_exact"] == 1
    assert report["scope_guards"] == {
        "corpus_accessed": False,
        "programs_accessed": False,
        "program_outputs_accessed": False,
        "scores_labels_or_outcomes_accessed": False,
    }


@pytest.mark.parametrize(
    "mutation",
    ["extra_key", "unknown_unit", "float", "missing_unit"],
)
def test_response_contract_is_strict(mutation: str) -> None:
    request = compile_alignment_request(_submission(), randomization_seed="seed")
    value = json.loads(_valid_response(request))
    if mutation == "extra_key":
        value["confidence"] = 1
    elif mutation == "unknown_unit":
        value["unmatched_unit_ids"][0] = "u_unknown"
    elif mutation == "float":
        value["confidence"] = 0.5
    else:
        value["unmatched_unit_ids"].pop()
    with pytest.raises(AlignmentError):
        parse_alignment_response(json.dumps(value), request=request)


def test_closed_submission_rejects_non_three_fleet_and_corpus_fields() -> None:
    two = _submission()
    two["fleets"] = two["fleets"][:2]
    with pytest.raises(AlignmentError, match="exactly three"):
        compile_alignment_request(two, randomization_seed="seed")
    contaminated = _submission()
    contaminated["corpus"] = ["item text"]
    with pytest.raises(AlignmentError, match="corpus"):
        compile_alignment_request(contaminated, randomization_seed="seed")


def test_compile_and_score_cli_without_model_call(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.json"
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.txt"
    report_path = tmp_path / "report.json"
    submission_path.write_text(json.dumps(_submission()), encoding="utf-8")
    assert main([
        "compile", str(submission_path), "--seed", "seed", "--output", str(request_path)
    ]) == 0
    request = json.loads(request_path.read_text(encoding="utf-8"))
    response_path.write_text(_valid_response(request), encoding="utf-8")
    assert main([
        "score", str(request_path), str(response_path), "--output", str(report_path)
    ]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["alignment_call_count"] == 1
