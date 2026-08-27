from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile

import pytest

from methods.metric_seam.adjudicate_math_construct_fidelity_r3_cross_audit import (
    A168_DEPTH_CAVEAT,
    A168_EMPTY_FIELD_CAVEAT,
    A198_SENTINEL_CAVEAT,
    A210_NONE_CAVEAT,
    A36_NONE_CAVEAT,
    A66_CLAMP_CAVEAT,
    CHANGE_SPECS,
    DEFAULT_OUT,
    DEFAULT_SOURCE_AUDIT,
    EXPECTED_SOURCE_AUDIT_SHA256,
    REPO_ROOT,
    build,
    validate,
)


A126_DISCOVERY = (
    "TB::math-stackexchange::general::R3::grandparent::17::"
    "6c6a55696fc92dcd5b7e"
)
A72_ECONOMY = (
    "TB::math-stackexchange::general::R3::merged_group::11::"
    "d5ce3c2432ca508dba8a"
)
A168_GRAMMAR = (
    "TB::math-stackexchange::general::R3::merged_group::16::"
    "bedf9a334646bda9ae1e"
)
A36_MODULAR = (
    "TB::math-stackexchange::general::R3::merged_group::14::"
    "da47b04ffaa9bf294ae9"
)
A180_NOTATION = (
    "TB::math-stackexchange::general::R3::merged_group::3::"
    "e7911c7b707a53bacba4"
)
A210_TYPESETTING = (
    "TB::math-stackexchange::general::R3::grandparent::16::"
    "570eed33fe5f1ce2a120"
)


@pytest.fixture(scope="module")
def source_bytes() -> bytes:
    return (REPO_ROOT / DEFAULT_SOURCE_AUDIT).read_bytes()


@pytest.fixture(scope="module")
def source(source_bytes: bytes) -> dict:
    return json.loads(source_bytes)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return build()


def test_frozen_source_and_complete_retrieved_coverage(
    source_bytes: bytes, artifact: dict
) -> None:
    assert hashlib.sha256(source_bytes).hexdigest() == EXPECTED_SOURCE_AUDIT_SHA256
    assert artifact["review_coverage"] == {
        "source_rows": 30,
        "retrieved_candidates_reviewed": 19,
        "retrieved_candidate_set_sha256": artifact["review_coverage"][
            "retrieved_candidate_set_sha256"
        ],
        "changed_rows": 7,
        "unchanged_retrieved_rows": 12,
        "all_retrieved_candidates_reviewed": True,
    }
    assert {change["cell_id"] for change in artifact["changes"]} == set(CHANGE_SPECS)


def test_overlay_lists_only_real_guarded_field_changes(
    source: dict, artifact: dict
) -> None:
    source_by_id = {row["cell_id"]: row for row in source["rows"]}
    for change in artifact["changes"]:
        row = source_by_id[change["cell_id"]]
        assert row["candidate"] is not None
        assert set(change["before"]) == set(change["after"])
        assert change["before"] != change["after"]
        for field, value in change["before"].items():
            assert row[field] == value
        assert change["candidate_guard"] == {
            "aspect_id": row["candidate"]["aspect_id"],
            "source_path": row["candidate"]["source_path"],
            "program_sha256": row["candidate"]["program_sha256"],
        }


def test_verdicts_are_retained_and_only_matched_depth_changes(
    source: dict, artifact: dict
) -> None:
    assert not [
        change for change in artifact["changes"] if "verdict" in change["after"]
    ]
    depth_changes = [
        change for change in artifact["changes"] if "audited_depth" in change["after"]
    ]
    assert [change["cell_id"] for change in depth_changes] == [A168_GRAMMAR]
    assert depth_changes[0]["before"]["audited_depth"] == 2
    assert depth_changes[0]["after"]["audited_depth"] == 1

    assert artifact["before_counts"]["retrieved_verdicts"] == {
        "mismatch": 4,
        "partial": 15,
    }
    assert artifact["after_counts_if_overlay_applied"] == {
        "retrieved_verdicts": {"mismatch": 4, "partial": 15},
        "retrieved_depths": {"1": 10, "2": 9},
        "eligible_depths": {"1": 6, "2": 9},
        "eligible_for_relation_local_execution": 15,
    }

    source_by_id = {row["cell_id"]: row for row in source["rows"]}
    # These are adversarial retentions, not unreviewed defaults. a126 has a
    # guess/check co-occurrence and local counterexample relation; a72 has only
    # bounded named anti-pattern burdens. Neither is whole-construct credit.
    for cell_id in (A126_DISCOVERY, A72_ECONOMY):
        assert source_by_id[cell_id]["verdict"] == "partial"
        assert source_by_id[cell_id]["eligible_for_relation_local_execution"] is True


def test_a168_depth_uses_only_requested_grammar_relation(artifact: dict) -> None:
    change = {row["cell_id"]: row for row in artifact["changes"]}[A168_GRAMMAR]
    implemented = " ".join(change["after"]["implemented_relations"]).lower()
    caveats = change["after"]["polarity_aggregation_applicability_caveats"]
    assert "congruence" not in implemented
    assert "bare logical" in implemented
    assert A168_DEPTH_CAVEAT in caveats
    assert A168_EMPTY_FIELD_CAVEAT in caveats

    program = (
        REPO_ROOT / "methods/metric_seam/hybrids/programs_math/a168_h0.py"
    ).read_text()
    assert "cong_mod_penalty" in program
    assert "W_FLAW * flaw_score" in program
    assert "W_CLAIM * claim_score" in program


def test_full_input_channel_disclosures_are_source_grounded(artifact: dict) -> None:
    by_id = {row["cell_id"]: row for row in artifact["changes"]}
    for cell_id in (A36_MODULAR, A180_NOTATION, A210_TYPESETTING):
        caveats = " ".join(
            by_id[cell_id]["after"]["polarity_aggregation_applicability_caveats"]
        ).lower()
        assert "question" in caveats
        assert "answer" in caveats

    for filename in ("a36_h0.py", "a180_h0.py", "a210_h0.py"):
        program = (
            REPO_ROOT / f"methods/metric_seam/hybrids/programs_math/{filename}"
        ).read_text()
        assert "Answer:" not in program

    a36_relations = " ".join(by_id[A36_MODULAR]["after"]["implemented_relations"])
    a180_relations = " ".join(by_id[A180_NOTATION]["after"]["implemented_relations"])
    assert "Question-plus-Answer" in a36_relations
    assert "Question-plus-Answer" in a180_relations


def test_field_interface_and_clamp_disclosures_match_control_flow(
    artifact: dict,
) -> None:
    by_id = {row["cell_id"]: row for row in artifact["changes"]}
    a66 = next(
        row for row in artifact["changes"]
        if row["candidate_guard"]["aspect_id"] == "a66"
    )
    assert A66_CLAMP_CAVEAT in a66["after"][
        "polarity_aggregation_applicability_caveats"
    ]

    a198 = [
        row for row in artifact["changes"]
        if row["candidate_guard"]["aspect_id"] == "a198"
    ]
    assert len(a198) == 2
    assert all(
        A198_SENTINEL_CAVEAT
        in row["after"]["polarity_aggregation_applicability_caveats"]
        for row in a198
    )
    assert A36_NONE_CAVEAT in by_id[A36_MODULAR]["after"][
        "polarity_aggregation_applicability_caveats"
    ]
    assert A210_NONE_CAVEAT in by_id[A210_TYPESETTING]["after"][
        "polarity_aggregation_applicability_caveats"
    ]

    a66_source = (
        REPO_ROOT / "methods/metric_seam/hybrids/programs_math/a66_h0.py"
    ).read_text()
    a198_source = (
        REPO_ROOT / "methods/metric_seam/hybrids/programs_math/a198_h1.py"
    ).read_text()
    a36_source = (
        REPO_ROOT / "methods/metric_seam/hybrids/programs_math/a36_h0.py"
    ).read_text()
    a210_source = (
        REPO_ROOT / "methods/metric_seam/hybrids/programs_math/a210_h0.py"
    ).read_text()
    assert "if not extracted:" in a66_source
    assert "_grounded_in_answer(cited_raw, ans)" in a198_source
    assert "return min(len(parts), 6)" in a36_source
    assert "llm_penalty = 0.08 if note else 0.0" in a210_source


def test_known_presence_function_rejections_remain_rejected(
    source: dict, artifact: dict
) -> None:
    changed = {row["cell_id"] for row in artifact["changes"]}
    mismatches = [
        row for row in source["rows"]
        if row.get("candidate") is not None and row["verdict"] == "mismatch"
    ]
    assert len(mismatches) == 4
    assert {row["candidate"]["aspect_id"] for row in mismatches} == {"a42", "a108"}
    assert all(row["cell_id"] not in changed for row in mismatches)


def test_validator_rejects_stale_before_and_candidate_guard(
    source: dict, artifact: dict
) -> None:
    stale = copy.deepcopy(artifact)
    first_field = next(iter(stale["changes"][0]["before"]))
    stale["changes"][0]["before"][first_field] = "tampered"
    with pytest.raises(ValueError, match="stale before guard"):
        validate(stale, source)

    wrong_candidate = copy.deepcopy(artifact)
    wrong_candidate["changes"][0]["candidate_guard"]["program_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="candidate guard mismatch"):
        validate(wrong_candidate, source)


def test_builder_rejects_rebased_source_bytes(source_bytes: bytes) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "source.json"
        path.write_bytes(source_bytes + b"\n")
        with pytest.raises(ValueError, match="source audit changed"):
            build(path)


def test_generated_artifact_is_reproducible_and_static(
    source: dict, artifact: dict
) -> None:
    output = json.loads((REPO_ROOT / DEFAULT_OUT).read_text())
    assert artifact == output
    assert output["forbidden_inputs_used"] is False
    assert output["candidate_execution_performed"] is False
    assert output["candidate_import_performed"] is False
    assert output["items_loaded"] is False
    assert output["model_or_api_calls_performed"] is False
    assert output["accelerators_used"] is False
    validate(output, source)
