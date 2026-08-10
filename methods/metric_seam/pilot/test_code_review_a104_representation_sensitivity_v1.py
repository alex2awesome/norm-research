"""Regression tests for the additive a104 representation sensitivity."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from methods.metric_seam.pilot.code_review_a104_representation_sensitivity_v1 import (
    DEFAULT_OUT,
    EXPECTED_FROZEN_PARSER_SOURCE_SHA256,
    EXPECTED_FROZEN_SANDBOX_SHA256,
    EXPECTED_MEASUREMENT_FREEZE_SHA256,
    EXPECTED_READOUT,
    FROZEN_SANDBOX_GIT_PROVENANCE,
    FROZEN_PARSER_SOURCE,
    SensitivityInputError,
    V3_AUDIT,
    _source_sha256,
    bind_frozen_parser,
    build_result,
    exact_prefix_crosswalk,
    expected_artifact,
)


@pytest.fixture(scope="module")
def result() -> dict:
    return build_result()


def test_vendored_parser_is_bound_to_the_independently_audited_dependency() -> None:
    assert _source_sha256(FROZEN_PARSER_SOURCE) == (
        EXPECTED_FROZEN_PARSER_SOURCE_SHA256
    )
    audit = json.loads(V3_AUDIT.read_text())
    assert audit["source_hashes_at_audit"]["deep_checker_sandbox_dependency"] == (
        EXPECTED_FROZEN_SANDBOX_SHA256
    )
    assert FROZEN_SANDBOX_GIT_PROVENANCE == {
        "commit": "e6018339153dfecf17dae9a51d3bea8c7c8257c2",
        "sandbox_blob_sha1": "c867caae1ae119f982803eaf785e7dcade04253f",
        "path": "methods/existing_metrics_runner/coded/sandbox.py",
    }
    provenance = FROZEN_SANDBOX_GIT_PROVENANCE
    historical = subprocess.check_output(
        ["git", "show", f"{provenance['commit']}:{provenance['path']}"],
        cwd=Path(__file__).resolve().parents[3],
    )
    git_blob = b"blob " + str(len(historical)).encode() + b"\0" + historical
    assert hashlib.sha1(git_blob).hexdigest() == provenance["sandbox_blob_sha1"]
    assert hashlib.sha256(historical).hexdigest() == EXPECTED_FROZEN_SANDBOX_SHA256
    source = historical.decode("utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "parse_diff_added_by_file"
    )
    assert ast.get_source_segment(source, node) + "\n" == FROZEN_PARSER_SOURCE


def test_binding_restores_the_live_parser_global() -> None:
    from methods.existing_metrics_runner.coded.metrics import a104_test_presence

    live = a104_test_presence.parse_diff_added_by_file
    with bind_frozen_parser(a104_test_presence):
        assert a104_test_presence.parse_diff_added_by_file is not live
    assert a104_test_presence.parse_diff_added_by_file is live
    try:
        with bind_frozen_parser(a104_test_presence):
            raise RuntimeError("exercise restoring finally")
    except RuntimeError:
        pass
    assert a104_test_presence.parse_diff_added_by_file is live


def test_exact_crosswalk_rejects_nonidentical_or_duplicate_prefixes() -> None:
    source = [
        {"datapoint_id": "a", "ctext": "abcdef"},
        {"datapoint_id": "b", "ctext": "uvwxyz"},
    ]
    hierarchy = [
        {"item_key": "x", "ctext": "abcd"},
        {"item_key": "y", "ctext": "uvwx"},
    ]
    assert exact_prefix_crosswalk(source, hierarchy, max_chars=4) == {
        "a": "x",
        "b": "y",
    }
    hierarchy[1]["ctext"] = "nope"
    with pytest.raises(SensitivityInputError, match="exact source-prefix multiset"):
        exact_prefix_crosswalk(source, hierarchy, max_chars=4)
    source[1]["ctext"] = "abcd-more"
    with pytest.raises(SensitivityInputError, match="projection is not unique"):
        exact_prefix_crosswalk(source, hierarchy, max_chars=4)


def test_frozen_replay_and_prefix_crosswalk_are_exact(result: dict) -> None:
    assert result["frozen_replay"]["n_items"] == 250
    assert result["frozen_replay"]["score_mismatches"] == 0
    assert result["crosswalk"]["exact_unique_prefix_matches"] == 250
    assert result["crosswalk"]["hierarchy_rows_at_cap"] == 205
    assert result["crosswalk"]["hierarchy_fraction_at_cap"] == 0.82
    assert not result["crosswalk"]["hierarchy_split_used_for_statistics"]


def test_common_support_readout_matches_the_frozen_parser_replay(result: dict) -> None:
    heldout = result["heldout_readout"]
    support = result["score_support_all_250"]
    assert heldout["common_support_n"] == EXPECTED_READOUT["common_heldout_n"]
    assert heldout["head_tail_rho_on_common"] == pytest.approx(
        EXPECTED_READOUT["head_tail_rho_on_common"], abs=5e-15
    )
    assert heldout["prefix4000_rho_on_common"] == pytest.approx(
        EXPECTED_READOUT["prefix_rho_on_common"], abs=5e-15
    )
    assert heldout["delta_prefix_minus_head_tail"] == pytest.approx(
        EXPECTED_READOUT["delta_prefix_minus_head_tail"], abs=5e-15
    )
    assert heldout["head_tail_prefix_program_vector_rho"] == pytest.approx(
        EXPECTED_READOUT["program_vector_rho"], abs=5e-15
    )
    assert support["applicability_status_changes"] == 12
    assert support["value_changes_on_common_scored"] == 118
    assert support["common_scored"] == 227


def test_result_keeps_the_claim_and_compute_boundary(result: dict) -> None:
    assert result["status"] == "complete_posthoc_exploratory"
    assert result["design_scope"] == "post_hoc_representation_sensitivity_no_new_gate"
    assert result["claim_status"] == {
        "post_hoc": True,
        "exploratory": True,
        "new_gate": False,
        "program_selection": "none_single_frozen_candidate",
        "criterion_selected_for_sensitivity_post_outcome": True,
        "supersedes_prior_artifact": False,
        "canonical_v4_modified": False,
    }
    assert not result["compute"]["model_or_api_calls"]
    assert not result["compute"]["gpu_used"]
    assert not result["compute"]["repository_or_under_review_test_execution"]
    blindness = result["blindness_and_reference_order"]
    assert blindness["classification"] == "label_unreferenced_not_label_inaccessible"
    assert blindness[
        "candidate_scores_and_split_frozen_before_llm_reference_values_parsed"
    ]
    assert blindness["measurement_freeze_sha256"] == (
        EXPECTED_MEASUREMENT_FREEZE_SHA256
    )
    assert blindness["reference_files_opened_only_after_measurement_freeze"]
    assert not blindness["outcomes_used_for_program_selection_or_tuning"]
    assert blindness["reference_representation"] == "historical_head_tail_ctext"
    assert not blindness["prefix_candidate_matches_reference_input"]
    assert blindness["prefix_arm"] == "one_sided_representation_mismatch_sensitivity"
    assert not blindness["direct_same_input_prefix_prompt_code_test"]
    assert "new confirmatory gate or promotion" in result["interpretation"][
        "not_permitted"
    ]
    assert "direct same-input prefix prompt/code isomorphism test" in result[
        "interpretation"
    ]["not_permitted"]


def test_checked_in_artifact_is_the_deterministic_recomputation() -> None:
    assert DEFAULT_OUT.read_bytes() == expected_artifact()


def test_direct_cli_check_works_from_repository_root() -> None:
    root = Path(__file__).resolve().parents[3]
    script = root / "methods/metric_seam/pilot/code_review_a104_representation_sensitivity_v1.py"
    checked = subprocess.run(
        [sys.executable, str(script), "--check"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr
    assert checked.stdout.startswith("ok ")
