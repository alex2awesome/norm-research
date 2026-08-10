from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_items import SCHEMA as ITEMS_SCHEMA
from methods.metric_seam.hierarchy_math_lclamp_gate import (
    MathLClampGateError,
    build_train_profile_gate,
)
from methods.metric_seam.hierarchy_math_lclamp_runner import (
    CAPABILITY_PATHS,
    MERGED_AUDIT_DESIGN_SCOPE,
    MERGED_AUDIT_SCHEMA,
    MERGED_AUDIT_STATUS,
    ROOT,
    MathLClampInputError,
    _expected_audit_summary,
    build_execution_plan,
    build_sentinel_profiles,
    execute_audit,
    execute_one_program,
    validate_items,
    validate_merged_audit,
    validate_profiles,
    worker_environment,
)


SYNTHETIC_PROGRAM = r'''"""Synthetic L-clamp contract fixture."""
import os

LLM_FIELDS = {"flag": "Synthetic constant field."}

def score(text, extracted, ops):
    forbidden = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN")
    if any(os.environ.get(name) for name in forbidden):
        raise RuntimeError("credential leaked")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("NVIDIA_VISIBLE_DEVICES") != "none":
        raise RuntimeError("accelerator visible")
    if text.startswith("abstain"):
        return None
    if text.startswith("bad"):
        return "not-a-number"
    if text.startswith("boom"):
        raise RuntimeError("synthetic item failure")
    base = 0.15 if extracted.get("flag") in ("", "NONE", "NO") else 0.35
    if text.startswith("high"):
        return base + 0.4
    if text.startswith("mid"):
        return base + 0.2
    return base
'''


def _write_program(root: Path, *, aspect_id: str = "a900", revision: int = 0) -> Path:
    path = root / f"{aspect_id}_h{revision}.py"
    path.write_text(SYNTHETIC_PROGRAM, encoding="utf-8")
    return path


def _capabilities():
    return [
        {
            "path": source,
            "sha256": hashlib.sha256((ROOT / source).read_bytes()).hexdigest(),
        }
        for source in CAPABILITY_PATHS
    ]


def _audit_sources():
    # Source-record shape is production-realistic; all five labels may point at
    # one harmless repository fixture in this synthetic-only test.
    record = {
        "path": CAPABILITY_PATHS[0],
        "sha256": hashlib.sha256((ROOT / CAPABILITY_PATHS[0]).read_bytes()).hexdigest(),
    }
    return {
        name: dict(record)
        for name in (
            "panel",
            "seed_map",
            "level_audit_1",
            "level_audit_2",
            "cross_audit_overlay",
        )
    }


def _candidate(program: Path):
    return {
        "aspect_id": "a900",
        "source_heading": "Synthetic conditional scorer",
        "selected_revision": 0,
        "source_path": str(program.resolve()),
        "program_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
        "historical_hybrid_provenance": "synthetic test fixture; never a scientific result",
        "llm_fields_excluded_from_implemented_relations": ["flag"],
    }


def _row(index: int, program: Path, *, eligible: bool):
    level = ("R1", "R2", "R3")[index // 30]
    if eligible:
        candidate = _candidate(program)
        verdict = "partial"
        scope = "subrelation_only"
        relations = ["Vary a synthetic text-length relation under a frozen field value."]
        depth = 1
    else:
        candidate = None
        verdict = "no_candidate_bounded_non_discovery"
        scope = "none"
        relations = []
        depth = None
    return {
        "cell_id": f"TB::math-stackexchange::synthetic::{level}::{index:03d}",
        "task": "math-stackexchange",
        "level": level,
        "metric_name": f"Synthetic metric {index}",
        "metric_description": "Synthetic metric used only to test the runner.",
        "candidate": candidate,
        "requested_relation": "Synthetic text to scalar relation.",
        "implemented_relations": relations,
        "residual_construct": "All scientific content is intentionally residual.",
        "verdict": verdict,
        "scope": scope,
        "eligible_for_relation_local_execution": eligible,
        "audited_depth": depth,
        "polarity_aggregation_applicability_caveats": ["Synthetic test only."],
        "justification": "Exercises the runner contract without a real item or program.",
        "interpretation": "No scientific claim.",
    }


def _audit(program: Path, *, n_eligible_rows: int = 3):
    rows = [_row(i, program, eligible=i < n_eligible_rows) for i in range(90)]
    payload = {
        "schema": MERGED_AUDIT_SCHEMA,
        "status": MERGED_AUDIT_STATUS,
        "task": "math-stackexchange",
        "design_scope": MERGED_AUDIT_DESIGN_SCOPE,
        "cross_audit": {
            "status": "complete",
            "n_guarded_changes": 0,
            "provisional_until_complete": False,
        },
        "sources": _audit_sources(),
        "panel_content_sha256": "0" * 64,
        "hierarchy_frame": {"kind": "synthetic test fixture"},
        "ops_math_source": CAPABILITY_PATHS[1],
        "ops_math_sha256": hashlib.sha256(
            (ROOT / CAPABILITY_PATHS[1]).read_bytes()
        ).hexdigest(),
        "execution_performed": False,
        "items_loaded": False,
        "reference_values_loaded": False,
        "outcome_labels_loaded": False,
        "program_outputs_loaded": False,
        "external_supervision": False,
        "depth_vocabulary": {"1": "synthetic parsed relation"},
        "capability_limit": "Synthetic fixture only.",
        "provenance": {"kind": "synthetic test fixture"},
        "interpretation": "Synthetic validation artifact only.",
        "summary": {},
        "rows": rows,
    }
    payload["summary"] = _expected_audit_summary(rows)
    return payload


def _item_root(root: Path):
    train = [
        {"item_key": "train_0001", "ctext": "low train"},
        {"item_key": "train_0002", "ctext": "mid train"},
        {"item_key": "train_0003", "ctext": "high train"},
    ]
    heldout = [
        {"item_key": "heldout_0001", "ctext": "low heldout"},
        {"item_key": "heldout_0002", "ctext": "high heldout"},
    ]
    manifest = {
        "schema": ITEMS_SCHEMA,
        "task": "math-stackexchange",
        "selection": {"train_n": len(train), "heldout_n": len(heldout)},
        "policy": {"outcome_columns_emitted": False},
    }
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "compiler_train.json").write_text(json.dumps(train), encoding="utf-8")
    (root / "sealed_heldout.json").write_text(json.dumps(heldout), encoding="utf-8")
    return train, heldout


def test_fixed_cartesian_grid_preserves_declared_field_identity():
    profiles = build_sentinel_profiles(["first", "second"])
    assert len(profiles) == 25
    assert profiles[0] == {
        "profile_id": "profile_000",
        "profile_index": 0,
        "assignments": [
            {"field_name": "first", "sentinel_id": "empty", "value": ""},
            {"field_name": "second", "sentinel_id": "empty", "value": ""},
        ],
    }
    assert profiles[-1]["assignments"][-1]["sentinel_id"] == "present"
    assert validate_profiles(profiles, ["first", "second"], require_complete_grid=True) == profiles

    poisoned = copy.deepcopy(profiles)
    poisoned[0]["assignments"][0]["field_name"] = "wrong"
    with pytest.raises(MathLClampInputError, match="fixed sentinel"):
        validate_profiles(poisoned, ["first", "second"], require_complete_grid=True)


def test_item_contract_rejects_any_reference_or_outcome_field():
    items = [{"item_key": "one", "ctext": "text", "reference": 0.5}]
    with pytest.raises(MathLClampInputError, match="exactly item_key and ctext"):
        validate_items(items)


def test_strict_audit_binds_source_digest_fields_and_groups_shared_program(tmp_path):
    program = _write_program(tmp_path)
    audit = _audit(program)
    validate_merged_audit(
        audit, program_root=tmp_path, require_canonical_programs=False
    )
    plans = build_execution_plan(
        audit, program_root=tmp_path, require_canonical_programs=False
    )
    assert len(plans) == 1
    assert plans[0]["aspect_id"] == "a900"
    assert plans[0]["llm_field_names"] == ["flag"]
    assert len(plans[0]["relations"]) == 3

    poisoned = copy.deepcopy(audit)
    poisoned["rows"][0]["candidate"]["program_sha256"] = "f" * 64
    with pytest.raises(MathLClampInputError, match="program changed"):
        validate_merged_audit(
            poisoned, program_root=tmp_path, require_canonical_programs=False
        )


def test_canonical_completed_merge_is_compatible_without_execution():
    path = (
        ROOT
        / "outputs/metric_seam_pilot/hierarchy_r123/"
        "math_stackexchange_construct_fidelity_merged_v1.json"
    )
    audit = json.loads(path.read_text(encoding="utf-8"))
    validate_merged_audit(audit)
    plans = build_execution_plan(audit)
    assert len(plans) == 16
    assert sum(len(plan["relations"]) for plan in plans) == 33
    assert {len(plan["llm_field_names"]) for plan in plans} == {1, 2}


def test_completed_merge_rejects_provisional_or_incoherent_metadata(tmp_path):
    program = _write_program(tmp_path)
    audit = _audit(program)

    poisoned = copy.deepcopy(audit)
    poisoned["cross_audit"]["provisional_until_complete"] = True
    with pytest.raises(MathLClampInputError, match="cross-audit is not complete"):
        validate_merged_audit(
            poisoned, program_root=tmp_path, require_canonical_programs=False
        )

    poisoned = copy.deepcopy(audit)
    poisoned["summary"]["eligible_for_relation_local_execution"] += 1
    with pytest.raises(MathLClampInputError, match="summary does not match"):
        validate_merged_audit(
            poisoned, program_root=tmp_path, require_canonical_programs=False
        )

    poisoned = copy.deepcopy(audit)
    poisoned["rows"][0]["candidate"][
        "llm_fields_excluded_from_implemented_relations"
    ] = ["other"]
    with pytest.raises(MathLClampInputError, match="field identity drift"):
        validate_merged_audit(
            poisoned, program_root=tmp_path, require_canonical_programs=False
        )


def test_three_state_accounting_is_separate_from_detailed_failure_type(tmp_path, monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "none")
    program = _write_program(tmp_path)
    plan = build_execution_plan(
        _audit(program, n_eligible_rows=1),
        program_root=tmp_path,
        require_canonical_programs=False,
    )[0]
    items = [
        {"item_key": "low", "ctext": "low"},
        {"item_key": "high", "ctext": "high"},
        {"item_key": "abstain", "ctext": "abstain"},
        {"item_key": "bad", "ctext": "bad"},
        {"item_key": "boom", "ctext": "boom"},
    ]
    result = execute_one_program(
        plan,
        items,
        build_sentinel_profiles(["flag"]),
        _capabilities(),
        program_root=tmp_path,
        require_canonical_programs=False,
        require_complete_grid=True,
    )
    assert result["worker_status"] == "completed"
    for profile in result["profiles"]:
        assert profile["summary"]["state_counts"] == {
            "measured": 2,
            "abstained": 1,
            "failed": 2,
        }
        statuses = {row["status"] for row in profile["rows"]}
        assert {"scored", "abstained", "contract_error", "execution_error"} <= statuses


def test_worker_environment_is_allowlisted_and_masks_all_accelerators(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setenv("LD_PRELOAD", "evil")
    env = worker_environment(tmp_path)
    assert not {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
        "LD_PRELOAD",
    } & set(env)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["NVIDIA_VISIBLE_DEVICES"] == "none"
    assert env["HIP_VISIBLE_DEVICES"] == "-1"


def test_synthetic_end_to_end_train_gate_then_heldout_uses_only_frozen_profile(
    tmp_path, monkeypatch
):
    program_root = tmp_path / "programs"
    program_root.mkdir()
    program = _write_program(program_root)
    item_root = tmp_path / "items"
    _item_root(item_root)
    audit = _audit(program)

    # These parent values must not survive into the subprocess. The synthetic
    # program raises if any do, so successful measurement is an end-to-end check.
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("GITHUB_TOKEN", "secret")

    train_execution = execute_audit(
        audit,
        item_root,
        phase="compiler_train",
        timeout_seconds=30,
        require_canonical_items=False,
        program_root=program_root,
        require_canonical_programs=False,
        audit_source_path="synthetic_audit",
    )
    assert train_execution["summary"]["n_unique_programs"] == 1
    assert train_execution["summary"]["n_profile_runs"] == 5
    assert train_execution["summary"]["three_state_totals"] == {
        "measured": 15,
        "abstained": 0,
        "failed": 0,
    }

    gate = build_train_profile_gate(
        train_execution,
        audit,
        min_measured=2,
        min_coverage=0.5,
        min_unique_scores=2,
        max_failed=0,
        execution_source="synthetic_train",
        audit_source="synthetic_audit",
        program_root=program_root,
        require_canonical_programs=False,
    )
    assert gate["reference_values_used"] is False
    assert gate["outcome_labels_used"] is False
    assert gate["heldout_items_or_outputs_used"] is False
    assert gate["capability_runtime"] == train_execution["capability_runtime"]
    assert gate["summary"]["n_selected_programs"] == 1
    assert gate["selected_program_profiles"][0]["profile"]["profile_id"] == "profile_000"

    heldout_execution = execute_audit(
        audit,
        item_root,
        phase="heldout_pre_reference",
        timeout_seconds=30,
        require_canonical_items=False,
        program_root=program_root,
        require_canonical_programs=False,
        audit_source_path="synthetic_audit",
        profile_selection=gate,
        profile_selection_source="synthetic_gate",
    )
    assert heldout_execution["summary"]["n_profile_runs"] == 1
    assert [
        profile["profile_id"]
        for profile in heldout_execution["programs"][0]["profiles"]
    ] == ["profile_000"]
    assert heldout_execution["reference_fields_passed_to_worker"] is False
    assert heldout_execution["actual_llm_extractions_passed_to_worker"] is False
    assert heldout_execution["accelerators_visible_to_worker"] is False

    poisoned_gate = copy.deepcopy(gate)
    poisoned_gate["heldout_items_or_outputs_used"] = True
    with pytest.raises(MathLClampInputError, match="forbidden field"):
        execute_audit(
            audit,
            item_root,
            phase="heldout_pre_reference",
            require_canonical_items=False,
            program_root=program_root,
            require_canonical_programs=False,
            profile_selection=poisoned_gate,
        )

    poisoned_gate = copy.deepcopy(gate)
    poisoned_gate["capability_runtime"][0]["sha256"] = "f" * 64
    with pytest.raises(MathLClampInputError, match="different capability runtime"):
        execute_audit(
            audit,
            item_root,
            phase="heldout_pre_reference",
            require_canonical_items=False,
            program_root=program_root,
            require_canonical_programs=False,
            profile_selection=poisoned_gate,
        )

    poisoned_gate = copy.deepcopy(gate)
    wrong_profile = build_sentinel_profiles(["flag"])[1]
    poisoned_gate["programs"][0]["selected_profile"] = wrong_profile
    poisoned_gate["selected_program_profiles"][0]["profile"] = wrong_profile
    with pytest.raises(MathLClampInputError, match="selection/tie-break drift"):
        execute_audit(
            audit,
            item_root,
            phase="heldout_pre_reference",
            require_canonical_items=False,
            program_root=program_root,
            require_canonical_programs=False,
            profile_selection=poisoned_gate,
        )


def test_gate_rejects_outcome_bearing_or_profile_drifted_execution(tmp_path):
    program_root = tmp_path / "programs"
    program_root.mkdir()
    program = _write_program(program_root)
    item_root = tmp_path / "items"
    _item_root(item_root)
    audit = _audit(program)
    execution = execute_audit(
        audit,
        item_root,
        phase="compiler_train",
        timeout_seconds=30,
        require_canonical_items=False,
        program_root=program_root,
        require_canonical_programs=False,
    )

    poisoned = copy.deepcopy(execution)
    poisoned["outcome_fields_passed_to_worker"] = True
    with pytest.raises(MathLClampGateError, match="outcome_fields"):
        build_train_profile_gate(
            poisoned,
            audit,
            min_measured=2,
            program_root=program_root,
            require_canonical_programs=False,
        )

    poisoned = copy.deepcopy(execution)
    poisoned["capability_runtime"][0]["sha256"] = "f" * 64
    with pytest.raises(MathLClampGateError, match="capability runtime drifted"):
        build_train_profile_gate(
            poisoned,
            audit,
            min_measured=2,
            program_root=program_root,
            require_canonical_programs=False,
        )

    poisoned = copy.deepcopy(execution)
    poisoned["programs"][0]["profiles"][0]["assignments"][0]["value"] = "HELDOUT_PICK"
    with pytest.raises(MathLClampGateError, match="noncanonical train profile grid"):
        build_train_profile_gate(
            poisoned,
            audit,
            min_measured=2,
            program_root=program_root,
            require_canonical_programs=False,
        )
