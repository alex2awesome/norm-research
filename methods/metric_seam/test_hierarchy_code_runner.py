from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_code_runner import (
    CANONICAL_ITEMS_ROOT,
    ExecutionInputError,
    _validate_worker_result,
    build_execution_plan,
    execute_audit,
    execute_one_program,
    load_bound_items,
    module_name_for_source,
    validate_items,
    worker_environment,
)


ROOT = Path(__file__).resolve().parents[2]
A104 = "methods/existing_metrics_runner/coded/metrics/a104_test_presence.py"
AUDIT = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/code_review_construct_fidelity_v1.json"


def _items():
    return [
        {
            "item_key": "one",
            "ctext": "diff --git a/x.py b/x.py\n+++ b/x.py\n@@ -0,0 +1 @@\n+def test_x(): pass\n",
        },
        {
            "item_key": "two",
            "ctext": "diff --git a/x.py b/x.py\n+++ b/x.py\n@@ -0,0 +1 @@\n+def f(): pass\n",
        },
    ]


def _audit():
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def _one_program_audit():
    audit = _audit()
    kept = False
    for row in audit["rows"]:
        if row["candidate"] and row["candidate"]["aspect_id"] == "a104" and not kept:
            row["verdict"] = "partial"
            row["scope"] = "subrelation_only"
            row["eligible_for_relation_local_execution"] = True
            kept = True
        elif row["eligible_for_relation_local_execution"]:
            row["verdict"] = "mismatch"
            row["scope"] = "none"
            row["eligible_for_relation_local_execution"] = False
    assert kept
    return audit


def test_item_contract_rejects_any_label_or_reference_field():
    bad = _items()
    bad[0]["label"] = 1
    with pytest.raises(ExecutionInputError, match="exactly item_key and ctext"):
        validate_items(bad)


def test_phase_is_bound_to_complete_official_split_panel(tmp_path):
    train, train_path = load_bound_items(CANONICAL_ITEMS_ROOT, "compiler_train")
    heldout, heldout_path = load_bound_items(CANONICAL_ITEMS_ROOT, "heldout_pre_reference")
    assert train_path.name == "compiler_train.json"
    assert heldout_path.name == "sealed_heldout.json"
    assert {row["ctext"] for row in train}.isdisjoint(row["ctext"] for row in heldout)
    with pytest.raises(ExecutionInputError, match="canonical items root"):
        load_bound_items(tmp_path, "compiler_train")


def test_module_path_must_remain_in_historical_metric_library():
    assert module_name_for_source(A104).endswith(".a104_test_presence")
    with pytest.raises(ExecutionInputError, match="outside the allowed"):
        module_name_for_source("methods/metric_seam/reconstruction_v2.py")


def test_real_canonical_audit_is_strict_and_plan_normalizes_program_identity():
    plan = build_execution_plan(_one_program_audit())
    assert len(plan) == 1
    assert plan[0]["aspect_id"] == "a104"
    assert plan[0]["source_path"] == A104
    assert plan[0]["relations"][0]["construct_fidelity_verdict"] == "partial"

    poisoned = _one_program_audit()
    poisoned["reference_values"] = [1, 2, 3]
    with pytest.raises(ExecutionInputError, match="top-level fields differ"):
        build_execution_plan(poisoned)


def test_strict_audit_rejects_bool_depth_and_scope_drift():
    poisoned = _one_program_audit()
    row = next(row for row in poisoned["rows"] if row["eligible_for_relation_local_execution"])
    row["audited_depth"] = True
    with pytest.raises(ExecutionInputError, match="invalid audited depth"):
        build_execution_plan(poisoned)

    poisoned = _one_program_audit()
    row = next(row for row in poisoned["rows"] if row["eligible_for_relation_local_execution"])
    row["scope"] = "whole_construct"
    with pytest.raises(ExecutionInputError, match="verdict/scope mismatch"):
        build_execution_plan(poisoned)


def test_real_structural_metric_runs_with_audited_source_identity():
    source_sha256 = hashlib.sha256((ROOT / A104).read_bytes()).hexdigest()
    result = execute_one_program(A104, "a104", source_sha256, _items())
    assert result["worker_status"] == "completed"
    assert result["summary"]["n_items"] == 2
    assert set(row["item_key"] for row in result["rows"]) == {"one", "two"}


def test_worker_environment_is_allowlisted_and_masks_accelerators(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setenv("DATABASE_URL", "secret")
    monkeypatch.setenv("LD_PRELOAD", "evil")
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    env = worker_environment(tmp_path)
    assert not {"AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "DATABASE_URL", "LD_PRELOAD"} & set(env)
    assert env["PYTHONPATH"] == str(ROOT)
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_parent_execution_uses_official_panel_and_narrow_provenance_claims():
    result = execute_audit(
        _one_program_audit(),
        CANONICAL_ITEMS_ROOT,
        phase="compiler_train",
        timeout_seconds=60,
        audit_source_path=str(AUDIT),
    )
    assert result["reference_fields_passed_to_worker"] is False
    assert result["outcome_fields_passed_to_worker"] is False
    assert result["worker_filesystem_isolated"] is False
    assert result["worker_network_isolated"] is False
    assert result["summary"]["n_planned_relation_mappings"] == 1
    assert result["programs"][0]["relations"][0]["scope"] == "subrelation_only"


def test_returned_item_set_is_checked():
    plan = build_execution_plan(_one_program_audit())[0]
    bad = {
        "aspect_id": "a104",
        "source_path": A104,
        "worker_status": "completed",
        "rows": [{"item_key": "wrong"}],
    }
    with pytest.raises(ExecutionInputError, match="item-set mismatch"):
        _validate_worker_result(bad, plan, ["expected"])
