from __future__ import annotations

import copy
import hashlib
import json

import pytest

from methods.codability.experiments.compile_fresh_name_arm_bank import (
    BREADTH_LEVELS,
    DEFAULT_BREADTH_TASKS,
    compile_breadth_bank,
    compile_metric_panel,
)
from methods.metric_seam.hierarchy_batch import (
    BRIEF_SCHEMA,
    COMPLETION_ARTIFACT_FIELDS,
    COMPLETION_GATES,
    COMPLETION_RECEIPT_SCHEMA,
    TERMINAL_ATTEMPT_RECEIPT_SCHEMA,
    build_readiness,
    compile_briefs,
)


@pytest.fixture(scope="module")
def panel():
    return compile_metric_panel()


def test_real_panel_compiles_990_label_free_metric_seam_briefs(panel):
    briefs = compile_briefs(panel)
    assert len(briefs) == 11 * 3 * 30
    assert {brief["schema"] for brief in briefs} == {BRIEF_SCHEMA}
    assert all(brief["compiler_view"]["reference_values_available"] is False for brief in briefs)
    assert all(brief["objective"]["external_supervision"] is False for brief in briefs)
    assert all(brief["candidate_subrelations"] for brief in briefs)
    assert all("0" in brief["program_contract"]["depth_vocabulary"] for brief in briefs)
    assert all("4" in brief["program_contract"]["depth_vocabulary"] for brief in briefs)


def test_prompt_arm_compilation_does_not_count_as_deep_code_completion(panel):
    bank = compile_breadth_bank(panel=panel)
    readiness = build_readiness(panel, prompt_bank=bank)
    assert readiness["n_cells"] == 990
    assert all(row["prompt_arms_compiled"] for row in readiness["rows"])
    assert all(row["prompt_arm_count"] >= 19 for row in readiness["rows"])
    assert not any(row["candidate_program_present"] for row in readiness["rows"])
    assert not any(row["completed_deep_metric_seam_run"] for row in readiness["rows"])
    assert all(
        cell == {"target": 30, "complete": 0, "remaining": 30}
        for task in readiness["matrix"].values()
        for cell in task.values()
    )


def test_path_only_registry_does_not_advance_a_scientific_completion(panel):
    first = panel["cells"][0]
    registry = {
        first["id"]: {
            "candidate_path": "candidate.py",
            "decomposition_path": "decomposition.json",
            "depth_record_path": "depth.json",
            "candidate_execution_path": "scores.json",
            "construct_fidelity_path": "construct_fidelity.json",
            "certificate_path": "certificate.json",
            "frozen_reference_path": "reference.json",
            "sealed_evaluation_path": "evaluation.json",
            "isomorphism_path": "isomorphism.json",
        }
    }
    readiness = build_readiness(panel, program_registry=registry)
    row = next(item for item in readiness["rows"] if item["cell_id"] == first["id"])
    assert row["candidate_program_declared"] is True
    assert row["candidate_program_present"] is False
    assert row["artifact_evidence"]["all_completion_artifacts_declared"] is True
    assert row["artifact_evidence"]["all_completion_artifacts_locally_present"] is False
    assert row["artifact_evidence"]["completion_receipt_valid"] is False
    assert row["completed_deep_metric_seam_run"] is False
    assert sum(item["completed_deep_metric_seam_run"] for item in readiness["rows"]) == 0


def test_bound_completion_receipt_advances_only_its_cell(panel, tmp_path):
    first = panel["cells"][0]
    registry_row = {}
    artifacts = {}
    for label, field in COMPLETION_ARTIFACT_FIELDS.items():
        suffix = ".py" if label == "candidate" else ".json"
        path = tmp_path / f"{label}{suffix}"
        path.write_text(f"artifact:{label}\n", encoding="utf-8")
        registry_row[field] = path.name
        artifacts[label] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    receipt = {
        "schema": COMPLETION_RECEIPT_SCHEMA,
        "cell_id": first["id"],
        "panel_content_sha256": panel["panel_content_sha256"],
        "status": "validated_complete",
        "external_supervised_target_used": False,
        "gates": {
            gate: {"status": "pass", "evidence": f"validated by {gate}"}
            for gate in COMPLETION_GATES
        },
        "isomorphism_outcome": "isomorphic",
        "artifacts": artifacts,
    }
    receipt_path = tmp_path / "completion_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    registry_row["completion_receipt_path"] = receipt_path.name

    readiness = build_readiness(
        panel,
        program_registry={first["id"]: registry_row},
        artifact_root=tmp_path,
    )
    row = next(item for item in readiness["rows"] if item["cell_id"] == first["id"])
    assert row["artifact_evidence"]["all_completion_artifacts_locally_present"] is True
    assert row["artifact_evidence"]["completion_receipt_valid"] is True
    assert row["completed_deep_metric_seam_run"] is True
    assert sum(item["completed_deep_metric_seam_run"] for item in readiness["rows"]) == 1

    (tmp_path / "candidate.py").write_text("changed after receipt\n", encoding="utf-8")
    drifted = build_readiness(
        panel,
        program_registry={first["id"]: registry_row},
        artifact_root=tmp_path,
    )
    drifted_row = next(
        item for item in drifted["rows"] if item["cell_id"] == first["id"]
    )
    assert drifted_row["artifact_evidence"]["completion_receipt_valid"] is False
    assert "content binding" in drifted_row["artifact_evidence"]["completion_receipt_error"]
    assert drifted_row["completed_deep_metric_seam_run"] is False


def test_bounded_non_discovery_is_separate_and_needs_no_candidate(panel, tmp_path):
    first = panel["cells"][0]
    receipt = {
        "schema": TERMINAL_ATTEMPT_RECEIPT_SCHEMA,
        "cell_id": first["id"],
        "panel_content_sha256": panel["panel_content_sha256"],
        "status": "bounded_non_discovery",
        "external_supervised_target_used": False,
        "search_scope": {
            "program_class": "typed deterministic Python DAGs",
            "capabilities": "base parser and symbolic normalization",
            "representation": "compiler-train ctext only",
            "budget": "one frozen authoring pass",
            "reason": "no construct-faithful executable relation was found",
        },
        "tacitness_claimed": False,
    }
    path = tmp_path / "terminal_attempt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    readiness = build_readiness(
        panel,
        program_registry={
            first["id"]: {"terminal_attempt_receipt_path": path.name}
        },
        artifact_root=tmp_path,
    )
    row = next(item for item in readiness["rows"] if item["cell_id"] == first["id"])
    assert row["candidate_program_declared"] is False
    assert row["bounded_non_discovery_recorded"] is True
    assert row["completed_deep_metric_seam_run"] is False


def test_prompt_bank_must_belong_to_exact_panel(panel):
    bank = compile_breadth_bank(panel=panel)
    wrong = copy.deepcopy(bank)
    wrong["metric_panel_content_sha256"] = "wrong"
    with pytest.raises(ValueError, match="different hierarchy panel"):
        build_readiness(panel, prompt_bank=wrong)


def test_every_task_and_level_is_represented(panel):
    readiness = build_readiness(panel)
    assert set(readiness["matrix"]) == set(DEFAULT_BREADTH_TASKS)
    assert all(set(levels) == set(BREADTH_LEVELS) for levels in readiness["matrix"].values())


def test_registry_rows_must_be_objects(panel):
    with pytest.raises(ValueError, match="non-object rows"):
        build_readiness(panel, program_registry={panel["cells"][0]["id"]: "candidate.py"})
