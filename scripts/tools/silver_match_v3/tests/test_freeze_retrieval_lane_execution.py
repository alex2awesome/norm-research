import json

import pytest

import scripts.tools.silver_match_v3.freeze_retrieval_lane_execution as module
from scripts.tools.silver_match_v3.freeze_retrieval_lane_execution import freeze


def test_freezes_exact_lane_from_task_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "validate_plan", lambda plan: None)
    monkeypatch.setattr(module, "validate_gpu_indices_for_host", lambda values: None)
    plan = tmp_path / "plan.json"
    runner = tmp_path / "scripts/tools/silver_match_v3/run_frozen_retrieval_lane.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("runner")
    command = ["python", "retrieve"]
    plan.write_text(
        json.dumps(
            {
                "task": "t",
                "execution": {"repo_root": str(tmp_path)},
                "steps": [
                    {
                        "kind": "retrieve",
                        "corpus": "c",
                        "system": "s",
                        "candidate": "/out/c.jsonl",
                        "audit": "/out/c.audit.json",
                        "expected_k": 7,
                        "command": command,
                    }
                ],
            }
        )
    )
    result = freeze(plan, "c", "s", 5)
    assert result["gpu_index"] == 5
    assert result["command"] == command
    assert result["scientific_plan_changed"] is False


def test_rejects_missing_lane(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "validate_plan", lambda plan: None)
    monkeypatch.setattr(module, "validate_gpu_indices_for_host", lambda values: None)
    plan = tmp_path / "plan.json"
    runner = tmp_path / "scripts/tools/silver_match_v3/run_frozen_retrieval_lane.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("runner")
    plan.write_text(
        json.dumps({"task": "t", "execution": {"repo_root": str(tmp_path)}, "steps": []})
    )
    with pytest.raises(ValueError, match="exactly one"):
        freeze(plan, "c", "s", 0)
