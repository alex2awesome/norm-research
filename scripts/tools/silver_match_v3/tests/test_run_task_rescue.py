import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.run_task_rescue import (
    _abstention_command,
    _validate_plan,
)


def _artifact(path: Path, text: str = "x") -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return {"path": str(path), "sha256": sha256_file(path)}


def test_abstention_verifier_uses_direct_batch_and_two_order_contract(tmp_path: Path) -> None:
    prompt = _artifact(tmp_path / "verify-abstain.txt")
    plan = {
        "manifest": {"path": "/data/manifest.json"},
        "abstention_verifier": {
            "prompt": prompt,
            "model": "/models/gemma",
            "max_model_len": 8192,
            "max_tokens": 180,
            "seed": 43,
        },
    }
    command = _abstention_command(
        plan=plan,
        gemma_python=Path("/env/gemma/bin/python"),
        audits=Path("/out/no-match.jsonl"),
        output=Path("/out/hashed.jsonl"),
        order="hashed",
        batch_size=128,
        gpu_memory_utilization=0.88,
    )
    assert "scripts.tools.silver_match_v3.verify_abstention_gemma" in command
    assert command[command.index("--order-mode") + 1] == "hashed"
    assert command[command.index("--model") + 1] == "/models/gemma"
    assert "--resume" in command
    assert not any("server" in value.lower() for value in command)


def test_rescue_plan_validation_rejects_single_capture_contract(tmp_path: Path) -> None:
    generic = _artifact(tmp_path / "generic")
    plan = {
        "schema_version": "silver-match-v3-task-rescue-plan-v2",
        "status": "FROZEN_READY_FOR_REPEATED_FULL_BANK_RESCUE",
        "manifest": generic,
        "production_plan": generic,
        "production_report": generic,
        "primary_final_pre_rescue": {
            "c": {"output": generic, "report": generic}
        },
        "candidate_systems": {
            "a": {"inputs": [{"candidate": generic, "audit": generic}]},
            "b": {"inputs": [{"candidate": generic, "audit": generic}]},
        },
        "abstention_verifier": {"prompt": generic},
        "blind_audit_exclusions": [generic],
        "implementations": {"one": generic},
        "rescue_policy": {
            "coverage_repeats": 1,
            "reinclude_primary": True,
            "include_all_abstentions": True,
            "strict_two_order_finalist_adjudication": True,
            "strict_two_order_contrastive_verification": True,
            "strict_two_order_typed_abstention_verification": True,
        },
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="weakens"):
        _validate_plan(path)
