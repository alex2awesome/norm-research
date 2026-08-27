import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.validate_mi_execution_matrix import validate_matrix


REPO_ROOT = Path(__file__).resolve().parents[4]
MATRIX = REPO_ROOT / "outputs/silver_match_v3/mi_execution_matrix_20260713_v1.json"


def test_real_alltask_mi_execution_matrix_passes_hash_and_scope_audit():
    result = validate_matrix(MATRIX, REPO_ROOT)
    assert result["status"] == "PASS"
    assert result["tasks"] == 8
    assert result["corpora"] == 23
    assert result["canonical_norms"] == 1_732_515
    assert result["runnable_now_tasks"] == 0
    assert result["conditional_tasks"] == 7
    assert result["remote_certificate_files_reverified"] == 0
    assert result["production_correlations_run"] == 0


def _tampered(tmp_path: Path, mutate):
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    mutate(payload)
    output = tmp_path / "matrix.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def test_matrix_rejects_premature_production_readiness(tmp_path):
    path = _tampered(
        tmp_path,
        lambda payload: payload["tasks"][0].__setitem__("runnable_now", True),
    )
    with pytest.raises(ValueError, match="prematurely authorized"):
        validate_matrix(path, REPO_ROOT)


def test_matrix_rejects_missing_or_reordered_task(tmp_path):
    path = _tampered(
        tmp_path,
        lambda payload: payload["tasks"].reverse(),
    )
    with pytest.raises(ValueError, match="Humor must be first"):
        validate_matrix(path, REPO_ROOT)


def test_matrix_rejects_changed_local_evidence_hash(tmp_path):
    path = _tampered(
        tmp_path,
        lambda payload: payload["local_evidence"]["task_correlation_script"].__setitem__(
            "sha256", "0" * 64
        ),
    )
    with pytest.raises(ValueError, match="local evidence changed"):
        validate_matrix(path, REPO_ROOT)

