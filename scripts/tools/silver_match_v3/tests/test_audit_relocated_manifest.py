import json

import pytest

from scripts.tools.silver_match_v3.audit_relocated_manifest import audit
from scripts.tools.silver_match_v3.common import sha256_file


def _manifests(tmp_path):
    canonical_root = tmp_path / "canonical"
    runtime_root = tmp_path / "runtime"
    canonical_root.mkdir()
    runtime_root.mkdir()
    for root in (canonical_root, runtime_root):
        (root / "bank.json").write_text('{"metrics": [{"metric_id": "m1"}]}')
        (root / "norms.jsonl").write_text('{"norm_uid": "u", "row": 0}\n')
    payload = {
        "schema_version": "x",
        "banks": {
            "t": {
                "path": str(canonical_root / "bank.json"),
                "sha256": sha256_file(canonical_root / "bank.json"),
                "count": 1,
            }
        },
        "corpora": {
            "c": {
                "path": str(canonical_root / "norms.jsonl"),
                "sha256": sha256_file(canonical_root / "norms.jsonl"),
                "task": "t",
                "count": 1,
            }
        },
    }
    source = tmp_path / "source.json"
    source.write_text(json.dumps(payload))
    payload["banks"]["t"]["path"] = str(runtime_root / "bank.json")
    payload["corpora"]["c"]["path"] = str(runtime_root / "norms.jsonl")
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps(payload))
    return source, runtime


def test_accepts_path_only_equivalent_manifest(tmp_path):
    source, runtime = _manifests(tmp_path)
    result = audit(source, runtime)
    assert result["artifact_count"] == 2
    assert result["all_artifact_hashes_equal"] is True


def test_rejects_scientific_metadata_change(tmp_path):
    source, runtime = _manifests(tmp_path)
    payload = json.loads(runtime.read_text())
    payload["corpora"]["c"]["count"] = 2
    runtime.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="scientific metadata"):
        audit(source, runtime)


def test_uses_source_inventory_for_bank_without_artifact_hash(tmp_path):
    source, runtime = _manifests(tmp_path)
    source_payload = json.loads(source.read_text())
    bank_sha = source_payload["banks"]["t"].pop("sha256")
    source.write_text(json.dumps(source_payload))
    runtime_payload = json.loads(runtime.read_text())
    runtime_payload["banks"]["t"].pop("sha256")
    runtime.write_text(json.dumps(runtime_payload))
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "source_manifest_sha256": sha256_file(source),
                "artifacts": [
                    {"section": "banks", "name": "t", "sha256": bank_sha}
                ],
            }
        )
    )
    result = audit(source, runtime, inventory)
    assert result["source_artifact_inventory"]["artifact_count"] == 1
