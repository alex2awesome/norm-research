import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file, write_jsonl
from scripts.tools.silver_match_v3.materialize_consensus_training_truth import materialize


def _fixture(tmp_path: Path, *, complete: bool = True, crossing: bool = False):
    pack = tmp_path / "pack"
    pack.mkdir()
    splits = [("train", "train"), ("dev", "dev"), ("blind", "test")]
    items = []
    for index, (role, split) in enumerate(splits):
        items.append({
            "norm_uid": f"u{index}", "task": "demo", "corpus": "c",
            "norm": f"n{index}", "collection_role": role, "split": split,
            "predeclared_split": split,
            "source_group": "same" if crossing and index < 2 else f"g{index}",
            "split_group": "same" if crossing and index < 2 else f"g{index}",
            "truth_hidden": True,
        })
    write_jsonl(pack / "items.jsonl", items)
    (pack / "bank.json").write_text(json.dumps({"metrics": [{"metric_id": "m"}]}) + "\n")
    validation = {
        "task": "demo",
        "bank_source_sha256": "bank-sha",
        "outputs": {
            "items": {"sha256": sha256_file(pack / "items.jsonl")},
            "bank": {"sha256": sha256_file(pack / "bank.json")},
        },
    }
    (pack / "validation.json").write_text(json.dumps(validation) + "\n")
    resolved = tmp_path / "resolved.jsonl"
    write_jsonl(resolved, [{
        "norm_uid": row["norm_uid"], "decision": "MATCH", "metric_id": "m",
        "confidence": "high", "reason": "r", "label_source": "two-pass",
        "agreement_sources": ["a", "b"],
    } for row in items])
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "schema_version": "silver-match-v3-exact-multi-pass-truth-report-v1",
        "complete": complete,
        "unresolved_count": 0 if complete else 1,
        "inputs": {"source_pack_validation": {"sha256": sha256_file(pack / "validation.json")}},
        "outputs": {"resolved": {"path": str(resolved), "sha256": sha256_file(resolved)}},
    }) + "\n")
    return pack, report


def test_restores_roles_and_never_trains_on_blind(tmp_path):
    pack, report = _fixture(tmp_path)
    manifest = materialize(pack, report, tmp_path / "out")
    assert manifest["split_counts"] == {"dev": 1, "test": 1, "train": 1}
    assert manifest["blind_rows_training_eligible"] == 0
    train = [json.loads(line) for line in (tmp_path / "out/truth.train.jsonl").read_text().splitlines()]
    test = [json.loads(line) for line in (tmp_path / "out/truth.test.jsonl").read_text().splitlines()]
    assert train[0]["training_eligible"] is True
    assert test[0]["training_eligible"] is False
    assert test[0]["blind_evaluation_only"] is True


def test_prefers_high_confidence_supporter_rationale(tmp_path):
    pack, report = _fixture(tmp_path)
    report_payload = json.loads(report.read_text())
    resolved_path = Path(report_payload["outputs"]["resolved"]["path"])
    rows = [json.loads(line) for line in resolved_path.read_text().splitlines()]
    for row in rows:
        row["agreement_sources"] = ["a", "b"]
        row["source_predictions"] = {
            "a": {"confidence": "medium", "reason": "medium reason"},
            "b": {"confidence": "high", "reason": "specific high reason"},
        }
    write_jsonl(resolved_path, rows)
    report_payload["outputs"]["resolved"]["sha256"] = sha256_file(resolved_path)
    report.write_text(json.dumps(report_payload) + "\n")
    materialize(pack, report, tmp_path / "out")
    train = json.loads((tmp_path / "out/truth.train.jsonl").read_text().splitlines()[0])
    assert train["reason"] == "specific high reason"
    assert train["teacher_reason_source"] == "b"


def test_requires_complete_consensus(tmp_path):
    pack, report = _fixture(tmp_path, complete=False)
    with pytest.raises(ValueError, match="not complete"):
        materialize(pack, report, tmp_path / "out")


def test_rejects_source_group_crossing(tmp_path):
    pack, report = _fixture(tmp_path, crossing=True)
    with pytest.raises(ValueError, match="cross frozen splits"):
        materialize(pack, report, tmp_path / "out")
