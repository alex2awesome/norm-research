import json
from pathlib import Path

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.materialize_full_bank_blind_candidates import (
    materialize,
)


def test_materializes_exact_full_bank_without_labels(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    items = pack / "items.jsonl"
    bank = pack / "bank.json"
    items.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3.0",
                "norm_uid": "u1",
                "corpus": "c",
                "task": "t",
                "row": 4,
                "norm": "human text",
            }
        )
        + "\n"
    )
    bank.write_text(
        json.dumps(
            {
                "task": "t",
                "source_sha256": "bank-sha",
                "metrics": [{"metric_id": "a2"}, {"metric_id": "a1"}],
            }
        )
        + "\n"
    )
    (pack / "validation.json").write_text(
        json.dumps(
            {
                "task": "t",
                "truth_hidden": True,
                "bank_source_sha256": "bank-sha",
                "outputs": {
                    "items": {"sha256": sha256_file(items)},
                    "bank": {"sha256": sha256_file(bank)},
                },
            }
        )
        + "\n"
    )
    output, report = tmp_path / "candidates.jsonl", tmp_path / "freeze.json"
    result = materialize(pack, output, report)
    row = json.loads(output.read_text())
    assert result["status"] == "FROZEN_BEFORE_INFERENCE"
    assert result["candidate_depth"] == 2
    assert [value["metric_id"] for value in row["candidates"]] == ["a2", "a1"]
    assert "decision" not in row and "metric_id" not in row
