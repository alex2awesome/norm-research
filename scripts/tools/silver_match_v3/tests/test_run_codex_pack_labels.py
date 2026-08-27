from __future__ import annotations

import json
from pathlib import Path

from scripts.tools.silver_match_v3.run_codex_pack_labels import (
    archive_invalid_output,
    valid_existing,
)


def test_variable_chunk_schema_supports_blind_audit_chunks() -> None:
    schema_path = (
        Path(__file__).parents[1] / "schemas" / "independent_labels_1_to_25.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    labels = schema["properties"]["labels"]
    assert labels["minItems"] == 1
    assert labels["maxItems"] == 25
    assert labels["items"]["properties"]["norm_uid"]["pattern"] == "^[0-9a-f]{64}$"


def test_existing_output_still_requires_exact_runtime_chunk_count(
    tmp_path: Path,
) -> None:
    output = tmp_path / "part-000.json"
    output.write_text(
        json.dumps(
            {
                "task": "peer-review",
                "chunk_id": "part-000",
                "labels": [{"norm_uid": f"{index:064x}"} for index in range(20)],
            }
        )
    )
    expected = [f"{index:064x}" for index in range(20)]
    assert valid_existing(output, "peer-review", "part-000", expected)
    assert not valid_existing(output, "peer-review", "part-000", expected[:-1])
    corrupted = list(expected)
    corrupted[-1] = "f" * 64
    assert not valid_existing(output, "peer-review", "part-000", corrupted)


def test_invalid_raw_output_and_log_are_archived_before_retry(tmp_path: Path) -> None:
    output, log, archive = (
        tmp_path / "raw" / "part-000.json",
        tmp_path / "logs" / "part-000.log",
        tmp_path / "invalid",
    )
    output.parent.mkdir()
    log.parent.mkdir()
    output.write_text('{"invalid": true}\n')
    log.write_text("old log\n")
    archive_invalid_output(output, log, archive)
    assert not output.exists()
    assert not log.exists()
    archived_json = list(archive.glob("part-000.*.json"))
    archived_log = list(archive.glob("part-000.*.log"))
    assert len(archived_json) == len(archived_log) == 1
    assert archived_json[0].read_text() == '{"invalid": true}\n'
    assert archived_log[0].read_text() == "old log\n"


def test_timeout_log_is_archived_even_without_raw_output(tmp_path: Path) -> None:
    output, log, archive = (
        tmp_path / "raw" / "part-000.json",
        tmp_path / "logs" / "part-000.log",
        tmp_path / "invalid",
    )
    log.parent.mkdir()
    log.write_text("timed out\n")
    archive_invalid_output(output, log, archive)
    assert not log.exists()
    archived = list(archive.glob("part-000.*.log"))
    assert len(archived) == 1
    assert archived[0].read_text() == "timed out\n"
