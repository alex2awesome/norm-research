from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file, write_jsonl
from scripts.tools.silver_match_v3.validate_independent_teacher_labels import main


def test_validator_accepts_explicit_dev_role_and_source_group_alias(
    tmp_path: Path, monkeypatch
) -> None:
    pack, raw = tmp_path / "pack", tmp_path / "raw"
    item = {
        "schema_version": "audit-v1",
        "norm_uid": "a" * 64,
        "corpus": "humor_multi",
        "task": "humor",
        "row": 3,
        "source_group": "post:3",
    }
    bank = {
        "task": "humor",
        "source_sha256": "bank-sha",
        "metrics": [{"metric_id": "m0", "name": "Metric"}],
    }
    write_jsonl(pack / "items.jsonl", [item])
    write_jsonl(pack / "chunks" / "part-000.jsonl", [item])
    (pack / "bank.json").write_text(json.dumps(bank))
    (pack / "validation.json").write_text(
        json.dumps(
            {
                "task": "humor",
                "bank_source_sha256": "bank-sha",
                "outputs": {
                    "items": {"sha256": sha256_file(pack / "items.jsonl")},
                    "bank": {"sha256": sha256_file(pack / "bank.json")},
                },
            }
        )
    )
    raw.mkdir()
    (raw / "part-000.json").write_text(
        json.dumps(
            {
                "task": "humor",
                "chunk_id": "part-000",
                "labels": [
                    {
                        "norm_uid": "a" * 64,
                        "decision": "MATCH",
                        "metric_id": "m0",
                        "confidence": "high",
                        "reason": "Exact criterion.",
                    }
                ],
            }
        )
    )
    transcript_audit = tmp_path / "transcript.audit.json"
    transcript_audit.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-isolated-labeler-transcript-audit-v1",
                "status": "PASS",
                "complete": True,
                "expected_chunks": 1,
                "audited_chunks": 1,
                "bank": {"sha256": sha256_file(pack / "bank.json")},
                "chunks": [
                    {
                        "chunk": "part-000",
                        "chunk_sha256": sha256_file(pack / "chunks" / "part-000.jsonl"),
                        "raw_label_sha256": sha256_file(raw / "part-000.json"),
                        "log_sha256": "log",
                        "command_count": 2,
                    }
                ],
                "violations": [],
            }
        )
    )
    output, report = tmp_path / "labels.jsonl", tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_independent_teacher_labels",
            "--pack-root",
            str(pack),
            "--raw-label-dir",
            str(raw),
            "--split-role",
            "dev",
            "--transcript-audit",
            str(transcript_audit),
            "--output",
            str(output),
            "--report",
            str(report),
        ],
    )
    main()
    row = next(read_jsonl(output))
    assert row["split_group"] == "post:3"
    assert row["split"] == "dev"
    validation = json.loads(report.read_text())
    assert validation["transcript_audit"]["status"] == "PASS"
