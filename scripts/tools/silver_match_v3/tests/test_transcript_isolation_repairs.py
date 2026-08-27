import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file, write_jsonl
from scripts.tools.silver_match_v3.prepare_transcript_isolation_repair_pack import build
from scripts.tools.silver_match_v3.promote_transcript_isolation_repairs import promote


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_source(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "source"
    rows = [
        {
            "schema_version": "silver-match-v3.0",
            "norm_uid": "u0",
            "corpus": "c",
            "task": "code-review",
            "row": 0,
            "split": "dev",
            "source_group": "g0",
        },
        {
            "schema_version": "silver-match-v3.0",
            "norm_uid": "u1",
            "corpus": "c",
            "task": "code-review",
            "row": 1,
            "split": "dev",
            "source_group": "g1",
        },
    ]
    write_jsonl(root / "items.jsonl", rows)
    write_jsonl(root / "chunks" / "part-000.jsonl", [rows[0]])
    write_jsonl(root / "chunks" / "part-001.jsonl", [rows[1]])
    dump(
        root / "bank.json",
        {
            "task": "code-review",
            "source_sha256": "bank-source",
            "metrics": [{"metric_id": "m0", "name": "M", "description": "D"}],
        },
    )
    dump(
        root / "validation.json",
        {
            "task": "code-review",
            "bank_source_sha256": "bank-source",
            "outputs": {
                "items": {"sha256": sha256_file(root / "items.jsonl")},
                "bank": {"sha256": sha256_file(root / "bank.json")},
            },
        },
    )
    audit = tmp_path / "failed_audit.json"
    dump(
        audit,
        {
            "schema_version": "silver-match-v3-isolated-labeler-transcript-audit-v1",
            "status": "FAIL",
            "complete": False,
            "bank": {"sha256": sha256_file(root / "bank.json")},
            "chunks": [
                {
                    "chunk": name,
                    "chunk_sha256": sha256_file(root / "chunks" / f"{name}.jsonl"),
                    "raw_label_sha256": f"raw-{name}",
                    "log_sha256": f"log-{name}",
                    "command_count": 1,
                }
                for name in ("part-000", "part-001")
            ],
            "violations": [
                {
                    "chunk": "part-001",
                    "kind": "UNAPPROVED_EXECUTION",
                    "detail": "rg used",
                }
            ],
        },
    )
    return root, audit


def label(uid: str, reason: str) -> dict:
    return {
        "schema_version": "silver-match-v3.0",
        "norm_uid": uid,
        "corpus": "c",
        "task": "code-review",
        "row": int(uid[-1]),
        "split_group": f"g{uid[-1]}",
        "split": "dev",
        "decision": "NO_EXPLICIT_CRITERION",
        "metric_id": None,
        "current_bank_source_sha256": "bank-source",
        "confidence": "high",
        "reason": reason,
    }


def test_builds_and_promotes_only_audit_selected_repairs(tmp_path: Path) -> None:
    source, failed_audit = make_source(tmp_path)
    repair = tmp_path / "repair"
    result = build(source, failed_audit, repair)
    assert result["selected_chunks"] == ["part-001"]
    assert result["label_content_read_for_selection"] is False
    assert [row["norm_uid"] for row in map(json.loads, (repair / "items.jsonl").read_text().splitlines())] == ["u1"]

    base_labels = tmp_path / "base.labels.jsonl"
    repair_labels = tmp_path / "repair.labels.jsonl"
    write_jsonl(base_labels, [label("u0", "base clean"), label("u1", "base excluded")])
    write_jsonl(repair_labels, [label("u1", "repair accepted")])
    repair_audit = tmp_path / "repair.audit.json"
    dump(
        repair_audit,
        {
            "schema_version": "silver-match-v3-isolated-labeler-transcript-audit-v1",
            "status": "PASS",
            "complete": True,
            "bank": {"sha256": sha256_file(repair / "bank.json")},
            "chunks": [
                {
                    "chunk": "part-001",
                    "chunk_sha256": sha256_file(repair / "chunks" / "part-001.jsonl"),
                    "raw_label_sha256": "repair-raw",
                    "log_sha256": "repair-log",
                    "command_count": 1,
                }
            ],
            "violations": [],
        },
    )
    base_validation = tmp_path / "base.validation.json"
    repair_validation = tmp_path / "repair.validation.json"
    dump(
        base_validation,
        {
            "schema_version": "silver-match-v3-independent-label-validation-v1",
            "complete": True,
            "pack_validation": {"sha256": sha256_file(source / "validation.json")},
            "output": {"sha256": sha256_file(base_labels)},
        },
    )
    dump(
        repair_validation,
        {
            "schema_version": "silver-match-v3-independent-label-validation-v1",
            "complete": True,
            "pack_validation": {"sha256": sha256_file(repair / "validation.json")},
            "transcript_audit": {"sha256": sha256_file(repair_audit)},
            "output": {"sha256": sha256_file(repair_labels)},
        },
    )
    output, report = tmp_path / "promoted.jsonl", tmp_path / "promotion.json"
    promoted = promote(
        source_pack=source,
        base_labels=base_labels,
        base_validation_path=base_validation,
        failed_audit_path=failed_audit,
        repair_pack=repair,
        repair_labels=repair_labels,
        repair_validation_path=repair_validation,
        repair_audit_path=repair_audit,
        output=output,
        report_path=report,
    )
    rows = list(map(json.loads, output.read_text().splitlines()))
    assert promoted["status"] == "PASS_COMPOSITE_TRANSCRIPT_CLEAN_LABELS"
    assert promoted["excluded_failed_base_uid_count"] == 1
    assert [row["reason"] for row in rows] == ["base clean", "repair accepted"]
    assert rows[0]["transcript_acceptance"]["source"] == "original_clean_chunk"
    assert rows[1]["transcript_acceptance"]["source"] == "audit_selected_isolation_repair"


def test_promotion_rejects_failed_repair_audit(tmp_path: Path) -> None:
    source, failed_audit = make_source(tmp_path)
    repair = tmp_path / "repair"
    build(source, failed_audit, repair)
    base_labels, repair_labels = tmp_path / "base.jsonl", tmp_path / "repair.jsonl"
    write_jsonl(base_labels, [label("u0", "a"), label("u1", "b")])
    write_jsonl(repair_labels, [label("u1", "c")])
    bad_audit = tmp_path / "bad_repair_audit.json"
    dump(
        bad_audit,
        {
            "schema_version": "silver-match-v3-isolated-labeler-transcript-audit-v1",
            "status": "FAIL",
            "complete": False,
            "bank": {"sha256": sha256_file(repair / "bank.json")},
            "chunks": [],
            "violations": [{"chunk": "part-001", "kind": "UNAPPROVED_EXECUTION"}],
        },
    )
    base_validation, repair_validation = tmp_path / "base.validation.json", tmp_path / "repair.validation.json"
    dump(
        base_validation,
        {
            "schema_version": "silver-match-v3-independent-label-validation-v1",
            "complete": True,
            "pack_validation": {"sha256": sha256_file(source / "validation.json")},
            "output": {"sha256": sha256_file(base_labels)},
        },
    )
    dump(
        repair_validation,
        {
            "schema_version": "silver-match-v3-independent-label-validation-v1",
            "complete": True,
            "pack_validation": {"sha256": sha256_file(repair / "validation.json")},
            "transcript_audit": {"sha256": sha256_file(bad_audit)},
            "output": {"sha256": sha256_file(repair_labels)},
        },
    )
    with pytest.raises(ValueError, match="repair transcript audit"):
        promote(
            source_pack=source,
            base_labels=base_labels,
            base_validation_path=base_validation,
            failed_audit_path=failed_audit,
            repair_pack=repair,
            repair_labels=repair_labels,
            repair_validation_path=repair_validation,
            repair_audit_path=bad_audit,
            output=tmp_path / "out.jsonl",
            report_path=tmp_path / "out.json",
        )
