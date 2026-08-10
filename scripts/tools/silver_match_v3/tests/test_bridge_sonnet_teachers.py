import json
from argparse import Namespace
from pathlib import Path

from scripts.tools.silver_match_v3.bridge_sonnet_teachers import bridge
from scripts.tools.silver_match_v3.common import stable_uid


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_strict_alias_row_unique_and_ambiguous_bridge(tmp_path):
    calibration = tmp_path / "calibration"
    production = tmp_path / "production"
    output = tmp_path / "output"
    legacy_rows = [
        "same row norm",
        "unique moved norm",
        "ambiguous repeated norm",
        "missing norm",
    ]
    calibration_rows = [
        {
            "schema_version": "silver-match-v3.0",
            "norm_uid": stable_uid("humor", row, norm),
            "corpus": "humor",
            "task": "humor",
            "row": row,
            "source_id": "",
            "norm": norm,
        }
        for row, norm in enumerate(legacy_rows)
    ]
    _jsonl(calibration / "norms/humor.jsonl", calibration_rows)

    production_norms = [
        "same row norm",
        "unrelated",
        "ambiguous repeated norm",
        "unique moved norm",
        "ambiguous repeated norm",
    ]
    _jsonl(
        production / "norms/humor_multi.jsonl",
        [
            {
                "schema_version": "silver-match-v3.0",
                "norm_uid": f"production-{row}",
                "corpus": "humor_multi",
                "task": "humor",
                "row": row,
                "source_id": f"source-{row}",
                "norm": norm,
            }
            for row, norm in enumerate(production_norms)
        ],
    )
    bank = {
        "schema_version": "silver-match-v3.0",
        "task": "humor",
        "source_sha256": "bank-sha",
        "metrics": [{"metric_id": "a0", "name": "Metric Zero"}],
    }
    (production / "banks").mkdir(parents=True)
    (production / "banks/humor.json").write_text(json.dumps(bank), encoding="utf-8")
    teacher_path = tmp_path / "sonnet.jsonl"
    _jsonl(
        teacher_path,
        [
            {
                "schema_version": "silver-match-v3.0",
                "norm_uid": row["norm_uid"],
                "corpus": "humor",
                "task": "humor",
                "row": row["row"],
                "decision": "MATCH",
                "metric_id": "a0",
                "current_bank_source_sha256": "bank-sha",
                "label_source": "sonnet_audit",
                "confidence": "high",
                "notes": {"current_metric_name": "Metric Zero"},
            }
            for row in calibration_rows
        ],
    )
    summary = bridge(
        Namespace(
            teachers=str(teacher_path),
            calibration_root=str(calibration),
            production_root=str(production),
            output_root=str(output),
        )
    )
    teachers = [
        json.loads(line)
        for line in (output / "teachers/sonnet.production.jsonl").read_text().splitlines()
    ]
    rejected = [
        json.loads(line)
        for line in (
            output / "teachers/sonnet.production_bridge_rejections.jsonl"
        ).read_text().splitlines()
    ]
    assert summary["production_teachers"] == 2
    assert {row["norm_uid"] for row in teachers} == {"production-0", "production-3"}
    assert {row["production_uid_bridge_method"] for row in teachers} == {
        "alias_row_and_unique_norm_exact",
        "alias_unique_norm_exact",
    }
    assert all(row["corpus"] == "humor_multi" for row in teachers)
    assert all(row["uid_universe"] == "production_canonical_v3" for row in teachers)
    assert {row["reason"] for row in rejected} == {
        "ambiguous_production_norm",
        "production_norm_missing",
    }


def test_source_identity_does_not_fallback_to_text_only(tmp_path):
    calibration = tmp_path / "calibration"
    production = tmp_path / "production"
    output = tmp_path / "output"
    norm = "identical text"
    old_uid = stable_uid("press_releases", 0, norm)
    _jsonl(
        calibration / "norms/press_releases.jsonl",
        [
            {
                "schema_version": "silver-match-v3.0",
                "norm_uid": old_uid,
                "corpus": "press_releases",
                "task": "press-releases",
                "row": 0,
                "source_id": "expected-source",
                "norm": norm,
            }
        ],
    )
    _jsonl(
        production / "norms/press_releases.jsonl",
        [
            {
                "schema_version": "silver-match-v3.0",
                "norm_uid": "production-0",
                "corpus": "press_releases",
                "task": "press-releases",
                "row": 0,
                "source_id": "different-source",
                "norm": norm,
            }
        ],
    )
    (production / "banks").mkdir(parents=True)
    (production / "banks/press-releases.json").write_text(
        json.dumps(
            {
                "source_sha256": "bank-sha",
                "metrics": [{"metric_id": "a0", "name": "Metric"}],
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "teachers.jsonl"
    _jsonl(
        source,
        [
            {
                "norm_uid": old_uid,
                "corpus": "press_releases",
                "task": "press-releases",
                "row": 0,
                "decision": "MATCH",
                "metric_id": "a0",
                "current_bank_source_sha256": "bank-sha",
                "label_source": "sonnet_full",
                "notes": {"current_metric_name": "Metric"},
            }
        ],
    )
    summary = bridge(
        Namespace(
            teachers=str(source),
            calibration_root=str(calibration),
            production_root=str(production),
            output_root=str(output),
        )
    )
    assert summary["production_teachers"] == 0
    assert summary["rejections_by_reason"] == {"production_norm_missing": 1}
