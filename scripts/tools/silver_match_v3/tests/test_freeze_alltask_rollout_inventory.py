from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.freeze_alltask_rollout_inventory import (
    FINAL_DECISION_TAXONOMY,
    ROLLOUT_TASK_ORDER,
    freeze_alltask_rollout_inventory,
    main,
)


def _uid(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _synthetic_eight_task_manifest(tmp_path: Path) -> Path:
    banks = {}
    corpora = {}
    routing = {}
    total_norms = 0
    for task_index, task in enumerate(ROLLOUT_TASK_ORDER):
        source_sha = hashlib.sha256(f"source:{task}".encode()).hexdigest()
        bank_path = _write_json(
            tmp_path / "banks" / f"{task}.json",
            {
                "task": task,
                "source_path": f"/frozen/sources/{task}.json",
                "source_sha256": source_sha,
                "metrics": [
                    {
                        "task": task,
                        "metric_id": "a0",
                        "metric_index": 0,
                        "name": f"{task} metric",
                    }
                ],
            },
        )
        banks[task] = {
            "path": str(bank_path),
            "count": 1,
            "source_path": f"/frozen/sources/{task}.json",
            "source_sha256": source_sha,
        }
        corpus = f"corpus_{task_index}"
        norm_path = tmp_path / "norms" / f"{corpus}.jsonl"
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        norm_path.write_text(
            json.dumps(
                {
                    "norm_uid": _uid(corpus),
                    "corpus": corpus,
                    "task": task,
                    "row": 0,
                    "source_id": f"source-{task_index}",
                    "norm": f"explicit {task} criterion",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        corpora[corpus] = {
            "task": task,
            "path": str(norm_path),
            "count": 1,
            "source_paths": [f"/frozen/extractions/{corpus}.jsonl"],
            "source_sha256": {
                f"/frozen/extractions/{corpus}.jsonl": _uid(f"source:{corpus}")
            },
            "coverage_complete": True,
            "missing_optional_segments": [],
        }
        routing[corpus] = task
        total_norms += 1
    return _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": "silver-match-v3.0",
            "source_mode": "canonical",
            "aliases": {},
            "routing": routing,
            "total_norms": total_norms,
            "total_corpora": len(corpora),
            "total_tasks": len(banks),
            "banks": banks,
            "corpora": corpora,
        },
    )


def test_freezes_exact_eight_task_scope_without_claiming_readiness(
    tmp_path: Path,
) -> None:
    manifest = _synthetic_eight_task_manifest(tmp_path)
    before = {
        path: sha256_file(path)
        for path in [
            manifest,
            *sorted(tmp_path.glob("banks/*")),
            *sorted(tmp_path.glob("norms/*")),
        ]
    }

    result = freeze_alltask_rollout_inventory(manifest)

    assert result["rollout_order"] == list(ROLLOUT_TASK_ORDER)
    assert list(result["tasks"]) == list(ROLLOUT_TASK_ORDER)
    assert result["rollout_order"][0] == "humor"
    assert result["rollout_order"][-1] == "notice-and-comment"
    assert result["totals"] == {
        "tasks": 8,
        "corpora": 8,
        "norms": 8,
        "metrics": 8,
    }
    assert result["release_ready"] is False
    assert result["readiness_evidence_evaluated"] is False
    assert len(result["scope_sha256"]) == 64
    assert result["final_decision_taxonomy"] == list(FINAL_DECISION_TAXONOMY)
    for task, task_record in result["tasks"].items():
        assert len(task_record["scope_sha256"]) == 64
        assert task_record["bank"]["sha256"] == sha256_file(
            Path(task_record["bank"]["path"])
        )
        assert task_record["required_artifacts"]["release"]["status"] == (
            "REQUIRED_NOT_EVALUATED"
        )
        assert task_record["norm_count"] == 1
    assert before == {path: sha256_file(path) for path in before}


def test_rejects_missing_task(tmp_path: Path) -> None:
    manifest = _synthetic_eight_task_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["banks"]["peer-review"]
    payload["total_tasks"] -= 1
    _write_json(manifest, payload)

    with pytest.raises(ValueError, match="manifest task set mismatch"):
        freeze_alltask_rollout_inventory(manifest)


def test_rejects_foreign_or_missing_corpus_routing(tmp_path: Path) -> None:
    manifest = _synthetic_eight_task_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["routing"]["corpus_0"] = "code-review"
    _write_json(manifest, payload)

    with pytest.raises(ValueError, match="manifest routing mismatch"):
        freeze_alltask_rollout_inventory(manifest)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["routing"]["corpus_0"]
    _write_json(manifest, payload)
    with pytest.raises(ValueError, match="routing/corpus set mismatch"):
        freeze_alltask_rollout_inventory(manifest)


def test_rejects_duplicate_corpus_artifact_routing(tmp_path: Path) -> None:
    manifest = _synthetic_eight_task_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["corpora"]["corpus_1"]["path"] = payload["corpora"]["corpus_0"]["path"]
    _write_json(manifest, payload)

    with pytest.raises(ValueError, match="duplicate canonical artifact path"):
        freeze_alltask_rollout_inventory(manifest)


def test_rejects_duplicate_json_corpus_routing_key(tmp_path: Path) -> None:
    manifest = _synthetic_eight_task_manifest(tmp_path)
    text = manifest.read_text(encoding="utf-8")
    text = text.replace(
        '"corpus_0": "humor"',
        '"corpus_0": "humor", "corpus_0": "code-review"',
        1,
    )
    manifest.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key 'corpus_0'"):
        freeze_alltask_rollout_inventory(manifest)


def test_rejects_foreign_row_and_duplicate_global_uid(tmp_path: Path) -> None:
    manifest = _synthetic_eight_task_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    norm_path = Path(payload["corpora"]["corpus_2"]["path"])
    row = json.loads(norm_path.read_text(encoding="utf-8"))
    row["corpus"] = "corpus_3"
    norm_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="foreign corpus routing"):
        freeze_alltask_rollout_inventory(manifest)

    manifest = _synthetic_eight_task_manifest(tmp_path / "duplicate_uid")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    first = json.loads(
        Path(payload["corpora"]["corpus_0"]["path"]).read_text(encoding="utf-8")
    )
    second_path = Path(payload["corpora"]["corpus_1"]["path"])
    second = json.loads(second_path.read_text(encoding="utf-8"))
    second["norm_uid"] = first["norm_uid"]
    second_path.write_text(json.dumps(second) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate global norm_uid"):
        freeze_alltask_rollout_inventory(manifest)


def test_rejects_bank_source_or_bank_payload_drift(tmp_path: Path) -> None:
    manifest = _synthetic_eight_task_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    bank_path = Path(payload["banks"]["humor"]["path"])
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bank["source_sha256"] = "0" * 64
    _write_json(bank_path, bank)

    with pytest.raises(ValueError, match="bank source provenance mismatch"):
        freeze_alltask_rollout_inventory(manifest)


def test_cli_writes_once_into_existing_parent(tmp_path: Path, monkeypatch) -> None:
    manifest = _synthetic_eight_task_manifest(tmp_path)
    output = tmp_path / "existing" / "rollout.json"
    output.parent.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_alltask_rollout_inventory.py",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )

    main()

    assert json.loads(output.read_text(encoding="utf-8"))["scope_frozen"] is True
    with pytest.raises(FileExistsError):
        main()
