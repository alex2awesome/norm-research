from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.tools.silver_match_v3.recover_humor_ce_status import parse_args, run
from scripts.tools.silver_match_v3.remote_recovery_core import (
    MODEL,
    PILOTS,
    REPORT_SCHEMA,
    classify_pilot,
)


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _running_fixture(base: Path, index: int = 0) -> tuple[Path, dict[str, str]]:
    spec = PILOTS[index]
    root = spec.root(base)
    split_sha = _write(root / "split_assignments.jsonl", {"split": "train"})
    config = {
        "schema_version": REPORT_SCHEMA,
        "model": str(MODEL),
        "seed": 20260713,
        "max_length": 1024,
        "exposure_budgets": [10000, 25000, 50000],
        "lora_learning_rate": spec.learning_rate,
        "head_learning_rate": 1e-3,
        "attention": "eager",
        "lora": {"rank": spec.rank, "alpha": spec.alpha},
        "split_assignments_sha256": split_sha,
    }
    config_sha = _write(root / "run_config.json", config)
    _write(root / "events.jsonl", {"event": "RUN_STARTED"})
    return root, {"config": config_sha, "split": split_sha}


def _artifact_rows(root: Path, name: str) -> list[dict[str, object]]:
    paths = {
        "run_config": (root / "run_config.json", "full"),
        "training_report": (root / "training_report.json", "full"),
        "reload_verification": (root / "reload_verification.json", "full"),
        "events": (root / "events.jsonl", "tail"),
        "split_assignments": (root / "split_assignments.jsonl", "hash_only"),
        "log": (root.parent / "logs" / f"{name}.log", "tail"),
    }
    rows = []
    for key, (path, mode) in paths.items():
        row: dict[str, object] = {
            "key": key,
            "path": str(path),
            "mode": mode,
            "exists": path.is_file(),
        }
        if path.is_file():
            data = path.read_bytes()
            row.update({"size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            if mode in {"full", "tail"}:
                row["content"] = data
        rows.append(row)
    return rows


def test_classifies_complete_running_failed_and_absent(tmp_path: Path) -> None:
    running_root, hashes = _running_fixture(tmp_path, 0)
    running = classify_pilot(
        PILOTS[0], tmp_path, True, _artifact_rows(running_root, PILOTS[0].name)
    )
    assert running["status"] == "RUNNING"

    reload_value = {
        "status": "PASS",
        "selected_checkpoint": str(running_root / "checkpoint-25000"),
    }
    _write(running_root / "reload_verification.json", reload_value)
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "COMPLETE",
        "reload_verification": reload_value,
        "input_sha256": {"run_config": hashes["config"]},
    }
    report_sha = _write(running_root / "training_report.json", report)
    (running_root / "events.jsonl").write_text(
        json.dumps({"event": "RUN_STARTED"})
        + "\n"
        + json.dumps({"event": "RUN_COMPLETE", "training_report_sha256": report_sha})
        + "\n"
    )
    complete = classify_pilot(
        PILOTS[0], tmp_path, True, _artifact_rows(running_root, PILOTS[0].name)
    )
    assert complete["status"] == "COMPLETE", complete

    failed_root, _ = _running_fixture(tmp_path, 1)
    with (failed_root / "events.jsonl").open("a") as handle:
        handle.write(json.dumps({"event": "RUN_FAILED", "error": "boom"}) + "\n")
    failed = classify_pilot(
        PILOTS[1], tmp_path, True, _artifact_rows(failed_root, PILOTS[1].name)
    )
    assert failed["status"] == "FAILED"

    absent_root = tmp_path / "missing"
    absent = classify_pilot(PILOTS[0], absent_root, False, [])
    assert absent["status"] == "ABSENT"


def test_local_fixture_emits_content_addressed_report_without_checkpoints(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    _running_fixture(fixture, 0)
    output = tmp_path / "out"
    args = parse_args(
        [
            "--local-fixture-root",
            str(fixture),
            "--output-root",
            str(output),
            "--pilot-host",
            "sk2",
            "--gpu-host",
            "sk2",
        ]
    )
    report, path = run(args)
    assert path is not None and path.is_file()
    assert path.name == f"status.{hashlib.sha256(path.read_bytes()).hexdigest()}.json"
    assert report["status_counts"] == {
        "COMPLETE": 0,
        "RUNNING": 1,
        "FAILED": 0,
        "ABSENT": 1,
    }
    assert report["checkpoints_read_or_copied"] is False
    assert report["mutation_attempted"] is False
    assert not any(
        "checkpoint" in item.name for item in (output / "artifacts").iterdir()
    )


def test_dry_run_makes_no_writes_and_lists_exact_recipes(tmp_path: Path) -> None:
    output = tmp_path / "never-created"
    args = parse_args(["--dry-run", "--output-root", str(output)])
    report, path = run(args)
    assert path is None
    assert not output.exists()
    assert [row["expected_gpu"] for row in report["pilots"]] == [2, 3]
    assert [Path(row["expected_root"]).name for row in report["pilots"]] == [
        spec.name for spec in PILOTS
    ]
    assert report["forbidden"] == {"host": "sk3", "gpu_indices": [1, 2, 3, 4]}
    assert report["mutation_commands"] == []
