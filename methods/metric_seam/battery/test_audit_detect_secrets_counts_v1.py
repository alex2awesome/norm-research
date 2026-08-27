"""Tests for the content-suppressing independent secret-scan receipt."""

import json
from pathlib import Path
import stat
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory

import pytest

from methods.metric_seam.battery import audit_detect_secrets_counts_v1 as audit


def _fake_run(command, *, check, capture_output, text):
    assert check is False and capture_output is True and text is True
    if command[-1] == "--version":
        return CompletedProcess(command, 0, stdout="9.9.9\n", stderr="")
    payload = {
        "version": "9.9.9",
        "plugins": [{"name": "SyntheticDetector"}],
        "filters": [{"path": "synthetic-filter"}],
        "results": {
            command[-1]: [
                {"hashed_secret": "never-bank-this-detail", "line_number": 4},
                {"hashed_secret": "nor-this-detail", "line_number": 8},
            ]
        },
    }
    return CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")


def test_receipt_records_only_aggregate_count_and_versions(monkeypatch):
    with TemporaryDirectory() as directory:
        artifact = Path(directory) / "bundle.json"
        artifact.write_text("synthetic artifact with no live secret")
        monkeypatch.setattr(audit.shutil, "which", lambda _name: "/tool/detect-secrets")
        monkeypatch.setattr(audit.subprocess, "run", _fake_run)
        receipt = audit.build_receipt(artifact_path=artifact)
        encoded = json.dumps(receipt)
        assert receipt["aggregate_finding_count"] == 2
        assert receipt["scanner_version"] == "9.9.9"
        assert receipt["scan_passed"] is False
        assert receipt["counts_only"] is True
        assert "never-bank-this-detail" not in encoded
        assert '"line_number":' not in encoded


def test_subprocess_failure_never_surfaces_captured_channels(monkeypatch):
    with TemporaryDirectory() as directory:
        artifact = Path(directory) / "bundle.json"
        artifact.write_text("safe synthetic input")
        monkeypatch.setattr(audit.shutil, "which", lambda _name: "/tool/detect-secrets")

        def fail(command, *, check, capture_output, text):
            return CompletedProcess(
                command,
                2,
                stdout="never expose stdout content",
                stderr="never expose stderr content",
            )

        monkeypatch.setattr(audit.subprocess, "run", fail)
        with pytest.raises(RuntimeError) as caught:
            audit.build_receipt(artifact_path=artifact)
        message = str(caught.value)
        assert "stdout content" not in message
        assert "stderr content" not in message
        assert "hidden output" in message


def test_writer_is_read_only_and_refuses_overwrite(monkeypatch):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        artifact = root / "bundle.json"
        artifact.write_text("safe synthetic input")
        receipt_path = root / "receipt.json"
        monkeypatch.setattr(audit.shutil, "which", lambda _name: "/tool/detect-secrets")
        monkeypatch.setattr(audit.subprocess, "run", _fake_run)
        audit.write_receipt(artifact_path=artifact, receipt_path=receipt_path)
        assert stat.S_IMODE(receipt_path.stat().st_mode) & stat.S_IWUSR == 0
        with pytest.raises(FileExistsError):
            audit.write_receipt(artifact_path=artifact, receipt_path=receipt_path)
