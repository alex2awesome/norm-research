#!/usr/bin/env python3
"""Durably watch and execute the frozen Humor overlay-to-MI handoff on sk2."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import sha256_file


ROOT = Path("/lfs/skampere2/0/alexspan/norm-research-silver-v3/runtime/humor_ce_binary_v1")
RUN = ROOT / "production_reducer/humor-k200-minus-joined22090.seed-2026071502.exposure100k.v1"
INPUTS = ROOT / "production_reducer/inputs_v1"
RELEASE = ROOT / "releases/humor_binary_typed_v1"
AUDIT_ROOT = ROOT / "audits/humor_postblind_hybrid_risk_v1"
PREDICTIONS = RUN / "production_predictions.normalized.55288.jsonl"
MANIFEST = INPUTS / "manifest.sk2.json"
TRUTH = INPUTS / "truth.joined.all.jsonl"
CERTIFICATE = ROOT / "analysis_inputs/mi_certificates/humor.json"
RISK = AUDIT_ROOT / "inputs/POSTBLIND_DEPLOYMENT_RISK.json"
SAMPLE_ROOT = AUDIT_ROOT / "sample_v1"
PLAN = RELEASE / "humor.production-plan.frozen.json"
FINAL = RELEASE / "humor_multi.final.canonical.jsonl"
FINAL_AUDIT = RELEASE / "humor_multi.final.audit.json"
OVERLAY_REPORT = RELEASE / "humor_multi.overlay.report.json"
TASK_RELEASE = RELEASE / "humor.task-analysis-release.json"
MI_HANDOFF = RELEASE / "humor.mi-handoff.json"
MI_OUTPUT = RELEASE / "mi/humor.mi-validation.json"
STATE_ROOT = ROOT / "handoff/humor_postdeployment_v1"
EVENTS = STATE_ROOT / "events.jsonl"
COMPLETE = STATE_ROOT / "COMPLETE.json"
FAILED = STATE_ROOT / "FAILED.json"

MANIFEST_SHA = "e235d11d4748412ca90f90018ea80fd676087dd32544b0d6b40ffcb88ccdb029"
TRUTH_SHA = "1e579269c92f87de6896f1320cb868868b6be16a65c6bfa51530ded7da6d1906"
CERTIFICATE_SHA = "c18a765d68529a9986a02c41012d54f4672deac790898d27dac57010b1b014c3"
RISK_SHA = "e867e8fb7ae04d53d6450ca2fb295636ce4d2c761e157a292daf750a23b1a35a"
BANK_SOURCE_SHA = "1b4a29d34b4ef4d999e0cb0b2d1125286372349ff6dfa21a6adc5bc8e76f0de9"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(event: str, **values: Any) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": _now(), "event": event, **values}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _run(module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    _append("COMMAND_START", command=command)
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    _append(
        "COMMAND_END",
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout[-4000:],
        stderr=completed.stderr[-4000:],
    )
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")


def _require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{label} missing or SHA mismatch: {path}")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_plan() -> None:
    payload = {
        "schema_version": "silver-match-v3-humor-production-plan-v1",
        "status": "FROZEN_READY_FOR_UNLABELED_PRODUCTION",
        "task": "humor",
        "manifest": {"path": str(MANIFEST), "sha256": MANIFEST_SHA},
        "bank_source_sha256": BANK_SOURCE_SHA,
        "canonical_rows": 77378,
        "analysis_excluded_rows": 22090,
        "production_rows": 55288,
        "analysis_firewall": {
            "matcher_is_immutable": True,
            "may_tune_matcher_from_mi": False,
            "truth_uids_excluded_from_mi": True,
        },
    }
    if PLAN.exists():
        if _json(PLAN) != payload:
            raise ValueError("existing frozen production plan differs")
    else:
        _write_new(PLAN, payload)
    _append("PLAN_FROZEN", path=str(PLAN), sha256=sha256_file(PLAN))


def _wait_for_predictions() -> str:
    last_notice = 0.0
    previous: tuple[int, int] | None = None
    while True:
        if PREDICTIONS.is_file():
            stat = PREDICTIONS.stat()
            current = (stat.st_size, stat.st_mtime_ns)
            if current == previous and stat.st_size > 0:
                with PREDICTIONS.open("rb") as handle:
                    handle.seek(-1, os.SEEK_END)
                    terminal_newline = handle.read(1) == b"\n"
                if terminal_newline:
                    rows = sum(1 for _ in PREDICTIONS.open("rb"))
                    if rows == 55288:
                        observed = sha256_file(PREDICTIONS)
                        _append(
                            "PREDICTIONS_SEALED",
                            path=str(PREDICTIONS),
                            rows=rows,
                            bytes=stat.st_size,
                            sha256=observed,
                        )
                        return observed
                    if rows > 55288:
                        raise ValueError(f"normalized predictions exceed 55,288 rows: {rows}")
            previous = current
        if time.monotonic() - last_notice >= 600:
            _append("WAITING_FOR_NORMALIZED_55288", path=str(PREDICTIONS))
            last_notice = time.monotonic()
        time.sleep(30)


def _overlay(prediction_sha: str) -> None:
    paths = [FINAL, FINAL_AUDIT, OVERLAY_REPORT]
    if not any(path.exists() for path in paths):
        _run(
            "scripts.tools.silver_match_v3.merge_humor_truth_production_overlay",
            "--manifest", str(MANIFEST),
            "--manifest-sha256", MANIFEST_SHA,
            "--truth", str(TRUTH),
            "--truth-sha256", TRUTH_SHA,
            "--predictions", str(PREDICTIONS),
            "--predictions-sha256", prediction_sha,
            "--output", str(FINAL),
            "--audit-output", str(FINAL_AUDIT),
            "--report-output", str(OVERLAY_REPORT),
        )
    if not all(path.is_file() for path in paths):
        raise ValueError("partial canonical overlay artifacts exist")
    report = _json(OVERLAY_REPORT)
    audit = _json(FINAL_AUDIT)
    if (
        report.get("status") != "COMPLETE_CANONICAL_AUTHORITATIVE_OVERLAY_AUDITED"
        or int((report.get("output") or {}).get("count", -1)) != 77378
        or (report.get("output") or {}).get("sha256") != sha256_file(FINAL)
        or audit.get("complete") is not True
        or int(audit.get("audited_rows", -1)) != 77378
    ):
        raise ValueError("canonical overlay validation failed")
    _append("CANONICAL_OVERLAY_AUDITED", final_sha256=sha256_file(FINAL))


def _sample_postproduction_risk() -> None:
    report_path = SAMPLE_ROOT / "REPORT.json"
    if not SAMPLE_ROOT.exists():
        _run(
            "scripts.tools.silver_match_v3.freeze_humor_postproduction_risk_audit",
            "sample-production",
            "--risk-record", str(RISK),
            "--production-predictions", str(PREDICTIONS),
            "--output-root", str(SAMPLE_ROOT),
        )
    if not report_path.is_file():
        raise ValueError("postproduction risk sample report missing")
    report = _json(report_path)
    if (
        report.get("status") != "COMPLETE_CREATE_ONLY_PRODUCTION_AUDIT_SAMPLE"
        or int(report.get("blind_or_test_rows_read", -1)) != 0
        or (report.get("production_predictions") or {}).get("sha256")
        != sha256_file(PREDICTIONS)
    ):
        raise ValueError("postproduction risk sample validation failed")
    _append("POSTPRODUCTION_RISK_SAMPLE_FROZEN", selected=report.get("selected"))


def _freeze_release() -> None:
    if not TASK_RELEASE.exists() and not MI_HANDOFF.exists():
        _run(
            "scripts.tools.silver_match_v3.freeze_humor_overlay_analysis_release",
            "--manifest", str(MANIFEST),
            "--plan", str(PLAN),
            "--final", str(FINAL),
            "--final-audit", str(FINAL_AUDIT),
            "--overlay-report", str(OVERLAY_REPORT),
            "--production-predictions", str(PREDICTIONS),
            "--analysis-exclusion", str(TRUTH),
            "--risk-record", str(RISK),
            "--production-audit-report", str(SAMPLE_ROOT / "REPORT.json"),
            "--mi-certificate", str(CERTIFICATE),
            "--mi-certificate-sha256", CERTIFICATE_SHA,
            "--output", str(TASK_RELEASE),
            "--mi-handoff-output", str(MI_HANDOFF),
        )
    if not TASK_RELEASE.is_file() or not MI_HANDOFF.is_file():
        raise ValueError("partial task release/MI handoff exists")
    release = _json(TASK_RELEASE)
    if (
        release.get("status") != "TASK_FROZEN_ANALYSIS_READY"
        or int((release.get("analysis_exclusions") or {}).get("count", -1)) != 22090
        or (release.get("coverage") or {}).get("analysis_eligible_rows") != 55288
        or release.get("precision_claim_supported") is not False
        or release.get("false_abstention_claim_supported") is not False
    ):
        raise ValueError("task analysis release validation failed")
    _append("TASK_ANALYSIS_RELEASE_FROZEN", release_sha256=sha256_file(TASK_RELEASE))


def _run_mi() -> None:
    if not MI_OUTPUT.exists():
        _run(
            "scripts.tools.silver_match_v3.silver_mi_validation_v3",
            "--release", str(TASK_RELEASE),
            "--certificate", str(CERTIFICATE),
            "--n-permutations", "2000",
            "--n-bootstrap", "1000",
            "--seed", "1729",
            "--output", str(MI_OUTPUT),
        )
    report = _json(MI_OUTPUT)
    if (
        report.get("status") != "TASK_FROZEN_ANALYSIS"
        or (report.get("certificate") or {}).get("sha256") != CERTIFICATE_SHA
        or int((report.get("certificate") or {}).get("joined_metrics", -1)) != 285
        or len(report.get("per_metric") or []) != 285
        or int(report.get("analysis_excluded_rows", -1)) != 22090
        or int(report.get("analysis_eligible_rows", -1)) != 55288
    ):
        raise ValueError("MI output validation failed")
    _append("MI_COMPLETE", output_sha256=sha256_file(MI_OUTPUT), primary=(report.get("results") or {}).get("source_presence", {}).get("OPT"))


def main() -> None:
    if COMPLETE.exists():
        print(COMPLETE.read_text(encoding="utf-8"), end="")
        return
    if FAILED.exists():
        raise RuntimeError(f"prior fail-closed marker exists: {FAILED}")
    try:
        _require_sha(MANIFEST, MANIFEST_SHA, "manifest")
        _require_sha(TRUTH, TRUTH_SHA, "joined truth")
        _require_sha(CERTIFICATE, CERTIFICATE_SHA, "MI certificate")
        _require_sha(RISK, RISK_SHA, "post-blind risk record")
        _prepare_plan()
        prediction_sha = _wait_for_predictions()
        _overlay(prediction_sha)
        _sample_postproduction_risk()
        _freeze_release()
        _run_mi()
        payload = {
            "status": "COMPLETE_HUMOR_CANONICAL_RELEASE_AND_MI",
            "completed_at": _now(),
            "predictions": {"path": str(PREDICTIONS), "sha256": prediction_sha, "rows": 55288},
            "final": {"path": str(FINAL), "sha256": sha256_file(FINAL), "rows": 77378},
            "release": {"path": str(TASK_RELEASE), "sha256": sha256_file(TASK_RELEASE)},
            "mi": {"path": str(MI_OUTPUT), "sha256": sha256_file(MI_OUTPUT)},
        }
        _write_new(COMPLETE, payload)
        print(json.dumps(payload, sort_keys=True))
    except BaseException as exc:
        payload = {"status": "FAILED_CLOSED", "failed_at": _now(), "error": repr(exc)}
        if not FAILED.exists():
            _write_new(FAILED, payload)
        _append("FAILED_CLOSED", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
