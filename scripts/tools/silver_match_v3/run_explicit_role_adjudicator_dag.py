#!/usr/bin/env python3
"""Resume and validate the adjudicator portion of a frozen explicit-role DAG."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file
from .gpu_host_policy import validate_gpu_indices_for_host, validate_launch_gpus


PLAN_SCHEMA = "silver-match-v3-explicit-role-task-local-gepa-plan-v1"
FREEZE_SCHEMA = "silver-match-v3-explicit-role-task-local-gepa-freeze-v1"
STAGES = {"adjudicator", "adjudicator_consensus", "adjudicator_score"}


def _arg(argv: list[str], flag: str) -> str:
    if argv.count(flag) != 1:
        raise ValueError(f"expected exactly one {flag}")
    index = argv.index(flag)
    if index + 1 == len(argv):
        raise ValueError(f"missing value for {flag}")
    return argv[index + 1]


def _validate_inference(output: Path) -> tuple[bool, str]:
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    if not output.is_file() or not meta_path.is_file():
        return False, "missing output/meta"
    try:
        meta = json.loads(meta_path.read_text())
        rows = list(read_jsonl(output))
    except Exception as exc:  # pragma: no cover - operational diagnostics
        return False, f"parse error: {exc}"
    uids = [str(row.get("norm_uid") or "") for row in rows]
    invalid = [
        row
        for row in rows
        if row.get("decision") == "INVALID_OUTPUT" or row.get("parse_error")
    ]
    hash_ok = sha256_file(output) == meta.get("output_sha256")
    ok = (
        len(rows) == int(meta.get("eligible_count", -1))
        and len(uids) == len(set(uids))
        and "" not in uids
        and not invalid
        and hash_ok
    )
    return (
        ok,
        f"rows={len(rows)} eligible={meta.get('eligible_count')} "
        f"unique={len(set(uids))} invalid={len(invalid)} sha_ok={hash_ok}",
    )


def _validate_cpu(stage: str, output: Path) -> tuple[bool, str]:
    if not output.is_file():
        return False, "missing output"
    try:
        if stage == "adjudicator_consensus":
            rows = list(read_jsonl(output))
            report = json.loads(
                output.with_suffix(output.suffix + ".report.json").read_text()
            )
            hash_ok = report.get("output", {}).get("sha256") == sha256_file(output)
            inputs_ok = all(
                Path(ref["path"]).is_file()
                and sha256_file(Path(ref["path"])) == ref["sha256"]
                for ref in report.get("inputs", {}).values()
            )
            ok = (
                hash_ok
                and inputs_ok
                and int(report.get("consensus_match_count", -1)) == len(rows)
                and len({row["norm_uid"] for row in rows}) == len(rows)
            )
            return ok, f"consensus={len(rows)} sha_ok={hash_ok} inputs_ok={inputs_ok}"
        payload = json.loads(output.read_text())
        refs: dict[str, Any] = payload.get("inputs", {})
        inputs_ok = all(
            Path(ref["path"]).is_file()
            and sha256_file(Path(ref["path"])) == ref["sha256"]
            for ref in refs.values()
        )
        ok = (
            payload.get("schema_version") == "silver-match-v3-two-order-gepa-score-v1"
            and inputs_ok
        )
        return ok, f"schema={payload.get('schema_version')} inputs_ok={inputs_ok}"
    except Exception as exc:  # pragma: no cover - operational diagnostics
        return False, f"parse error: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    gpu_indices = [
        int(value.strip())
        for value in args.cuda_visible_devices.split(",")
        if value.strip()
    ]
    validate_gpu_indices_for_host(gpu_indices)
    gpu_launch_guard = validate_launch_gpus(gpu_indices)
    plan_path = Path(args.plan).resolve()
    plan_sha = sha256_file(plan_path)
    if plan_sha != args.expected_plan_sha256:
        raise ValueError("command plan hash drift")
    plan = json.loads(plan_path.read_text())
    freeze_path = plan_path.with_name("FREEZE.json")
    freeze = json.loads(freeze_path.read_text())
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "FROZEN_BEFORE_TASK_LOCAL_GEPA_INFERENCE"
        or freeze.get("schema_version") != FREEZE_SCHEMA
        or freeze.get("command_plan", {}).get("sha256") != plan_sha
        or freeze.get("test_or_blind_audit_consumed") is not False
        or freeze.get("production_consumed") is not False
    ):
        raise ValueError("plan and role freeze are not cleanly linked")

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
            "PYTHONPATH": ".",
            "HF_HOME": "/lfs/skampere3/0/shared_hf_cache",
        }
    )
    print(json.dumps({"gpu_launch_guard": gpu_launch_guard}, sort_keys=True), flush=True)
    for cell in plan["commands"]:
        stage = str(cell.get("stage") or "")
        if stage not in STAGES:
            continue
        command = (
            cell["direct_batch_command"] if stage == "adjudicator" else cell["command"]
        )
        argv = [str(value) for value in command["argv"]]
        output = Path(_arg(argv, "--output")).resolve()
        validator = _validate_inference if stage == "adjudicator" else lambda value: _validate_cpu(stage, value)
        ok, detail = validator(output)
        label = ":".join(
            str(cell.get(key) or "")
            for key in ("stage", "variant", "role", "order")
        )
        if ok:
            print(f"SKIP_VALID {label} {detail}", flush=True)
            continue
        if stage != "adjudicator" and output.exists():
            raise RuntimeError(f"invalid existing CPU artifact: {label}: {detail}")
        print(f"START_OR_RESUME {label} {detail}", flush=True)
        started = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-m", command["module"], *argv],
            cwd=Path(args.repo_root).resolve(),
            env=environment,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(f"command failed: {label}: rc={result.returncode}")
        ok, detail = validator(output)
        print(
            f"DONE {label} seconds={time.monotonic() - started:.1f} {detail}",
            flush=True,
        )
        if not ok:
            raise RuntimeError(f"post-validation failed: {label}: {detail}")
    print("ALL_ADJUDICATOR_DAG_CELLS_VALID", flush=True)


if __name__ == "__main__":
    main()
