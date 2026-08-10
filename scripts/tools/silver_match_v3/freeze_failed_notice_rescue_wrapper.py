#!/usr/bin/env python3
"""Freeze proof that the stale N&C GPU wrapper ended before new inference."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from .common import sha256_file


PIN = re.compile(r"^PINNED_(ADJUDICATE|VERIFY)_SHA=([0-9a-f]{64})$", re.MULTILINE)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _live_wrapper_processes(script: Path) -> list[dict[str, str | int]]:
    needle = str(script.resolve())
    relative_needle = "/".join(script.parts[-4:])
    output = []
    proc = Path("/proc")
    if not proc.is_dir():
        return output
    for child in proc.iterdir():
        if not child.name.isdigit() or int(child.name) == os.getpid():
            continue
        try:
            cmdline = (child / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
            status = (child / "status").read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        uid_match = re.search(r"^Uid:\s+(\d+)", status, re.MULTILINE)
        if (
            (needle in cmdline or relative_needle in cmdline)
            and uid_match
            and int(uid_match.group(1)) == os.getuid()
        ):
            output.append({"pid": int(child.name), "command": cmdline.strip()})
    return sorted(output, key=lambda row: int(row["pid"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper-script", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--prewrapper-audit", required=True)
    parser.add_argument("--postwrapper-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    wrapper = Path(args.wrapper_script).resolve()
    repo = Path(args.repo).resolve()
    log = Path(args.log).resolve()
    pre_path = Path(args.prewrapper_audit).resolve()
    post_path = Path(args.postwrapper_audit).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    for path in (wrapper, repo, pre_path, post_path):
        if not path.exists():
            raise FileNotFoundError(path)
    pre, post = _json(pre_path), _json(post_path)
    if (
        pre.get("status") != "PASS_SEALED_GPU_ARTIFACTS_CONTENT_REVALIDATED"
        or post.get("status") != "PASS_SEALED_GPU_ARTIFACTS_CONTENT_REVALIDATED"
        or pre.get("artifacts") != post.get("artifacts")
    ):
        raise ValueError("pre/post sealed artifact audits do not prove zero drift")
    live = _live_wrapper_processes(wrapper)
    if live:
        raise RuntimeError(f"stale wrapper remains live: {live}")
    text = wrapper.read_text(encoding="utf-8")
    pins = {name.lower(): value for name, value in PIN.findall(text)}
    sources = {
        "adjudicate": repo / "scripts/tools/silver_match_v3/adjudicate_gemma.py",
        "verify": repo / "scripts/tools/silver_match_v3/verify_gemma.py",
    }
    source_state = {
        name: {
            "path": str(path),
            "pinned_sha256": pins.get(name),
            "observed_sha256": sha256_file(path),
            "matches_pin": pins.get(name) == sha256_file(path),
        }
        for name, path in sources.items()
    }
    if source_state["adjudicate"]["matches_pin"] is not False:
        raise ValueError("expected stale adjudicator pin mismatch is absent")
    setup_start = text.find("while test ! -f")
    first_check = text.find("\ncheck_sources\n", setup_start)
    first_worker = text.find("run_trial_worker \"$RESCUE_GPU_A\"")
    if first_check < 0 or first_worker < 0 or first_check >= first_worker:
        raise ValueError("wrapper no longer checks source pins before GPU workers")
    report = {
        "schema_version": "silver-match-v3-notice-stale-wrapper-failure-v1",
        "status": "STALE_WRAPPER_FAILED_BEFORE_INFERENCE",
        "gpu_inference_started": False,
        "basis": {
            "source_pin_check_precedes_first_gpu_worker": True,
            "adjudicator_pin_mismatch": True,
            "no_live_scoped_wrapper_process": True,
            "sealed_artifact_universe_unchanged_pre_to_post": True,
        },
        "wrapper": {"path": str(wrapper), "sha256": sha256_file(wrapper)},
        "source_state": source_state,
        "prewrapper_audit": {"path": str(pre_path), "sha256": sha256_file(pre_path)},
        "postwrapper_audit": {"path": str(post_path), "sha256": sha256_file(post_path)},
        "log": (
            {
                "path": str(log),
                "sha256": sha256_file(log),
                "bytes": log.stat().st_size,
            }
            if log.is_file()
            else {"path": str(log), "exists": False}
        ),
        "live_scoped_processes": live,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "output_sha256": sha256_file(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
