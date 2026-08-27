#!/usr/bin/env python3
"""SSH and content-addressed report layer for Humor CE recovery."""

from __future__ import annotations
import argparse
import base64
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from scripts.tools.silver_match_v3.remote_recovery_core import (
    FORBIDDEN_SK3_GPUS,
    MAX_FULL_BYTES,
    PILOTS,
    REMOTE_ROOT,
    SCHEMA,
    TAIL_BYTES,
    artifact_specs,
    canonical,
    classify_pilot,
    inspection_plan,
    local_probe,
    sha,
)


def _ssh(h: str, t: int) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={t}",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=1",
        h,
    ]


def wait_for_ssh(
    h: str, *, polls: int, poll_seconds: float, timeout: int
) -> dict[str, Any]:
    errors = []
    for attempt in range(1, polls + 1):
        z = subprocess.run([*_ssh(h, timeout), "true"], text=True, capture_output=True)
        if z.returncode == 0:
            return {"status": "AVAILABLE", "attempts": attempt}
        errors.append((z.stderr or z.stdout).strip()[-1000:])
        if attempt < polls:
            time.sleep(poll_seconds)
    return {"status": "UNAVAILABLE", "attempts": polls, "errors": errors[-3:]}


def _remote(h: str, b: Path, t: int, pilots: bool) -> dict[str, Any]:
    payload = {
        "max": MAX_FULL_BYTES,
        "tail": TAIL_BYTES,
        "pilots": [
            {"name": p.name, "root": str(p.root(b)), "artifacts": artifact_specs(b, p)}
            for p in PILOTS
        ]
        if pilots
        else [],
    }
    data = base64.b64encode(canonical(payload)).decode()
    script = Path(__file__).with_name("remote_recovery_probe.py").read_text()
    z = subprocess.run(
        [*_ssh(h, t), "python3", "-", data],
        input=script,
        text=True,
        capture_output=True,
    )
    if z.returncode:
        raise RuntimeError((z.stderr or z.stdout).strip())
    out = json.loads(z.stdout)
    for rows in out.get("artifacts", {}).values():
        for row in rows:
            if "content_b64" in row:
                row["content"] = base64.b64decode(row.pop("content_b64"))
    return out


def _store(root: Path, probe: Mapping[str, Any]) -> dict[str, Any]:
    ledger = {}
    directory = root / "artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    for pilot, rows in probe.get("artifacts", {}).items():
        ledger[pilot] = {}
        for original in rows:
            row = dict(original)
            content = row.pop("content", None)
            if isinstance(content, bytes):
                digest = sha(content)
                suffix = "json" if row["mode"] == "full" else "tail"
                path = directory / (digest + "." + suffix)
                if path.exists() and path.read_bytes() != content:
                    raise ValueError("content-address collision")
                if not path.exists():
                    path.write_bytes(content)
                row.update(
                    local_content_path=str(path.resolve()), local_content_sha256=digest
                )
            ledger[pilot][row["key"]] = row
    return ledger


def _connect(h: str, a: argparse.Namespace) -> dict[str, Any]:
    return wait_for_ssh(
        h, polls=a.max_polls, poll_seconds=a.poll_seconds, timeout=a.connect_timeout
    )


def run(a: argparse.Namespace) -> tuple[dict[str, Any], Path | None]:
    base = Path(a.local_fixture_root).resolve() if a.local_fixture_root else REMOTE_ROOT
    if a.dry_run:
        return inspection_plan(base, a.pilot_host, a.gpu_host), None
    connectivity = {}
    gpus = {}
    if a.local_fixture_root:
        probe = local_probe(base)
        connectivity[a.pilot_host] = {"status": "LOCAL_FIXTURE", "attempts": 0}
        gpus[a.pilot_host] = {"available": False, "fixture": True}
    else:
        connection = _connect(a.pilot_host, a)
        connectivity[a.pilot_host] = connection
        if connection["status"] == "AVAILABLE":
            probe = _remote(a.pilot_host, base, a.connect_timeout, True)
            gpus[a.pilot_host] = probe.get("gpu_probe", {})
        else:
            probe = {"roots": {}, "artifacts": {}}
            gpus[a.pilot_host] = {"available": False, "ssh_unavailable": True}
        for host in dict.fromkeys(a.gpu_host):
            if host == a.pilot_host:
                continue
            connection = _connect(host, a)
            connectivity[host] = connection
            gpus[host] = (
                _remote(host, base, a.connect_timeout, False).get("gpu_probe", {})
                if connection["status"] == "AVAILABLE"
                else {"available": False, "ssh_unavailable": True}
            )
    pilots = [
        classify_pilot(
            p,
            base,
            bool(probe.get("roots", {}).get(p.name)),
            probe.get("artifacts", {}).get(p.name, []),
        )
        for p in PILOTS
    ]
    report = {
        "schema_version": SCHEMA,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "release_ready": False,
        "pilot_host": a.pilot_host,
        "remote_root": str(base),
        "connectivity": connectivity,
        "pilots": pilots,
        "status_counts": {
            s: sum(p["status"] == s for p in pilots)
            for s in ("COMPLETE", "RUNNING", "FAILED", "ABSENT")
        },
        "artifact_ledger": _store(Path(a.output_root), probe),
        "candidate_gpu_hosts": gpus,
        "sk3_forbidden_gpu_indices": list(FORBIDDEN_SK3_GPUS),
        "mutation_attempted": False,
        "jobs_launched": False,
        "jobs_killed": False,
        "checkpoints_read_or_copied": False,
    }
    data = canonical(report)
    directory = Path(a.output_root) / "status"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("status." + sha(data) + ".json")
    if path.exists() and path.read_bytes() != data:
        raise ValueError("content-address collision")
    if not path.exists():
        path.write_bytes(data)
    return report, path.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot-host", default="sk2")
    p.add_argument("--gpu-host", action="append", default=[])
    p.add_argument(
        "--output-root", default="outputs/silver_match_v3/humor_ce_remote_recovery"
    )
    p.add_argument("--max-polls", type=int, default=12)
    p.add_argument("--poll-seconds", type=float, default=5.0)
    p.add_argument("--connect-timeout", type=int, default=10)
    p.add_argument("--local-fixture-root")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.max_polls < 1 or a.poll_seconds < 0 or a.connect_timeout < 1:
        p.error("invalid polling configuration")
    if not a.gpu_host:
        a.gpu_host = [a.pilot_host, "sk1", "sk3"]
    return a


def main(argv: Sequence[str] | None = None) -> int:
    report, path = run(parse_args(argv))
    print(
        json.dumps(
            {"report": report, "status_report": str(path) if path else None},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
