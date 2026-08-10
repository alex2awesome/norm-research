#!/usr/bin/env python3
"""Run one hash-frozen Nemotron adapter evaluation on external dev only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    args = parser.parse_args()
    queue_path = Path(args.queue).resolve()
    queue = json.loads(queue_path.read_text())
    if (
        queue.get("schema_version")
        != "silver-match-v3-frozen-nemotron-external-dev-queue-v1"
        or queue.get("status") != "FROZEN_READY"
    ):
        raise ValueError("unsupported or unready queue")
    observed = {}
    for row in queue["bindings"]:
        path = Path(row["path"])
        actual = sha256_file(path)
        if actual != row["sha256"]:
            raise ValueError(f"binding mismatch: {row['name']}")
        observed[row["name"]] = actual
    labels = Path(queue["external_dev"])
    seen = 0
    for row in read_jsonl(labels):
        seen += 1
        if row.get("split") != "dev":
            raise ValueError("external-dev artifact contains a non-dev row")
    if not seen:
        raise ValueError("external-dev artifact is empty")
    command = [str(value) for value in queue["command"]]
    if command[1:4] != ["-u", "-m", "scripts.tools.silver_match_v3.evaluate_nemotron_adapter"]:
        raise ValueError("unexpected evaluator entry point")
    if "--split" not in command or command[command.index("--split") + 1] != "dev":
        raise ValueError("queue is not a dev-only evaluation")
    output = Path(command[command.index("--output") + 1])
    log = Path(queue["log"])
    record = Path(queue["run_record"])
    if output.exists() or log.exists() or record.exists():
        raise FileExistsError("external-dev output already exists")
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update({str(k): str(v) for k, v in queue["environment"].items()})
    started = datetime.now(timezone.utc).isoformat()
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=queue["repo"],
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    payload: dict[str, Any] = {
        "schema_version": "silver-match-v3-frozen-nemotron-external-dev-run-v1",
        "status": "COMPLETED" if completed.returncode == 0 else "FAILED_CLOSED",
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "queue": str(queue_path),
        "queue_sha256": sha256_file(queue_path),
        "verified_bindings": observed,
        "external_dev_rows": seen,
        "external_test_consumed": False,
        "returncode": completed.returncode,
        "output": str(output),
        "output_sha256": sha256_file(output) if output.exists() else None,
        "log_sha256": sha256_file(log),
    }
    record.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
