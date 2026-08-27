#!/usr/bin/env python3
"""Run independent structured Codex labeling over every chunk with bounded concurrency."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def valid_existing(path: Path, task: str, chunk: str, expected_uids: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    labels = payload.get("labels")
    observed_uids = (
        [str(row.get("norm_uid") or "") for row in labels]
        if isinstance(labels, list)
        else []
    )
    return (
        payload.get("task") == task
        and payload.get("chunk_id") == chunk
        and len(observed_uids) == len(expected_uids)
        and len(set(observed_uids)) == len(observed_uids)
        and set(observed_uids) == set(expected_uids)
    )


def archive_invalid_output(output: Path, log: Path, archive_root: Path) -> None:
    """Preserve an invalid raw response and its log before a clean retry."""
    if not output.exists() and not log.exists():
        return
    digest_source = output.read_bytes() if output.exists() else log.read_bytes()
    digest = hashlib.sha256(digest_source).hexdigest()
    archive_root.mkdir(parents=True, exist_ok=True)
    if output.exists():
        archived_output = archive_root / f"{output.stem}.{digest[:16]}.json"
        if archived_output.exists():
            if archived_output.read_bytes() != output.read_bytes():
                raise FileExistsError(
                    f"invalid-output archive hash collision: {archived_output}"
                )
            output.unlink()
        else:
            output.replace(archived_output)
    if log.exists():
        archived_log = archive_root / f"{log.stem}.{digest[:16]}.log"
        if archived_log.exists():
            if archived_log.read_bytes() != log.read_bytes():
                raise FileExistsError(f"invalid-log archive collision: {archived_log}")
            log.unlink()
        else:
            shutil.move(str(log), str(archived_log))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--pass-name", required=True)
    parser.add_argument("--boundary-guide", action="append", default=[])
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=900,
        help="Fail and preserve a chunk for retry if one Codex process exceeds this wall time.",
    )
    parser.add_argument(
        "--chunk-attempts",
        type=int,
        default=2,
        help="Retry a timed-out or runtime-invalid chunk from scratch this many times.",
    )
    parser.add_argument(
        "--output-schema",
        default="scripts/tools/silver_match_v3/schemas/independent_labels_1_to_25.schema.json",
        help="Repo-relative variable-size schema supporting every frozen chunk size from 1 to 25.",
    )
    args = parser.parse_args()
    if args.timeout_seconds < 1 or args.chunk_attempts < 1:
        parser.error("--timeout-seconds and --chunk-attempts must be positive")
    root = Path(args.pack_root).resolve()
    repo = Path.cwd().resolve()
    schema = Path(args.output_schema)
    if not schema.is_absolute():
        schema = repo / schema
    schema = schema.resolve()
    raw_root, log_root = root / "raw_labels", root / "logs"
    invalid_root = root / "invalid_raw_labels"
    raw_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    chunks = sorted((root / "chunks").glob("part-*.jsonl"))
    if not chunks:
        raise ValueError("pack has no chunks")
    bank = root / "bank.json"
    guides = [repo / value for value in args.boundary_guide]
    for path in [bank, schema, *guides]:
        if not path.exists():
            raise FileNotFoundError(path)

    def run(chunk_path: Path) -> dict:
        chunk = chunk_path.stem
        chunk_rows = [
            json.loads(line)
            for line in chunk_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        expected_uids = [str(row["norm_uid"]) for row in chunk_rows]
        if len(expected_uids) != len(set(expected_uids)):
            raise ValueError(f"chunk contains duplicate UIDs: {chunk_path}")
        count = len(expected_uids)
        output = raw_root / f"{chunk}.json"
        log = log_root / f"{chunk}.log"
        if valid_existing(output, args.task, chunk, expected_uids):
            return {"chunk": chunk, "count": count, "status": "skipped_valid"}
        guide_text = " ".join(
            f"Read {path.relative_to(repo)} for audited exact-leaf boundary guidance."
            for path in guides
        )
        prompt = (
            f"Act as independent hidden-ID full-bank annotator {args.pass_name} for a "
            "high-precision silver norm-to-metric pool. Read "
            "scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md. "
            f"{guide_text} Read the frozen, order-permuted {args.task} bank at "
            f"{bank.relative_to(repo)} and label all {count} items in "
            f"{chunk_path.relative_to(repo)} from scratch. Consider the entire bank for "
            "every item. Distinguish exact MATCH from related-family-only and typed "
            "bank-gap/no-criterion abstentions; never force a leaf. "
            f"Set task to {args.task} and chunk_id to {chunk}. Return only "
            "schema-conforming JSON. Do not search for any prior labels, proposals, "
            "audits, or truth; use only the item text/context, guides, and this bank."
        )
        command = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--dangerously-bypass-hook-trust",
            "-m",
            args.model,
            "-c",
            f'model_reasoning_effort="{args.reasoning_effort}"',
            "--output-schema",
            str(schema),
            "-o",
            str(output),
            prompt,
        ]
        started = time.time()
        last_failure = None
        for attempt in range(1, args.chunk_attempts + 1):
            archive_invalid_output(output, log, invalid_root)
            timed_out = False
            child_env = os.environ.copy()
            # The runner may be imported from a different repository via
            # PYTHONPATH while its deliberately minimal labeling workspace is
            # the current directory.  Do not expose that import path to the
            # independent Codex annotator: its admissible evidence is the
            # staged bank, items, and guide only.
            child_env.pop("PYTHONPATH", None)
            child_env.pop("OLDPWD", None)
            child_env["PWD"] = str(repo)
            with log.open("w", encoding="utf-8") as handle:
                try:
                    completed = subprocess.run(
                        command,
                        cwd=repo,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                        timeout=args.timeout_seconds,
                        env=child_env,
                    )
                except subprocess.TimeoutExpired:
                    timed_out = True
                    completed = None
            if not timed_out and completed is not None and completed.returncode == 0 and valid_existing(
                output, args.task, chunk, expected_uids
            ):
                return {
                    "chunk": chunk,
                    "count": count,
                    "status": "completed",
                    "attempts": attempt,
                    "elapsed_seconds": time.time() - started,
                }
            last_failure = {
                "failure": "timeout" if timed_out else "invalid_or_nonzero",
                "returncode": None if completed is None else completed.returncode,
            }
        return {
            "chunk": chunk,
            "count": count,
            "status": "failed",
            "attempts": args.chunk_attempts,
            "timeout_seconds": args.timeout_seconds,
            "elapsed_seconds": time.time() - started,
            **(last_failure or {}),
        }

    started = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(run, path): path for path in chunks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    failures = [row for row in results if row["status"] == "failed"]
    summary = {
        "task": args.task,
        "pass_name": args.pass_name,
        "chunks": len(chunks),
        "completed": sum(row["status"] == "completed" for row in results),
        "skipped_valid": sum(row["status"] == "skipped_valid" for row in results),
        "failed": len(failures),
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
