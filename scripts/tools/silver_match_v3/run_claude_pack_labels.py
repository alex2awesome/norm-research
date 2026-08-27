#!/usr/bin/env python3
"""Run independent structured Claude labeling over a frozen full-bank pack."""

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
from typing import Any

from .adjudicate_gemma import CONFIDENCES, DECISIONS
from .common import sha256_file


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _chunk_uids(path: Path) -> list[str]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    uids = [str(row.get("norm_uid") or "") for row in rows]
    if not uids or "" in uids or len(uids) != len(set(uids)):
        raise ValueError(f"empty, missing, or duplicate chunk UIDs: {path}")
    return uids


def validate_payload(
    payload: dict[str, Any],
    *,
    task: str,
    chunk_id: str,
    expected_uids: list[str],
    bank_ids: set[str],
) -> None:
    labels = payload.get("labels")
    if (
        payload.get("task") != task
        or payload.get("chunk_id") != chunk_id
        or not isinstance(labels, list)
    ):
        raise ValueError("label envelope task/chunk/schema drift")
    observed_uids = [str(row.get("norm_uid") or "") for row in labels]
    if (
        len(observed_uids) != len(expected_uids)
        or len(observed_uids) != len(set(observed_uids))
        or set(observed_uids) != set(expected_uids)
    ):
        raise ValueError("label envelope does not exactly cover chunk UIDs")
    for row in labels:
        decision = str(row.get("decision") or "")
        metric_id = row.get("metric_id")
        reason = str(row.get("reason") or "")
        if (
            decision not in DECISIONS
            or row.get("confidence") not in CONFIDENCES
            or not 8 <= len(reason) <= 600
            or (decision == "MATCH" and metric_id not in bank_ids)
            or (decision != "MATCH" and metric_id is not None)
        ):
            raise ValueError(f"invalid independent label: {row.get('norm_uid')}")


def _final_result(transcript: Path) -> dict[str, Any]:
    finals: list[dict[str, Any]] = []
    for line in transcript.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict) and value.get("type") == "result":
            finals.append(value)
    if len(finals) != 1:
        raise ValueError(f"expected one final Claude result: {transcript}")
    final = finals[0]
    if final.get("subtype") != "success" or final.get("is_error") is not False:
        raise ValueError(f"Claude result is not successful: {transcript}")
    structured = final.get("structured_output")
    if isinstance(structured, dict):
        return structured
    result = final.get("result")
    if not isinstance(result, str):
        raise ValueError(f"Claude result has no structured payload: {transcript}")
    value = json.loads(result)
    if not isinstance(value, dict):
        raise ValueError(f"Claude result payload is not an object: {transcript}")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    pack = Path(args.pack_root).resolve()
    workspace = pack.parent.resolve()
    repo = Path(args.repo).resolve()
    schema_path = Path(args.output_schema)
    if not schema_path.is_absolute():
        schema_path = repo / schema_path
    schema_path = schema_path.resolve()
    guides = [repo / "scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md"]
    guides.extend(
        Path(value).resolve() if Path(value).is_absolute() else (repo / value).resolve()
        for value in args.boundary_guide
    )
    bank_path = pack / "bank.json"
    validation_path = pack / "validation.json"
    chunks = sorted((pack / "chunks").glob("part-*.jsonl"))
    for path in [pack, schema_path, bank_path, validation_path, *guides]:
        if not path.exists():
            raise FileNotFoundError(path)
    if not chunks:
        raise ValueError("pack has no chunks")
    validation = _load_json(validation_path)
    bank = _load_json(bank_path)
    bank_rows = list(bank.get("metrics") or bank.get("bank") or [])
    bank_ids = {str(row.get("metric_id") or "") for row in bank_rows}
    if (
        validation.get("status") != "FROZEN_TRUTH_HIDDEN_BEFORE_LABELING"
        or validation.get("truth_hidden") is not True
        or validation.get("task") != args.task
        or not bank_ids
        or "" in bank_ids
        or len(bank_ids) != len(bank_rows)
    ):
        raise ValueError("source is not an exact frozen truth-hidden full-bank pack")
    schema = _load_json(schema_path)
    schema.pop("$schema", None)
    schema_text = json.dumps(schema, separators=(",", ":"), sort_keys=True)
    claude = shutil.which("claude")
    if not claude:
        raise FileNotFoundError("claude")

    output_root = pack / args.output_namespace
    raw_root = output_root / "raw_labels"
    transcript_root = output_root / "transcripts"
    request_root = output_root / "requests"
    stderr_root = output_root / "stderr"
    for path in (raw_root, transcript_root, request_root, stderr_root):
        path.mkdir(parents=True, exist_ok=True)

    def one(chunk_path: Path) -> dict[str, Any]:
        chunk_id = chunk_path.stem
        expected_uids = _chunk_uids(chunk_path)
        raw_path = raw_root / f"{chunk_id}.json"
        request_path = request_root / f"{chunk_id}.json"
        if raw_path.exists():
            payload = _load_json(raw_path)
            validate_payload(
                payload,
                task=args.task,
                chunk_id=chunk_id,
                expected_uids=expected_uids,
                bank_ids=bank_ids,
            )
            return {"chunk": chunk_id, "rows": len(expected_uids), "status": "skipped_valid"}
        if request_path.exists():
            raise FileExistsError(f"orphan request without valid label: {request_path}")
        count = len(expected_uids)
        guide_clause = " ".join(
            f"Read {path.relative_to(workspace)} for audited exact-leaf boundary guidance."
            for path in guides[1:]
        )
        prompt = (
            f"Act as independent hidden-ID full-bank annotator {args.pass_name} for a "
            "high-precision silver norm-to-metric pool. Read "
            f"{guides[0].relative_to(workspace)}. {guide_clause} Read the frozen, "
            f"order-permuted {args.task} bank at {bank_path.relative_to(workspace)} and "
            f"label all {count} items in {chunk_path.relative_to(workspace)} from scratch. "
            "Consider the entire bank for every item. Distinguish exact MATCH from "
            "related-family-only and typed bank-gap/no-criterion abstentions; never force "
            f"a leaf. Set task to {args.task} and chunk_id to {chunk_id}. Return only "
            "schema-conforming JSON. Do not search for any prior labels, proposals, "
            "audits, truth, model outputs, MI, or outcomes; use only the item text/context, "
            "guides, and this bank."
        )
        command = [
            claude,
            "-p",
            "--safe-mode",
            "--model",
            args.model,
            "--effort",
            args.effort,
            "--tools",
            "Read",
            "--allowedTools",
            "Read",
            "--disallowedTools",
            "Bash,Edit,Write,Glob,Grep,WebFetch,WebSearch,Task,Skill",
            "--permission-mode",
            "dontAsk",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--no-session-persistence",
            "--output-format",
            "stream-json",
            "--verbose",
            "--json-schema",
            schema_text,
            prompt,
        ]
        request = {
            "schema_version": "silver-match-v3-independent-claude-label-request-v1",
            "status": "FROZEN_BEFORE_REQUEST",
            "task": args.task,
            "pass_name": args.pass_name,
            "chunk_id": chunk_id,
            "model": args.model,
            "effort": args.effort,
            "cwd": str(workspace),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "tool_contract": {
                "available": ["Read"],
                "network_tools_available": False,
                "write_or_shell_tools_available": False,
                "safe_mode": True,
                "session_persistence": False,
            },
            "inputs": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in [*guides, bank_path, chunk_path, schema_path, validation_path]
            ],
        }
        request_path.write_text(
            json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        started = time.time()
        last_error = ""
        for attempt in range(1, args.chunk_attempts + 1):
            transcript = transcript_root / f"{chunk_id}.attempt-{attempt:04d}.jsonl"
            stderr = stderr_root / f"{chunk_id}.attempt-{attempt:04d}.log"
            if transcript.exists() or stderr.exists():
                raise FileExistsError(f"attempt artifacts already exist: {chunk_id}/{attempt}")
            child_env = os.environ.copy()
            for key in ("PYTHONPATH", "OLDPWD"):
                child_env.pop(key, None)
            child_env["PWD"] = str(workspace)
            child_env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
            timed_out = False
            with transcript.open("xb") as stdout, stderr.open("xb") as stderr_handle:
                try:
                    completed = subprocess.run(
                        command,
                        cwd=workspace,
                        env=child_env,
                        stdout=stdout,
                        stderr=stderr_handle,
                        timeout=args.timeout_seconds,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    timed_out = True
                    completed = None
            try:
                if timed_out or completed is None or completed.returncode != 0:
                    raise RuntimeError(
                        "timeout"
                        if timed_out
                        else f"exit={completed.returncode if completed else None}"
                    )
                payload = _final_result(transcript)
                validate_payload(
                    payload,
                    task=args.task,
                    chunk_id=chunk_id,
                    expected_uids=expected_uids,
                    bank_ids=bank_ids,
                )
                if raw_path.exists():
                    raise FileExistsError(raw_path)
                raw_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                return {
                    "chunk": chunk_id,
                    "rows": count,
                    "status": "completed",
                    "attempt": attempt,
                    "elapsed_seconds": time.time() - started,
                    "raw_label_sha256": sha256_file(raw_path),
                    "transcript_sha256": sha256_file(transcript),
                }
            except Exception as exc:  # preserve the exact failed attempt for audit
                last_error = f"{type(exc).__name__}: {exc}"
        return {
            "chunk": chunk_id,
            "rows": count,
            "status": "failed",
            "attempts": args.chunk_attempts,
            "elapsed_seconds": time.time() - started,
            "error": last_error,
        }

    started = time.time()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(one, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    failures = [row for row in results if row["status"] == "failed"]
    summary = {
        "schema_version": "silver-match-v3-independent-claude-pack-run-v1",
        "status": "COMPLETE" if not failures else "INCOMPLETE",
        "task": args.task,
        "pass_name": args.pass_name,
        "model": args.model,
        "effort": args.effort,
        "pack": {"path": str(pack), "validation_sha256": sha256_file(validation_path)},
        "chunks": len(chunks),
        "rows": sum(row["rows"] for row in results if row["status"] != "failed"),
        "completed": sum(row["status"] == "completed" for row in results),
        "skipped_valid": sum(row["status"] == "skipped_valid" for row in results),
        "failed": len(failures),
        "elapsed_seconds": time.time() - started,
        "results": sorted(results, key=lambda row: row["chunk"]),
    }
    summary_path = output_root / "RUN_SUMMARY.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"{len(failures)} Claude chunks failed")
    return {**summary, "summary_sha256": sha256_file(summary_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--task", required=True)
    parser.add_argument("--pass-name", required=True)
    parser.add_argument("--boundary-guide", action="append", default=[])
    parser.add_argument(
        "--output-schema",
        default="scripts/tools/silver_match_v3/schemas/independent_labels_1_to_25.schema.json",
    )
    parser.add_argument("--output-namespace", default="claude_sonnet_v1")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument(
        "--effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default="high",
    )
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--chunk-attempts", type=int, default=1)
    args = parser.parse_args()
    if min(args.concurrency, args.timeout_seconds, args.chunk_attempts) < 1:
        parser.error("concurrency, timeout, and attempts must be positive")
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
