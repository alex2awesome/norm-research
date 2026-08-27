#!/usr/bin/env python3
"""Execute the frozen Math-a12 LLM verifier with an exact smoke STOP.

This runner is intentionally specific to the independently authored Math
request contract.  It uses fresh Claude CLI ``-p`` invocations, never resumes
a session, logs one append-only response envelope per request, and will not
start production unless the first ten frozen requests are 10/10 valid.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable, Sequence
import uuid

from methods.metric_seam.verifiers.math_a12_llm_contract import (
    REQUEST_SCHEMA,
    SYSTEM_PROMPT,
    validate_response_envelope,
)
from methods.metric_seam.verifiers.schema import SchemaError


SMOKE_SIZE = 10
MAX_CONCURRENCY = 4
ENVELOPE_SCHEMA = "metric-seam.math-a12-llm-response-envelope.v1"


class MathHarnessError(ValueError):
    pass


class StopProduction(MathHarnessError):
    pass


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_bundle(path: Path, *, model: str) -> list[dict]:
    requests: list[dict] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise MathHarnessError(f"bundle line {line_number}: blank line")
        request = json.loads(line)
        if request.get("schema") != REQUEST_SCHEMA:
            raise MathHarnessError(f"bundle line {line_number}: wrong request schema")
        if request.get("model") != model or request.get("split") != "compiler_train":
            raise MathHarnessError(f"bundle line {line_number}: model/split mismatch")
        if request.get("system_prompt") != SYSTEM_PROMPT:
            raise MathHarnessError(f"bundle line {line_number}: prompt drift")
        digest = request.get("request_sha256")
        if not isinstance(digest, str) or digest in seen:
            raise MathHarnessError(f"bundle line {line_number}: invalid/duplicate digest")
        # The contract independently recomputes the digest when a response is
        # validated.  Validate the request now with a deliberately
        # nonapplicable schema-valid placeholder response.
        validate_response_envelope(
            {
                "request_sha256": digest,
                "raw_response": '{"applies":false,"violated":false,"witnesses":[]}',
            },
            request,
        )
        seen.add(digest)
        requests.append(request)
    if len(requests) < SMOKE_SIZE:
        raise MathHarnessError("bundle is too small for the exact smoke")
    return requests


def _invoke(request: dict, timeout_seconds: float) -> tuple[int | None, str, str, bool]:
    argv = [
        "claude",
        "--model",
        request["model"],
        "--output-format",
        "text",
        "--system-prompt",
        request["system_prompt"],
        "-p",
        request["user_prompt"],
    ]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return completed.returncode, completed.stdout, completed.stderr, False
    except subprocess.TimeoutExpired as exc:
        return (
            None,
            exc.stdout if isinstance(exc.stdout, str) else "",
            exc.stderr if isinstance(exc.stderr, str) else "",
            True,
        )


Invoker = Callable[[dict, float], tuple[int | None, str, str, bool]]


def _load_existing(path: Path, requests: Sequence[dict]) -> dict[str, dict]:
    known = {row["request_sha256"] for row in requests}
    values: dict[str, dict] = {}
    if not path.exists():
        return values
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        digest = row.get("request_sha256")
        if row.get("schema") != ENVELOPE_SCHEMA or digest not in known or digest in values:
            raise MathHarnessError(f"response line {line_number}: invalid or duplicate envelope")
        request = next(value for value in requests if value["request_sha256"] == digest)
        if row.get("status") == "valid":
            replay = validate_response_envelope(row, request)
            if row.get("validated_response") != replay:
                raise MathHarnessError(f"response line {line_number}: replay mismatch")
        values[digest] = row
    return values


def _execute_one(
    request: dict,
    *,
    index: int,
    phase: str,
    timeout_seconds: float,
    run_id: str,
    invoker: Invoker,
) -> dict:
    started = _utc_now()
    returncode, stdout, stderr, timed_out = invoker(request, timeout_seconds)
    completed = _utc_now()
    status = "process_error"
    validated = None
    contract_error = None
    if returncode == 0 and not timed_out:
        try:
            validated = validate_response_envelope(
                {
                    "request_sha256": request["request_sha256"],
                    "raw_response": stdout,
                },
                request,
            )
            status = "valid"
        except (SchemaError, ValueError) as exc:
            status = "contract_error"
            contract_error = f"{type(exc).__name__}: {exc}"
    return {
        "schema": ENVELOPE_SCHEMA,
        "run_id": run_id,
        "phase": phase,
        "request_index": index,
        "request_sha256": request["request_sha256"],
        "pair_id": request["pair"]["pair_id"],
        "pass_index": request["pass_index"],
        "model": request["model"],
        "split": request["split"],
        "attempt_index": 1,
        "started_at": started,
        "completed_at": completed,
        "status": status,
        "returncode": returncode,
        "timed_out": timed_out,
        "raw_response": stdout,
        "stderr": stderr[-4000:],
        "contract_error": contract_error,
        "validated_response": validated,
    }


def run_phase(
    *,
    requests: Sequence[dict],
    responses_path: Path,
    phase: str,
    concurrency: int,
    timeout_seconds: float,
    invoker: Invoker = _invoke,
) -> dict:
    if phase not in {"smoke", "production"}:
        raise MathHarnessError("phase must be smoke or production")
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise MathHarnessError("concurrency must be between 1 and 4")
    existing = _load_existing(responses_path, requests)
    smoke = requests[:SMOKE_SIZE]
    if phase == "smoke":
        if any(row["request_sha256"] in existing for row in smoke):
            raise MathHarnessError("smoke is one-shot; an envelope already exists")
        selected = list(enumerate(smoke))
    else:
        smoke_rows = [existing.get(row["request_sha256"]) for row in smoke]
        if any(row is None or row.get("status") != "valid" for row in smoke_rows):
            valid = sum(row is not None and row.get("status") == "valid" for row in smoke_rows)
            raise StopProduction(f"STOP: smoke must be 10/10 valid; observed {valid}/10")
        selected = [
            (index, request)
            for index, request in enumerate(requests)
            if index >= SMOKE_SIZE and request["request_sha256"] not in existing
        ]
    run_id = str(uuid.uuid4())
    results: list[dict] = []
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    with responses_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    _execute_one,
                    request,
                    index=index,
                    phase=phase,
                    timeout_seconds=timeout_seconds,
                    run_id=run_id,
                    invoker=invoker,
                ): index
                for index, request in selected
            }
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                # Commit every completed invocation before scheduling state can
                # be lost; a long production run is resumable from this ledger.
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                handle.flush()
        results.sort(key=lambda row: row["request_index"])
    return {
        "phase": phase,
        "selected": len(selected),
        "valid": sum(row["status"] == "valid" for row in results),
        "contract_error": sum(row["status"] == "contract_error" for row in results),
        "process_error": sum(row["status"] == "process_error" for row in results),
    }


def recover_smoke(
    *,
    requests: Sequence[dict],
    source_responses_path: Path,
    recovered_responses_path: Path,
) -> dict:
    """Replay one failed smoke through a corrected parser, without model calls."""

    if recovered_responses_path.exists():
        raise MathHarnessError("recovered smoke output already exists")
    source_rows = [
        json.loads(line)
        for line in source_responses_path.read_text(encoding="utf-8").splitlines()
    ]
    by_index = {row.get("request_index"): row for row in source_rows}
    if set(by_index) != set(range(SMOKE_SIZE)) or len(source_rows) != SMOKE_SIZE:
        raise MathHarnessError("source smoke must contain exactly request indices 0..9")
    recovered: list[dict] = []
    for index, request in enumerate(requests[:SMOKE_SIZE]):
        source = by_index[index]
        if source.get("request_sha256") != request["request_sha256"]:
            raise MathHarnessError("source smoke request digest mismatch")
        validated = validate_response_envelope(source, request)
        recovered.append(
            {
                **source,
                "schema": ENVELOPE_SCHEMA,
                "status": "valid",
                "contract_error": None,
                "validated_response": validated,
                "recovery": {
                    "kind": "cpu_parser_replay_no_model_call",
                    "source_responses_path": str(source_responses_path),
                    "source_responses_sha256": _file_sha(source_responses_path),
                },
            }
        )
    recovered_responses_path.parent.mkdir(parents=True, exist_ok=True)
    with recovered_responses_path.open("x", encoding="utf-8") as handle:
        for row in recovered:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return {"recovered": len(recovered), "valid": len(recovered), "model_calls": 0}


def import_completed(
    *,
    requests: Sequence[dict],
    source_responses_path: Path,
    imported_responses_path: Path,
) -> dict:
    """Reuse byte-identical completed requests in a smaller additive bundle."""

    if imported_responses_path.exists():
        raise MathHarnessError("imported response output already exists")
    source_rows = [
        json.loads(line)
        for line in source_responses_path.read_text(encoding="utf-8").splitlines()
    ]
    valid_by_digest = {
        row["request_sha256"]: row
        for row in source_rows
        if row.get("status") == "valid"
    }
    imported: list[dict] = []
    for index, request in enumerate(requests):
        source = valid_by_digest.get(request["request_sha256"])
        if source is None:
            continue
        validated = validate_response_envelope(source, request)
        imported.append(
            {
                **source,
                "schema": ENVELOPE_SCHEMA,
                "request_index": index,
                "phase": "smoke" if index < SMOKE_SIZE else "production",
                "status": "valid",
                "contract_error": None,
                "validated_response": validated,
                "recovery": {
                    "kind": "byte_identical_request_reuse_no_model_call",
                    "source_responses_path": str(source_responses_path),
                    "source_responses_sha256": _file_sha(source_responses_path),
                },
            }
        )
    imported_indices = {row["request_index"] for row in imported}
    if not set(range(SMOKE_SIZE)) <= imported_indices:
        raise MathHarnessError("source responses do not cover the new bundle's exact smoke")
    imported_responses_path.parent.mkdir(parents=True, exist_ok=True)
    with imported_responses_path.open("x", encoding="utf-8") as handle:
        for row in imported:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return {"imported": len(imported), "model_calls": 0}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--phase",
        choices=("smoke", "recover-smoke", "import-completed", "production"),
        required=True,
    )
    parser.add_argument("--source-responses", type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args(argv)
    requests = load_bundle(args.requests, model=args.model)
    if args.phase in {"recover-smoke", "import-completed"}:
        if args.source_responses is None:
            raise MathHarnessError(f"{args.phase} requires --source-responses")
        if args.phase == "recover-smoke":
            summary = recover_smoke(
                requests=requests,
                source_responses_path=args.source_responses,
                recovered_responses_path=args.responses,
            )
        else:
            summary = import_completed(
                requests=requests,
                source_responses_path=args.source_responses,
                imported_responses_path=args.responses,
            )
    else:
        summary = run_phase(
            requests=requests,
            responses_path=args.responses,
            phase=args.phase,
            concurrency=args.concurrency,
            timeout_seconds=args.timeout_seconds,
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
