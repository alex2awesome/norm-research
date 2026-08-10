#!/usr/bin/env python3
"""Compile, execute, and analyze the bounded Patent pre-authoring probe."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence
import uuid

from methods.metric_seam.verifiers.construct_probe import (
    PATENT_ANTECEDENT_PROPOSAL,
    REQUEST_SCHEMA,
    SYSTEM_PROMPT,
    compile_sample_requests,
    validate_response_envelope,
)
from methods.metric_seam.verifiers.lifecycle import evaluate_pre_authoring_probe
from methods.metric_seam.verifiers.schema import SchemaError, Verdict


ENVELOPE_SCHEMA = "metric-seam.construct-base-rate-response-envelope.v1"
SMOKE_SIZE = 5
MAX_CONCURRENCY = 4


def _rows(bundle: object) -> list[dict[str, object]]:
    if isinstance(bundle, list):
        rows = bundle
    elif isinstance(bundle, dict):
        rows = next((bundle[key] for key in ("train_items", "items", "rows") if isinstance(bundle.get(key), list)), None)
    else:
        rows = None
    if not isinstance(rows, list) or not rows:
        raise ValueError("bundle has no TRAIN rows")
    values = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("item_key"), str) or not isinstance(row.get("ctext"), str):
            raise ValueError("TRAIN rows require item_key and ctext")
        values.append({"item_key": row["item_key"], "ctext": row["ctext"]})
    return values


def compile_bundle(source: Path, proposal_output: Path, requests_output: Path, *, model: str) -> None:
    if "heldout" in str(source).casefold():
        raise ValueError("pre-authoring probe refuses held-out paths")
    rows = _rows(json.loads(source.read_text(encoding="utf-8")))
    requests = compile_sample_requests(PATENT_ANTECEDENT_PROPOSAL, rows, model=model)
    proposal_output.parent.mkdir(parents=True, exist_ok=True)
    proposal_output.write_text(json.dumps(PATENT_ANTECEDENT_PROPOSAL.to_json_value(), indent=2, sort_keys=True) + "\n")
    requests_output.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in requests))


def _load_requests(path: Path, *, model: str) -> list[dict[str, object]]:
    requests = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(requests) != 32 or any(
        row.get("schema") != REQUEST_SCHEMA or row.get("model") != model
        or row.get("system_prompt") != SYSTEM_PROMPT or row.get("split") != "compiler_train"
        for row in requests
    ):
        raise ValueError("request bundle is not the frozen 32-item TRAIN probe")
    return requests


def _existing(path: Path, requests: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    known = {row["request_sha256"]: row for row in requests}
    values: dict[str, dict[str, object]] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        digest = row.get("request_sha256")
        if row.get("schema") != ENVELOPE_SCHEMA or digest not in known or digest in values:
            raise ValueError("invalid or duplicate retained response")
        if row.get("status") == "valid":
            replay = validate_response_envelope(row, known[digest])
            if row.get("validated_response") != replay:
                raise ValueError("retained response replay mismatch")
        values[digest] = row
    return values


def _invoke(request: Mapping[str, object], timeout: float) -> tuple[int | None, str, str, bool]:
    try:
        result = subprocess.run(
            ["claude", "--model", str(request["model"]), "--output-format", "text",
             "--system-prompt", str(request["system_prompt"]), "-p", str(request["user_prompt"])],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        return result.returncode, result.stdout, result.stderr, False
    except subprocess.TimeoutExpired as exc:
        return None, exc.stdout if isinstance(exc.stdout, str) else "", exc.stderr if isinstance(exc.stderr, str) else "", True


def _execute(request: Mapping[str, object], *, index: int, phase: str, timeout: float, run_id: str) -> dict[str, object]:
    started = datetime.now(timezone.utc).isoformat()
    returncode, stdout, stderr, timed_out = _invoke(request, timeout)
    status, validated, error = "process_error", None, None
    if returncode == 0 and not timed_out:
        try:
            validated = validate_response_envelope({"request_sha256": request["request_sha256"], "raw_response": stdout}, request)
            status = "valid"
        except (SchemaError, ValueError) as exc:
            status, error = "contract_error", f"{type(exc).__name__}: {exc}"
    return {
        "schema": ENVELOPE_SCHEMA, "run_id": run_id, "phase": phase,
        "request_index": index, "request_sha256": request["request_sha256"],
        "item_key": request["item_key"], "model": request["model"], "split": request["split"],
        "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": status, "returncode": returncode, "timed_out": timed_out,
        "raw_response": stdout, "stderr": stderr[-4000:], "contract_error": error,
        "validated_response": validated,
    }


def run_phase(requests_path: Path, responses_path: Path, *, model: str, phase: str, concurrency: int, timeout: float) -> dict[str, object]:
    if phase not in {"smoke", "production"} or not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError("invalid phase or concurrency")
    requests = _load_requests(requests_path, model=model)
    existing = _existing(responses_path, requests)
    if phase == "smoke":
        if existing:
            raise ValueError("smoke is one-shot")
        selected = list(enumerate(requests[:SMOKE_SIZE]))
    else:
        smoke = [existing.get(row["request_sha256"]) for row in requests[:SMOKE_SIZE]]
        if any(row is None or row.get("status") != "valid" for row in smoke):
            raise ValueError("STOP: production requires a 5/5 valid smoke")
        selected = [(i, row) for i, row in enumerate(requests) if row["request_sha256"] not in existing]
    run_id = str(uuid.uuid4())
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    completed = []
    with responses_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_execute, row, index=i, phase=phase, timeout=timeout, run_id=run_id): i for i, row in selected}
        for future in as_completed(futures):
            value = future.result()
            completed.append(value)
            handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
    counts = Counter(str(row["status"]) for row in completed)
    return {"phase": phase, "executed": len(completed), "status_counts": dict(sorted(counts.items()))}


def analyze(requests_path: Path, responses_path: Path) -> dict[str, object]:
    requests = _load_requests(requests_path, model=json.loads(requests_path.read_text().splitlines()[0])["model"])
    existing = _existing(responses_path, requests)
    verdicts: dict[str, Verdict | None] = {}
    statuses = Counter()
    for request in requests:
        row = existing.get(request["request_sha256"])
        status = "missing" if row is None else str(row["status"])
        statuses[status] += 1
        verdicts[str(request["item_key"])] = (
            Verdict.from_json(row["validated_response"]["verdict"])
            if row is not None and row.get("status") == "valid" else None
        )
    result = evaluate_pre_authoring_probe(
        PATENT_ANTECEDENT_PROPOSAL,
        [str(row["item_key"]) for row in requests], verdicts,
    )
    result["response_status_counts"] = dict(sorted(statuses.items()))
    result["model"] = requests[0]["model"]
    result["claim_limits"] = [
        "This cheap prompt-based TRAIN screen licenses detector authorship; it is not a verifiability result.",
        "No code detector, code output, held-out item, or supervised external anchor was supplied to the judge.",
    ]
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--source", type=Path, required=True); c.add_argument("--proposal", type=Path, required=True); c.add_argument("--requests", type=Path, required=True); c.add_argument("--model", required=True)
    r = sub.add_parser("run")
    r.add_argument("--requests", type=Path, required=True); r.add_argument("--responses", type=Path, required=True); r.add_argument("--model", required=True); r.add_argument("--phase", choices=("smoke", "production"), required=True); r.add_argument("--concurrency", type=int, default=4); r.add_argument("--timeout", type=float, default=180)
    a = sub.add_parser("analyze")
    a.add_argument("--requests", type=Path, required=True); a.add_argument("--responses", type=Path, required=True); a.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "compile":
        compile_bundle(args.source, args.proposal, args.requests, model=args.model)
    elif args.command == "run":
        print(json.dumps(run_phase(args.requests, args.responses, model=args.model, phase=args.phase, concurrency=args.concurrency, timeout=args.timeout), sort_keys=True))
    else:
        result = analyze(args.requests, args.responses)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
