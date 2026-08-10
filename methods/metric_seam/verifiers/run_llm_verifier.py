"""Execute frozen LLM-verifier requests behind an exact smoke/production gate.

This additive harness is deliberately narrow:

* it reads digest-pinned requests produced by :mod:`llm_contract`;
* it runs each Claude invocation as a fresh ``-p`` subprocess (never a resumed
  session), with one model pinned for the whole bundle;
* it appends one raw, digest-bound response envelope per attempt;
* it validates retained responses through ``llm_contract``; and
* it refuses production until the first ten bundle requests are 10/10 valid.

The harness has no provider SDK and performs no work unless explicitly invoked.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence
import uuid

from .llm_contract import (
    PARSER_VERSION,
    REQUEST_SCHEMA,
    SYSTEM_PROMPT,
    UnitContract,
    smoke_passes,
    validate_response_envelope,
)
from .schema import SchemaError


ENVELOPE_SCHEMA = "metric-seam.verifier-llm-response-envelope.v2"
FREEZE_RECEIPT_SCHEMA = "metric-seam.verifier-heldout-freeze-receipt.v1"
SMOKE_SIZE = 10
MAX_CONCURRENCY = 4
DEFAULT_COMMAND_TEMPLATE = (
    "claude",
    "--model",
    "{model}",
    "--output-format",
    "text",
    "--system-prompt",
    "{system_prompt}",
    "-p",
    "{user_prompt}",
)
_REQUEST_IDENTITY_KEYS = (
    "schema",
    "unit",
    "item_key",
    "pass_index",
    "model",
    "split",
    "system_prompt",
    "user_prompt",
)
_REQUEST_KEYS = set(_REQUEST_IDENTITY_KEYS) | {
    "request_sha256",
    "response_contract",
}
_FORBIDDEN_SESSION_OPTIONS = {"--resume", "--continue", "-c"}


class HarnessError(ValueError):
    """Raised when a bundle, retained envelope, or run setting is invalid."""


class StopProduction(HarnessError):
    """Raised before execution when the exact ten-row smoke has not passed."""


@dataclass(frozen=True)
class InvocationResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    exception_type: str | None = None


Invoker = Callable[[Sequence[str], float], InvocationResult]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_REQUIRED_FREEZE_ROLES = {
    "v_ast_implementation",
    "train_gate_readout",
    "llm_contract_source",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_freeze_receipt(
    receipt_path: Path,
    *,
    heldout_bundle_path: Path,
    model: str,
) -> dict:
    """Validate a pre-existing freeze receipt and every artifact it binds.

    Relative artifact paths are resolved from the receipt directory.  The
    receipt is deliberately produced outside this runner: heldout execution
    may consume a freeze decision, but may not manufacture one after seeing a
    heldout response.
    """

    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError("heldout freeze receipt is missing or invalid") from exc
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema",
        "status",
        "model",
        "heldout_bundle_sha256",
        "frozen_artifacts",
    }:
        raise HarnessError("heldout freeze receipt keys do not match contract")
    if receipt["schema"] != FREEZE_RECEIPT_SCHEMA:
        raise HarnessError("unsupported heldout freeze receipt schema")
    if receipt["status"] != "frozen_before_sealed_heldout":
        raise HarnessError("heldout freeze receipt is not finalized")
    if receipt["model"] != model:
        raise HarnessError("heldout freeze receipt model mismatch")
    if receipt["heldout_bundle_sha256"] != _file_sha256(heldout_bundle_path):
        raise HarnessError("heldout bundle changed after freeze")
    artifacts = receipt["frozen_artifacts"]
    if not isinstance(artifacts, list):
        raise HarnessError("frozen_artifacts must be an array")
    roles: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict) or set(artifact) != {"role", "path", "sha256"}:
            raise HarnessError(f"freeze artifact {index} has invalid keys")
        role = artifact["role"]
        raw_path = artifact["path"]
        digest = artifact["sha256"]
        if not isinstance(role, str) or not role or role in roles:
            raise HarnessError(f"freeze artifact {index} has duplicate/invalid role")
        if not isinstance(raw_path, str) or not raw_path:
            raise HarnessError(f"freeze artifact {index} has invalid path")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise HarnessError(f"freeze artifact {index} has invalid SHA-256")
        artifact_path = Path(raw_path)
        if not artifact_path.is_absolute():
            artifact_path = receipt_path.parent / artifact_path
        if not artifact_path.is_file() or _file_sha256(artifact_path) != digest:
            raise HarnessError(f"freeze artifact {role!r} changed or is missing")
        roles.add(role)
    missing_roles = sorted(_REQUIRED_FREEZE_ROLES - roles)
    if missing_roles:
        raise HarnessError(f"heldout freeze receipt omits roles: {missing_roles}")
    return receipt


def _validate_request(request: object, *, model: str, line_number: int) -> dict:
    if not isinstance(request, dict) or set(request) != _REQUEST_KEYS:
        raise HarnessError(f"bundle line {line_number}: request keys do not match contract")
    if request.get("schema") != REQUEST_SCHEMA:
        raise HarnessError(f"bundle line {line_number}: unsupported request schema")
    if request.get("model") != model:
        raise HarnessError(
            f"bundle line {line_number}: request model is not pinned to {model!r}"
        )
    if request.get("system_prompt") != SYSTEM_PROMPT:
        raise HarnessError(f"bundle line {line_number}: system prompt is not frozen")
    unit = request.get("unit")
    if not isinstance(unit, dict) or set(unit) != {
        "unit_id",
        "metric_name",
        "relation",
        "cuf_executor",
        "cuf_node_id",
    }:
        raise HarnessError(f"bundle line {line_number}: invalid unit contract")
    try:
        UnitContract(**unit)
    except (TypeError, ValueError) as exc:
        raise HarnessError(f"bundle line {line_number}: invalid unit contract") from exc
    if not isinstance(request.get("item_key"), str) or not request["item_key"]:
        raise HarnessError(f"bundle line {line_number}: invalid item key")
    if request.get("pass_index") not in (1, 2):
        raise HarnessError(f"bundle line {line_number}: invalid pass index")
    if request.get("split") not in {"compiler_train", "sealed_heldout"}:
        raise HarnessError(f"bundle line {line_number}: invalid split")
    user_prompt = request.get("user_prompt")
    if not isinstance(user_prompt, str) or "ITEM (unified diff):\n" not in user_prompt:
        raise HarnessError(f"bundle line {line_number}: request has no unified diff")
    response_contract = request.get("response_contract")
    if response_contract != {
        "parser_version": PARSER_VERSION,
        "floats_allowed": False,
        "witnesses_must_address_visible_new_side_lines": True,
    }:
        raise HarnessError(f"bundle line {line_number}: parser contract is not pinned")
    identity = {key: request[key] for key in _REQUEST_IDENTITY_KEYS}
    if request.get("request_sha256") != _canonical_sha256(identity):
        raise HarnessError(f"bundle line {line_number}: request digest mismatch")
    return request


def load_frozen_bundle(path: Path, *, model: str) -> list[dict]:
    """Load and independently validate one direct-request JSONL bundle."""

    if not model.strip():
        raise HarnessError("model must be nonempty")
    requests: list[dict] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise HarnessError(f"bundle line {line_number}: blank lines are forbidden")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HarnessError(f"bundle line {line_number}: invalid JSON") from exc
            request = _validate_request(value, model=model, line_number=line_number)
            digest = request["request_sha256"]
            if digest in seen:
                raise HarnessError(f"bundle line {line_number}: duplicate request digest")
            seen.add(digest)
            requests.append(request)
    if len(requests) < SMOKE_SIZE:
        raise HarnessError(f"bundle requires at least {SMOKE_SIZE} requests")
    splits = {request["split"] for request in requests}
    if len(splits) != 1:
        raise HarnessError("a frozen bundle may contain exactly one split")
    return requests


def parse_command_template(template: str | None) -> tuple[str, ...]:
    """Parse shell-like syntax once, before prompt substitution.

    Substitution occurs token-by-token and the subprocess never uses a shell,
    so quotes or metacharacters inside a diff cannot create new argv entries.
    """

    tokens = tuple(shlex.split(template)) if template is not None else DEFAULT_COMMAND_TEMPLATE
    if not tokens:
        raise HarnessError("command template is empty")
    joined = "\0".join(tokens)
    for placeholder in ("{model}", "{system_prompt}", "{user_prompt}"):
        if placeholder not in joined:
            raise HarnessError(f"command template is missing {placeholder}")
    return tokens


def build_command(template: Sequence[str], request: Mapping[str, object]) -> tuple[str, ...]:
    substitutions = {
        "model": str(request["model"]),
        "system_prompt": str(request["system_prompt"]),
        "user_prompt": str(request["user_prompt"]),
        "request_sha256": str(request["request_sha256"]),
        "item_key": str(request["item_key"]),
    }
    try:
        argv = tuple(token.format_map(substitutions) for token in template)
    except (KeyError, ValueError) as exc:
        raise HarnessError("command template contains an unsupported placeholder") from exc
    if "-p" not in argv:
        raise HarnessError("every verifier invocation must be a fresh -p call")
    forbidden = [
        token
        for token in argv
        if token in _FORBIDDEN_SESSION_OPTIONS or token.startswith("--resume=")
    ]
    if forbidden:
        raise HarnessError(f"session-resuming options are forbidden: {forbidden}")
    model_values: list[str] = []
    for index, token in enumerate(argv):
        if token == "--model" and index + 1 < len(argv):
            model_values.append(argv[index + 1])
        elif token.startswith("--model="):
            model_values.append(token.split("=", 1)[1])
    if model_values != [str(request["model"])]:
        raise HarnessError(
            "rendered command must contain exactly one --model option with the pinned model"
        )
    return argv


def _invoke_subprocess(argv: Sequence[str], timeout_seconds: float) -> InvocationResult:
    try:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return InvocationResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        return InvocationResult(
            returncode=None,
            stdout=exc.stdout if isinstance(exc.stdout, str) else "",
            stderr=exc.stderr if isinstance(exc.stderr, str) else "",
            timed_out=True,
            exception_type=type(exc).__name__,
        )
    except OSError as exc:
        return InvocationResult(
            returncode=None,
            stdout="",
            stderr=str(exc),
            exception_type=type(exc).__name__,
        )


def _load_envelopes(path: Path, requests: Sequence[dict]) -> dict[str, list[dict]]:
    by_digest = {request["request_sha256"]: [] for request in requests}
    index_by_digest = {
        request["request_sha256"]: index for index, request in enumerate(requests)
    }
    request_by_digest = {request["request_sha256"]: request for request in requests}
    if not path.exists():
        return by_digest
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HarnessError(f"response line {line_number}: invalid JSON") from exc
            if not isinstance(row, dict) or row.get("schema") != ENVELOPE_SCHEMA:
                raise HarnessError(f"response line {line_number}: invalid envelope schema")
            digest = row.get("request_sha256")
            if digest not in by_digest:
                raise HarnessError(f"response line {line_number}: unknown request digest")
            if row.get("request_index") != index_by_digest[digest]:
                raise HarnessError(f"response line {line_number}: request index mismatch")
            phase = row.get("phase")
            expected_phase = (
                "heldout_finalize"
                if request_by_digest[digest]["split"] == "sealed_heldout"
                else (
                    "smoke"
                    if index_by_digest[digest] < SMOKE_SIZE
                    else "production"
                )
            )
            if phase != expected_phase:
                raise HarnessError(
                    f"response line {line_number}: expected {expected_phase} phase"
                )
            attempts = by_digest[digest]
            if row.get("attempt_index") != len(attempts) + 1:
                raise HarnessError(f"response line {line_number}: nonsequential attempt")
            if any(attempt.get("status") == "valid" for attempt in attempts):
                raise HarnessError(
                    f"response line {line_number}: request re-executed after valid response"
                )
            status = row.get("status")
            if status not in {"valid", "contract_error", "process_error"}:
                raise HarnessError(f"response line {line_number}: invalid status")
            if row.get("model") != request_by_digest[digest]["model"]:
                raise HarnessError(f"response line {line_number}: model mismatch")
            if row.get("split") != request_by_digest[digest]["split"]:
                raise HarnessError(f"response line {line_number}: split mismatch")
            if status == "valid":
                try:
                    validated = validate_response_envelope(
                        row, request_by_digest[digest]
                    )
                except (SchemaError, ValueError) as exc:
                    raise HarnessError(
                        f"response line {line_number}: retained valid response fails replay"
                    ) from exc
                if row.get("validated_response") != validated:
                    raise HarnessError(
                        f"response line {line_number}: validated response mismatch"
                    )
            attempts.append(row)
    return by_digest


def _smoke_rows(
    requests: Sequence[dict], attempts: Mapping[str, Sequence[dict]]
) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for request in requests[:SMOKE_SIZE]:
        request_attempts = attempts[request["request_sha256"]]
        valid = [row for row in request_attempts if row.get("status") == "valid"]
        if valid and valid[0].get("phase") == "smoke":
            rows.append(valid[0])
        elif request_attempts:
            rows.append(request_attempts[-1])
    return rows


def _production_gate(
    requests: Sequence[dict], attempts: Mapping[str, Sequence[dict]]
) -> None:
    rows = _smoke_rows(requests, attempts)
    if not smoke_passes(rows, expected=SMOKE_SIZE):
        valid = sum(row.get("status") == "valid" for row in rows)
        raise StopProduction(
            f"STOP: production requires the same first {SMOKE_SIZE} frozen requests "
            f"to be {SMOKE_SIZE}/{SMOKE_SIZE} valid; observed {valid}/{SMOKE_SIZE}"
        )


def _make_envelope(
    *,
    request: dict,
    request_index: int,
    phase: str,
    attempt_index: int,
    run_id: str,
    argv: Sequence[str],
    result: InvocationResult,
    started_at: str,
    completed_at: str,
    duration_ms: int,
    freeze_receipt_sha256: str | None = None,
) -> dict:
    envelope: dict[str, object] = {
        "schema": ENVELOPE_SCHEMA,
        "request_sha256": request["request_sha256"],
        "request_index": request_index,
        "item_key": request["item_key"],
        "unit_id": request["unit"]["unit_id"],
        "pass_index": request["pass_index"],
        "model": request["model"],
        "split": request["split"],
        "phase": phase,
        "attempt_index": attempt_index,
        "run_id": run_id,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "duration_ms": duration_ms,
        "command_program": argv[0],
        "command_sha256": _canonical_sha256(list(argv)),
        "fresh_print_mode": "-p" in argv,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "exception_type": result.exception_type,
        "raw_response": result.stdout,
        "stderr": result.stderr,
    }
    if freeze_receipt_sha256 is not None:
        envelope["freeze_receipt_sha256"] = freeze_receipt_sha256
    if result.returncode != 0 or result.timed_out or result.exception_type:
        envelope["status"] = "process_error"
        envelope["validation_error"] = "subprocess did not complete successfully"
        return envelope
    try:
        validated = validate_response_envelope(envelope, request)
    except (SchemaError, ValueError) as exc:
        envelope["status"] = "contract_error"
        envelope["validation_error"] = f"{type(exc).__name__}: {exc}"
    else:
        envelope["status"] = "valid"
        envelope["validated_response"] = validated
    return envelope


def _append_envelope(path: Path, envelope: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(envelope, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        )
        handle.flush()


def run_bundle(
    *,
    bundle_path: Path,
    output_path: Path,
    phase: str,
    model: str,
    command_template: str | None = None,
    max_concurrency: int = 1,
    max_attempts: int = 1,
    timeout_seconds: float = 300.0,
    dry_run: bool = False,
    freeze_receipt_path: Path | None = None,
    invoker: Invoker = _invoke_subprocess,
) -> dict:
    """Execute one gated phase and return exact accounting."""

    if phase not in {"smoke", "production", "heldout_finalize"}:
        raise HarnessError("phase must be smoke, production, or heldout_finalize")
    if not 1 <= max_concurrency <= MAX_CONCURRENCY:
        raise HarnessError(f"max_concurrency must be between 1 and {MAX_CONCURRENCY}")
    if max_attempts < 1:
        raise HarnessError("max_attempts must be positive")
    if phase in {"smoke", "heldout_finalize"} and max_attempts != 1:
        raise HarnessError(f"{phase} requires exactly one attempt per request")
    if timeout_seconds <= 0:
        raise HarnessError("timeout_seconds must be positive")
    requests = load_frozen_bundle(bundle_path, model=model)
    bundle_split = str(requests[0]["split"])
    if phase in {"smoke", "production"} and bundle_split != "compiler_train":
        raise HarnessError(f"{phase} accepts compiler_train bundles only")
    if phase == "heldout_finalize" and bundle_split != "sealed_heldout":
        raise HarnessError("heldout_finalize accepts sealed_heldout bundles only")
    freeze_receipt: dict | None = None
    if phase == "heldout_finalize":
        if freeze_receipt_path is None:
            raise HarnessError("heldout_finalize requires --freeze-receipt")
        freeze_receipt = validate_freeze_receipt(
            freeze_receipt_path,
            heldout_bundle_path=bundle_path,
            model=model,
        )
        if output_path.exists():
            raise HarnessError(
                "sealed-heldout execution ledger already exists; a second execution is forbidden"
            )
    template = parse_command_template(command_template)
    attempts = _load_envelopes(output_path, requests)
    if phase == "production":
        _production_gate(requests, attempts)
        scoped = list(enumerate(requests))[SMOKE_SIZE:]
    elif phase == "heldout_finalize":
        scoped = list(enumerate(requests))
    else:
        scoped = list(enumerate(requests))[:SMOKE_SIZE]

    def eligible(index: int, request: dict) -> bool:
        existing = attempts[request["request_sha256"]]
        return (
            not any(row.get("status") == "valid" for row in existing)
            and len(existing) < max_attempts
        )

    pending = [(index, request) for index, request in scoped if eligible(index, request)]
    summary: dict[str, object] = {
        "phase": phase,
        "dry_run": dry_run,
        "bundle_path": str(bundle_path),
        "bundle_sha256": _file_sha256(bundle_path),
        "output_path": str(output_path),
        "model": model,
        "split": bundle_split,
        "request_count": len(requests),
        "phase_request_count": len(scoped),
        "already_valid_count": sum(
            any(row.get("status") == "valid" for row in attempts[request["request_sha256"]])
            for _, request in scoped
        ),
        "planned_request_count": len(pending),
        "planned_request_sha256": [request["request_sha256"] for _, request in pending],
        "attempts_appended": 0,
        "status_counts_appended": {},
        "max_concurrency": max_concurrency,
        "max_attempts_per_request": max_attempts,
    }
    freeze_receipt_sha256: str | None = None
    if freeze_receipt_path is not None:
        freeze_receipt_sha256 = _file_sha256(freeze_receipt_path)
        summary["freeze_receipt_path"] = str(freeze_receipt_path)
        summary["freeze_receipt_sha256"] = freeze_receipt_sha256
    if dry_run:
        if phase == "smoke":
            summary["smoke_passed"] = smoke_passes(
                _smoke_rows(requests, attempts), expected=SMOKE_SIZE
            )
        return summary

    if phase == "heldout_finalize":
        # Claim the one-shot ledger before any sealed request is invoked.  A
        # crash may leave a partial ledger, but it can never authorize a rerun.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            output_path.open("x", encoding="utf-8").close()
        except FileExistsError as exc:  # concurrent finalizer fail-closed
            raise HarnessError(
                "sealed-heldout execution ledger was claimed concurrently"
            ) from exc

    run_id = str(uuid.uuid4())
    appended: list[dict] = []
    while pending:
        futures = {}
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            for index, request in pending:
                argv = build_command(template, request)
                started_at = _utc_now()
                started_clock = time.monotonic()
                future = executor.submit(invoker, argv, timeout_seconds)
                futures[future] = (
                    index,
                    request,
                    argv,
                    started_at,
                    started_clock,
                )
            for future in as_completed(futures):
                index, request, argv, started_at, started_clock = futures[future]
                try:
                    result = future.result()
                    if not isinstance(result, InvocationResult):
                        raise TypeError("invoker did not return InvocationResult")
                except Exception as exc:  # retained as process accounting
                    result = InvocationResult(
                        returncode=None,
                        stdout="",
                        stderr=str(exc),
                        exception_type=type(exc).__name__,
                    )
                envelope = _make_envelope(
                    request=request,
                    request_index=index,
                    phase=phase,
                    attempt_index=len(attempts[request["request_sha256"]]) + 1,
                    run_id=run_id,
                    argv=argv,
                    result=result,
                    started_at=started_at,
                    completed_at=_utc_now(),
                    duration_ms=max(0, round((time.monotonic() - started_clock) * 1000)),
                    freeze_receipt_sha256=freeze_receipt_sha256,
                )
                _append_envelope(output_path, envelope)
                attempts[request["request_sha256"]].append(envelope)
                appended.append(envelope)
        pending = [
            (index, request) for index, request in scoped if eligible(index, request)
        ]

    status_counts: dict[str, int] = {}
    for row in appended:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    summary["attempts_appended"] = len(appended)
    summary["status_counts_appended"] = dict(sorted(status_counts.items()))
    summary["run_id"] = run_id
    summary["final_valid_count"] = sum(
        any(row.get("status") == "valid" for row in attempts[request["request_sha256"]])
        for _, request in scoped
    )
    if phase == "smoke":
        summary["smoke_passed"] = smoke_passes(
            _smoke_rows(requests, attempts), expected=SMOKE_SIZE
        )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("smoke", "production", "heldout_finalize"),
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--command-template",
        help=(
            "Shell-like argv template containing {model}, {system_prompt}, and "
            "{user_prompt}; parsed before substitution and never run through a shell"
        ),
    )
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--freeze-receipt",
        type=Path,
        help="required pre-existing freeze receipt for heldout_finalize",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run_bundle(
            bundle_path=args.bundle,
            output_path=args.output,
            phase=args.phase,
            model=args.model,
            command_template=args.command_template,
            max_concurrency=args.max_concurrency,
            max_attempts=args.max_attempts,
            timeout_seconds=args.timeout_seconds,
            dry_run=args.dry_run,
            freeze_receipt_path=args.freeze_receipt,
        )
    except StopProduction as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except (HarnessError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if args.phase == "smoke" and not args.dry_run and not summary["smoke_passed"]:
        print("STOP: smoke was not 10/10 valid; production remains disabled", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
