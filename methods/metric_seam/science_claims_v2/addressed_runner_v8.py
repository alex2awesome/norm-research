#!/usr/bin/env python3
"""Serial-ready, hash-bound API transport for science v8.

Importing this module performs no network operation.  The CLI additionally requires an
explicit ``--execute`` flag.  Tests inject a sender; the default sender is the only code
path that opens a network connection.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from . import addressed_pipeline_v8 as pipeline


ROOT = pipeline.ROOT
RESULT_SCHEMA = pipeline.RESULT_SCHEMA

PROMPT_RESPONSE_JSON_SCHEMA = {
    "name": "science_articulability_addressed_v8",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["paper_id", "selections"],
        "properties": {
            "paper_id": {"type": "string"},
            "selections": {
                "type": "array",
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "claim_sentence_id", "evidence_sentence_id", "decision",
                        "relation", "quantity_state", "comparison_state",
                        "evidence_kind", "quantity_count", "comparison_present",
                    ],
                    "properties": {
                        "claim_sentence_id": {
                            "type": "string", "pattern": "^A[0-9]{4,}$"
                        },
                        "evidence_sentence_id": {
                            "anyOf": [
                                {"type": "string", "pattern": "^B[0-9]{4,}$"},
                                {"type": "null"},
                            ]
                        },
                        "decision": {
                            "type": "string",
                            "enum": sorted(pipeline.v7.ALLOWED_DECISIONS),
                        },
                        "relation": {
                            "type": "string",
                            "enum": sorted(pipeline.v7.ALLOWED_RELATIONS),
                        },
                        "quantity_state": {
                            "type": "string",
                            "enum": sorted(pipeline.v7.ALLOWED_QUANTITY_STATES),
                        },
                        "comparison_state": {
                            "type": "string",
                            "enum": sorted(pipeline.v7.ALLOWED_COMPARISON_STATES),
                        },
                        "evidence_kind": {
                            "type": "string",
                            "enum": sorted(pipeline.v7.ALLOWED_EVIDENCE_KINDS),
                        },
                        "quantity_count": {"type": "integer", "minimum": 0},
                        "comparison_present": {"type": "boolean"},
                    },
                },
            },
        },
    },
}


def api_payload_for_request(
    request: dict[str, Any], model_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Return the exact future API payload whose canonical hash is bound in results."""

    pipeline._validate_model(model_manifest)
    return {
        "model": model_manifest["model"],
        "temperature": model_manifest["temperature"],
        "max_tokens": model_manifest["max_output_tokens"],
        "reasoning": model_manifest["reasoning"],
        "provider": {"require_parameters": True},
        "messages": [
            {"role": "system", "content": request["system_prompt"]},
            {"role": "user", "content": request["user_prompt"]},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": PROMPT_RESPONSE_JSON_SCHEMA,
        },
    }


_TELEMETRY_KEYS = {
    "physical_attempt_count", "attempts", "usage", "finish_reason",
    "reasoning", "provider_response_model", "provider_name",
}
_ATTEMPT_KEYS = {"attempt_index", "outcome", "http_status", "error_type"}


def validate_telemetry(value: Any, *, max_attempts: int | None = None) -> None:
    if not isinstance(value, dict) or set(value) != _TELEMETRY_KEYS:
        raise ValueError("v8 telemetry keys differ from the frozen contract")
    count = value["physical_attempt_count"]
    attempts = value["attempts"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("physical_attempt_count must be a positive integer")
    if not isinstance(attempts, list) or len(attempts) != count:
        raise ValueError("physical attempt ledger length mismatch")
    if max_attempts is not None and count > max_attempts:
        raise ValueError("physical attempt count exceeds frozen max_attempts")
    for index, attempt in enumerate(attempts, 1):
        if not isinstance(attempt, dict) or set(attempt) != _ATTEMPT_KEYS:
            raise ValueError("attempt telemetry keys differ from the frozen contract")
        if attempt["attempt_index"] != index:
            raise ValueError("physical attempts must be consecutively numbered")
        if attempt["outcome"] not in {"success", "error"}:
            raise ValueError("invalid physical attempt outcome")
        status = attempt["http_status"]
        if status is not None and (
            not isinstance(status, int) or isinstance(status, bool) or status < 100
        ):
            raise ValueError("invalid HTTP status telemetry")
        error = attempt["error_type"]
        if error is not None and not isinstance(error, str):
            raise ValueError("invalid attempt error_type")
    for attempt in attempts[:-1]:
        if attempt["outcome"] != "error":
            raise ValueError("only the final physical attempt may be successful")
        if not attempt["error_type"]:
            raise ValueError("failed physical attempts require an error_type")
    final = attempts[-1]
    if final["outcome"] != "success":
        raise ValueError("a completed result must end in a successful physical attempt")
    if (
        not isinstance(final["http_status"], int)
        or isinstance(final["http_status"], bool)
        or not 200 <= final["http_status"] < 300
    ):
        raise ValueError("final successful physical attempt requires HTTP 2xx")
    if final["error_type"] is not None:
        raise ValueError("final successful physical attempt cannot have an error_type")
    if not isinstance(value["usage"], dict):
        raise ValueError("usage telemetry must be an object")
    if value["finish_reason"] is not None and not isinstance(
        value["finish_reason"], str
    ):
        raise ValueError("finish_reason telemetry must be string or null")
    reasoning = value["reasoning"]
    if not isinstance(reasoning, dict) or set(reasoning) != {
        "requested", "reported_reasoning_tokens",
        "provider_returned_reasoning_field", "trace_retained",
    }:
        raise ValueError("reasoning telemetry keys differ from the frozen contract")
    if reasoning["requested"] is not False or reasoning["trace_retained"] is not False:
        raise ValueError("v8 must not request or retain a reasoning trace")
    tokens = reasoning["reported_reasoning_tokens"]
    if tokens is not None and (
        not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0
    ):
        raise ValueError("reported reasoning tokens must be nonnegative or null")
    if not isinstance(reasoning["provider_returned_reasoning_field"], bool):
        raise ValueError("provider reasoning-field telemetry must be boolean")
    for key in ("provider_response_model", "provider_name"):
        if value[key] is not None and not isinstance(value[key], str):
            raise ValueError(f"{key} must be string or null")


def _reasoning_tokens(usage: dict[str, Any]) -> int | None:
    direct = usage.get("reasoning_tokens")
    details = usage.get("completion_tokens_details")
    nested = details.get("reasoning_tokens") if isinstance(details, dict) else None
    value = direct if direct is not None else nested
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def telemetry_from_response(
    response_payload: dict[str, Any], attempts: list[dict[str, Any]]
) -> dict[str, Any]:
    choices = response_payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    message = message if isinstance(message, dict) else {}
    usage = response_payload.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    value = {
        "physical_attempt_count": len(attempts),
        "attempts": attempts,
        "usage": usage,
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "reasoning": {
            "requested": False,
            "reported_reasoning_tokens": _reasoning_tokens(usage),
            "provider_returned_reasoning_field": (
                "reasoning" in message or "reasoning_details" in message
            ),
            "trace_retained": False,
        },
        "provider_response_model": response_payload.get("model"),
        "provider_name": response_payload.get("provider"),
    }
    validate_telemetry(value)
    return value


def default_sender(
    endpoint: str, api_key: str, payload: dict[str, Any], timeout_seconds: float
) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status = int(response.status)
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("provider response must be a JSON object")
    return status, value


def _response_content(response_payload: dict[str, Any]) -> Any:
    try:
        content = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("provider response lacks choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("provider response content must be a nonempty JSON string")
    return content


def _load_completed(
    output_path: Path,
    *,
    requests: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    manifest_sha: str,
) -> set[str]:
    completed: set[str] = set()
    if not output_path.exists():
        return completed
    for line_number, row in enumerate(pipeline._read_jsonl(output_path), 1):
        rid = row.get("request_id") if isinstance(row, dict) else None
        if rid in completed:
            raise ValueError(f"duplicate completed request at output line {line_number}")
        if rid not in requests:
            raise ValueError(f"completed result outside v8 bundle at line {line_number}")
        pipeline.verify_bound_result(
            row,
            request=requests[rid],
            manifest=manifest,
            bundle_manifest_sha256=manifest_sha,
        )
        completed.add(rid)
    return completed


Sender = Callable[[str, str, dict[str, Any], float], tuple[int, dict[str, Any]]]


def run_serial(
    bundle: Path,
    output_path: Path,
    failure_path: Path,
    *,
    api_key: str,
    max_requests: int,
    max_attempts: int | None = None,
    timeout_seconds: float | None = None,
    sender: Sender = default_sender,
) -> dict[str, int]:
    """Execute at most ``max_requests`` requests serially with append-only results."""

    if not api_key:
        raise ValueError("an API key is required for explicit execution")
    if max_requests < 1:
        raise ValueError("max_requests must be positive")
    manifest, requests, _ = pipeline.verify_bundle(bundle)
    manifest_sha = pipeline.hash_file(bundle / "manifest.json")
    completed = _load_completed(
        output_path,
        requests=requests,
        manifest=manifest,
        manifest_sha=manifest_sha,
    )
    model_manifest = manifest["model_manifest"]["identity"]
    frozen_attempts = model_manifest["max_attempts"]
    frozen_timeout = float(model_manifest["request_timeout_seconds"])
    max_attempts = frozen_attempts if max_attempts is None else max_attempts
    timeout_seconds = frozen_timeout if timeout_seconds is None else timeout_seconds
    if max_attempts != frozen_attempts or float(timeout_seconds) != frozen_timeout:
        raise ValueError(
            "runner retry/timeout settings must equal the hash-bound model manifest"
        )
    endpoint = model_manifest["endpoint"]
    ordered = sorted(requests.values(), key=lambda row: row["source_index"])
    launched = succeeded = failed = physical_attempts = 0
    for request in ordered:
        if request["request_id"] in completed or launched >= max_requests:
            continue
        launched += 1
        payload = api_payload_for_request(request, model_manifest)
        attempts: list[dict[str, Any]] = []
        response_payload: dict[str, Any] | None = None
        content: Any = None
        parsed: dict[str, Any] | None = None
        telemetry: dict[str, Any] | None = None
        last_error: Exception | None = None
        for attempt_index in range(1, max_attempts + 1):
            physical_attempts += 1
            attempt_status: int | None = None
            try:
                status, response_payload = sender(
                    endpoint, api_key, payload, timeout_seconds
                )
                attempt_status = status
                if status < 200 or status >= 300:
                    last_error = RuntimeError(f"provider HTTP status {status}")
                    attempts.append({
                        "attempt_index": attempt_index,
                        "outcome": "error",
                        "http_status": status,
                        "error_type": type(last_error).__name__,
                    })
                    response_payload = None
                    continue
                successful_attempt = {
                    "attempt_index": attempt_index,
                    "outcome": "success",
                    "http_status": status,
                    "error_type": None,
                }
                # A 2xx transport is not a successful physical attempt until its
                # content, schema, hydration, audit replay, and telemetry validate.
                content = _response_content(response_payload)
                parsed = pipeline._extract_json(content)
                pipeline.hydrate_response(parsed, request)
                candidate_attempts = [*attempts, successful_attempt]
                telemetry = telemetry_from_response(
                    response_payload, candidate_attempts
                )
                attempts = candidate_attempts
                last_error = None
                break
            except Exception as exc:  # transport failures are explicitly ledgered
                last_error = exc
                http_status = (
                    int(exc.code) if isinstance(exc, urllib.error.HTTPError) else None
                )
                attempts.append({
                    "attempt_index": attempt_index,
                    "outcome": "error",
                    "http_status": (
                        http_status
                        if http_status is not None
                        else attempt_status
                    ),
                    "error_type": type(exc).__name__,
                })
                response_payload = None
                content = None
                parsed = None
                telemetry = None
        if (
            last_error is not None
            or response_payload is None
            or parsed is None
            or telemetry is None
        ):
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            pipeline._write_jsonl(failure_path, [{
                "schema_version": "science-articulability-addressed-failure-v8",
                "request_id": request["request_id"],
                "request_sha256": request["request_sha256"],
                "model_manifest_sha256": manifest["model_manifest"][
                    "canonical_sha256"
                ],
                "bundle_manifest_sha256": manifest_sha,
                "api_payload_sha256": pipeline.hash_value(payload),
                "runner_sha256": pipeline.hash_file(Path(__file__)),
                "provider": model_manifest["backend"],
                "model": model_manifest["model"],
                "physical_attempt_count": len(attempts),
                "attempts": attempts,
                "terminal_error_type": type(last_error).__name__,
                "response_content_retained": False,
            }], mode="a")
            failed += 1
            continue
        row = {
            "schema_version": RESULT_SCHEMA,
            "request_id": request["request_id"],
            "request_sha256": request["request_sha256"],
            "model_manifest_sha256": manifest["model_manifest"]["canonical_sha256"],
            "bundle_manifest_sha256": manifest_sha,
            "runner_sha256": pipeline.hash_file(Path(__file__)),
            "api_payload_sha256": pipeline.hash_value(payload),
            "provider": model_manifest["backend"],
            "model": model_manifest["model"],
            "response": content,
            "parsed_response_sha256": pipeline.hash_value(parsed),
            "telemetry": telemetry,
        }
        # Exercise the same full binding/replay check used on resume.
        pipeline.verify_bound_result(
            row,
            request=request,
            manifest=manifest,
            bundle_manifest_sha256=manifest_sha,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline._write_jsonl(output_path, [row], mode="a")
        succeeded += 1
    return {
        "already_completed": len(completed),
        "logical_requests_launched": launched,
        "successful_results_appended": succeeded,
        "terminal_failures": failed,
        "physical_attempt_count_including_retries": physical_attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=pipeline.DEFAULT_OUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, required=True)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing network execution without explicit --execute")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    summary = run_serial(
        args.bundle.resolve(), args.output.resolve(), args.failures.resolve(),
        api_key=api_key, max_requests=args.max_requests,
        max_attempts=args.max_attempts, timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
