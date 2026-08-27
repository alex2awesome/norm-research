#!/usr/bin/env python3
"""Hash-bound serial runner for the prepared a407 prompt and hybrid arms.

Importing this module performs no network operation.  The command line requires
an explicit ``--execute`` flag and an API key.  Logical requests run strictly
serially.  Every physical attempt is ledgered, schema-invalid 2xx responses are
failed attempts, and append-only resume replays all request/result bindings
before skipping a completed request.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any, Callable

try:
    from . import a407_dual_channel_pipeline_v1 as pipeline
except ImportError:  # pragma: no cover - direct-script compatibility
    import a407_dual_channel_pipeline_v1 as pipeline  # type: ignore[no-redef]


RESULT_SCHEMA = "metric-seam.a407-dual-channel-model-result.v1"
FAILURE_SCHEMA = "metric-seam.a407-dual-channel-model-failure.v1"


def _score_schema() -> dict[str, Any]:
    return {
        "anyOf": [
            {"type": "number", "minimum": 0.0, "maximum": 1.0},
            {"type": "null"},
        ]
    }


def _relation_object_schema(relations: tuple[str, ...], value_schema: dict) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(relations),
        "properties": {relation: value_schema for relation in relations},
    }


def response_json_schema(arm: str) -> dict[str, Any]:
    if arm == "raw_prompt":
        relations = pipeline.ALL_RELATIONS
        spec = pipeline._load_json_object(pipeline.RAW_PROMPT_SPEC_PATH)
        schema_name = "a407_raw_prompt_articulability_v1"
    elif arm == "hybrid":
        relations = pipeline.HYBRID_RELATIONS
        spec = pipeline._load_json_object(pipeline.HYBRID_PROMPT_SPEC_PATH)
        schema_name = "a407_hybrid_seam_v1"
    else:
        raise ValueError("unknown request arm")
    reasons = spec.get("abstention_reasons")
    if not isinstance(reasons, list) or not all(isinstance(value, str) for value in reasons):
        raise ValueError("prompt spec abstention reasons are invalid")
    return {
        "name": schema_name,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "item_key",
                "abstained",
                "abstention_reason",
                "declared_holistic_score",
                "relation_scores",
                "relation_abstentions",
            ],
            "properties": {
                "item_key": {"type": "string", "pattern": "^heldout_[0-9]{4}$"},
                "abstained": {"type": "boolean"},
                "abstention_reason": {"type": "string", "enum": reasons},
                "declared_holistic_score": _score_schema(),
                "relation_scores": _relation_object_schema(
                    relations, _score_schema()
                ),
                "relation_abstentions": _relation_object_schema(
                    relations, {"type": "boolean"}
                ),
            },
        },
    }


def api_payload_for_request(
    request: dict[str, Any], model: dict[str, Any], arm: str
) -> dict[str, Any]:
    return {
        "model": model["model"],
        "temperature": model["temperature"],
        "max_tokens": model["max_output_tokens"],
        "reasoning": model["reasoning"],
        "provider": {"require_parameters": model["provider_require_parameters"]},
        "messages": [
            {"role": "system", "content": request["system_prompt"]},
            {"role": "user", "content": request["user_prompt"]},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": response_json_schema(arm),
        },
    }


def _is_score(value: Any) -> bool:
    return pipeline._unit_interval_or_none(value)


def validate_response(
    value: Any, *, request: dict[str, Any], arm: str
) -> dict[str, Any]:
    relations = (
        pipeline.ALL_RELATIONS if arm == "raw_prompt" else pipeline.HYBRID_RELATIONS
    )
    spec_path = (
        pipeline.RAW_PROMPT_SPEC_PATH
        if arm == "raw_prompt"
        else pipeline.HYBRID_PROMPT_SPEC_PATH
    )
    reasons = set(pipeline._load_json_object(spec_path)["abstention_reasons"])
    keys = {
        "item_key",
        "abstained",
        "abstention_reason",
        "declared_holistic_score",
        "relation_scores",
        "relation_abstentions",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("response keys differ from the frozen schema")
    if value["item_key"] != request["item_key"]:
        raise ValueError("response item key mismatch")
    if not isinstance(value["abstained"], bool):
        raise ValueError("response abstained field must be boolean")
    if value["abstention_reason"] not in reasons:
        raise ValueError("response abstention reason is invalid")
    if not _is_score(value["declared_holistic_score"]):
        raise ValueError("response holistic score is invalid")
    scores = value["relation_scores"]
    abstentions = value["relation_abstentions"]
    if not isinstance(scores, dict) or set(scores) != set(relations):
        raise ValueError("response relation-score keys differ")
    if not isinstance(abstentions, dict) or set(abstentions) != set(relations):
        raise ValueError("response relation-abstention keys differ")
    for relation in relations:
        if not _is_score(scores[relation]):
            raise ValueError("response relation score is invalid")
        if not isinstance(abstentions[relation], bool):
            raise ValueError("response relation abstention is invalid")
        if abstentions[relation] != (scores[relation] is None):
            raise ValueError("relation score/abstention fields are inconsistent")
    if value["abstained"]:
        if value["abstention_reason"] == "none":
            raise ValueError("global abstention requires a reason")
        if value["declared_holistic_score"] is not None:
            raise ValueError("global abstention requires a null holistic score")
        if not all(abstentions.values()):
            raise ValueError("global abstention requires all relations to abstain")
    else:
        if value["abstention_reason"] != "none":
            raise ValueError("nonabstained response must use reason none")
        if value["declared_holistic_score"] is None:
            raise ValueError("nonabstained response requires a holistic score")
        if all(abstentions.values()):
            raise ValueError("nonabstained response requires relation evidence")
    # The fixed schema admits no rationale, quote, path, identifier, or source
    # text field.  The only free strings are a bound opaque alias and enum.
    return value


def parse_response_content(
    content: Any, *, request: dict[str, Any], arm: str
) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("provider content must be a nonempty JSON string")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("provider content is not exact JSON") from exc
    return validate_response(parsed, request=request, arm=arm)


_ATTEMPT_KEYS = {"attempt_index", "outcome", "http_status", "error_type"}
_TELEMETRY_KEYS = {
    "physical_attempt_count",
    "attempts",
    "usage",
    "finish_reason",
    "reasoning",
    "provider_response_model",
    "provider_name",
}


def validate_telemetry(value: Any, *, max_attempts: int) -> None:
    if not isinstance(value, dict) or set(value) != _TELEMETRY_KEYS:
        raise ValueError("telemetry keys differ from the frozen contract")
    count = value["physical_attempt_count"]
    attempts = value["attempts"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("physical attempt count must be positive")
    if count > max_attempts or not isinstance(attempts, list) or len(attempts) != count:
        raise ValueError("physical attempt ledger length is invalid")
    for index, attempt in enumerate(attempts, 1):
        if not isinstance(attempt, dict) or set(attempt) != _ATTEMPT_KEYS:
            raise ValueError("physical attempt keys differ")
        if attempt["attempt_index"] != index:
            raise ValueError("physical attempt indices must be consecutive")
        if attempt["outcome"] not in {"error", "success"}:
            raise ValueError("physical attempt outcome is invalid")
        status = attempt["http_status"]
        if status is not None and (
            not isinstance(status, int) or isinstance(status, bool) or status < 100
        ):
            raise ValueError("physical attempt HTTP status is invalid")
        error = attempt["error_type"]
        if error is not None and not isinstance(error, str):
            raise ValueError("physical attempt error type is invalid")
    if any(attempt["outcome"] != "error" for attempt in attempts[:-1]):
        raise ValueError("only the final physical attempt may succeed")
    if any(not attempt["error_type"] for attempt in attempts[:-1]):
        raise ValueError("failed physical attempts require an error type")
    final = attempts[-1]
    if final["outcome"] != "success":
        raise ValueError("completed telemetry must end in success")
    if not isinstance(final["http_status"], int) or not 200 <= final["http_status"] < 300:
        raise ValueError("successful physical attempt requires HTTP 2xx")
    if final["error_type"] is not None:
        raise ValueError("successful physical attempt cannot have an error")
    if not isinstance(value["usage"], dict):
        raise ValueError("usage telemetry must be an object")
    if value["finish_reason"] is not None and not isinstance(value["finish_reason"], str):
        raise ValueError("finish reason must be a string or null")
    reasoning = value["reasoning"]
    if not isinstance(reasoning, dict) or set(reasoning) != {
        "requested_effort",
        "reported_reasoning_tokens",
        "provider_returned_reasoning_field",
        "trace_retained",
    }:
        raise ValueError("reasoning telemetry keys differ")
    if reasoning["requested_effort"] != "none" or reasoning["trace_retained"] is not False:
        raise ValueError("runner must request and retain no reasoning trace")
    tokens = reasoning["reported_reasoning_tokens"]
    if tokens is not None and (
        not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0
    ):
        raise ValueError("reasoning-token telemetry is invalid")
    if not isinstance(reasoning["provider_returned_reasoning_field"], bool):
        raise ValueError("provider reasoning-field telemetry is invalid")
    for key in ("provider_response_model", "provider_name"):
        if value[key] is not None and not isinstance(value[key], str):
            raise ValueError("provider telemetry must be string or null")


def _reasoning_tokens(usage: dict[str, Any]) -> int | None:
    direct = usage.get("reasoning_tokens")
    details = usage.get("completion_tokens_details")
    nested = details.get("reasoning_tokens") if isinstance(details, dict) else None
    value = direct if direct is not None else nested
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def telemetry_from_response(
    response: dict[str, Any], attempts: list[dict[str, Any]], *, max_attempts: int
) -> dict[str, Any]:
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    message = message if isinstance(message, dict) else {}
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    value = {
        "physical_attempt_count": len(attempts),
        "attempts": attempts,
        "usage": usage,
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "reasoning": {
            "requested_effort": "none",
            "reported_reasoning_tokens": _reasoning_tokens(usage),
            "provider_returned_reasoning_field": (
                "reasoning" in message or "reasoning_details" in message
            ),
            "trace_retained": False,
        },
        "provider_response_model": response.get("model"),
        "provider_name": response.get("provider"),
    }
    validate_telemetry(value, max_attempts=max_attempts)
    return value


def _response_content(response: dict[str, Any]) -> Any:
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("provider response lacks message content") from exc


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
        raise ValueError("provider response must be an object")
    return status, value


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def verify_bound_result(
    row: dict[str, Any],
    *,
    request: dict[str, Any],
    arm: str,
    manifest_sha256: str,
    model: dict[str, Any],
) -> None:
    required = {
        "schema",
        "arm",
        "request_id",
        "request_sha256",
        "preparation_manifest_sha256",
        "model_spec_sha256",
        "runner_sha256",
        "api_payload_sha256",
        "provider",
        "model",
        "response",
        "parsed_response_sha256",
        "telemetry",
    }
    if not isinstance(row, dict) or set(row) != required:
        raise ValueError("completed result keys differ from the frozen contract")
    if row["schema"] != RESULT_SCHEMA or row["arm"] != arm:
        raise ValueError("completed result identity mismatch")
    if row["request_id"] != request["request_id"] or row["request_sha256"] != request["request_sha256"]:
        raise ValueError("completed result request binding mismatch")
    if row["preparation_manifest_sha256"] != manifest_sha256:
        raise ValueError("completed result preparation binding mismatch")
    if row["model_spec_sha256"] != pipeline.hash_value(model):
        raise ValueError("completed result model binding mismatch")
    if row["runner_sha256"] != pipeline.hash_file(Path(__file__)):
        raise ValueError("completed result runner binding mismatch")
    payload = api_payload_for_request(request, model, arm)
    if row["api_payload_sha256"] != pipeline.hash_value(payload):
        raise ValueError("completed result API payload binding mismatch")
    parsed = parse_response_content(row["response"], request=request, arm=arm)
    if row["parsed_response_sha256"] != pipeline.hash_value(parsed):
        raise ValueError("completed result parsed-response binding mismatch")
    if row["provider"] != model["backend"] or row["model"] != model["model"]:
        raise ValueError("completed result provider/model identity mismatch")
    validate_telemetry(row["telemetry"], max_attempts=model["max_attempts"])


def _load_completed(
    path: Path,
    *,
    requests: dict[str, dict[str, Any]],
    arm: str,
    manifest_sha256: str,
    model: dict[str, Any],
) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for row in pipeline.read_jsonl(path):
        request_id = row.get("request_id")
        if request_id in completed or request_id not in requests:
            raise ValueError("completed result request ID is duplicate or foreign")
        verify_bound_result(
            row,
            request=requests[request_id],
            arm=arm,
            manifest_sha256=manifest_sha256,
            model=model,
        )
        completed.add(request_id)
    return completed


Sender = Callable[[str, str, dict[str, Any], float], tuple[int, dict[str, Any]]]


def run_serial(
    preparation_dir: Path,
    *,
    arm: str,
    output_path: Path,
    failure_path: Path,
    api_key: str,
    max_requests: int,
    max_attempts: int | None = None,
    timeout_seconds: float | None = None,
    sender: Sender = default_sender,
) -> dict[str, int]:
    if arm not in {"raw_prompt", "hybrid"}:
        raise ValueError("arm must be raw_prompt or hybrid")
    if not api_key:
        raise ValueError("an API key is required for explicit execution")
    if max_requests < 1:
        raise ValueError("max_requests must be positive")
    _manifest, arms, model = pipeline.verify_preparation_bundle(preparation_dir)
    requests_list = arms[arm]
    requests = {row["request_id"]: row for row in requests_list}
    manifest_sha = pipeline.hash_file(preparation_dir / "preparation_manifest.json")
    frozen_attempts = int(model["max_attempts"])
    frozen_timeout = float(model["request_timeout_seconds"])
    max_attempts = frozen_attempts if max_attempts is None else max_attempts
    timeout_seconds = frozen_timeout if timeout_seconds is None else timeout_seconds
    if max_attempts != frozen_attempts or float(timeout_seconds) != frozen_timeout:
        raise ValueError("retry and timeout settings must match the frozen model")
    completed = _load_completed(
        output_path,
        requests=requests,
        arm=arm,
        manifest_sha256=manifest_sha,
        model=model,
    )

    launched = succeeded = failed = physical_attempts = 0
    for request in sorted(requests_list, key=lambda row: row["heldout_ordinal"]):
        if request["request_id"] in completed or launched >= max_requests:
            continue
        launched += 1
        payload = api_payload_for_request(request, model, arm)
        attempts: list[dict[str, Any]] = []
        parsed: dict[str, Any] | None = None
        content: str | None = None
        telemetry: dict[str, Any] | None = None
        last_error: Exception | None = None
        for attempt_index in range(1, max_attempts + 1):
            physical_attempts += 1
            attempt_status: int | None = None
            try:
                status, response = sender(
                    model["endpoint"], api_key, payload, timeout_seconds
                )
                attempt_status = status
                if not 200 <= status < 300:
                    raise RuntimeError("provider returned a non-success status")
                candidate_content = _response_content(response)
                candidate_parsed = parse_response_content(
                    candidate_content, request=request, arm=arm
                )
                candidate_attempts = [
                    *attempts,
                    {
                        "attempt_index": attempt_index,
                        "outcome": "success",
                        "http_status": status,
                        "error_type": None,
                    },
                ]
                candidate_telemetry = telemetry_from_response(
                    response,
                    candidate_attempts,
                    max_attempts=max_attempts,
                )
                attempts = candidate_attempts
                content = candidate_content
                parsed = candidate_parsed
                telemetry = candidate_telemetry
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                http_status = (
                    int(exc.code) if isinstance(exc, urllib.error.HTTPError) else attempt_status
                )
                attempts.append({
                    "attempt_index": attempt_index,
                    "outcome": "error",
                    "http_status": http_status,
                    "error_type": type(exc).__name__,
                })
                parsed = None
                content = None
                telemetry = None
        if last_error is not None or parsed is None or content is None or telemetry is None:
            _append_jsonl(failure_path, {
                "schema": FAILURE_SCHEMA,
                "arm": arm,
                "request_id": request["request_id"],
                "request_sha256": request["request_sha256"],
                "preparation_manifest_sha256": manifest_sha,
                "model_spec_sha256": pipeline.hash_value(model),
                "runner_sha256": pipeline.hash_file(Path(__file__)),
                "api_payload_sha256": pipeline.hash_value(payload),
                "physical_attempt_count": len(attempts),
                "attempts": attempts,
                "terminal_error_type": type(last_error).__name__,
                "response_content_retained": False,
                "source_text_retained": False,
            })
            failed += 1
            continue
        result = {
            "schema": RESULT_SCHEMA,
            "arm": arm,
            "request_id": request["request_id"],
            "request_sha256": request["request_sha256"],
            "preparation_manifest_sha256": manifest_sha,
            "model_spec_sha256": pipeline.hash_value(model),
            "runner_sha256": pipeline.hash_file(Path(__file__)),
            "api_payload_sha256": pipeline.hash_value(payload),
            "provider": model["backend"],
            "model": model["model"],
            "response": content,
            "parsed_response_sha256": pipeline.hash_value(parsed),
            "telemetry": telemetry,
        }
        verify_bound_result(
            result,
            request=request,
            arm=arm,
            manifest_sha256=manifest_sha,
            model=model,
        )
        _append_jsonl(output_path, result)
        succeeded += 1
    return {
        "already_completed": len(completed),
        "logical_requests_launched": launched,
        "successful_results_appended": succeeded,
        "terminal_failures": failed,
        "physical_attempt_count_including_retries": physical_attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("raw_prompt", "hybrid"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, required=True)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing network execution without explicit --execute")
    summary = run_serial(
        args.preparation_dir.resolve(),
        arm=args.arm,
        output_path=args.output.resolve(),
        failure_path=args.failures.resolve(),
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        max_requests=args.max_requests,
        max_attempts=args.max_attempts,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
