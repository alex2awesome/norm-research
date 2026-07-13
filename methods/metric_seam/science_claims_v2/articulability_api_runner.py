#!/usr/bin/env python3
"""Optional CPU/API executor for a prepared science articulability bundle.

This is intentionally separate from preparation and evaluation. It never opens the
source dataset, code-verifier results, or labels; all call material comes from the
sealed request bundle. Results retain the bindings required by ``ingest``.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

from .articulability_pipeline import RESULT_SCHEMA, hash_file, verify_bundle


DEFAULT_KEYS = (
    "~/.z-ai-api-key.txt",
    "~/.z-ai-api-key-alexander-spangher.txt",
    "~/.z-ai-api-key-spangher.txt",
)
OPENROUTER_KEYS = ("~/.openrouter-api-key.txt",)


def read_key(key_file: str | None, backend: str | None = None) -> str:
    defaults = OPENROUTER_KEYS if backend == "openrouter" else DEFAULT_KEYS
    candidates = (key_file,) if key_file else defaults
    for candidate in candidates:
        path = Path(os.path.expanduser(candidate))
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
    raise FileNotFoundError(f"no API key found in {list(candidates)}")


def api_payload(request_row: dict, model: dict) -> dict:
    if model.get("system_prompt_transport") != "system":
        raise ValueError("unsupported system prompt transport")
    payload = {
        "model": model["model"],
        "max_tokens": model["max_output_tokens"],
        "temperature": model["temperature"],
    }
    if model.get("protocol") == "anthropic_messages":
        payload.update({
            "system": request_row["system_prompt"],
            "messages": [{"role": "user", "content": request_row["user_prompt"]}],
        })
        return payload
    if model.get("protocol") == "openai_chat_completions":
        payload.update({
            "messages": [
                {"role": "system", "content": request_row["system_prompt"]},
                {"role": "user", "content": request_row["user_prompt"]},
            ],
            "response_format": {"type": "json_object"},
        })
        if model.get("provider_require_parameters"):
            payload["provider"] = {"require_parameters": True}
        if model.get("reasoning") is not None:
            payload["reasoning"] = model["reasoning"]
        return payload
    raise ValueError("unsupported API protocol")


def call(endpoint: str, api_key: str, payload: dict, protocol: str, *,
         request_timeout_seconds: float, tries: int) -> dict:
    """Execute one bounded request, including retries and backoff in one deadline.

    ``urllib``'s timeout applies to one blocking operation, not to the complete retry
    lifecycle. The monotonic deadline below therefore caps the full logical request.
    """
    if request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    if tries < 1:
        raise ValueError("tries must be positive")
    encoded = json.dumps(payload).encode("utf-8")
    if protocol == "anthropic_messages":
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    elif protocol == "openai_chat_completions":
        headers = {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
    else:
        raise ValueError("unsupported API protocol")
    delay = 2.0
    deadline = time.monotonic() + request_timeout_seconds
    for attempt in range(tries):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"request exceeded total deadline of {request_timeout_seconds:g}s"
            )
        try:
            req = urlrequest.Request(endpoint, data=encoded, headers=headers)
            with urlrequest.urlopen(req, timeout=remaining) as response:
                return json.loads(response.read())
        except urlerror.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 529} or attempt == tries - 1:
                raise
        except Exception:
            if attempt == tries - 1:
                raise
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"request exceeded total deadline of {request_timeout_seconds:g}s"
            )
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


def provider_diagnostic(payload: dict) -> dict:
    """Return response-shape evidence without retaining hidden reasoning text."""
    choices = payload.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    content = message.get("content")
    reasoning = message.get("reasoning")
    return {
        "top_level_keys": sorted(payload),
        "id": payload.get("id"),
        "model": payload.get("model"),
        "provider": payload.get("provider"),
        "error": payload.get("error"),
        "choice_error": choice.get("error"),
        "finish_reason": choice.get("finish_reason"),
        "native_finish_reason": choice.get("native_finish_reason"),
        "message_keys": sorted(message) if isinstance(message, dict) else [],
        "content_type": type(content).__name__,
        "content_characters": len(content) if isinstance(content, str) else None,
        "reasoning_characters": len(reasoning) if isinstance(reasoning, str) else None,
        "usage": payload.get("usage"),
    }


def response_text(payload: dict, protocol: str = "anthropic_messages") -> str:
    if protocol == "anthropic_messages":
        parts = payload.get("content") or []
        text = "".join(str(part.get("text") or "") for part in parts
                       if part.get("type") in (None, "text"))
    elif protocol == "openai_chat_completions":
        choices = payload.get("choices") or []
        text = str((choices[0].get("message") or {}).get("content") or "") if choices else ""
    else:
        raise ValueError("unsupported API protocol")
    if not text.strip():
        raise ValueError(
            "API response contains no text content: "
            + json.dumps(provider_diagnostic(payload), ensure_ascii=False, sort_keys=True)
        )
    return text.strip()


def load_completed(path: Path, request_ids: set[str]) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rid = row.get("request_id")
            if rid not in request_ids:
                raise ValueError(f"output contains request outside bundle: {rid}")
            completed.add(rid)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--errors", type=Path)
    parser.add_argument("--limit", type=int, default=0,
                        help="execute first N remaining requests; 0 means all")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--key-file")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    manifest, by_id = verify_bundle(args.bundle.resolve())
    bundle_manifest_sha = hash_file(args.bundle.resolve() / "manifest.json")
    model = manifest["model_manifest"]["identity"]
    model_sha = manifest["model_manifest"]["canonical_sha256"]
    endpoint = model.get("endpoint")
    if not endpoint:
        raise ValueError("model manifest has no endpoint")
    request_timeout_seconds = float(model.get("request_timeout_seconds", 120))
    max_attempts = int(model.get("max_attempts", 2))
    if request_timeout_seconds <= 0 or max_attempts < 1:
        raise ValueError("model manifest has invalid request bounds")
    ordered = sorted(by_id.values(), key=lambda row: row["sequence_index"])
    completed = load_completed(args.out.resolve(), set(by_id))
    todo = [row for row in ordered if row["request_id"] not in completed]
    if args.limit:
        todo = todo[:args.limit]
    print(f"resume={len(completed)} calls={len(todo)} model={model['model']}", flush=True)
    api_key = read_key(args.key_file, model.get("backend"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    errors_path = (args.errors or args.out.with_suffix(".errors.jsonl")).resolve()
    lock = threading.Lock()
    successes = failures = 0

    def work(row: dict) -> None:
        nonlocal successes, failures
        raw = None
        try:
            protocol = model["protocol"]
            raw = call(
                endpoint, api_key, api_payload(row, model), protocol,
                request_timeout_seconds=request_timeout_seconds, tries=max_attempts,
            )
            result = {
                "schema_version": RESULT_SCHEMA,
                "request_id": row["request_id"],
                "request_sha256": row["request_sha256"],
                "model_manifest_sha256": model_sha,
                "bundle_manifest_sha256": bundle_manifest_sha,
                "response": response_text(raw, protocol),
                "provider_metadata": {
                    "id": raw.get("id"), "model": raw.get("model"),
                    "stop_reason": (
                        raw.get("stop_reason")
                        if protocol == "anthropic_messages"
                        else (((raw.get("choices") or [{}])[0]).get("finish_reason"))
                    ),
                    "usage": raw.get("usage"),
                },
            }
            with lock, args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                successes += 1
        except Exception as exc:
            failure = {"request_id": row["request_id"],
                       "request_sha256": row["request_sha256"],
                       "error_type": type(exc).__name__, "error": str(exc)[:4000]}
            if isinstance(raw, dict):
                failure["provider_response_diagnostic"] = provider_diagnostic(raw)
            with lock, errors_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, sort_keys=True) + "\n")
                failures += 1

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        list(executor.map(work, todo))
    print(json.dumps({"successes": successes, "failures": failures,
                      "remaining_after_successes": len(by_id) - len(completed) - successes},
                     sort_keys=True), flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
