"""Execute a frozen hierarchy prompt bundle through the z.ai Anthropic endpoint.

The runner deliberately projects each frozen job down to ``request.system`` and
``request.user``.  Audit metadata is used only for local filtering and preflight
checks and is never included in the provider request.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping

from methods.metric_seam.hierarchy_prompt_batch import validate_prompt_response


ZAI_ANTHROPIC_URL = "https://api.z.ai/api/anthropic/v1/messages"
MAX_ATTEMPTS = 8
EXPECTED_CELLS = 18
EXPECTED_ITEMS_PER_CELL = 125
EXPECTED_PASSES = {1, 2}


class PromptJobError(ValueError):
    """Raised when the frozen job bundle or resume file violates its contract."""


@dataclass(frozen=True)
class PreflightSummary:
    selected_jobs: int
    cell_ids: tuple[str, ...]
    request_ids: frozenset[str]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def request_sha256(request: Mapping[str, str]) -> str:
    """Hash the exact two-field model-visible request projection."""

    return hashlib.sha256(_canonical_json_bytes(dict(request))).hexdigest()


def resolve_zai_key(environ: Mapping[str, str] | None = None) -> str:
    """Resolve the z.ai credential in the repository's frozen precedence order."""

    env = os.environ if environ is None else environ
    configured = env.get("ZAI_KEY_FILE")
    if configured:
        candidate = Path(os.path.expanduser(configured))
        if candidate.is_file():
            key = candidate.read_text(encoding="utf-8").strip()
        else:
            # Preserve the existing repository convention: the environment value
            # may itself contain a key even though its preferred use is a file.
            key = configured.strip()
        if key:
            return key
    for name in (
        "~/.z-ai-api-key-alexander-spangher.txt",
        "~/.z-ai-api-key-spangher.txt",
    ):
        candidate = Path(name).expanduser()
        if candidate.is_file():
            key = candidate.read_text(encoding="utf-8").strip()
            if key:
                return key
    raise FileNotFoundError(
        "no z.ai key (set ZAI_KEY_FILE or create one of the repository key files)"
    )


def _pass_id(metadata: Mapping[str, object]) -> int | None:
    value = metadata.get("pass_id", metadata.get("pass_index"))
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _iter_jsonl_gz(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PromptJobError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise PromptJobError(f"job at {path}:{line_number} is not an object")
            yield row


def _selected(row: Mapping[str, object], channel: str) -> bool:
    metadata = row.get("audit_metadata")
    return isinstance(metadata, Mapping) and metadata.get("channel") == channel


def iter_selected_jobs(path: Path, channel: str) -> Iterable[dict]:
    """Stream only the locally selected channel from a compressed job bundle."""

    for row in _iter_jsonl_gz(path):
        if _selected(row, channel):
            yield row


def preflight_jobs(
    path: Path,
    *,
    channel: str,
    expected_jobs: int,
) -> PreflightSummary:
    """Scan the bundle once and assert the complete frozen execution design."""

    cell_items: dict[str, set[str]] = defaultdict(set)
    cell_item_pass_counts: dict[str, Counter[tuple[str, int]]] = defaultdict(Counter)
    request_ids: set[str] = set()
    selected_jobs = 0

    for row in iter_selected_jobs(path, channel):
        selected_jobs += 1
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise PromptJobError("selected job has no nonempty request_id")
        if request_id in request_ids:
            raise PromptJobError(f"duplicate selected request_id: {request_id}")
        request_ids.add(request_id)

        request = row.get("request")
        if not isinstance(request, Mapping) or set(request) != {"system", "user"}:
            raise PromptJobError(f"request {request_id} is not the two-field projection")
        if not all(isinstance(request[key], str) for key in ("system", "user")):
            raise PromptJobError(f"request {request_id} has non-text prompt fields")

        metadata = row.get("audit_metadata")
        assert isinstance(metadata, Mapping)  # guaranteed by iter_selected_jobs
        cell_id = metadata.get("cell_id")
        item_key = metadata.get("item_key")
        pass_id = _pass_id(metadata)
        if not isinstance(cell_id, str) or not isinstance(item_key, str):
            raise PromptJobError(f"request {request_id} lacks cell/item identity")
        if pass_id not in EXPECTED_PASSES:
            raise PromptJobError(f"request {request_id} has invalid pass identity")
        cell_items[cell_id].add(item_key)
        cell_item_pass_counts[cell_id][(item_key, pass_id)] += 1

    if selected_jobs != expected_jobs:
        raise PromptJobError(
            f"selected {selected_jobs} jobs for {channel!r}; expected {expected_jobs}"
        )
    if len(cell_items) != EXPECTED_CELLS:
        raise PromptJobError(
            f"selected {len(cell_items)} cell IDs; expected {EXPECTED_CELLS}"
        )
    for cell_id in sorted(cell_items):
        items = cell_items[cell_id]
        if len(items) != EXPECTED_ITEMS_PER_CELL:
            raise PromptJobError(
                f"cell {cell_id} has {len(items)} item keys; "
                f"expected {EXPECTED_ITEMS_PER_CELL}"
            )
        pair_counts = cell_item_pass_counts[cell_id]
        expected_pairs = {
            (item_key, pass_id)
            for item_key in items
            for pass_id in EXPECTED_PASSES
        }
        if set(pair_counts) != expected_pairs or set(pair_counts.values()) != {1}:
            raise PromptJobError(
                f"cell {cell_id} does not contain exactly two unique passes per item"
            )

    return PreflightSummary(
        selected_jobs=selected_jobs,
        cell_ids=tuple(sorted(cell_items)),
        request_ids=frozenset(request_ids),
    )


def _anthropic_body(
    request: Mapping[str, str],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> bytes:
    # No job field outside request.system/request.user is projected here.
    return _canonical_json_bytes(
        {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": request["system"],
            "messages": [{"role": "user", "content": request["user"]}],
        }
    )


def _post_request(body: bytes, key: str, timeout: float) -> bytes:
    provider_request = urllib.request.Request(
        ZAI_ANTHROPIC_URL,
        data=body,
        method="POST",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(provider_request, timeout=timeout) as response:
        return response.read()


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    return isinstance(exc, urllib.error.URLError) and isinstance(
        exc.reason, (TimeoutError, socket.timeout)
    )


def _is_retryable(exc: BaseException) -> bool:
    if _is_timeout(exc):
        return True
    return isinstance(exc, urllib.error.HTTPError) and (
        exc.code == 429 or 500 <= exc.code <= 599
    )


def _safe_transport_message(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    if _is_timeout(exc):
        return "timeout"
    return type(exc).__name__


def _response_text(envelope: Mapping[str, object]) -> str:
    content = envelope.get("content")
    if not isinstance(content, list) or not content:
        raise PromptJobError("provider response has no content blocks")
    texts: list[str] = []
    for block in content:
        if isinstance(block, Mapping) and isinstance(block.get("text"), str):
            texts.append(block["text"])
    if not texts:
        raise PromptJobError("provider response has no text content")
    return "".join(texts).strip()


_FENCE = re.compile(r"\A\s*```(?:json|JSON)?\s*\n(?P<body>.*?)\n?\s*```\s*\Z", re.DOTALL)


def deserialize_response(raw_response: str) -> object:
    """Decode the transport envelope's text into the JSON value the model emitted.

    This is deserialization, not repair.  Two transport-layer facts are handled,
    both lossless and applied uniformly to every row before any content is read:

    * A Markdown code fence is a serialization wrapper, not model content.  An
      unfenced and a fenced emission of the same object carry identical meaning.
    * ``strict=False`` admits literal tab/newline bytes inside JSON strings.
      Evidence spans quote tab-indented source, so a strict parse rejects rows
      non-randomly -- it selects against tab-indented languages.

    Nothing else is attempted.  Malformed JSON is not repaired, no object is
    hunted out of surrounding prose, and no value is coerced.  The frozen
    ``never coerce or retry selectively`` policy governs the parsed payload,
    which is still handed unchanged to ``validate_prompt_response``.
    """

    match = _FENCE.match(raw_response)
    body = match.group("body") if match else raw_response
    return json.loads(body, strict=False)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def execute_one(
    job: Mapping[str, object],
    *,
    key: str,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float = 180.0,
    max_attempts: int = MAX_ATTEMPTS,
    post: Callable[[bytes, str, float], bytes] = _post_request,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Execute one stateless request and return exactly one terminal row."""

    if provider != "zai_anthropic":
        raise PromptJobError(f"unsupported backend: {provider}")
    request_id = job["request_id"]
    request = job["request"]
    assert isinstance(request_id, str) and isinstance(request, Mapping)
    visible_request = {"system": request["system"], "user": request["user"]}
    body = _anthropic_body(
        visible_request,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    attempts = 0
    response_bytes: bytes | None = None
    terminal_error = ""

    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        try:
            response_bytes = post(body, key, timeout)
            break
        except Exception as exc:  # convert every request into a terminal row
            terminal_error = _safe_transport_message(exc)
            if not _is_retryable(exc) or attempt == max_attempts:
                break
            sleep(min(60.0, float(2 ** (attempt - 1))))

    base = {
        "request_id": request_id,
        "request_sha256": request_sha256(visible_request),
        "status": "transport_error",
        "raw_response": terminal_error,
        "parsed_response": {},
        "provider": provider,
        "requested_model": model,
        "returned_model": "",
        "attempts": attempts,
        "usage": {},
        "completed_at": _timestamp(),
    }
    if response_bytes is None:
        return base

    try:
        envelope = json.loads(response_bytes.decode("utf-8"))
        if not isinstance(envelope, Mapping):
            raise PromptJobError("provider response envelope is not an object")
        returned_model = envelope.get("model")
        usage = envelope.get("usage")
        base["returned_model"] = returned_model if isinstance(returned_model, str) else ""
        base["usage"] = dict(usage) if isinstance(usage, Mapping) else {}
        raw_response = _response_text(envelope)
        base["raw_response"] = raw_response
        parsed = deserialize_response(raw_response)
        base["parsed_response"] = validate_prompt_response(parsed)
        base["status"] = "valid"
    except (UnicodeError, json.JSONDecodeError, PromptJobError, ValueError, TypeError):
        # A successful transport gets exactly one strict validation attempt against
        # the frozen response schema.  Deserialization unwraps the transport's own
        # framing first; the payload itself is never repaired or retried.
        base["status"] = "contract_error"
    return base


def load_completed_request_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PromptJobError(f"invalid resume JSON at {path}:{line_number}") from exc
            request_id = row.get("request_id") if isinstance(row, Mapping) else None
            if not isinstance(request_id, str) or not request_id:
                raise PromptJobError(f"resume row {line_number} has no request_id")
            if request_id in completed:
                raise PromptJobError(f"duplicate request_id in resume output: {request_id}")
            completed.add(request_id)
    return completed


def _append_result(handle, result: Mapping[str, object]) -> None:
    handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()


def run_jobs(
    *,
    jobs_path: Path,
    channel: str,
    backend: str,
    model: str,
    temperature: float,
    max_tokens: int,
    concurrency: int,
    expected_jobs: int,
    output_path: Path,
    limit: int | None = None,
    key: str | None = None,
    timeout: float = 180.0,
    post: Callable[[bytes, str, float], bytes] = _post_request,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    """Preflight, stream, execute, and immediately append terminal results."""

    if concurrency < 1:
        raise PromptJobError("concurrency must be positive")
    if limit is not None and limit < 0:
        raise PromptJobError("limit must be nonnegative")
    preflight = preflight_jobs(
        jobs_path, channel=channel, expected_jobs=expected_jobs
    )
    completed = load_completed_request_ids(output_path)
    unknown = completed - set(preflight.request_ids)
    if unknown:
        raise PromptJobError(
            f"resume output contains {len(unknown)} request IDs outside selected jobs"
        )
    credential = resolve_zai_key() if key is None else key
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scheduled = 0
    written = 0
    status_counts: Counter[str] = Counter()

    def submit_job(pool: ThreadPoolExecutor, row: dict) -> Future:
        return pool.submit(
            execute_one,
            row,
            key=credential,
            provider=backend,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            post=post,
            sleep=sleep,
        )

    with output_path.open("a", encoding="utf-8", buffering=1) as handle:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending: set[Future] = set()
            for row in iter_selected_jobs(jobs_path, channel):
                if row["request_id"] in completed:
                    continue
                if limit is not None and scheduled >= limit:
                    break
                pending.add(submit_job(pool, row))
                scheduled += 1
                if len(pending) >= concurrency:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        result = future.result()
                        _append_result(handle, result)
                        written += 1
                        status_counts[result["status"]] += 1
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    _append_result(handle, result)
                    written += 1
                    status_counts[result["status"]] += 1

    return {
        "selected": preflight.selected_jobs,
        "previously_completed": len(completed),
        "scheduled": scheduled,
        "written": written,
        "valid": status_counts["valid"],
        "contract_error": status_counts["contract_error"],
        "transport_error": status_counts["transport_error"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--backend", default="zai_anthropic", choices=["zai_anthropic"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--expected-jobs", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_jobs(
        jobs_path=args.jobs,
        channel=args.channel,
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        expected_jobs=args.expected_jobs,
        output_path=args.output,
        limit=args.limit,
        timeout=args.timeout,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
