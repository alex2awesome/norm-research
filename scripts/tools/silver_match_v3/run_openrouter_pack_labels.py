#!/usr/bin/env python3
"""Label a frozen full-bank pack through OpenRouter with auditable requests.

The runner is deliberately narrower than the general API adjudicator: its only
admissible semantic inputs are the supplied guides, one frozen bank, and one
frozen chunk.  Every request and response is written without credentials so a
separate audit can reconstruct exactly what the remote model saw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from .adjudicate_gemma import CONFIDENCES, DECISIONS
from .common import read_jsonl, sha256_file


TRANSCRIPT_SCHEMA = "silver-match-v3-openrouter-labeler-transcript-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_messages(
    *,
    task: str,
    chunk_id: str,
    guides: Sequence[tuple[str, str]],
    bank_text: str,
    chunk_text: str,
    pass_name: str,
) -> list[dict[str, str]]:
    guide_text = "\n\n".join(
        f"===== {name} =====\n{text.rstrip()}" for name, text in guides
    )
    system = (
        "You are an independent hidden-ID full-bank annotator for a final risk "
        "audit. Your admissible evidence is exactly the guides, frozen bank, and "
        "frozen item chunk included in this request. You have no access to system "
        "predictions, sample keys, prior labels, proposals, MI, outcomes, or the "
        "other independent pass. Label every item from scratch.\n\n"
        f"PASS NAME: {pass_name}\nTASK: {task}\n\n{guide_text}\n\n"
        "===== COMPLETE FROZEN BANK =====\n"
        f"{bank_text.rstrip()}"
    )
    user = (
        f"Label all items in chunk {chunk_id}. Consider the complete bank for every "
        "item. Return exactly one schema-conforming JSON object with task set to "
        f"{json.dumps(task)} and chunk_id set to {json.dumps(chunk_id)}. Preserve "
        "every norm_uid verbatim and exactly once. MATCH requires one exact bank "
        "leaf that wins the nearest-sibling contrast; otherwise use the precise "
        "typed abstention and metric_id null. Do not force yield.\n\n"
        "===== FROZEN ITEM CHUNK =====\n"
        f"{chunk_text.rstrip()}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_request(
    *,
    model: str,
    messages: Sequence[dict[str, str]],
    schema: dict[str, Any],
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    # OpenRouter passes the standard JSON-schema response contract through to
    # supported providers.  The prompt still contains the full contract so a
    # provider-side schema rejection can be retried without changing semantics.
    return {
        "model": model,
        "messages": list(messages),
        "temperature": 0.0,
        "seed": seed,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "independent_full_bank_labels",
                "strict": True,
                "schema": provider_schema(schema),
            },
        },
        "usage": {"include": True},
    }


def provider_schema(value: Any) -> Any:
    """Remove validation-only keywords unsupported by Anthropic structured output.

    Exact UID/count/metric validation remains fail-closed in ``validate_payload``;
    this transport schema is only the provider-side JSON-shape constraint.
    """
    unsupported = {"$schema", "minItems", "maxItems", "minLength", "maxLength", "pattern"}
    if isinstance(value, dict):
        return {
            key: provider_schema(child)
            for key, child in value.items()
            if key not in unsupported
        }
    if isinstance(value, list):
        return [provider_schema(child) for child in value]
    return value


def parse_json_content(content: str) -> dict[str, Any]:
    value = content.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    return parsed


def validate_payload(
    payload: dict[str, Any],
    *,
    task: str,
    chunk_id: str,
    expected_uids: Sequence[str],
    bank_ids: set[str],
) -> None:
    if payload.get("task") != task or payload.get("chunk_id") != chunk_id:
        raise ValueError("task/chunk mismatch")
    labels = payload.get("labels")
    if not isinstance(labels, list) or len(labels) != len(expected_uids):
        raise ValueError("label count mismatch")
    observed = [str(row.get("norm_uid") or "") for row in labels]
    if len(set(observed)) != len(observed) or set(observed) != set(expected_uids):
        raise ValueError("UID coverage mismatch")
    for row in labels:
        uid = str(row.get("norm_uid") or "")
        decision = str(row.get("decision") or "").upper()
        confidence = str(row.get("confidence") or "").lower()
        metric_id = row.get("metric_id")
        metric_id = None if metric_id is None else str(metric_id)
        reason = str(row.get("reason") or "").strip()
        if decision not in DECISIONS or confidence not in CONFIDENCES:
            raise ValueError(f"invalid decision/confidence: {uid}")
        if len(reason) < 8 or len(reason) > 600:
            raise ValueError(f"invalid reason: {uid}")
        if decision == "MATCH":
            if metric_id not in bank_ids:
                raise ValueError(f"MATCH metric absent from bank: {uid}/{metric_id}")
        elif metric_id is not None:
            raise ValueError(f"abstention carries metric ID: {uid}")


class Budget:
    def __init__(self, max_requests: int, max_reported_cost: float) -> None:
        self.max_requests = max_requests
        self.max_reported_cost = max_reported_cost
        self.requests = 0
        self.reported_cost = 0.0
        self.lock = threading.Lock()

    def reserve(self) -> None:
        with self.lock:
            if self.requests + 1 > self.max_requests:
                raise RuntimeError("OpenRouter request cap exceeded")
            if self.reported_cost >= self.max_reported_cost:
                raise RuntimeError("OpenRouter reported-cost cap reached")
            self.requests += 1

    def record(self, usage: dict[str, Any]) -> None:
        raw = usage.get("cost")
        try:
            cost = float(raw or 0.0)
        except (TypeError, ValueError):
            cost = 0.0
        with self.lock:
            self.reported_cost += cost
            if self.reported_cost > self.max_reported_cost:
                raise RuntimeError(
                    f"OpenRouter reported-cost cap exceeded: {self.reported_cost:.6f}"
                )


def call_api(
    *,
    endpoint: str,
    api_key: str,
    body: dict[str, Any],
    timeout: float,
    budget: Budget,
) -> dict[str, Any]:
    budget.reserve()
    request = urllib.request.Request(
        endpoint,
        data=canonical_json(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/silver-match-v3/audit",
            "X-Title": "silver-match-v3-final-risk-audit",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    if not isinstance(result, dict) or not result.get("choices"):
        raise ValueError("OpenRouter response lacks choices")
    budget.record(result.get("usage") or {})
    return result


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--pass-name", required=True)
    parser.add_argument("--guide", action="append", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--api-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model", default="anthropic/claude-sonnet-5")
    parser.add_argument("--seed", type=int, default=7193)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument(
        "--chunk-id",
        action="append",
        default=[],
        help="Run only the named chunk(s), for a bounded preflight or repair.",
    )
    parser.add_argument("--timeout", type=float, default=360.0)
    parser.add_argument("--max-api-requests", type=int, required=True)
    parser.add_argument("--max-reported-cost-usd", type=float, required=True)
    args = parser.parse_args()
    if min(args.max_tokens, args.concurrency, args.attempts, args.max_api_requests) < 1:
        parser.error("token/concurrency/attempt/request values must be positive")
    if args.max_reported_cost_usd <= 0:
        parser.error("reported cost cap must be positive")

    root = Path(args.pack_root).resolve()
    validation_path = root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("task") != args.task or validation.get("truth_hidden") is not True:
        raise ValueError("pack is not a truth-hidden pack for this task")
    bank_path, items_path = root / "bank.json", root / "items.jsonl"
    if sha256_file(bank_path) != validation["outputs"]["bank"]["sha256"]:
        raise ValueError("bank hash mismatch")
    if sha256_file(items_path) != validation["outputs"]["items"]["sha256"]:
        raise ValueError("items hash mismatch")
    if list(root.glob("*.key.jsonl")) or list(root.glob("*candidate*")):
        raise ValueError("pack exposes a key or candidate artifact")

    guide_paths = [Path(value).resolve() for value in args.guide]
    schema_path = Path(args.schema).resolve()
    runner_path = Path(__file__).resolve()
    guides = [(path.name, path.read_text(encoding="utf-8")) for path in guide_paths]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    bank_text = bank_path.read_text(encoding="utf-8")
    bank = json.loads(bank_text)
    bank_ids = {str(row["metric_id"]) for row in bank.get("metrics") or []}
    if not bank_ids or bank.get("task") != args.task:
        raise ValueError("invalid frozen bank")
    key = Path(args.api_key_file).expanduser().read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError("empty API key")
    endpoint = args.api_base_url.rstrip("/") + "/chat/completions"
    budget = Budget(args.max_api_requests, args.max_reported_cost_usd)
    chunks = sorted((root / "chunks").glob("part-*.jsonl"))
    recorded_by_name = {
        Path(path).name: digest
        for path, digest in (validation["outputs"].get("chunks") or {}).items()
    }
    if not chunks or {path.name for path in chunks} != set(recorded_by_name):
        raise ValueError("chunk inventory mismatch")
    for path in chunks:
        if sha256_file(path) != recorded_by_name[path.name]:
            raise ValueError(f"chunk hash mismatch: {path}")
    if args.chunk_id:
        requested = set(args.chunk_id)
        available = {path.stem for path in chunks}
        if not requested.issubset(available):
            raise ValueError(f"unknown requested chunks: {sorted(requested - available)}")
        chunks = [path for path in chunks if path.stem in requested]

    raw_root, transcript_root = root / "raw_labels", root / "api_transcripts"
    raw_root.mkdir(exist_ok=True)
    transcript_root.mkdir(exist_ok=True)

    def run_chunk(chunk_path: Path) -> dict[str, Any]:
        chunk_id = chunk_path.stem
        raw_path = raw_root / f"{chunk_id}.json"
        transcript_path = transcript_root / f"{chunk_id}.json"
        expected_uids = [str(row["norm_uid"]) for row in read_jsonl(chunk_path)]
        if raw_path.exists() and transcript_path.exists():
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            validate_payload(
                payload,
                task=args.task,
                chunk_id=chunk_id,
                expected_uids=expected_uids,
                bank_ids=bank_ids,
            )
            return {"chunk": chunk_id, "status": "skipped_valid", "count": len(expected_uids)}
        if raw_path.exists() or transcript_path.exists():
            raise FileExistsError(f"partial prior output for {chunk_id}")
        messages = build_messages(
            task=args.task,
            chunk_id=chunk_id,
            guides=guides,
            bank_text=bank_text,
            chunk_text=chunk_path.read_text(encoding="utf-8"),
            pass_name=args.pass_name,
        )
        attempts: list[dict[str, Any]] = []
        current_messages = messages
        final_payload = None
        for ordinal in range(1, args.attempts + 1):
            body = build_request(
                model=args.model,
                messages=current_messages,
                schema=schema,
                max_tokens=args.max_tokens,
                seed=args.seed,
            )
            started = time.time()
            response = call_api(
                endpoint=endpoint,
                api_key=key,
                body=body,
                timeout=args.timeout,
                budget=budget,
            )
            message = (response["choices"][0] or {}).get("message") or {}
            content = message.get("content")
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text") or "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            content = str(content or "")
            error = None
            try:
                parsed = parse_json_content(content)
                validate_payload(
                    parsed,
                    task=args.task,
                    chunk_id=chunk_id,
                    expected_uids=expected_uids,
                    bank_ids=bank_ids,
                )
                final_payload = parsed
            except Exception as exc:  # preserve exact invalid response before retry
                error = f"{type(exc).__name__}: {exc}"
            attempts.append(
                {
                    "ordinal": ordinal,
                    "request": body,
                    "request_sha256": sha256_text(canonical_json(body)),
                    "response_id": response.get("id"),
                    "response_model": response.get("model"),
                    "response_content": content,
                    "response_content_sha256": sha256_text(content),
                    "usage": response.get("usage") or {},
                    "elapsed_seconds": time.time() - started,
                    "validation_error": error,
                }
            )
            if final_payload is not None:
                break
            current_messages = [
                *messages,
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "The prior response violated the frozen JSON/UID/metric contract: "
                        f"{error}. Return the complete corrected JSON object only."
                    ),
                },
            ]
        if final_payload is None:
            raise RuntimeError(f"all structured-output attempts failed for {chunk_id}")
        atomic_json(raw_path, final_payload)
        transcript = {
            "schema_version": TRANSCRIPT_SCHEMA,
            "status": "COMPLETE",
            "truth_hidden": True,
            "task": args.task,
            "pass_name": args.pass_name,
            "chunk_id": chunk_id,
            "model": args.model,
            "api_base_url": args.api_base_url,
            "input_artifacts": {
                "runner": {"path": str(runner_path), "sha256": sha256_file(runner_path)},
                "pack_validation": {
                    "path": str(validation_path),
                    "sha256": sha256_file(validation_path),
                },
                "bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
                "items": {"path": str(items_path), "sha256": sha256_file(items_path)},
                "chunk": {"path": str(chunk_path), "sha256": sha256_file(chunk_path)},
                "schema": {"path": str(schema_path), "sha256": sha256_file(schema_path)},
                "guides": [
                    {"path": str(path), "sha256": sha256_file(path)} for path in guide_paths
                ],
            },
            "api_key_logged": False,
            "attempts": attempts,
            "raw_label": {"path": str(raw_path), "sha256": sha256_file(raw_path)},
        }
        atomic_json(transcript_path, transcript)
        return {
            "chunk": chunk_id,
            "status": "completed",
            "count": len(expected_uids),
            "attempts": len(attempts),
            "reported_cost": sum(float((row.get("usage") or {}).get("cost") or 0.0) for row in attempts),
        }

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(run_chunk, path): path for path in chunks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    summary = {
        "task": args.task,
        "pass_name": args.pass_name,
        "model": args.model,
        "chunks": len(chunks),
        "rows": sum(row["count"] for row in results),
        "completed": sum(row["status"] == "completed" for row in results),
        "skipped_valid": sum(row["status"] == "skipped_valid" for row in results),
        "api_requests": budget.requests,
        "reported_cost_usd": budget.reported_cost,
        "max_api_requests": args.max_api_requests,
        "max_reported_cost_usd": args.max_reported_cost_usd,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
