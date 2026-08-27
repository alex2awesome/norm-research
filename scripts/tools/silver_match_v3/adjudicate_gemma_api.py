#!/usr/bin/env python3
"""Run the frozen Gemma adjudicator through an owned OpenAI-compatible API.

This is the low-footprint counterpart to :mod:`adjudicate_gemma`.  Prompt
rendering, candidate ordering, response parsing, output rows, and resume
semantics are intentionally shared with the local-vLLM runner; only inference
transport differs.  It is intended for small train/dev GEPA rounds, never for
silently consuming a frozen test split.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from .adjudicate_gemma import (
    append_rows,
    batched_work,
    iter_work_items,
    load_inputs,
    parse_response,
    prompt_equivalence_groups,
    prompt_sha256,
    scan_candidate_input,
)
from .common import read_jsonl, sha256_file
from .config import DEFAULT_OUTPUT_ROOT


def chat_completion(
    *,
    base_url: str,
    model: str,
    messages: Sequence[dict[str, str]],
    max_tokens: int,
    seed: int,
    timeout: float,
    transport_retries: int,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    reasoning_exclude: bool = True,
    force_json_object: bool = False,
) -> str:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    if reasoning_effort is not None:
        payload["reasoning"] = {
            "effort": reasoning_effort,
            "exclude": reasoning_exclude,
        }
    if force_json_object:
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or 'EMPTY'}",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(transport_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            choices = payload.get("choices") or []
            if not choices:
                raise ValueError("API response lacks choices")
            return str(((choices[0].get("message") or {}).get("content")) or "")
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= transport_retries:
                break
            time.sleep(min(2.0, 0.25 * (2**attempt)))
    raise RuntimeError(f"Gemma API request failed after retries: {last_error}")


def infer_many(
    conversations: Sequence[Sequence[dict[str, str]]], args: argparse.Namespace
) -> list[str]:
    reserve_api_requests(args, len(conversations))

    def infer(messages: Sequence[dict[str, str]]) -> str:
        return chat_completion(
            base_url=args.api_base_url,
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            seed=args.seed,
            timeout=args.request_timeout,
            transport_retries=args.transport_retries,
            api_key=args._api_key,
            reasoning_effort=args.reasoning_effort,
            reasoning_exclude=args.reasoning_exclude,
            force_json_object=args.force_json_object,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        return list(executor.map(infer, conversations))


def configure_api_access(args: argparse.Namespace) -> None:
    """Load credentials without exposing them and initialize a hard request cap."""
    is_openrouter = "openrouter.ai" in args.api_base_url.lower()
    if is_openrouter and not args.api_key_file:
        raise ValueError("OpenRouter requires --api-key-file")
    if is_openrouter and args.max_api_requests < 1:
        raise ValueError("OpenRouter requires a positive --max-api-requests cap")
    api_key = None
    if args.api_key_file:
        key_path = Path(args.api_key_file).expanduser().resolve()
        api_key = key_path.read_text(encoding="utf-8").strip()
        if not api_key:
            raise ValueError(f"empty API key file: {key_path}")
    args._api_key = api_key
    args._api_request_count = 0
    args._api_request_lock = threading.Lock()


def reserve_api_requests(args: argparse.Namespace, count: int) -> None:
    """Atomically reserve calls so concurrent batches cannot exceed the spend cap."""
    if count < 0:
        raise ValueError("request reservation cannot be negative")
    with args._api_request_lock:
        proposed = args._api_request_count + count
        if args.max_api_requests and proposed > args.max_api_requests:
            raise RuntimeError(
                f"API request cap exceeded: proposed={proposed} "
                f"cap={args.max_api_requests}"
            )
        args._api_request_count = proposed


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.split_role not in {"train", "dev"}:
        raise ValueError("API GEPA runner is restricted to declared train/dev roles")
    configure_api_access(args)
    manifest_path = Path(args.manifest).resolve()
    candidates_path = Path(args.candidates).resolve()
    output_path = Path(args.output).resolve()
    prompt_paths = [Path(args.prompt).resolve(), *[Path(p).resolve() for p in args.prompt_addon]]
    system_prompt = "\n\n".join(
        path.read_text(encoding="utf-8").rstrip() for path in prompt_paths
    ) + "\n"
    prompt_hash = prompt_sha256(system_prompt)
    done = (
        {str(row["norm_uid"]) for row in read_jsonl(output_path)}
        if args.resume and output_path.exists()
        else set()
    )
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --resume")
    corpora, eligible_count = scan_candidate_input(
        candidates_path,
        done=done,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    manifest, norms_by_corpus, banks = load_inputs(manifest_path, corpora)
    work = iter_work_items(
        candidates_path,
        done=done,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        max_candidates=args.max_candidates,
        order_mode=args.order_mode,
        system_prompt=system_prompt,
        norms_by_corpus=norms_by_corpus,
        banks=banks,
        context_chars=args.context_chars,
        description_chars=args.description_chars,
        example_chars=args.example_chars,
        max_examples=args.max_examples,
    )
    started = time.time()
    written = invalid = unique_prompt_inferences = retry_prompt_inferences = 0
    for batch in batched_work(work, args.batch_size):
        representatives, representative_for, group_sizes = prompt_equivalence_groups(batch)
        inference_batch = [batch[index] for index in representatives]
        unique_prompt_inferences += len(inference_batch)
        conversations = [[{"role": "user", "content": item[3]}] for item in inference_batch]
        raw_outputs = infer_many(conversations, args)
        representative_values: dict[int, tuple[dict[str, Any] | None, str | None, str]] = {}
        retry_indices: list[int] = []
        for representative, item, raw in zip(representatives, inference_batch, raw_outputs):
            parsed, error = parse_response(raw, {row["metric_id"] for row in item[2]})
            representative_values[representative] = (parsed, error, raw)
            if parsed is None:
                retry_indices.append(representative)
        if retry_indices:
            retry_prompt_inferences += len(retry_indices)
            retry_conversations = []
            for representative in retry_indices:
                original = batch[representative][3]
                raw = representative_values[representative][2]
                retry_conversations.append(
                    [
                        {"role": "user", "content": original},
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Your prior answer violated the JSON contract. Return only a valid "
                                "object. MATCH must use an ID from this item's cards; every abstention "
                                "must use metric_id null."
                            ),
                        },
                    ]
                )
            retry_outputs = infer_many(retry_conversations, args)
            for representative, raw in zip(retry_indices, retry_outputs):
                parsed, error = parse_response(
                    raw, {row["metric_id"] for row in batch[representative][2]}
                )
                representative_values[representative] = (parsed, error, raw)
        rows = []
        for item_index, item in enumerate(batch):
            candidate_row, norm, candidates, rendered_prompt = item
            representative = representative_for[item_index]
            parsed, error, raw = representative_values[representative]
            if parsed is None:
                invalid += 1
                parsed = {
                    "decision": "INVALID_OUTPUT",
                    "metric_id": None,
                    "confidence": "low",
                    "reason": error,
                }
            rows.append(
                {
                    "schema_version": manifest["schema_version"],
                    "norm_uid": norm["norm_uid"],
                    "corpus": norm["corpus"],
                    "task": norm["task"],
                    "row": norm["row"],
                    "decision": parsed["decision"],
                    "metric_id": parsed["metric_id"],
                    "confidence": parsed["confidence"],
                    "reason": parsed["reason"],
                    "candidate_ids": [row["metric_id"] for row in candidates],
                    "candidate_bank_source_sha256": candidate_row["bank_source_sha256"],
                    "prompt_sha256": prompt_hash,
                    "model": args.model,
                    "order_mode": args.order_mode,
                    "parse_error": error if parsed["decision"] == "INVALID_OUTPUT" else None,
                    "raw_response": raw if args.keep_raw or parsed["decision"] == "INVALID_OUTPUT" else None,
                    "item_prompt_sha256": prompt_sha256(rendered_prompt),
                    "inference_representative_norm_uid": batch[representative][1]["norm_uid"],
                    "inference_equivalence_size": group_sizes[representative],
                    "rescue_trial": candidate_row.get("rescue_trial"),
                    "rescue_lane": candidate_row.get("rescue_lane"),
                    "rescue_system": candidate_row.get("rescue_system"),
                    "rescue_coverage_before": candidate_row.get("rescue_coverage_before"),
                    "rescue_coverage_after": candidate_row.get("rescue_coverage_after"),
                    "rescue_bank_count": candidate_row.get("rescue_bank_count"),
                }
            )
        append_rows(output_path, rows)
        written += len(rows)
        print(
            f"adjudicated={written}/{eligible_count} invalid={invalid} "
            f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )
    meta = {
        "schema_version": manifest["schema_version"],
        "selection_role": args.split_role,
        "backend": "owned_openai_compatible_api",
        "api_base_url": args.api_base_url,
        "input_candidates": str(candidates_path),
        "input_candidates_sha256": sha256_file(candidates_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path) if output_path.exists() else None,
        "prompt": str(prompt_paths[0]),
        "prompt_addons": [str(path) for path in prompt_paths[1:]],
        "prompt_component_sha256": {str(path): sha256_file(path) for path in prompt_paths},
        "prompt_sha256": prompt_hash,
        "model": args.model,
        "order_mode": args.order_mode,
        "max_candidates": args.max_candidates,
        "prompt_rendering": {
            "context_chars": args.context_chars,
            "description_chars": args.description_chars,
            "example_chars": args.example_chars,
            "max_examples": args.max_examples,
        },
        "new_count": written,
        "eligible_count": eligible_count,
        "unique_prompt_inferences": unique_prompt_inferences,
        "deduplicated_prompt_count": written - unique_prompt_inferences,
        "retry_prompt_inferences": retry_prompt_inferences,
        "invalid_count": invalid,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "concurrency": args.concurrency,
        "api_request_count": args._api_request_count,
        "max_api_requests": args.max_api_requests,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_exclude": args.reasoning_exclude,
        "force_json_object": args.force_json_object,
        "api_key_file_supplied": bool(args.api_key_file),
        "elapsed_seconds": time.time() - started,
    }
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifest.json"))
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-role", choices=("train", "dev"), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-addon", action="append", default=[])
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8006/v1")
    parser.add_argument(
        "--api-key-file",
        help="read a bearer token from this file; the value is never logged",
    )
    parser.add_argument(
        "--max-api-requests",
        type=int,
        default=0,
        help="hard cap across initial and parse-retry calls; required for OpenRouter",
    )
    parser.add_argument("--model", default="gemma")
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--context-chars", type=int, default=1200)
    parser.add_argument("--description-chars", type=int, default=260)
    parser.add_argument("--example-chars", type=int, default=80)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--transport-retries", type=int, default=2)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high"),
        help="optional OpenRouter reasoning.effort; omitted for local vLLM",
    )
    parser.add_argument(
        "--include-reasoning",
        dest="reasoning_exclude",
        action="store_false",
        help="return reasoning tokens in the provider response (excluded by default)",
    )
    parser.set_defaults(reasoning_exclude=True)
    parser.add_argument("--force-json-object", action="store_true")
    parser.add_argument("--order-mode", choices=("original", "reverse", "hashed"), default="original")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard-id must be in [0, --num-shards)")
    if args.max_candidates < 1 or args.concurrency < 1 or args.batch_size < 1:
        parser.error("candidate, concurrency, and batch sizes must be positive")
    if min(args.context_chars, args.description_chars, args.example_chars) < 1:
        parser.error("prompt truncation lengths must be positive")
    if args.max_examples < 0 or args.transport_retries < 0 or args.max_api_requests < 0:
        parser.error("example and retry counts must be nonnegative")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
