#!/usr/bin/env python3
"""Run the independent Gemma match verifier through an owned chat API.

This preserves :mod:`verify_gemma` prompt rendering, candidate ordering, JSON
validation, row schema, and resume semantics.  Only the inference transport is
replaced.  It is restricted to declared train/dev calibration roles so a frozen
test cannot be consumed accidentally.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from .adjudicate_gemma import append_rows, batched_work, load_inputs, prompt_sha256, scan_candidate_input
from .adjudicate_gemma_api import (
    chat_completion,
    configure_api_access,
    reserve_api_requests,
)
from .common import read_jsonl, sha256_file
from .config import DEFAULT_OUTPUT_ROOT
from .retrieve import stable_shard
from .verify_gemma import (
    iter_verification_work,
    parse_response,
    verification_prompt_equivalence_groups,
)


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.split_role not in {"train", "dev"}:
        raise ValueError("API verifier calibration is restricted to declared train/dev roles")
    configure_api_access(args)
    manifest_path = Path(args.manifest).resolve()
    candidates_path = Path(args.candidates).resolve()
    primary_path = Path(args.primary).resolve()
    output_path = Path(args.output).resolve()
    prompt_paths = [Path(args.prompt).resolve(), *[Path(path).resolve() for path in args.prompt_addon]]
    system_prompt = "\n\n".join(
        path.read_text(encoding="utf-8").rstrip() for path in prompt_paths
    ) + "\n"
    prompt_hash = prompt_sha256(system_prompt)

    primary_rows = list(read_jsonl(primary_path))
    primary_by_uid = {str(row["norm_uid"]): row for row in primary_rows}
    if len(primary_by_uid) != len(primary_rows):
        raise ValueError("duplicate norm_uid in primary adjudications")
    done = (
        {str(row["norm_uid"]) for row in read_jsonl(output_path)}
        if args.resume and output_path.exists()
        else set()
    )
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --resume")

    corpora, _ = scan_candidate_input(
        candidates_path, done=set(), shard_id=0, num_shards=1
    )
    manifest, norms_by_corpus, banks = load_inputs(manifest_path, corpora)
    expected_uids = {
        uid
        for uid, primary in primary_by_uid.items()
        if primary.get("decision") == "MATCH"
        and uid not in done
        and stable_shard(uid, args.num_shards) == args.shard_id
    }
    seen_expected: set[str] = set()
    work = iter_verification_work(
        candidates_path,
        primary_by_uid=primary_by_uid,
        expected_uids=expected_uids,
        seen_expected=seen_expected,
        norms_by_corpus=norms_by_corpus,
        banks=banks,
        system_prompt=system_prompt,
        order_mode=args.order_mode,
        max_alternatives=args.max_alternatives,
        context_chars=args.context_chars,
        description_chars=args.description_chars,
        example_chars=args.example_chars,
        max_examples=args.max_examples,
    )

    started = time.time()
    written = invalid = unique_prompt_inferences = retry_prompt_inferences = 0
    for batch in batched_work(work, args.batch_size):
        representatives, representative_for, group_sizes = (
            verification_prompt_equivalence_groups(batch)
        )
        inference_batch = [batch[index] for index in representatives]
        unique_prompt_inferences += len(inference_batch)
        conversations = [[{"role": "user", "content": item[4]}] for item in inference_batch]
        raw_outputs = infer_many(conversations, args)
        representative_values: dict[int, tuple[dict[str, Any] | None, str | None, str]] = {}
        retry_indices: list[int] = []
        for representative, item, raw in zip(representatives, inference_batch, raw_outputs):
            _, primary, _, alternatives, _ = item
            parsed, error = parse_response(
                raw,
                str(primary["metric_id"]),
                {str(row["metric_id"]) for row in alternatives},
            )
            representative_values[representative] = (parsed, error, raw)
            if parsed is None:
                retry_indices.append(representative)

        if retry_indices:
            retry_prompt_inferences += len(retry_indices)
            retry_conversations = []
            for representative in retry_indices:
                raw = representative_values[representative][2]
                retry_conversations.append(
                    [
                        {"role": "user", "content": batch[representative][4]},
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Your prior answer violated the JSON contract. Return only a valid "
                                "object. CONFIRM_MATCH must repeat the proposal ID; BETTER_CANDIDATE "
                                "must use an alternative ID; every abstention must use metric_id null."
                            ),
                        },
                    ]
                )
            retry_outputs = infer_many(retry_conversations, args)
            for representative, raw in zip(retry_indices, retry_outputs):
                _, primary, _, alternatives, _ = batch[representative]
                parsed, error = parse_response(
                    raw,
                    str(primary["metric_id"]),
                    {str(row["metric_id"]) for row in alternatives},
                )
                representative_values[representative] = (parsed, error, raw)

        rows = []
        for item_index, item in enumerate(batch):
            candidate_row, primary, norm, alternatives, rendered_prompt = item
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
                    "primary_metric_id": primary["metric_id"],
                    "decision": parsed["decision"],
                    "metric_id": parsed["metric_id"],
                    "confidence": parsed["confidence"],
                    "reason": parsed["reason"],
                    "alternative_ids": [row["metric_id"] for row in alternatives],
                    "candidate_bank_source_sha256": candidate_row["bank_source_sha256"],
                    "primary_prompt_sha256": primary.get("prompt_sha256"),
                    "prompt_sha256": prompt_hash,
                    "model": args.model,
                    "order_mode": args.order_mode,
                    "parse_error": error if parsed["decision"] == "INVALID_OUTPUT" else None,
                    "raw_response": raw if args.keep_raw or parsed["decision"] == "INVALID_OUTPUT" else None,
                    "item_prompt_sha256": prompt_sha256(rendered_prompt),
                    "inference_representative_norm_uid": batch[representative][2]["norm_uid"],
                    "inference_equivalence_size": group_sizes[representative],
                }
            )
        append_rows(output_path, rows)
        written += len(rows)
        print(
            f"verified={written}/{len(expected_uids)} invalid={invalid} "
            f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )

    missing_candidates = expected_uids - seen_expected
    if missing_candidates:
        raise ValueError(
            f"primary MATCH rows lack candidates: {len(missing_candidates)}; "
            f"sample={sorted(missing_candidates)[:3]}"
        )
    meta = {
        "schema_version": manifest["schema_version"],
        "selection_role": args.split_role,
        "backend": "owned_openai_compatible_api",
        "api_base_url": args.api_base_url,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "input_candidates": str(candidates_path),
        "input_candidates_sha256": sha256_file(candidates_path),
        "primary": str(primary_path),
        "primary_sha256": sha256_file(primary_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path) if output_path.exists() else None,
        "prompt": str(prompt_paths[0]),
        "prompt_addons": [str(path) for path in prompt_paths[1:]],
        "prompt_component_sha256": {str(path): sha256_file(path) for path in prompt_paths},
        "prompt_sha256": prompt_hash,
        "model": args.model,
        "order_mode": args.order_mode,
        "new_count": written,
        "eligible_count": len(expected_uids),
        "unique_prompt_inferences": unique_prompt_inferences,
        "deduplicated_prompt_count": written - unique_prompt_inferences,
        "retry_prompt_inferences": retry_prompt_inferences,
        "invalid_count": invalid,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "max_alternatives": args.max_alternatives,
        "batch_size": args.batch_size,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": None,
        "seed": args.seed,
        "context_chars": args.context_chars,
        "description_chars": args.description_chars,
        "example_chars": args.example_chars,
        "max_examples": args.max_examples,
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
    parser.add_argument("--primary", required=True)
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
    parser.add_argument("--max-alternatives", type=int, default=49)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--seed", type=int, default=29)
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
    parser.add_argument("--context-chars", type=int, default=1200)
    parser.add_argument("--description-chars", type=int, default=260)
    parser.add_argument("--example-chars", type=int, default=180)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard-id must be in [0, --num-shards)")
    if args.max_alternatives < 1 or args.batch_size < 1 or args.concurrency < 1:
        parser.error("alternatives, batch size, and concurrency must be positive")
    if min(args.context_chars, args.description_chars, args.example_chars) < 1:
        parser.error("prompt truncation lengths must be positive")
    if args.max_examples < 0 or args.transport_retries < 0 or args.max_api_requests < 0:
        parser.error("example and retry counts must be nonnegative")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
