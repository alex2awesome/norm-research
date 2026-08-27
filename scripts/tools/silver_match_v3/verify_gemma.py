#!/usr/bin/env python3
"""Independent Gemma verification of proposed norm-to-metric matches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

from .adjudicate_gemma import (
    CONFIDENCES,
    append_rows,
    batched_work,
    load_inputs,
    ordered_candidates,
    prompt_sha256,
    scan_candidate_input,
    truncate,
)
from .common import read_jsonl, sha256_file
from .config import DEFAULT_OUTPUT_ROOT, GEMMA4
from .retrieve import stable_shard


DECISIONS = {
    "CONFIRM_MATCH",
    "AMBIGUOUS_MATCH",
    "BETTER_CANDIDATE",
    "NO_EXPLICIT_CRITERION",
    "CONTEXT_NEEDED",
    "GENERIC_VERDICT",
    "NO_CANDIDATE_FITS",
    "NOISE",
}


def verification_prompt_equivalence_groups(
    batch: list[
        tuple[
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            list[dict[str, Any]],
            str,
        ]
    ],
) -> tuple[list[int], list[int], dict[int, int]]:
    representatives: list[int] = []
    representative_for: list[int] = []
    sizes: dict[int, int] = {}
    seen: dict[tuple[str, str, tuple[str, ...]], int] = {}
    for index, (_, primary, _, alternatives, prompt) in enumerate(batch):
        key = (
            prompt,
            str(primary.get("metric_id") or ""),
            tuple(str(row["metric_id"]) for row in alternatives),
        )
        representative = seen.get(key)
        if representative is None:
            representative = index
            seen[key] = index
            representatives.append(index)
            sizes[index] = 0
        representative_for.append(representative)
        sizes[representative] += 1
    return representatives, representative_for, sizes


def build_verification_prompt(
    system_prompt: str,
    norm: dict[str, Any],
    primary_metric: dict[str, Any],
    alternatives: list[dict[str, Any]],
    metric_by_id: dict[str, dict[str, Any]],
    *,
    context_chars: int,
    description_chars: int,
    example_chars: int,
    max_examples: int,
) -> str:
    lines = [system_prompt.rstrip(), "", f"TASK BANK: {norm['task']}"]
    lines.append(f'HUMAN STATEMENT: "{norm["norm"]}"')
    context = truncate(norm.get("context"), context_chars)
    if context and context != str(norm["norm"]).strip():
        lines.append(f'EVIDENCE PASSAGE FROM THE HUMAN FEEDBACK: "{context}"')
    lines.extend(["", "PROPOSED METRIC:"])

    def add_card(metric: dict[str, Any], label: str) -> None:
        lines.append(f"{label} [{metric['metric_id']}] {metric['name']}")
        lines.append(f"  Definition: {truncate(metric['description'], description_chars)}")
        examples = metric.get("examples") or []
        if examples and max_examples:
            rendered = "; ".join(
                truncate(example, example_chars) for example in examples[:max_examples]
            )
            lines.append(f"  Bank examples: {rendered}")

    add_card(primary_metric, "PROPOSAL")
    lines.extend(["", "STRONGEST ALTERNATIVES:"])
    for row in alternatives:
        add_card(metric_by_id[row["metric_id"]], "ALTERNATIVE")
    lines.extend(["", "Return the JSON verification now."])
    return "\n".join(lines)


def parse_response(
    raw: str,
    primary_metric_id: str,
    alternative_ids: set[str],
) -> tuple[dict[str, Any] | None, str | None]:
    decoder = json.JSONDecoder()
    objects = []
    for start, char in enumerate(raw or ""):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode((raw or "")[start:])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict) and "decision" in value:
            objects.append(value)
    if not objects:
        return None, "no_json"
    value = objects[-1]
    decision = str(value.get("decision") or "").strip().upper()
    metric_id = value.get("metric_id")
    metric_id = None if metric_id in (None, "", "null", "None") else str(metric_id).strip()
    confidence = str(value.get("confidence") or "").strip().lower()
    reason = str(value.get("reason") or "").strip()
    if decision not in DECISIONS:
        return None, "unknown_decision"
    if confidence not in CONFIDENCES:
        return None, "unknown_confidence"
    if decision == "CONFIRM_MATCH" and metric_id != primary_metric_id:
        return None, "confirm_metric_mismatch"
    if decision == "BETTER_CANDIDATE" and metric_id not in alternative_ids:
        return None, "better_metric_not_alternative"
    if decision not in {"CONFIRM_MATCH", "BETTER_CANDIDATE"} and metric_id is not None:
        return None, "metric_on_abstention"
    if not reason:
        return None, "missing_reason"
    return {
        "decision": decision,
        "metric_id": metric_id,
        "confidence": confidence,
        "reason": reason,
    }, None


def iter_verification_work(
    candidates_path: Path,
    *,
    primary_by_uid: dict[str, dict[str, Any]],
    expected_uids: set[str],
    seen_expected: set[str],
    norms_by_corpus: dict[str, dict[str, Any]],
    banks: dict[str, dict[str, Any]],
    system_prompt: str,
    order_mode: str,
    max_alternatives: int,
    context_chars: int,
    description_chars: int,
    example_chars: int,
    max_examples: int,
) -> Iterator[
    tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        list[dict[str, Any]],
        str,
    ]
]:
    for candidate_row in read_jsonl(candidates_path):
        uid = str(candidate_row["norm_uid"])
        if uid not in expected_uids:
            continue
        if uid in seen_expected:
            raise ValueError(f"duplicate candidate norm_uid for verification: {uid}")
        primary = primary_by_uid[uid]
        corpus = str(candidate_row["corpus"])
        norm = norms_by_corpus[corpus].get(uid)
        if norm is None:
            raise KeyError(f"candidate norm_uid missing from manifest: {uid}")
        task_bank = banks[norm["task"]]
        primary_id = str(primary.get("metric_id") or "")
        if primary_id not in task_bank:
            raise KeyError(f"primary metric absent from bank for {uid}: {primary_id}")
        candidate_ids = [
            str(row["metric_id"]) for row in candidate_row.get("candidates") or []
        ]
        if primary_id not in candidate_ids:
            raise ValueError(f"primary metric absent from candidate slate for {uid}")
        alternatives = [
            row
            for row in ordered_candidates(
                candidate_row["candidates"], order_mode, uid
            )
            if row["metric_id"] != primary_id
        ][:max_alternatives]
        prompt = build_verification_prompt(
            system_prompt,
            norm,
            task_bank[primary_id],
            alternatives,
            task_bank,
            context_chars=context_chars,
            description_chars=description_chars,
            example_chars=example_chars,
            max_examples=max_examples,
        )
        seen_expected.add(uid)
        yield candidate_row, primary, norm, alternatives, prompt


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    candidates_path = Path(args.candidates)
    primary_path = Path(args.primary)
    output_path = Path(args.output)
    prompt_paths = [Path(args.prompt), *[Path(path) for path in args.prompt_addon]]
    system_prompt = "\n\n".join(
        path.read_text(encoding="utf-8").rstrip() for path in prompt_paths
    ) + "\n"
    prompt_hash = prompt_sha256(system_prompt)
    primary_rows = list(read_jsonl(primary_path))
    primary_by_uid = {row["norm_uid"]: row for row in primary_rows}
    if len(primary_by_uid) != len(primary_rows):
        raise ValueError("duplicate norm_uid in primary adjudications")
    done = (
        {row["norm_uid"] for row in read_jsonl(output_path)}
        if args.resume and output_path.exists()
        else set()
    )
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --resume")
    corpora, _ = scan_candidate_input(
        candidates_path,
        done=set(),
        shard_id=0,
        num_shards=1,
    )
    manifest, norms_by_corpus, banks = load_inputs(manifest_path, corpora)
    expected_uids = {
        str(uid)
        for uid, primary in primary_by_uid.items()
        if primary.get("decision") == "MATCH"
        and uid not in done
        and stable_shard(str(uid), args.num_shards) == args.shard_id
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

    os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/shared_hf_cache")
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/lfs/skampere3/0/alexspan")
    os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)
    started = time.time()
    written = invalid = unique_prompt_inferences = retry_prompt_inferences = 0
    for batch in batched_work(work, args.batch_size):
        representatives, representative_for, group_sizes = (
            verification_prompt_equivalence_groups(batch)
        )
        inference_batch = [batch[index] for index in representatives]
        unique_prompt_inferences += len(inference_batch)
        conversations = [
            [{"role": "user", "content": item[4]}] for item in inference_batch
        ]
        outputs = llm.chat(conversations, sampling, use_tqdm=False)
        rows = []
        representative_values: dict[
            int, tuple[dict[str, Any] | None, str | None, str]
        ] = {}
        retry_indices = []
        for representative, item, output in zip(
            representatives, inference_batch, outputs
        ):
            candidate_row, primary, norm, alternatives, _ = item
            raw = output.outputs[0].text if output.outputs else ""
            parsed, error = parse_response(
                raw,
                str(primary["metric_id"]),
                {row["metric_id"] for row in alternatives},
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
            retry_outputs = llm.chat(retry_conversations, sampling, use_tqdm=False)
            for representative, output in zip(retry_indices, retry_outputs):
                _, primary, _, alternatives, _ = batch[representative]
                raw = output.outputs[0].text if output.outputs else ""
                parsed, error = parse_response(
                    raw,
                    str(primary["metric_id"]),
                    {row["metric_id"] for row in alternatives},
                )
                representative_values[representative] = (parsed, error, raw)

        parsed_values = [
            representative_values[representative]
            for representative in representative_for
        ]
        for item_index, (item, (parsed, error, raw)) in enumerate(
            zip(batch, parsed_values)
        ):
            candidate_row, primary, norm, alternatives, prompt = item
            representative = representative_for[item_index]
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
                    "item_prompt_sha256": prompt_sha256(prompt),
                    "inference_representative_norm_uid": batch[representative][2][
                        "norm_uid"
                    ],
                    "inference_equivalence_size": group_sizes[representative],
                }
            )
        append_rows(output_path, rows)
        written += len(rows)
        print(
            f"verified={written}/{len(expected_uids)} written={written} "
            f"invalid={invalid} elapsed={time.time() - started:.0f}s",
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
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "input_candidates": str(candidates_path),
        "input_candidates_sha256": sha256_file(candidates_path),
        "primary": str(primary_path),
        "primary_sha256": sha256_file(primary_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path) if output_path.exists() else None,
        "prompt": str(args.prompt),
        "prompt_addons": [str(path) for path in args.prompt_addon],
        "prompt_component_sha256": {
            str(path): sha256_file(path) for path in prompt_paths
        },
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
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
        "context_chars": args.context_chars,
        "description_chars": args.description_chars,
        "example_chars": args.example_chars,
        "max_examples": args.max_examples,
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
    parser.add_argument(
        "--prompt",
        default=str(Path(__file__).with_name("prompts") / "verify_match_v1.txt"),
    )
    parser.add_argument(
        "--prompt-addon",
        action="append",
        default=[],
        help="append a task-specific GEPA verification instruction (repeatable)",
    )
    parser.add_argument("--model", default=GEMMA4)
    parser.add_argument("--max-alternatives", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="disable CUDA graph/compile paths for runtime compatibility audits",
    )
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--order-mode", choices=("original", "reverse", "hashed"), default="reverse")
    parser.add_argument("--context-chars", type=int, default=1400)
    parser.add_argument("--description-chars", type=int, default=520)
    parser.add_argument("--example-chars", type=int, default=180)
    parser.add_argument("--max-examples", type=int, default=2)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()
    if not (0 <= args.shard_id < args.num_shards):
        parser.error("--shard-id must be in [0, --num-shards)")
    if args.max_alternatives < 1:
        parser.error("--max-alternatives must be positive")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
