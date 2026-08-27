#!/usr/bin/env python3
"""Run truth-blind paired base/PEFT-LoRA Gemma adjudication with batch vLLM.

The base model is loaded once with vLLM LoRA support enabled.  Every rendered
prompt is then decoded once without an adapter and once with the named adapter,
under identical candidates, order, sampling parameters, and retry policy.  The
two arms are appended together in one row, preventing partially paired output.

This module has no truth argument and never imports or discovers a truth file.
Scoring is intentionally a separate post-inference operation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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


ORDERS = ("original", "hashed")
SCHEMA = "silver-match-v3-paired-gemma4-lora-truth-blind-inference-v1"
EXPECTED_MODEL_CONTENT_SHA256 = (
    "f06399f0164b3feeb55e2de43831e699d1443481afb6d6a1b0164053d86d13ae"
)
EXPECTED_MODEL_FILE_COUNT = 12


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _identity_files(root: Path, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for name in names:
        path = root / name
        if path.is_file():
            values[name] = _artifact(path)
    if not values:
        raise FileNotFoundError(f"no identity files found under {root}")
    return values


def _adapter_inventory(adapter: Path) -> dict[str, dict[str, Any]]:
    required = ("adapter_config.json", "adapter_model.safetensors")
    if any(not (adapter / name).is_file() for name in required):
        raise FileNotFoundError("adapter lacks config or safetensors weights")
    return {
        str(path.relative_to(adapter)): _artifact(path)
        for path in sorted(adapter.rglob("*"))
        if path.is_file()
    }


def validate_model_inventory(path: Path, model: Path) -> dict[str, Any]:
    inventory = json.loads(path.read_text(encoding="utf-8"))
    if (
        inventory.get("status") != "FROZEN_CONTENT_HASH_INVENTORY"
        or Path(str(inventory.get("root") or "")).resolve() != model.resolve()
        or int(inventory.get("file_count", -1)) != EXPECTED_MODEL_FILE_COUNT
        or inventory.get("content_inventory_sha256")
        != EXPECTED_MODEL_CONTENT_SHA256
    ):
        raise ValueError("Gemma-4 base-model content inventory mismatch")
    return inventory


def _index_output(path: Path, order: str) -> set[str]:
    if not path.exists():
        return set()
    rows = list(read_jsonl(path))
    uids = [str(row.get("norm_uid") or "") for row in rows]
    if "" in uids or len(uids) != len(set(uids)):
        raise ValueError(f"partial {order} output has missing/duplicate UIDs")
    if any(
        row.get("schema_version") != SCHEMA or row.get("order_mode") != order
        for row in rows
    ):
        raise ValueError(f"partial {order} output violates paired schema")
    return set(uids)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def infer_manifest_task(
    candidate_rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> str:
    """Return the one exact manifest task represented by candidate rows."""

    candidate_tasks = {str(row.get("task") or "") for row in candidate_rows}
    candidate_corpora = {str(row.get("corpus") or "") for row in candidate_rows}
    corpora = manifest.get("corpora") or {}
    routed_tasks = {
        str(corpora.get(corpus, {}).get("task") or "")
        for corpus in candidate_corpora
    }
    if (
        not candidate_rows
        or len(candidate_tasks) != 1
        or "" in candidate_tasks
        or candidate_tasks != routed_tasks
        or "" in candidate_corpora
    ):
        raise ValueError("candidate input does not route to exactly one manifest task")
    return next(iter(candidate_tasks))


def _freeze_contract(args: argparse.Namespace, prompt_paths: list[Path]) -> dict[str, Any]:
    model = Path(args.model).resolve()
    model_inventory = Path(args.model_inventory).resolve()
    adapter = Path(args.adapter).resolve()
    validate_model_inventory(model_inventory, model)
    adapter_config = json.loads(
        (adapter / "adapter_config.json").read_text(encoding="utf-8")
    )
    rank = int(adapter_config.get("r", -1))
    if rank < 1 or rank > args.max_lora_rank:
        raise ValueError(
            f"adapter rank {rank} exceeds --max-lora-rank {args.max_lora_rank}"
        )
    return {
        "schema_version": "silver-match-v3-paired-gemma4-lora-inference-freeze-v1",
        "status": "FROZEN_BEFORE_PAIRED_MODEL_INFERENCE",
        "task": args.task,
        "backend": "direct_batch_vllm_not_openai_server",
        "truth_firewall": {
            "truth_read": False,
            "truth_path_argument_exists": False,
            "resolved_or_unresolved_label_artifacts_read": False,
            "scoring_in_separate_process_after_predictions": True,
        },
        "inputs": {
            "manifest": _artifact(Path(args.manifest)),
            "candidates": _artifact(Path(args.candidates)),
            "prompt_components": [_artifact(path) for path in prompt_paths],
            "runner_script": _artifact(Path(__file__)),
            "model_inventory": _artifact(model_inventory),
            "model_identity": {
                "path": str(model),
                "files": _identity_files(
                    model,
                    (
                        "config.json",
                        "generation_config.json",
                        "model.safetensors.index.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                    ),
                ),
            },
            "adapter_identity": {
                "path": str(adapter),
                "files": _adapter_inventory(adapter),
            },
        },
        "paired_contract": {
            "systems": ["base", "lora"],
            "orders": list(ORDERS),
            "same_model_instance": True,
            "same_candidate_set_and_rendered_prompt_within_each_pair": True,
            "same_sampling_and_retry_policy": True,
            "adapter_name": args.adapter_name,
            "adapter_id": args.adapter_id,
            "no_hyperparameter_or_seed_search": True,
        },
        "rendering": {
            "max_candidates": args.max_candidates,
            "context_chars": args.context_chars,
            "description_chars": args.description_chars,
            "example_chars": args.example_chars,
            "max_examples": args.max_examples,
        },
        "generation": {
            "temperature": 0.0,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "dtype": "bfloat16",
        },
        "runtime": {
            "batch_size": args.batch_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_lora_rank": args.max_lora_rank,
            "python": str(Path(sys.executable).resolve()),
        },
    }


def _prediction_payload(
    parsed: dict[str, Any] | None,
    error: str | None,
    raw: str,
    *,
    keep_raw: bool,
) -> dict[str, Any]:
    if parsed is None:
        parsed = {
            "decision": "INVALID_OUTPUT",
            "metric_id": None,
            "confidence": "low",
            "reason": error,
        }
    return {
        "decision": parsed["decision"],
        "metric_id": parsed["metric_id"],
        "confidence": parsed["confidence"],
        "reason": parsed["reason"],
        "parse_error": error if parsed["decision"] == "INVALID_OUTPUT" else None,
        "raw_response": (
            raw if keep_raw or parsed["decision"] == "INVALID_OUTPUT" else None
        ),
    }


def _infer_representatives(
    llm: Any,
    conversations: list[list[dict[str, str]]],
    candidate_id_sets: list[set[str]],
    sampling: Any,
    *,
    lora_request: Any | None,
) -> tuple[list[tuple[dict[str, Any] | None, str | None, str]], int]:
    kwargs: dict[str, Any] = {"use_tqdm": False}
    if lora_request is not None:
        kwargs["lora_request"] = lora_request
    outputs = llm.chat(conversations, sampling, **kwargs)
    values: list[tuple[dict[str, Any] | None, str | None, str]] = []
    retry_indices: list[int] = []
    for index, (output, candidate_ids) in enumerate(zip(outputs, candidate_id_sets)):
        raw = output.outputs[0].text if output.outputs else ""
        parsed, error = parse_response(raw, candidate_ids)
        values.append((parsed, error, raw))
        if parsed is None:
            retry_indices.append(index)
    if retry_indices:
        retries = []
        for index in retry_indices:
            retries.append(
                [
                    *conversations[index],
                    {"role": "assistant", "content": values[index][2]},
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
        retry_kwargs: dict[str, Any] = {"use_tqdm": False}
        if lora_request is not None:
            retry_kwargs["lora_request"] = lora_request
        retry_outputs = llm.chat(retries, sampling, **retry_kwargs)
        for index, output in zip(retry_indices, retry_outputs):
            raw = output.outputs[0].text if output.outputs else ""
            parsed, error = parse_response(raw, candidate_id_sets[index])
            values[index] = (parsed, error, raw)
    return values, len(retry_indices)


def paired_rows_for_batch(
    batch: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str]],
    base_values: Sequence[tuple[dict[str, Any] | None, str | None, str]],
    lora_values: Sequence[tuple[dict[str, Any] | None, str | None, str]],
    representative_for: Sequence[int],
    representatives: Sequence[int],
    *,
    order: str,
    prompt_hash: str,
    freeze_hash: str,
    model: str,
    adapter: str,
    adapter_name: str,
    keep_raw: bool,
) -> list[dict[str, Any]]:
    """Materialize paired rows; split out for CPU-only contract testing."""

    position = {representative: index for index, representative in enumerate(representatives)}
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(batch):
        candidate_row, norm, candidates, rendered_prompt = item
        rep_position = position[representative_for[index]]
        base = _prediction_payload(*base_values[rep_position], keep_raw=keep_raw)
        lora = _prediction_payload(*lora_values[rep_position], keep_raw=keep_raw)
        item_hash = prompt_sha256(rendered_prompt)
        rows.append(
            {
                "schema_version": SCHEMA,
                "norm_uid": norm["norm_uid"],
                "corpus": norm["corpus"],
                "task": norm["task"],
                "row": norm["row"],
                "order_mode": order,
                "candidate_ids": [str(value["metric_id"]) for value in candidates],
                "candidate_bank_source_sha256": candidate_row[
                    "bank_source_sha256"
                ],
                "prompt_sha256": prompt_hash,
                "base_item_prompt_sha256": item_hash,
                "lora_item_prompt_sha256": item_hash,
                "inference_freeze_sha256": freeze_hash,
                "model": model,
                "adapter": adapter,
                "adapter_name": adapter_name,
                "base": base,
                "lora": lora,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    candidates_path = Path(args.candidates).resolve()
    prompt_paths = [
        Path(args.prompt).resolve(),
        *[Path(value).resolve() for value in args.prompt_addon],
    ]
    model_path = Path(args.model).resolve()
    adapter_path = Path(args.adapter).resolve()
    output_root = Path(args.output_root).resolve()
    outputs = {
        order: output_root / f"paired.{order}.jsonl" for order in ORDERS
    }
    freeze_path = output_root / "truth_blind_inference.freeze.json"
    meta_path = output_root / "paired_inference.meta.json"
    if meta_path.exists():
        raise FileExistsError(f"paired inference is already sealed: {meta_path}")
    if not args.resume and any(path.exists() for path in (*outputs.values(), freeze_path)):
        raise FileExistsError("refusing to overwrite paired inference artifacts; pass --resume")
    output_root.mkdir(parents=True, exist_ok=True)

    candidate_rows = list(read_jsonl(candidates_path))
    candidate_uids = [str(row.get("norm_uid") or "") for row in candidate_rows]
    if (
        not candidate_rows
        or "" in candidate_uids
        or len(candidate_uids) != len(set(candidate_uids))
        or any(len(row.get("candidates") or []) < args.max_candidates for row in candidate_rows)
    ):
        raise ValueError("candidate input lacks unique complete slates")
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    args.task = infer_manifest_task(candidate_rows, manifest_payload)
    contract = _freeze_contract(args, prompt_paths)
    if freeze_path.exists():
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        comparable = dict(frozen)
        comparable.pop("frozen_at", None)
        if comparable != contract:
            raise ValueError("resume arguments drift from the pre-inference freeze")
    else:
        _write_new_json(freeze_path, {**contract, "frozen_at": _utc_now()})
    freeze_hash = sha256_file(freeze_path)

    done_by_order = {order: _index_output(outputs[order], order) for order in ORDERS}
    if any(not done <= set(candidate_uids) for done in done_by_order.values()):
        raise ValueError("partial output contains a UID absent from candidates")

    system_prompt = "\n\n".join(
        path.read_text(encoding="utf-8").rstrip() for path in prompt_paths
    ) + "\n"
    system_prompt_hash = prompt_sha256(system_prompt)
    corpora, _ = scan_candidate_input(
        candidates_path, done=set(), shard_id=0, num_shards=1
    )
    manifest, norms_by_corpus, banks = load_inputs(manifest_path, corpora)

    pending = sum(len(candidate_rows) - len(done_by_order[order]) for order in ORDERS)
    started = time.time()
    counts = {
        order: {
            "resumed_count": len(done_by_order[order]),
            "new_count": 0,
            "base_invalid_count": 0,
            "lora_invalid_count": 0,
            "base_retry_count": 0,
            "lora_retry_count": 0,
        }
        for order in ORDERS
    }

    llm = sampling = lora_request = None
    if pending:
        os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        llm = LLM(
            model=str(model_path),
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            enable_lora=True,
            max_loras=1,
            max_lora_rank=args.max_lora_rank,
        )
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        lora_request = LoRARequest(
            args.adapter_name, args.adapter_id, str(adapter_path)
        )

    for order in ORDERS:
        work = iter_work_items(
            candidates_path,
            done=done_by_order[order],
            shard_id=0,
            num_shards=1,
            max_candidates=args.max_candidates,
            order_mode=order,
            system_prompt=system_prompt,
            norms_by_corpus=norms_by_corpus,
            banks=banks,
            context_chars=args.context_chars,
            description_chars=args.description_chars,
            example_chars=args.example_chars,
            max_examples=args.max_examples,
        )
        for batch in batched_work(work, args.batch_size):
            representatives, representative_for, _ = prompt_equivalence_groups(batch)
            inference_batch = [batch[index] for index in representatives]
            conversations = [
                [{"role": "user", "content": item[3]}] for item in inference_batch
            ]
            candidate_sets = [
                {str(value["metric_id"]) for value in item[2]}
                for item in inference_batch
            ]
            base_values, base_retries = _infer_representatives(
                llm,
                conversations,
                candidate_sets,
                sampling,
                lora_request=None,
            )
            lora_values, lora_retries = _infer_representatives(
                llm,
                conversations,
                candidate_sets,
                sampling,
                lora_request=lora_request,
            )
            rows = paired_rows_for_batch(
                batch,
                base_values,
                lora_values,
                representative_for,
                representatives,
                order=order,
                prompt_hash=system_prompt_hash,
                freeze_hash=freeze_hash,
                model=str(model_path),
                adapter=str(adapter_path),
                adapter_name=args.adapter_name,
                keep_raw=args.keep_raw,
            )
            append_rows(outputs[order], rows)
            counts[order]["new_count"] += len(rows)
            counts[order]["base_retry_count"] += base_retries
            counts[order]["lora_retry_count"] += lora_retries
            counts[order]["base_invalid_count"] += sum(
                row["base"]["decision"] == "INVALID_OUTPUT" for row in rows
            )
            counts[order]["lora_invalid_count"] += sum(
                row["lora"]["decision"] == "INVALID_OUTPUT" for row in rows
            )
            print(
                f"order={order} paired={counts[order]['new_count']}/"
                f"{len(candidate_rows) - counts[order]['resumed_count']} "
                f"elapsed={time.time() - started:.0f}s",
                flush=True,
            )

    for order, path in outputs.items():
        completed = _index_output(path, order)
        if completed != set(candidate_uids):
            raise ValueError(f"paired {order} output did not reach exact coverage")
    meta = {
        "schema_version": "silver-match-v3-paired-gemma4-lora-inference-meta-v1",
        "status": "COMPLETE_TRUTH_BLIND_PAIRED_INFERENCE",
        "completed_at": _utc_now(),
        "task": args.task,
        "truth_read": False,
        "backend": "direct_batch_vllm_not_openai_server",
        "same_loaded_base_model_instance_for_both_arms": True,
        "inference_freeze": _artifact(freeze_path),
        "outputs": {
            order: {**_artifact(path), "count": len(candidate_rows)}
            for order, path in outputs.items()
        },
        "counts": counts,
        "prompt_sha256": system_prompt_hash,
        "model": str(model_path),
        "adapter": str(adapter_path),
        "elapsed_seconds": time.time() - started,
    }
    _write_new_json(meta_path, meta)
    return {**meta, "meta_sha256": sha256_file(meta_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-addon", action="append", default=[])
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-inventory", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--adapter-name", default="humor_typed_v1")
    parser.add_argument("--adapter-id", type=int, default=1)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--context-chars", type=int, default=1400)
    parser.add_argument("--description-chars", type=int, default=520)
    parser.add_argument("--example-chars", type=int, default=180)
    parser.add_argument("--max-examples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(
        args.max_candidates,
        args.context_chars,
        args.description_chars,
        args.example_chars,
        args.batch_size,
        args.max_model_len,
        args.max_tokens,
        args.max_lora_rank,
        args.adapter_id,
    ) < 1:
        parser.error("positive integer arguments must be positive")
    if args.max_examples < 0:
        parser.error("--max-examples must be nonnegative")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1)")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
