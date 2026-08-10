#!/usr/bin/env python3
"""Run one deterministic shard of paired-order Humor c2 batch-vLLM deployment."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .common import read_jsonl, sha256_file
from .run_nemotron_ce import pair_shard, verify_base_manifest
from .run_paired_gemma_lora_batch import _infer_representatives, _prediction_payload


PROMPT_SCHEMA = "silver-match-v3-humor-c2-production-paired-prompt-v1"
PREDICTION_SCHEMA = "silver-match-v3-humor-c2-production-paired-prediction-v1"
META_SCHEMA = "silver-match-v3-humor-c2-production-paired-inference-meta-v1"
EXPECTED_UIDS = 55_288
EXPECTED_PROMPTS = 2 * EXPECTED_UIDS


def artifact(path: Path, **extra: Any) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, **extra}


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def infer(args: argparse.Namespace) -> dict[str, Any]:
    prompt_path = Path(args.prompts).resolve()
    if sha256_file(prompt_path) != args.prompts_sha256:
        raise ValueError("paired production prompt SHA differs")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard coordinates")
    model, inventory, adapter = map(Path, (args.model, args.model_inventory, args.adapter))
    model, inventory, adapter = model.resolve(), inventory.resolve(), adapter.resolve()
    verify_base_manifest(model, inventory, args.model_inventory_sha256)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / name).is_file():
            raise FileNotFoundError(adapter / name)
    root = Path(args.output_root).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=False)
    outputs = {order: root / f"typed.{order}.jsonl" for order in ("original", "reordered")}

    rows = []
    all_rows = all_uids = 0
    all_seen: set[str] = set()
    shard_orders: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(prompt_path):
        all_rows += 1
        uid = str(row.get("norm_uid") or "")
        order = str(row.get("order_mode") or "")
        ids = [str(value) for value in row.get("candidate_metric_ids") or []]
        if (
            row.get("schema_version") != PROMPT_SCHEMA or row.get("split") != "production"
            or not uid or order not in {"original", "reordered"} or not ids
            or len(ids) != len(set(ids)) or "messages" in row
            or not isinstance(row.get("conversation"), list)
        ):
            raise ValueError(f"invalid production prompt row: {uid}/{order}")
        if uid not in all_seen:
            all_seen.add(uid); all_uids += 1
        if pair_shard(uid, args.num_shards) == args.shard_id:
            if order in shard_orders[uid]:
                raise ValueError(f"duplicate shard order: {uid}/{order}")
            shard_orders[uid].add(order); rows.append(row)
    if all_rows != EXPECTED_PROMPTS or all_uids != EXPECTED_UIDS:
        raise ValueError("global production prompt coverage differs")
    if not rows or any(orders != {"original", "reordered"} for orders in shard_orders.values()):
        raise ValueError("paired shard coverage differs")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    maximum_tokens = 0
    maximum_uid = None
    for row in rows:
        encoded = tokenizer.apply_chat_template(
            row["conversation"], tokenize=True, add_generation_prompt=True
        )
        length = len(encoded)
        if length > maximum_tokens:
            maximum_tokens, maximum_uid = length, row["norm_uid"]
    if maximum_tokens > args.max_model_len:
        raise ValueError(f"production prompt exceeds frozen context: {maximum_uid}/{maximum_tokens}")
    del tokenizer

    contract = {
        "schema_version": "silver-match-v3-humor-c2-production-paired-contract-v1",
        "status": "FROZEN_PRODUCTION_DEPLOYMENT_BEFORE_INFERENCE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "deployment_claim": "DEV_FROZEN_DEPLOYMENT_BLIND_P855",
        "blind_gate": {"precision": 0.855, "wilson_lower": 0.753, "promotion_passed": False},
        "test_or_blind_rows_read": 0, "prompts": artifact(prompt_path, rows=all_rows, norm_uids=all_uids),
        "model": str(model), "model_inventory": artifact(inventory),
        "adapter": str(adapter), "adapter_files": {name: artifact(adapter / name) for name in ("adapter_config.json", "adapter_model.safetensors")},
        "shard": {"shard_id": args.shard_id, "num_shards": args.num_shards,
                  "norm_uids": len(shard_orders), "prompt_rows": len(rows)},
        "token_audit": {"maximum": maximum_tokens, "maximum_uid": maximum_uid,
                        "max_allowed": args.max_model_len, "all_within_limit": True},
        "decoding": {"temperature": 0.0, "seed": args.seed, "max_tokens": args.max_tokens},
        "backend": "direct_batch_vllm_not_openai_server",
    }
    contract_path = root / "INFERENCE_CONTRACT.json"; write_json_new(contract_path, contract)

    os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(
        model=str(model), dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, trust_remote_code=True, enable_lora=True,
        max_loras=1, max_lora_rank=args.max_lora_rank,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)
    request = LoRARequest(args.adapter_name, 1, str(adapter))
    counts = Counter(); started = time.time()
    handles = {order: path.open("x", encoding="utf-8") for order, path in outputs.items()}
    try:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            values, retries = _infer_representatives(
                llm, [row["conversation"] for row in batch],
                [set(row["candidate_metric_ids"]) for row in batch], sampling,
                lora_request=request,
            )
            counts["retries"] += retries
            for row, value in zip(batch, values):
                prediction = _prediction_payload(*value, keep_raw=False)
                counts[f"order:{row['order_mode']}"] += 1
                counts["invalid"] += prediction["decision"] == "INVALID_OUTPUT"
                output = {
                    "schema_version": PREDICTION_SCHEMA, "task": "humor", "corpus": "humor_multi",
                    "norm_uid": row["norm_uid"], "source_group": row["source_group"],
                    "split": "production", "order_mode": row["order_mode"],
                    "candidate_metric_ids": row["candidate_metric_ids"], **prediction,
                }
                handles[row["order_mode"]].write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
            for handle in handles.values():
                handle.flush(); os.fsync(handle.fileno())
            print(json.dumps({"completed": start + len(batch), "total": len(rows),
                              "shard_id": args.shard_id}), flush=True)
    finally:
        for handle in handles.values(): handle.close()
    expected = len(shard_orders)
    for order, path in outputs.items():
        values = list(read_jsonl(path))
        if len(values) != expected or len({row["norm_uid"] for row in values}) != expected:
            raise ValueError(f"incomplete production shard/order: {order}")
    meta = {
        "schema_version": META_SCHEMA, "status": "COMPLETE_C2_PRODUCTION_PAIRED_INFERENCE",
        "deployment_claim": "DEV_FROZEN_DEPLOYMENT_BLIND_P855", "test_or_blind_rows_read": 0,
        "contract": artifact(contract_path), "shard_id": args.shard_id, "num_shards": args.num_shards,
        "outputs": {order: artifact(path, rows=expected) for order, path in outputs.items()},
        "counts": dict(sorted(counts.items())), "elapsed_seconds": time.time() - started,
    }
    meta_path = root / "INFERENCE_META.json"; write_json_new(meta_path, meta)
    print(json.dumps(meta, sort_keys=True)); return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True); parser.add_argument("--prompts-sha256", required=True)
    parser.add_argument("--model", required=True); parser.add_argument("--model-inventory", required=True)
    parser.add_argument("--model-inventory-sha256", required=True); parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-root", required=True); parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True); parser.add_argument("--adapter-name", default="humor_c2_production")
    parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=192); parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-lora-rank", type=int, default=16); parser.add_argument("--seed", type=int, default=94137)
    infer(parser.parse_args())


if __name__ == "__main__":
    main()
