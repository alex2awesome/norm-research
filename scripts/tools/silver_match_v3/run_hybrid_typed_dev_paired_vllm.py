#!/usr/bin/env python3
"""Prepare and run one paired-order typed-LoRA batch-vLLM dev inference.

This is deliberately dev-only.  ``prepare`` projects the frozen compact dev
rows into prompt-only original and deterministic hashed-order records.  ``infer``
loads one adapter once and decodes both orders in the same direct batch-vLLM
process.  Test/blind/heldout rows are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import read_jsonl, sha256_file
from .run_nemotron_ce import verify_base_manifest
from .run_paired_gemma_lora_batch import _infer_representatives, _prediction_payload


CARD_LINE = re.compile(r"^\[([^\]]+)\] .+$")
PREPARE_SCHEMA = "silver-match-v3-humor-typed-dev-paired-order-prepare-v1"
PREDICTION_SCHEMA = "silver-match-v3-humor-typed-dev-paired-order-prediction-v1"


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def _reorder_compact_prompt(content: str, uid: str, expected_ids: Sequence[str]) -> tuple[str, list[str]]:
    marker = "CANDIDATE METRIC CARDS (no examples):\n"
    suffix = "\n\nReturn the JSON decision now."
    if marker not in content or suffix not in content:
        raise ValueError(f"compact card markers absent: {uid}")
    prefix, remainder = content.split(marker, 1)
    cards, tail = remainder.split(suffix, 1)
    if tail:
        raise ValueError(f"unexpected content after compact prompt suffix: {uid}")
    lines = cards.splitlines()
    parsed: dict[str, str] = {}
    for line in lines:
        match = CARD_LINE.match(line)
        if not match or match.group(1) in parsed:
            raise ValueError(f"invalid/duplicate compact card: {uid}/{line[:80]}")
        parsed[match.group(1)] = line
    expected = [str(value) for value in expected_ids]
    if set(parsed) != set(expected) or len(parsed) != len(expected):
        raise ValueError(f"compact cards differ from candidate IDs: {uid}")
    ordered = sorted(
        expected,
        key=lambda metric_id: hashlib.sha256(f"{uid}\0{metric_id}".encode()).hexdigest(),
    )
    if len(ordered) > 1 and ordered == expected:
        ordered = ordered[1:] + ordered[:1]
    return marker.join((prefix, "\n".join(parsed[value] for value in ordered))) + suffix, ordered


def prepare(args: argparse.Namespace) -> None:
    source = Path(args.dev_dataset).resolve()
    if sha256_file(source) != args.dev_sha256:
        raise ValueError("compact dev SHA mismatch")
    root = Path(args.output_root).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=False)
    prompt_path = root / "paired_order.prompts.jsonl"
    rows = []
    uids: set[str] = set()
    groups: set[str] = set()
    for line_number, row in enumerate(read_jsonl(source), 1):
        uid = str(row.get("norm_uid") or "")
        messages = row.get("messages") or []
        candidates = [str(value) for value in row.get("candidate_metric_ids") or []]
        if (
            not uid or uid in uids or row.get("split") != "dev"
            or row.get("gradient_eligible") is not False
            or not row.get("source_group")
            or len(messages) != 2 or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"
            or not candidates or len(candidates) != len(set(candidates))
        ):
            raise ValueError(f"invalid compact dev row: {line_number}")
        original = str(messages[0].get("content") or "")
        reordered, reordered_ids = _reorder_compact_prompt(original, uid, candidates)
        if sorted(candidates) != sorted(reordered_ids):
            raise AssertionError("candidate order changed candidate membership")
        for order, content, ids in (
            ("original", original, candidates),
            ("reordered", reordered, reordered_ids),
        ):
            rows.append(
                {
                    "schema_version": "silver-match-v3-humor-typed-dev-prompt-v1",
                    "task": row.get("task"), "corpus": row.get("corpus"),
                    "norm_uid": uid, "source_group": row["source_group"],
                    "split": "dev", "order_mode": order,
                    "candidate_metric_ids": ids,
                    "conversation": [{"role": "user", "content": content}],
                }
            )
        uids.add(uid)
        groups.add(str(row["source_group"]))
    if not uids:
        raise ValueError("empty compact dev")
    write_jsonl_new(prompt_path, rows)
    manifest = {
        "schema_version": PREPARE_SCHEMA,
        "status": "COMPLETE_DEV_ONLY_PAIRED_ORDER_PROMPT_PROJECTION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_or_blind_rows_read": 0,
        "source": {**artifact(source), "expected_sha256": args.dev_sha256},
        "prompts": {**artifact(prompt_path), "rows": len(rows), "unique_norm_uids": len(uids)},
        "source_groups": len(groups),
        "orders": ["original", "reordered"],
        "reordering": "sha256(norm_uid + NUL + metric_id), rotate one if identity order",
        "assistant_targets_projected": False,
    }
    manifest_path = root / "PREPARE.json"
    write_json_new(manifest_path, manifest)
    print(json.dumps({**manifest, "manifest": artifact(manifest_path)}, sort_keys=True))


def infer(args: argparse.Namespace) -> None:
    manifest_path = Path(args.prepare_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PREPARE_SCHEMA or manifest.get("test_or_blind_rows_read") != 0:
        raise ValueError("invalid dev prompt preparation manifest")
    prompt_path = Path(manifest["prompts"]["path"])
    if sha256_file(prompt_path) != manifest["prompts"]["sha256"]:
        raise ValueError("dev prompt projection drift")
    model = Path(args.model).resolve()
    inventory = Path(args.model_inventory).resolve()
    verify_base_manifest(model, inventory, args.model_inventory_sha256)
    adapter = Path(args.adapter).resolve()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / name).is_file():
            raise FileNotFoundError(adapter / name)
    root = Path(args.output_root).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=False)
    contract_path = root / "INFERENCE_CONTRACT.json"
    outputs = {order: root / f"typed.{order}.jsonl" for order in ("original", "reordered")}
    contract = {
        "schema_version": "silver-match-v3-humor-typed-dev-paired-vllm-contract-v1",
        "status": "FROZEN_BEFORE_DEV_ONLY_DIRECT_BATCH_VLLM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_or_blind_rows_read": 0,
        "prepare_manifest": artifact(manifest_path), "prompts": artifact(prompt_path),
        "model": str(model), "model_inventory": artifact(inventory),
        "adapter": str(adapter),
        "adapter_files": {name: artifact(adapter / name) for name in ("adapter_config.json", "adapter_model.safetensors")},
        "decoding": {"temperature": 0.0, "seed": args.seed, "max_tokens": args.max_tokens},
        "backend": "direct_batch_vllm_not_openai_server",
    }
    write_json_new(contract_path, contract)

    os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = list(read_jsonl(prompt_path))
    llm = LLM(
        model=str(model), dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, trust_remote_code=True, enable_lora=True,
        max_loras=1, max_lora_rank=args.max_lora_rank,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)
    request = LoRARequest(args.adapter_name, 1, str(adapter))
    counts = Counter()
    started = time.time()
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
                    "schema_version": PREDICTION_SCHEMA,
                    "task": row.get("task"), "corpus": row.get("corpus"),
                    "norm_uid": row["norm_uid"], "source_group": row["source_group"],
                    "split": "dev", "order_mode": row["order_mode"],
                    "candidate_metric_ids": row["candidate_metric_ids"],
                    **prediction,
                }
                handles[row["order_mode"]].write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
            for handle in handles.values():
                handle.flush(); os.fsync(handle.fileno())
            print(json.dumps({"completed": start + len(batch), "total": len(rows)}), flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    expected = int(manifest["prompts"]["unique_norm_uids"])
    for order, path in outputs.items():
        values = list(read_jsonl(path))
        if len(values) != expected or len({row["norm_uid"] for row in values}) != expected:
            raise ValueError(f"incomplete {order} dev inference")
    meta = {
        "schema_version": "silver-match-v3-humor-typed-dev-paired-vllm-meta-v1",
        "status": "COMPLETE_DEV_ONLY_PAIRED_ORDER_INFERENCE",
        "test_or_blind_rows_read": 0, "contract": artifact(contract_path),
        "outputs": {order: {**artifact(path), "rows": expected} for order, path in outputs.items()},
        "counts": dict(sorted(counts.items())), "elapsed_seconds": time.time() - started,
    }
    meta_path = root / "INFERENCE_META.json"
    write_json_new(meta_path, meta)
    print(json.dumps({**meta, "meta": artifact(meta_path)}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--dev-dataset", required=True)
    prepare_parser.add_argument("--dev-sha256", required=True)
    prepare_parser.add_argument("--output-root", required=True)
    infer_parser = sub.add_parser("infer")
    infer_parser.add_argument("--prepare-manifest", required=True)
    infer_parser.add_argument("--model", required=True)
    infer_parser.add_argument("--model-inventory", required=True)
    infer_parser.add_argument("--model-inventory-sha256", required=True)
    infer_parser.add_argument("--adapter", required=True)
    infer_parser.add_argument("--output-root", required=True)
    infer_parser.add_argument("--adapter-name", default="humor_typed_hybrid_dev")
    infer_parser.add_argument("--batch-size", type=int, default=128)
    infer_parser.add_argument("--max-model-len", type=int, default=2048)
    infer_parser.add_argument("--max-tokens", type=int, default=192)
    infer_parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    infer_parser.add_argument("--max-lora-rank", type=int, default=16)
    infer_parser.add_argument("--seed", type=int, default=94137)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare": prepare(args)
    else: infer(args)


if __name__ == "__main__":
    main()
