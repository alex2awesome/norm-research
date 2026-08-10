#!/usr/bin/env python3
"""Regenerate one frozen LoRA checkpoint on dev and emit a manual error packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .train_gemma4_typed_lora import (
    _left_padded_prompts,
    _parse_typed_response,
    directory_ref,
    file_ref,
    read_examples,
    tokenize_dataset,
)


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stable_key(row: dict[str, Any], category: str) -> str:
    value = f"humor-c2-dev-audit-v1\0{category}\0{row['norm_uid']}"
    return hashlib.sha256(value.encode()).hexdigest()


def _prompt_fields(prompt: str) -> dict[str, Any]:
    statement = re.search(
        r'HUMAN STATEMENT \(verbatim\):\n(?P<x>.*?)\nCONTEXT \(capped at 600 characters\):',
        prompt,
        re.S,
    )
    context = re.search(
        r'CONTEXT \(capped at 600 characters\):\n(?P<x>.*?)\n\nCANDIDATE METRIC CARDS',
        prompt,
        re.S,
    )
    cards: dict[str, str] = {}
    card_block = prompt.split("CANDIDATE METRIC CARDS (no examples):\n", 1)[-1]
    card_block = card_block.split("\n\nReturn the JSON decision now.", 1)[0]
    for line in card_block.splitlines():
        match = re.match(r"\[([^]]+)\]\s+(.*)", line.strip())
        if match:
            cards[match.group(1)] = match.group(2)
    return {
        "norm_statement": statement.group("x").strip() if statement else None,
        "context": context.group("x").strip() if context else None,
        "candidate_cards": cards,
    }


def _gold(row: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(row["messages"][-1]["content"])
    if value.get("decision") != row.get("decision") or value.get("metric_id") != row.get("metric_id"):
        raise ValueError(f"gold target/row mismatch: {row.get('norm_uid')}")
    return value


def _category(gold: dict[str, Any], prediction: dict[str, Any] | None) -> str | None:
    if (
        gold["decision"] == "MATCH"
        and prediction
        and prediction["decision"] == "MATCH"
        and prediction["metric_id"] == gold["metric_id"]
    ):
        return "correct_exact_match"
    if prediction and prediction["decision"] == "MATCH" and not (
        gold["decision"] == "MATCH" and prediction["metric_id"] == gold["metric_id"]
    ):
        return "false_match"
    if gold["decision"] == "MATCH" and (not prediction or prediction["decision"] != "MATCH"):
        return "missed_gold_match"
    if gold["decision"] != "MATCH" and (
        prediction is None
        or (prediction["decision"] != "MATCH" and prediction["decision"] != gold["decision"])
    ):
        return "abstention_type_error"
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("cumulative_exposure") != args.expected_exposure:
        raise ValueError("checkpoint exposure differs")
    if checkpoint.get("test_or_blind_data_read") is not False:
        raise ValueError("checkpoint is not dev-only")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(args.model).resolve()
    adapter_path = Path(args.adapter).resolve()
    dev_path = Path(args.dev_dataset).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = read_examples(dev_path)
    encoded, _ = tokenize_dataset(tokenizer, examples, args.max_length)
    base = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    device = torch.device("cuda", 0)
    model.to(device).eval()

    records: list[dict[str, Any]] = []
    parse_errors: Counter[str] = Counter()
    with torch.inference_mode():
        for start in range(0, len(encoded), args.batch_size):
            batch = encoded[start : start + args.batch_size]
            input_ids, attention_mask = _left_padded_prompts(batch, int(tokenizer.pad_token_id))
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=int(tokenizer.pad_token_id),
                eos_token_id=tokenizer.eos_token_id,
            )
            suffixes = generated[:, input_ids.shape[1] :]
            for row, token_row, source in zip(batch, suffixes, examples[start : start + len(batch)]):
                raw = tokenizer.decode(token_row, skip_special_tokens=True)
                prediction, error = _parse_typed_response(raw, set(row["candidate_metric_ids"]))
                if error:
                    parse_errors[error] += 1
                gold = _gold(source)
                prompt = source["messages"][0]["content"]
                fields = _prompt_fields(prompt)
                category = _category(gold, prediction)
                records.append(
                    {
                        "norm_uid": source["norm_uid"],
                        "source_group": source.get("source_group"),
                        "view": source.get("view"),
                        **fields,
                        "candidate_metric_ids": source["candidate_metric_ids"],
                        "gold": gold,
                        "gold_metric_card": fields["candidate_cards"].get(gold.get("metric_id")),
                        "prediction": prediction,
                        "predicted_metric_card": fields["candidate_cards"].get(
                            prediction.get("metric_id") if prediction else None
                        ),
                        "raw_prediction": raw,
                        "parse_error": error,
                        "audit_category": category,
                    }
                )

    valid = sum(row["prediction"] is not None for row in records)
    accepted = [row for row in records if row["prediction"] and row["prediction"]["decision"] == "MATCH"]
    correct = [
        row for row in accepted
        if row["gold"]["decision"] == "MATCH"
        and row["prediction"]["metric_id"] == row["gold"]["metric_id"]
    ]
    gate = checkpoint["confidence_gate"]
    if (
        valid != checkpoint["generation"]["valid_predictions"]
        or len(accepted) != gate["predicted_exact_count"]
        or len(correct) != gate["correct_exact_count"]
    ):
        raise ValueError(
            f"regeneration differs from checkpoint: valid={valid}, accepted={len(accepted)}, "
            f"correct={len(correct)}"
        )

    predictions_path = output_root / "dev_predictions.c2.jsonl"
    with predictions_path.open("x", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    requested = {
        "correct_exact_match": args.correct,
        "false_match": args.false_match,
        "missed_gold_match": args.missed,
        "abstention_type_error": args.abstention_error,
    }
    selected: list[dict[str, Any]] = []
    available: dict[str, int] = {}
    for category, count in requested.items():
        rows = [row for row in records if row["audit_category"] == category]
        rows.sort(key=lambda row: _stable_key(row, category))
        available[category] = len(rows)
        for rank, row in enumerate(rows[:count], 1):
            selected.append({"sample_category": category, "sample_rank": rank, **row})
    packet_path = output_root / "manual_audit_packet.c2.jsonl"
    with packet_path.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    report = {
        "schema_version": "silver-match-v3-humor-typed-lora-c2-dev-error-packet-v1",
        "status": "COMPLETE_DEV_ONLY_DETERMINISTIC_MANUAL_AUDIT_PACKET",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "method": "ascending_sha256(humor-c2-dev-audit-v1\\0category\\0norm_uid)",
            "requested": requested,
            "available": available,
            "selected": dict(Counter(row["sample_category"] for row in selected)),
        },
        "checkpoint": file_ref(checkpoint_path),
        "adapter": directory_ref(adapter_path),
        "dev_dataset": file_ref(dev_path),
        "model": str(model_path),
        "regeneration": {
            "rows": len(records),
            "valid_predictions": valid,
            "parse_errors": dict(sorted(parse_errors.items())),
            "predicted_match": len(accepted),
            "correct_exact_match": len(correct),
            "matches_checkpoint_aggregate": True,
            "decoding": "greedy_temperature_zero",
        },
        "artifacts": {
            "predictions": file_ref(predictions_path),
            "manual_packet": file_ref(packet_path),
        },
        "test_or_blind_data_read": False,
    }
    report_path = output_root / "REPORT.json"
    _write_json_new(report_path, report)
    return {**report, "report": file_ref(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-exposure", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--dev-dataset", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--correct", type=int, default=3)
    parser.add_argument("--false-match", type=int, default=4)
    parser.add_argument("--missed", type=int, default=3)
    parser.add_argument("--abstention-error", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
