#!/usr/bin/env python3
"""Evaluate Gemma similarity adapters and compare pooled/task-local variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import socket
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from methods.codability.lexicon_distill.dataset import LABELS
from methods.codability.lexicon_distill.train_gemma4_similarity_lora import (
    collate,
    encode_rows,
    file_ref,
    read_protocols,
    write_json_new,
)


def assert_sk2_host() -> None:
    host = socket.gethostname().split(".", 1)[0].lower()
    if host not in {"sk2", "skampere2"} and not host.startswith("skampere2-"):
        raise RuntimeError(f"similarity evaluation is sk2-only; refusing host {socket.gethostname()}")


def read_eval_rows(path: Path, *, level: str, task: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("level") != level or (task and row.get("task") != task):
                continue
            if row.get("split") == "train":
                raise ValueError("evaluation dataset contains training rows")
            rows.append(row)
    if not rows:
        raise ValueError("no evaluation rows selected")
    return rows


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def cohen_kappa(truth: Sequence[int], predictions: Sequence[int]) -> float:
    if len(truth) != len(predictions) or not truth:
        raise ValueError("kappa inputs must be nonempty and aligned")
    n = len(truth)
    observed = sum(a == b for a, b in zip(truth, predictions)) / n
    truth_counts = Counter(truth)
    prediction_counts = Counter(predictions)
    expected = sum(truth_counts[label] * prediction_counts[label] for label in range(3)) / (n * n)
    return _safe_div(observed - expected, 1.0 - expected)


def metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    truth = [int(row["truth"]) for row in rows]
    predicted = [int(row["prediction"]) for row in rows]
    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label, name in enumerate(LABELS):
        tp = sum(a == label and b == label for a, b in zip(truth, predicted))
        fp = sum(a != label and b == label for a, b in zip(truth, predicted))
        fn = sum(a == label and b != label for a, b in zip(truth, predicted))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": truth.count(label)}
    brier = sum(
        sum((float(row["probabilities"][label]) - float(row["target_probs"][label])) ** 2 for label in range(3))
        for row in rows
    ) / len(rows)
    ordinal_mae = sum(abs(a - b) for a, b in zip(truth, predicted)) / len(rows)
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(10)]
    for row in rows:
        confidence = max(float(value) for value in row["probabilities"])
        bins[min(9, int(confidence * 10))].append((confidence, int(row["prediction"]) == int(row["truth"])))
    ece = sum(
        len(bucket) / len(rows) * abs(sum(conf for conf, _correct in bucket) / len(bucket) - sum(correct for _conf, correct in bucket) / len(bucket))
        for bucket in bins if bucket
    )
    return {
        "n": len(rows),
        "accuracy": sum(a == b for a, b in zip(truth, predicted)) / len(rows),
        "macro_f1": sum(f1_values) / 3,
        "cohen_kappa": cohen_kappa(truth, predicted),
        "same_precision": per_class["SAME"]["precision"],
        "same_recall": per_class["SAME"]["recall"],
        "same_f1": per_class["SAME"]["f1"],
        "ordinal_mae": ordinal_mae,
        "brier": brier,
        "ece": ece,
        "order_consistency": sum(bool(row["order_consistent"]) for row in rows) / len(rows),
        "per_class": per_class,
        "truth_counts": dict(Counter(LABELS[value] for value in truth)),
        "prediction_counts": dict(Counter(LABELS[value] for value in predicted)),
    }


def _load_model(model_path: str, adapter: str | None) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.config.use_cache = True
    return model


def run_evaluation(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    dataset = Path(args.dataset).resolve()
    protocols_path = Path(args.protocols).resolve()
    source_rows = read_eval_rows(dataset, level=args.level, task=args.task)
    protocols = read_protocols(protocols_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded, tokenization, label_ids = encode_rows(
        tokenizer, source_rows, protocols, max_length=args.max_length, augment_order=True
    )
    model = _load_model(args.model, args.adapter)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", 0)
    model.to(device)
    model.eval()
    label_tensor = torch.tensor(label_ids, dtype=torch.long, device=device)
    views: dict[str, list[list[float]]] = defaultdict(list)
    with torch.inference_mode():
        for start in range(0, len(encoded), args.batch_size):
            selected = encoded[start : start + args.batch_size]
            batch = {key: value.to(device, non_blocking=True) for key, value in collate(selected, int(tokenizer.pad_token_id)).items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            last = batch["attention_mask"].sum(dim=1) - 1
            indices = torch.arange(logits.shape[0], device=device)
            probabilities = torch.softmax(logits[indices, last][:, label_tensor].float(), dim=-1).cpu().tolist()
            for row, probability in zip(selected, probabilities):
                views[str(row["example_id"])].append([float(value) for value in probability])
    by_id = {str(row["example_id"]): row for row in source_rows}
    predictions: list[dict[str, Any]] = []
    for example_id, row in by_id.items():
        pair_views = views[example_id]
        if len(pair_views) != 2:
            raise AssertionError(f"expected two order views for {example_id}")
        probability = [(pair_views[0][index] + pair_views[1][index]) / 2 for index in range(3)]
        prediction = max(range(3), key=lambda index: probability[index])
        truth = max(range(3), key=lambda index: float(row["target_probs"][index]))
        predictions.append(
            {
                "example_id": example_id,
                "task": row["task"],
                "level": row["level"],
                "protocol_id": row["protocol_id"],
                "split": row["split"],
                "truth": truth,
                "prediction": prediction,
                "target_probs": row["target_probs"],
                "probabilities": probability,
                "view_probabilities": pair_views,
                "order_consistent": max(range(3), key=lambda i: pair_views[0][i]) == max(range(3), key=lambda i: pair_views[1][i]),
            }
        )
    prediction_path = Path(args.predictions).resolve()
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    if prediction_path.exists():
        raise FileExistsError(prediction_path)
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    overall = metrics(predictions)
    by_task = {task: metrics([row for row in predictions if row["task"] == task]) for task in sorted({row["task"] for row in predictions})}
    by_split = {split: metrics([row for row in predictions if row["split"] == split]) for split in sorted({row["split"] for row in predictions})}
    by_protocol = {
        protocol: metrics([row for row in predictions if row["protocol_id"] == protocol])
        for protocol in sorted({row["protocol_id"] for row in predictions})
    }
    report = {
        "schema_version": "gemma4-similarity-eval-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "adapter": file_ref(Path(args.adapter) / "adapter_model.safetensors") if args.adapter else None,
        "dataset": file_ref(dataset),
        "protocols": file_ref(protocols_path),
        "selection": {"level": args.level, "task": args.task},
        "tokenization": tokenization,
        "overall": overall,
        "by_task": by_task,
        "by_split": by_split,
        "by_protocol": by_protocol,
        "predictions": file_ref(prediction_path),
    }
    write_json_new(Path(args.report), report)
    print(json.dumps({"status": "COMPLETE", "overall": overall, "report": args.report}), flush=True)


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result = {str(row["example_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate prediction IDs")
    return result


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def compare_predictions(args: argparse.Namespace) -> None:
    pooled = read_predictions(Path(args.pooled_predictions))
    task = read_predictions(Path(args.task_predictions))
    common = sorted(set(pooled) & set(task))
    if len(common) < 100:
        raise ValueError(f"paired comparison underpowered: only {len(common)} examples")
    for example_id in common:
        if pooled[example_id]["truth"] != task[example_id]["truth"]:
            raise ValueError(f"truth drift for {example_id}")
    pooled_rows = [pooled[example_id] for example_id in common]
    task_rows = [task[example_id] for example_id in common]
    pooled_metrics = metrics(pooled_rows)
    task_metrics = metrics(task_rows)
    rng = random.Random(args.seed)
    kappa_deltas: list[float] = []
    same_f1_deltas: list[float] = []
    for _ in range(args.bootstrap_samples):
        sample = [rng.randrange(len(common)) for _ in common]
        pooled_sample = [pooled_rows[index] for index in sample]
        task_sample = [task_rows[index] for index in sample]
        kappa_deltas.append(metrics(task_sample)["cohen_kappa"] - metrics(pooled_sample)["cohen_kappa"])
        same_f1_deltas.append(metrics(task_sample)["same_f1"] - metrics(pooled_sample)["same_f1"])
    delta_kappa = task_metrics["cohen_kappa"] - pooled_metrics["cohen_kappa"]
    delta_same_f1 = task_metrics["same_f1"] - pooled_metrics["same_f1"]
    precision_drop = pooled_metrics["same_precision"] - task_metrics["same_precision"]
    recall_drop = pooled_metrics["same_recall"] - task_metrics["same_recall"]
    kappa_ci = [percentile(kappa_deltas, 0.025), percentile(kappa_deltas, 0.975)]
    same_f1_ci = [percentile(same_f1_deltas, 0.025), percentile(same_f1_deltas, 0.975)]
    cold_ids = [
        index for index, row in enumerate(pooled_rows)
        if str(row.get("split", "")).startswith("cold_test")
    ]
    cold_comparison = None
    cold_gate = True
    if len(cold_ids) >= 100:
        pooled_cold = metrics([pooled_rows[index] for index in cold_ids])
        task_cold = metrics([task_rows[index] for index in cold_ids])
        cold_delta = task_cold["cohen_kappa"] - pooled_cold["cohen_kappa"]
        cold_gate = cold_delta >= -0.02
        cold_comparison = {
            "n": len(cold_ids), "pooled": pooled_cold, "task_specific": task_cold,
            "delta_cohen_kappa": cold_delta, "minimum_allowed_delta": -0.02,
        }
    promoted = (
        delta_kappa >= 0.03
        and delta_same_f1 >= 0.03
        and kappa_ci[0] > 0
        and precision_drop <= 0.02
        and recall_drop <= 0.02
        and cold_gate
    )
    report = {
        "schema_version": "gemma4-similarity-pooled-task-comparison-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n": len(common),
        "pooled": pooled_metrics,
        "task_specific": task_metrics,
        "delta": {
            "cohen_kappa": delta_kappa,
            "same_f1": delta_same_f1,
            "same_precision": task_metrics["same_precision"] - pooled_metrics["same_precision"],
            "same_recall": task_metrics["same_recall"] - pooled_metrics["same_recall"],
        },
        "paired_bootstrap_95ci": {"cohen_kappa": kappa_ci, "same_f1": same_f1_ci},
        "cold_concept_comparison": cold_comparison,
        "promotion_gate": {
            "promoted": promoted,
            "requirements": {
                "minimum_kappa_delta": 0.03,
                "minimum_same_f1_delta": 0.03,
                "kappa_ci_lower_above_zero": True,
                "maximum_same_precision_drop": 0.02,
                "maximum_same_recall_drop": 0.02,
                "minimum_cold_kappa_delta": -0.02,
            },
        },
    }
    write_json_new(Path(args.report), report)
    print(json.dumps(report, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--protocols", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--adapter")
    evaluate.add_argument("--level", required=True, choices=("R1", "R2", "R3"))
    evaluate.add_argument("--task")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--report", required=True)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--max-length", type=int, default=1024)
    compare = sub.add_parser("compare")
    compare.add_argument("--pooled-predictions", required=True)
    compare.add_argument("--task-predictions", required=True)
    compare.add_argument("--report", required=True)
    compare.add_argument("--bootstrap-samples", type=int, default=1000)
    compare.add_argument("--seed", type=int, default=94137)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "evaluate":
        assert_sk2_host()
        run_evaluation(args)
    else:
        compare_predictions(args)


if __name__ == "__main__":
    main()
