#!/usr/bin/env python3
"""Freeze, infer, and score one selected typed LoRA on test and blind.

The three subcommands enforce a truth firewall:

* ``freeze`` is allowed only after dev-only checkpoint selection.  It projects
  held-out datasets into separate prompt-only and gold-only artifacts.
* ``infer`` reads only the prompt projection and runs paired base/LoRA direct
  batch-vLLM inference.
* ``score`` reads gold only after inference is sealed.

All artifacts are create-only.  A partial inference may be resumed without
decoding an already-written row again.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import read_jsonl, sha256_file
from .run_nemotron_ce import verify_base_manifest
from .run_paired_gemma_lora_batch import (
    _infer_representatives,
    _prediction_payload,
)


FREEZE_SCHEMA = "silver-match-v3-typed-lora-heldout-freeze-v1"
PREDICTION_SCHEMA = "silver-match-v3-typed-lora-heldout-prediction-v1"
INFERENCE_SCHEMA = "silver-match-v3-typed-lora-heldout-inference-meta-v1"
SCORE_SCHEMA = "silver-match-v3-typed-lora-heldout-score-v1"
ROLES = ("test", "blind")
DECISIONS = (
    "MATCH",
    "MATCH_FAMILY_ONLY",
    "NO_EXPLICIT_CRITERION",
    "CONTEXT_NEEDED",
    "GENERIC_VERDICT",
    "NO_CANDIDATE_FITS",
    "NOISE",
)
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
INVALID = "INVALID_OUTPUT"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _directory_files(path: Path) -> dict[str, dict[str, Any]]:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    values = {
        child.relative_to(path).as_posix(): _artifact(child)
        for child in sorted(path.rglob("*"))
        if child.is_file()
    }
    if not values:
        raise ValueError(f"empty directory: {path}")
    return values


def _selection_contract(adapter: Path, report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selection_path = adapter / "DEV_SELECTION.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        report.get("status")
        != "COMPLETE_DEV_SELECTED_ADAPTER_FRESH_RELOAD_VERIFIED"
        or selection.get("status") != "SELECTED_ON_DEV_ONLY"
        or selection.get("selection_split") != "dev"
        or selection.get("test_or_blind_data_read") is not False
        or (report.get("selection") or {}).get("test_or_blind_data_read") is not False
    ):
        raise ValueError("adapter lacks a completed dev-only fresh-reload selection")
    report_adapter = (report.get("adapter") or {}).get("directory")
    if Path(str(report_adapter or "")).resolve() != adapter.resolve():
        raise ValueError("training report refers to a different adapter")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        ref = ((report.get("adapter") or {}).get("content") or {}).get("files") or []
        expected = next(
            (row for row in ref if row.get("relative_path") == name), None
        )
        path = adapter / name
        if expected is None or sha256_file(path) != expected.get("sha256"):
            raise ValueError(f"selected adapter identity drift: {name}")
    gate = ((selection.get("chosen_dev_report") or {}).get("confidence_gate"))
    if not isinstance(gate, Mapping) or gate.get("minimum_confidence") not in CONFIDENCE_RANK:
        raise ValueError("dev selection lacks a frozen confidence threshold")
    return {
        "training_report": _artifact(report_path),
        "dev_selection": _artifact(selection_path),
        "adapter": str(adapter.resolve()),
        "adapter_files": _directory_files(adapter),
        "chosen_cumulative_exposure": selection.get("chosen_cumulative_exposure"),
        "chosen_dev_confidence_gate": dict(gate),
    }


def _project_role(
    role: str,
    path: Path,
    expected_sha: str,
    *,
    compact: bool = False,
    tokenizer: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if sha256_file(path) != expected_sha:
        raise ValueError(f"{role} dataset SHA256 mismatch")
    prompts: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    seen: set[str] = set()
    maximum_tokens = 0
    maximum_uid: str | None = None
    for line_number, row in enumerate(read_jsonl(path), 1):
        source_row = row
        if compact:
            if tokenizer is None:
                raise ValueError("compact held-out projection requires a tokenizer")
            from .build_compact_typed_llama_dataset import _compact_row
            from .train_gemma4_typed_lora import tokenize_example

            row = _compact_row(row)
            encoded = tokenize_example(tokenizer, row, 2048)
            if int(encoded["length"]) > maximum_tokens:
                maximum_tokens = int(encoded["length"])
                maximum_uid = str(row.get("norm_uid") or "")
            if row["messages"][-1]["content"] != source_row["messages"][-1]["content"]:
                raise AssertionError("compact projection changed held-out assistant target")
        uid = str(row.get("norm_uid") or "")
        messages = row.get("messages")
        candidates = row.get("candidate_metric_ids")
        decision = str(row.get("decision") or "")
        metric_id = row.get("metric_id")
        if (
            not uid
            or uid in seen
            or row.get("split") != role
            or row.get("gradient_eligible") is not False
            or row.get("view") != "retrieval_order"
            or not isinstance(messages, list)
            or len(messages) != 2
            or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"
            or not str(messages[0].get("content") or "").strip()
            or not isinstance(candidates, list)
            or not candidates
            or len(candidates) != len(set(candidates))
            or decision not in DECISIONS
            or ((decision == "MATCH") != bool(metric_id))
        ):
            raise ValueError(f"invalid frozen {role} row at line {line_number}")
        seen.add(uid)
        prompts.append(
            {
                "schema_version": "silver-match-v3-typed-heldout-prompt-v1",
                "task": row.get("task"),
                "corpus": row.get("corpus"),
                "norm_uid": uid,
                "split": role,
                "source_group": row.get("source_group"),
                "candidate_metric_ids": candidates,
                "conversation": [
                    {"role": "user", "content": messages[0]["content"]}
                ],
            }
        )
        gold.append(
            {
                "schema_version": "silver-match-v3-typed-heldout-gold-v1",
                "task": row.get("task"),
                "corpus": row.get("corpus"),
                "norm_uid": uid,
                "split": role,
                "decision": decision,
                "metric_id": metric_id,
                "truth_decision": row.get("truth_decision"),
                "target_relation": row.get("target_relation"),
            }
        )
    if not prompts:
        raise ValueError(f"empty {role} dataset")
    return prompts, gold, {
        "compact_prompt": compact,
        "maximum_tokens": maximum_tokens if compact else None,
        "maximum_token_norm_uid": maximum_uid if compact else None,
        "all_rows_within_2048": True if compact else None,
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=False)
    selection = _selection_contract(
        Path(args.adapter).resolve(), Path(args.training_report).resolve()
    )
    tokenizer = None
    compact_projector = None
    compact_enabled = bool(getattr(args, "compact_prompt", False))
    model_arg = getattr(args, "model", None)
    projector_sha = getattr(args, "compact_projector_sha256", None)
    if compact_enabled:
        if not model_arg or not projector_sha:
            raise ValueError("compact freeze requires --model and projector SHA256")
        from transformers import AutoTokenizer
        from . import build_compact_typed_llama_dataset as compact_module

        projector_path = Path(inspect.getfile(compact_module)).resolve()
        if sha256_file(projector_path) != projector_sha:
            raise ValueError("compact prompt projector identity mismatch")
        tokenizer = AutoTokenizer.from_pretrained(model_arg)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        compact_projector = _artifact(projector_path)
    inputs: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    for role in ROLES:
        dataset = Path(getattr(args, f"{role}_dataset")).resolve()
        expected_sha = getattr(args, f"{role}_sha256")
        prompts, gold, prompt_audit = _project_role(
            role,
            dataset,
            expected_sha,
            compact=compact_enabled,
            tokenizer=tokenizer,
        )
        prompt_path = root / f"{role}.prompts.jsonl"
        gold_path = root / f"{role}.gold.sealed.jsonl"
        _write_jsonl_new(prompt_path, prompts)
        _write_jsonl_new(gold_path, gold)
        inputs[role] = {**_artifact(dataset), "expected_sha256": expected_sha}
        outputs[role] = {
            "count": len(prompts),
            "prompts": _artifact(prompt_path),
            "gold_sealed": _artifact(gold_path),
            "prompt_audit": prompt_audit,
        }
    manifest = {
        "schema_version": FREEZE_SCHEMA,
        "status": "FROZEN_AFTER_DEV_SELECTION_BEFORE_HELDOUT_INFERENCE",
        "created_at": _now(),
        "truth_firewall": {
            "inference_reads_only_prompt_projection": True,
            "gold_projection_sealed_until_inference_complete": True,
            "test_or_blind_used_for_checkpoint_or_threshold_selection": False,
        },
        "selection": selection,
        "compact_prompt_projection": {
            "enabled": compact_enabled,
            "model": str(Path(model_arg).resolve()) if model_arg else None,
            "projector": compact_projector,
        },
        "heldout_inputs": inputs,
        "projections": outputs,
    }
    manifest_path = root / "FREEZE.json"
    _write_json_new(manifest_path, manifest)
    return {**manifest, "freeze": _artifact(manifest_path)}


def _load_freeze(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != FREEZE_SCHEMA
        or value.get("status")
        != "FROZEN_AFTER_DEV_SELECTION_BEFORE_HELDOUT_INFERENCE"
    ):
        raise ValueError("unknown or incomplete held-out freeze")
    for role in ROLES:
        for key in ("prompts", "gold_sealed"):
            ref = value["projections"][role][key]
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                raise ValueError(f"frozen {role} {key} drifted")
    return value


def infer(args: argparse.Namespace) -> dict[str, Any]:
    freeze_path = Path(args.freeze).resolve()
    frozen = _load_freeze(freeze_path)
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / "predictions.jsonl"
    meta_path = root / "INFERENCE_META.json"
    contract_path = root / "INFERENCE_CONTRACT.json"
    if meta_path.exists():
        raise FileExistsError("held-out inference is already sealed")
    adapter = Path(args.adapter).resolve()
    if adapter != Path(frozen["selection"]["adapter"]).resolve():
        raise ValueError("adapter differs from held-out freeze")
    for relative, ref in frozen["selection"]["adapter_files"].items():
        if sha256_file(adapter / relative) != ref["sha256"]:
            raise ValueError(f"adapter drift after freeze: {relative}")
    verify_base_manifest(
        Path(args.model), Path(args.model_inventory), args.model_inventory_sha256
    )
    contract = {
        "schema_version": "silver-match-v3-typed-lora-inference-contract-v1",
        "status": "FROZEN_BEFORE_DIRECT_BATCH_VLLM",
        "freeze": _artifact(freeze_path),
        "model": str(Path(args.model).resolve()),
        "model_inventory": _artifact(Path(args.model_inventory)),
        "adapter": str(adapter),
        "adapter_config": _artifact(adapter / "adapter_config.json"),
        "backend": "direct_batch_vllm_not_openai_server",
        "systems": ["base", "lora"],
        "temperature": 0.0,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "test_or_blind_gold_read": False,
    }
    if contract_path.exists():
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
        comparable = dict(observed)
        comparable.pop("created_at", None)
        if comparable != contract:
            raise ValueError("resume inference contract drift")
    else:
        _write_json_new(contract_path, {**contract, "created_at": _now()})
    done: set[tuple[str, str]] = set()
    if output_path.exists():
        if not args.resume:
            raise FileExistsError(output_path)
        for row in read_jsonl(output_path):
            key = (str(row.get("split")), str(row.get("norm_uid")))
            if key in done or row.get("schema_version") != PREDICTION_SCHEMA:
                raise ValueError("partial prediction output is invalid")
            done.add(key)
    pending: list[dict[str, Any]] = []
    expected: set[tuple[str, str]] = set()
    for role in ROLES:
        prompt_ref = frozen["projections"][role]["prompts"]
        for row in read_jsonl(Path(prompt_ref["path"])):
            key = (role, str(row.get("norm_uid") or ""))
            if (
                not key[1]
                or key in expected
                or row.get("split") != role
                or any(message.get("role") == "assistant" for message in row["conversation"])
            ):
                raise ValueError("prompt projection violates truth firewall")
            expected.add(key)
            if key not in done:
                pending.append(row)
    if not done <= expected:
        raise ValueError("partial output contains rows outside the freeze")

    started = time.time()
    retry_counts = Counter()
    invalid_counts = Counter()
    if pending:
        os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        llm = LLM(
            model=str(Path(args.model).resolve()),
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            enable_lora=True,
            max_loras=1,
            max_lora_rank=args.max_lora_rank,
        )
        sampling = SamplingParams(
            temperature=0.0, max_tokens=args.max_tokens, seed=args.seed
        )
        lora_request = LoRARequest(args.adapter_name, 1, str(adapter))
        for start in range(0, len(pending), args.batch_size):
            rows = pending[start : start + args.batch_size]
            conversations = [row["conversation"] for row in rows]
            candidate_sets = [set(row["candidate_metric_ids"]) for row in rows]
            base_values, base_retries = _infer_representatives(
                llm, conversations, candidate_sets, sampling, lora_request=None
            )
            lora_values, lora_retries = _infer_representatives(
                llm,
                conversations,
                candidate_sets,
                sampling,
                lora_request=lora_request,
            )
            retry_counts.update({"base": base_retries, "lora": lora_retries})
            output_rows = []
            for row, base_value, lora_value in zip(rows, base_values, lora_values):
                base = _prediction_payload(*base_value, keep_raw=False)
                lora = _prediction_payload(*lora_value, keep_raw=False)
                invalid_counts.update(
                    {
                        "base": int(base["decision"] == INVALID),
                        "lora": int(lora["decision"] == INVALID),
                    }
                )
                output_rows.append(
                    {
                        "schema_version": PREDICTION_SCHEMA,
                        "task": row.get("task"),
                        "corpus": row.get("corpus"),
                        "norm_uid": row["norm_uid"],
                        "split": row["split"],
                        "candidate_metric_ids": row["candidate_metric_ids"],
                        "base": base,
                        "lora": lora,
                    }
                )
            _append_jsonl(output_path, output_rows)
            print(
                json.dumps(
                    {
                        "completed": len(done) + start + len(rows),
                        "total": len(expected),
                        "elapsed_seconds": time.time() - started,
                    }
                ),
                flush=True,
            )
    final_rows = list(read_jsonl(output_path))
    final_keys = {(str(row["split"]), str(row["norm_uid"])) for row in final_rows}
    if final_keys != expected or len(final_rows) != len(expected):
        raise ValueError("held-out inference did not reach exact frozen coverage")
    meta = {
        "schema_version": INFERENCE_SCHEMA,
        "status": "COMPLETE_TRUTH_BLIND_PAIRED_INFERENCE",
        "completed_at": _now(),
        "freeze": _artifact(freeze_path),
        "contract": _artifact(contract_path),
        "predictions": {**_artifact(output_path), "count": len(final_rows)},
        "counts_by_split": dict(sorted(Counter(row["split"] for row in final_rows).items())),
        "new_retry_counts": dict(retry_counts),
        "new_invalid_counts": dict(invalid_counts),
        "test_or_blind_gold_read": False,
        "elapsed_seconds_this_process": time.time() - started,
    }
    _write_json_new(meta_path, meta)
    return {**meta, "meta": _artifact(meta_path)}


def _wilson_lower(successes: int, total: int) -> float | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z2 / (4 * total)) / total) / denominator
    return max(0.0, center - radius)


def _metrics(
    gold: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    system: str,
    minimum_confidence: str,
) -> dict[str, Any]:
    threshold = CONFIDENCE_RANK[minimum_confidence]
    confusion: dict[str, Counter[str]] = {decision: Counter() for decision in DECISIONS}
    accepted = correct = raw_exact = invalid = raw_match = 0
    gold_match = sum(row["decision"] == "MATCH" for row in gold)
    for truth, output in zip(gold, predictions):
        prediction = output[system]
        predicted_decision = str(prediction.get("decision") or INVALID)
        if predicted_decision == INVALID:
            invalid += 1
        confusion[str(truth["decision"])][predicted_decision] += 1
        same_leaf = (
            predicted_decision == truth["decision"]
            and (
                predicted_decision != "MATCH"
                or prediction.get("metric_id") == truth.get("metric_id")
            )
        )
        raw_exact += int(same_leaf)
        predicts_match = predicted_decision == "MATCH"
        raw_match += int(predicts_match)
        accept = predicts_match and CONFIDENCE_RANK.get(
            str(prediction.get("confidence") or ""), -1
        ) >= threshold
        accepted += int(accept)
        correct += int(
            accept
            and truth["decision"] == "MATCH"
            and prediction.get("metric_id") == truth.get("metric_id")
        )
    precision = correct / accepted if accepted else None
    recall = correct / gold_match if gold_match else None
    f05 = (
        1.25 * precision * recall / (0.25 * precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0
    )
    per_typed_decision: dict[str, dict[str, Any]] = {}
    prediction_counts = Counter()
    for counts in confusion.values():
        prediction_counts.update(counts)
    for decision in DECISIONS:
        true_positive = confusion[decision][decision]
        support = sum(confusion[decision].values())
        predicted = prediction_counts[decision]
        typed_precision = true_positive / predicted if predicted else None
        typed_recall = true_positive / support if support else None
        typed_f1 = (
            2 * typed_precision * typed_recall / (typed_precision + typed_recall)
            if typed_precision is not None
            and typed_recall is not None
            and typed_precision + typed_recall
            else 0.0
        )
        per_typed_decision[decision] = {
            "support": support,
            "predicted": predicted,
            "true_positive": true_positive,
            "precision": typed_precision,
            "recall": typed_recall,
            "f1": typed_f1,
        }
    return {
        "n": len(gold),
        "raw_exact_typed_and_leaf_accuracy": raw_exact / len(gold),
        "raw_exact_typed_and_leaf_correct": raw_exact,
        "raw_predicted_match_count": raw_match,
        "invalid_count": invalid,
        "invalid_rate": invalid / len(gold),
        "dev_selected_match_gate": {
            "minimum_confidence": minimum_confidence,
            "accepted_count": accepted,
            "correct_exact_leaf_count": correct,
            "gold_match_count": gold_match,
            "exact_precision": precision,
            "exact_precision_wilson_95_lower": _wilson_lower(correct, accepted),
            "exact_recall": recall,
            "exact_f_beta_0_5": f05,
            "abstention_rate": 1 - accepted / len(gold),
        },
        "prediction_decision_counts": dict(sorted(prediction_counts.items())),
        "per_typed_decision": per_typed_decision,
        "typed_decision_confusion": {
            decision: dict(sorted(confusion[decision].items())) for decision in DECISIONS
        },
    }


def score(args: argparse.Namespace) -> dict[str, Any]:
    freeze_path = Path(args.freeze).resolve()
    frozen = _load_freeze(freeze_path)
    meta_path = Path(args.inference_meta).resolve()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        meta.get("schema_version") != INFERENCE_SCHEMA
        or meta.get("status") != "COMPLETE_TRUTH_BLIND_PAIRED_INFERENCE"
        or meta.get("test_or_blind_gold_read") is not False
        or meta.get("freeze", {}).get("sha256") != sha256_file(freeze_path)
    ):
        raise ValueError("inference is not a complete truth-blind run for this freeze")
    predictions_path = Path(meta["predictions"]["path"])
    if sha256_file(predictions_path) != meta["predictions"]["sha256"]:
        raise ValueError("sealed predictions drifted")
    predictions = list(read_jsonl(predictions_path))
    prediction_index = {
        (str(row["split"]), str(row["norm_uid"])): row for row in predictions
    }
    if len(prediction_index) != len(predictions):
        raise ValueError("duplicate prediction keys")
    minimum_confidence = frozen["selection"]["chosen_dev_confidence_gate"][
        "minimum_confidence"
    ]
    gold_by_role: dict[str, list[dict[str, Any]]] = {}
    predictions_by_role: dict[str, list[dict[str, Any]]] = {}
    for role in ROLES:
        gold_ref = frozen["projections"][role]["gold_sealed"]
        gold_rows = list(read_jsonl(Path(gold_ref["path"])))
        role_predictions = []
        for row in gold_rows:
            key = (role, str(row["norm_uid"]))
            if key not in prediction_index:
                raise ValueError(f"prediction missing frozen gold row: {key}")
            role_predictions.append(prediction_index[key])
        gold_by_role[role] = gold_rows
        predictions_by_role[role] = role_predictions
    report_systems: dict[str, Any] = {}
    for system in ("base", "lora"):
        report_systems[system] = {
            role: _metrics(
                gold_by_role[role],
                predictions_by_role[role],
                system,
                minimum_confidence,
            )
            for role in ROLES
        }
        pooled_gold = [row for role in ROLES for row in gold_by_role[role]]
        pooled_predictions = [
            row for role in ROLES for row in predictions_by_role[role]
        ]
        report_systems[system]["pooled"] = _metrics(
            pooled_gold, pooled_predictions, system, minimum_confidence
        )
    report = {
        "schema_version": SCORE_SCHEMA,
        "status": "COMPLETE_ONE_SHOT_TEST_AND_BLIND_SCORE",
        "created_at": _now(),
        "freeze": _artifact(freeze_path),
        "inference_meta": _artifact(meta_path),
        "dev_selection": {
            "chosen_cumulative_exposure": frozen["selection"][
                "chosen_cumulative_exposure"
            ],
            "confidence_gate": frozen["selection"][
                "chosen_dev_confidence_gate"
            ],
        },
        "counts": {role: len(gold_by_role[role]) for role in ROLES},
        "target_relation_counts": {
            role: dict(
                sorted(Counter(row.get("target_relation") for row in gold_by_role[role]).items())
            )
            for role in ROLES
        },
        "systems": report_systems,
        "test_or_blind_used_for_selection": False,
        "post_heldout_threshold_tuning": False,
    }
    output = Path(args.output).resolve()
    _write_json_new(output, report)
    return {**report, "report": _artifact(output)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freezer = subparsers.add_parser("freeze")
    freezer.add_argument("--test-dataset", required=True)
    freezer.add_argument("--test-sha256", required=True)
    freezer.add_argument("--blind-dataset", required=True)
    freezer.add_argument("--blind-sha256", required=True)
    freezer.add_argument("--adapter", required=True)
    freezer.add_argument("--training-report", required=True)
    freezer.add_argument("--output-root", required=True)
    freezer.add_argument("--compact-prompt", action="store_true")
    freezer.add_argument("--model")
    freezer.add_argument("--compact-projector-sha256")
    runner = subparsers.add_parser("infer")
    runner.add_argument("--freeze", required=True)
    runner.add_argument("--model", required=True)
    runner.add_argument("--model-inventory", required=True)
    runner.add_argument("--model-inventory-sha256", required=True)
    runner.add_argument("--adapter", required=True)
    runner.add_argument("--adapter-name", default="humor_typed_llama31_v1")
    runner.add_argument("--output-root", required=True)
    runner.add_argument("--batch-size", type=int, default=128)
    runner.add_argument("--max-model-len", type=int, default=8192)
    runner.add_argument("--max-tokens", type=int, default=192)
    runner.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    runner.add_argument("--max-lora-rank", type=int, default=16)
    runner.add_argument("--seed", type=int, default=94137)
    runner.add_argument("--resume", action="store_true")
    scorer = subparsers.add_parser("score")
    scorer.add_argument("--freeze", required=True)
    scorer.add_argument("--inference-meta", required=True)
    scorer.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "infer":
        if min(args.batch_size, args.max_model_len, args.max_tokens, args.max_lora_rank) < 1:
            parser.error("positive inference sizes must be positive")
        if not 0 < args.gpu_memory_utilization < 1:
            parser.error("--gpu-memory-utilization must be in (0,1)")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        result = freeze(args)
    elif args.command == "infer":
        result = infer(args)
    else:
        result = score(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
