#!/usr/bin/env python3
"""Freeze, infer, and score one preselected CE+typed hybrid on blind only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping

from .build_compact_typed_llama_dataset import INSTRUCTION, _between, _cards, _clip
from .common import read_jsonl, sha256_file
from .run_hybrid_typed_dev_paired_vllm import _reorder_compact_prompt
from .run_nemotron_ce import verify_base_manifest
from .run_paired_gemma_lora_batch import _infer_representatives, _prediction_payload


FREEZE_SCHEMA = "silver-match-v3-humor-hybrid-blind-freeze-v1"
TYPED_META_SCHEMA = "silver-match-v3-humor-hybrid-blind-typed-meta-v1"
SCORE_SCHEMA = "silver-match-v3-humor-hybrid-blind-score-v1"
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


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


def compact_user_prompt(original: str, candidate_ids: list[str]) -> str:
    norm = _between(original, "HUMAN STATEMENT:", "\nEVIDENCE PASSAGE FROM THE HUMAN FEEDBACK:")
    context = _between(
        original,
        "EVIDENCE PASSAGE FROM THE HUMAN FEEDBACK:",
        "\nEXTRACTED POLARITY (does not determine metric):",
    )
    cards = _cards(original, candidate_ids)
    card_text = "\n".join(
        f"[{card['metric_id']}] {card['name']} — {card['definition']}" for card in cards
    )
    return (
        f"{INSTRUCTION}\nTASK BANK: humor\nHUMAN STATEMENT (verbatim):\n{norm}\n"
        f"CONTEXT (capped at 600 characters):\n{_clip(context, 600)}\n\n"
        f"CANDIDATE METRIC CARDS (no examples):\n{card_text}\n\n"
        "Return the JSON decision now."
    )


def freeze(args: argparse.Namespace) -> None:
    root = Path(args.output_root).resolve()
    if root.exists():
        raise FileExistsError(root)
    gate_path = Path(args.dev_gate).resolve()
    if sha256_file(gate_path) != args.dev_gate_sha256:
        raise ValueError("dev gate SHA mismatch")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    selected = gate.get("selected_gate") or {}
    expected_rule = {
        "minimum_typed_confidence": "high",
        "minimum_ce_exact_probability": 0.9871788024902344,
        "minimum_ce_top_margin": 0.0,
    }
    if gate.get("status") != "PASS_VALIDATED_DEV_HYBRID_GATE" or any(
        selected.get(key) != value for key, value in expected_rule.items()
    ):
        raise ValueError("dev gate is not the predeclared strict hybrid rule")
    typed_source = Path(args.typed_blind).resolve()
    ce_source = Path(args.ce_blind_pairs).resolve()
    if sha256_file(typed_source) != args.typed_blind_sha256:
        raise ValueError("typed blind source SHA mismatch")
    if sha256_file(ce_source) != args.ce_blind_pairs_sha256:
        raise ValueError("CE blind pair source SHA mismatch")

    root.mkdir(parents=True, exist_ok=False)
    typed_paths = {mode: root / f"typed.{mode}.prompts.jsonl" for mode in ("original", "reordered")}
    typed_handles = {mode: path.open("x", encoding="utf-8") for mode, path in typed_paths.items()}
    typed_uids: set[str] = set()
    try:
        with typed_source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                # Gold-bearing keys are intentionally never accessed in this phase.
                row = json.loads(line)
                uid = str(row.get("norm_uid") or "")
                candidates = [str(value) for value in row.get("candidate_metric_ids") or []]
                messages = row.get("messages") or []
                if (
                    not uid or uid in typed_uids or row.get("split") != "blind"
                    or row.get("gradient_eligible") is not False or row.get("view") != "retrieval_order"
                    or len(messages) < 1 or messages[0].get("role") != "user"
                    or not candidates or len(candidates) != len(set(candidates))
                ):
                    raise ValueError(f"invalid typed blind projection row {line_number}")
                original = compact_user_prompt(str(messages[0].get("content") or ""), candidates)
                reordered, reordered_ids = _reorder_compact_prompt(original, uid, candidates)
                for mode, content, ids in (
                    ("original", original, candidates),
                    ("reordered", reordered, reordered_ids),
                ):
                    projected = {
                        "schema_version": "silver-match-v3-humor-hybrid-blind-typed-prompt-v1",
                        "task": row.get("task"), "corpus": row.get("corpus"),
                        "norm_uid": uid, "source_group": row.get("source_group"),
                        "split": "blind", "order_mode": mode,
                        "candidate_metric_ids": ids,
                        "conversation": [{"role": "user", "content": content}],
                    }
                    typed_handles[mode].write(json.dumps(projected, ensure_ascii=False, sort_keys=True) + "\n")
                typed_uids.add(uid)
        for handle in typed_handles.values():
            handle.flush(); os.fsync(handle.fileno())
    finally:
        for handle in typed_handles.values():
            handle.close()
    if len(typed_uids) != 1000:
        raise ValueError(f"typed blind cardinality differs: {len(typed_uids)}")

    ce_path = root / "ce.pairs.truth_blind.jsonl"
    ce_uids: set[str] = set()
    ce_pairs: set[tuple[str, str]] = set()
    with ce_source.open(encoding="utf-8") as source, ce_path.open("x", encoding="utf-8") as output:
        for line_number, line in enumerate(source, 1):
            # Relation/label/acceptable IDs/decision are never read or projected.
            row = json.loads(line)
            uid = str(row.get("norm_uid") or "")
            metric_id = str(row.get("metric_id") or row.get("candidate_metric_id") or "")
            identity = (uid, metric_id)
            query = str(row.get("query") or row.get("norm") or row.get("statement") or "")
            metric_card = str(row.get("metric_card") or "")
            if (
                not all(identity) or identity in ce_pairs or row.get("split") != "blind"
                or not row.get("source_group") or not query or not metric_card
            ):
                raise ValueError(f"invalid CE blind projection row {line_number}: {identity}")
            projected = {
                "schema_version": "silver-match-v3-humor-hybrid-blind-ce-pair-v1",
                "norm_uid": uid, "metric_id": metric_id,
                "source_group": row["source_group"], "split": "blind",
                "query": query, "metric_card": metric_card,
            }
            output.write(json.dumps(projected, ensure_ascii=False, sort_keys=True) + "\n")
            ce_pairs.add(identity); ce_uids.add(uid)
        output.flush(); os.fsync(output.fileno())
    if not ce_uids or not ce_uids < typed_uids:
        raise ValueError(
            f"expected a nonempty strict CE subset of typed blind UIDs: "
            f"{len(typed_uids)}/{len(ce_uids)}"
        )

    manifest = {
        "schema_version": FREEZE_SCHEMA,
        "status": "FROZEN_STRICT_DEV_RULE_AND_TRUTH_BLIND_PROJECTIONS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dev_gate": artifact(gate_path),
        "frozen_rule": {
            "acceptance": "same typed MATCH leaf both orders, both typed confidence high, CE same top1, CE P(EXACT)>=0.9871788024902344, top margin>=0",
            **expected_rule,
            "model_or_threshold_selection_after_freeze": False,
        },
        "typed_projection": {
            mode: {**artifact(path), "rows": 1000} for mode, path in typed_paths.items()
        },
        "ce_projection": {**artifact(ce_path), "rows": len(ce_pairs), "norm_uids": len(ce_uids)},
        "coverage_contract": {
            "typed_blind_norm_uids": len(typed_uids),
            "ce_blind_norm_uids": len(ce_uids),
            "missing_ce_norm_uids_forced_to_abstain": len(typed_uids - ce_uids),
        },
        "source_identities": {
            "typed_blind": {"path": str(typed_source), "sha256": args.typed_blind_sha256},
            "ce_blind_pairs": {"path": str(ce_source), "sha256": args.ce_blind_pairs_sha256},
        },
        "truth_firewall": {
            "gold_fields_projected": False,
            "inference_reads_only_truth_blind_projections": True,
            "blind_gold_may_be_opened_only_after_both_inferences_sealed": True,
            "test_rows_read": 0,
        },
    }
    write_json_new(root / "FREEZE.json", manifest)
    print(json.dumps({"status": manifest["status"], "typed_rows": len(typed_uids), "ce_pairs": len(ce_pairs)}, sort_keys=True))


def infer_typed(args: argparse.Namespace) -> None:
    freeze_path = Path(args.freeze).resolve()
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if frozen.get("schema_version") != FREEZE_SCHEMA:
        raise ValueError("invalid hybrid blind freeze")
    root = Path(args.output_root).resolve()
    if root.exists():
        raise FileExistsError(root)
    model = Path(args.model).resolve(); inventory = Path(args.model_inventory).resolve()
    adapter = Path(args.adapter).resolve()
    verify_base_manifest(model, inventory, args.model_inventory_sha256)
    root.mkdir(parents=True, exist_ok=False)
    outputs = {mode: root / f"typed.{mode}.jsonl" for mode in ("original", "reordered")}
    contract = {
        "schema_version": "silver-match-v3-humor-hybrid-blind-typed-contract-v1",
        "status": "FROZEN_BEFORE_BLIND_TYPED_INFERENCE",
        "freeze": artifact(freeze_path), "model": str(model),
        "model_inventory": artifact(inventory), "adapter": str(adapter),
        "adapter_config": artifact(adapter / "adapter_config.json"),
        "adapter_weights": artifact(adapter / "adapter_model.safetensors"),
        "decoding": {"temperature": 0.0, "seed": args.seed, "max_tokens": args.max_tokens},
        "backend": "direct_batch_vllm_not_openai_server", "blind_gold_read": False,
    }
    write_json_new(root / "INFERENCE_CONTRACT.json", contract)
    os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=str(model), dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, trust_remote_code=True, enable_lora=True,
        max_loras=1, max_lora_rank=args.max_lora_rank,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)
    request = LoRARequest("humor_hybrid_blind", 1, str(adapter))
    counts = Counter(); started = time.time()
    for mode in ("original", "reordered"):
        prompt_ref = frozen["typed_projection"][mode]
        prompt_path = Path(prompt_ref["path"])
        if sha256_file(prompt_path) != prompt_ref["sha256"]:
            raise ValueError("typed prompt projection drift")
        rows = list(read_jsonl(prompt_path))
        with outputs[mode].open("x", encoding="utf-8") as handle:
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
                    counts[f"{mode}:invalid"] += prediction["decision"] == "INVALID_OUTPUT"
                    handle.write(json.dumps({
                        "schema_version": "silver-match-v3-humor-hybrid-blind-typed-prediction-v1",
                        "norm_uid": row["norm_uid"], "source_group": row["source_group"],
                        "split": "blind", "order_mode": mode,
                        "candidate_metric_ids": row["candidate_metric_ids"], **prediction,
                    }, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                print(json.dumps({"mode": mode, "completed": start + len(batch), "total": len(rows)}), flush=True)
    meta = {
        "schema_version": TYPED_META_SCHEMA,
        "status": "COMPLETE_SEALED_BLIND_TYPED_TWO_ORDER_INFERENCE",
        "blind_gold_read": False, "contract": artifact(root / "INFERENCE_CONTRACT.json"),
        "outputs": {mode: {**artifact(path), "rows": 1000} for mode, path in outputs.items()},
        "counts": dict(sorted(counts.items())), "elapsed_seconds": time.time() - started,
    }
    write_json_new(root / "INFERENCE_META.json", meta)
    print(json.dumps({"status": meta["status"], "outputs": meta["outputs"]}, sort_keys=True))


def wilson(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = NormalDist().inv_cdf(.975); p = successes / total; d = 1 + z*z/total
    center = (p + z*z/(2*total))/d
    radius = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/d
    return [max(0.0, center-radius), min(1.0, center+radius)]


def keyed(path: Path, split: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        uid = str(row.get("norm_uid") or "")
        if not uid or uid in result or row.get("split") != split:
            raise ValueError(f"invalid/duplicate {split} row: {uid}")
        result[uid] = row
    return result


def score(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    freeze_path = Path(args.freeze).resolve(); frozen = json.loads(freeze_path.read_text())
    typed_meta_path = Path(args.typed_meta).resolve(); typed_meta = json.loads(typed_meta_path.read_text())
    ce_meta_path = Path(args.ce_meta).resolve(); ce_meta = json.loads(ce_meta_path.read_text())
    if (
        frozen.get("schema_version") != FREEZE_SCHEMA
        or typed_meta.get("status") != "COMPLETE_SEALED_BLIND_TYPED_TWO_ORDER_INFERENCE"
        or typed_meta.get("blind_gold_read") is not False
        or int(ce_meta.get("norm_group_count", -1))
        != int(frozen["coverage_contract"]["ce_blind_norm_uids"])
        or ce_meta.get("classification_mode") != "binary"
    ):
        raise ValueError("inference artifacts are not sealed blind contracts")
    original_path = Path(typed_meta["outputs"]["original"]["path"])
    reordered_path = Path(typed_meta["outputs"]["reordered"]["path"])
    ce_path = Path(args.ce_scores).resolve()
    if (
        sha256_file(original_path) != typed_meta["outputs"]["original"]["sha256"]
        or sha256_file(reordered_path) != typed_meta["outputs"]["reordered"]["sha256"]
        or sha256_file(ce_path) != ce_meta.get("output_sha256")
        or ce_meta.get("input_pairs_sha256") != frozen["ce_projection"]["sha256"]
    ):
        raise ValueError("sealed inference artifact drift")
    original = keyed(original_path, "blind"); reordered = keyed(reordered_path, "blind")
    ce_groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
    pairs = set()
    for row in read_jsonl(ce_path):
        uid, metric = str(row.get("norm_uid") or ""), str(row.get("metric_id") or "")
        if row.get("split") != "blind" or (uid, metric) in pairs:
            raise ValueError("invalid blind CE score")
        pairs.add((uid, metric)); ce_groups[uid].append((metric, float(row["probabilities"]["EXACT"])))
    if set(original) != set(reordered) or not set(ce_groups) < set(original):
        raise ValueError("blind inference UID universes differ")
    rule = frozen["frozen_rule"]
    accepted: dict[str, str] = {}
    for uid in original:
        if uid not in ce_groups:
            continue
        left, right = original[uid], reordered[uid]
        if not (
            left.get("decision") == right.get("decision") == "MATCH"
            and left.get("metric_id") == right.get("metric_id")
            and left.get("metric_id") not in (None, "")
            and CONFIDENCE_RANK.get(str(left.get("confidence")), -1) >= CONFIDENCE_RANK[rule["minimum_typed_confidence"]]
            and CONFIDENCE_RANK.get(str(right.get("confidence")), -1) >= CONFIDENCE_RANK[rule["minimum_typed_confidence"]]
        ):
            continue
        metric = str(left["metric_id"])
        ranked = sorted(ce_groups[uid], key=lambda value: (-value[1], value[0]))
        top = ranked[0]; second = ranked[1][1] if len(ranked) > 1 else 0.0
        if (
            top[0] == metric
            and top[1] >= float(rule["minimum_ce_exact_probability"])
            and top[1] - second >= float(rule["minimum_ce_top_margin"])
        ):
            accepted[uid] = metric

    # First and only post-inference gold access. Test is never opened.
    blind_source = Path(args.blind_gold_source).resolve()
    if sha256_file(blind_source) != frozen["source_identities"]["typed_blind"]["sha256"]:
        raise ValueError("blind gold source drift")
    gold: dict[str, set[str]] = {}
    with blind_source.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            uid = str(row["norm_uid"])
            gold[uid] = ({str(row["metric_id"])} if row.get("decision") == "MATCH" else set())
    if set(gold) != set(original):
        raise ValueError("blind gold/prediction UID universe differs")
    correct = sum(metric in gold[uid] for uid, metric in accepted.items())
    predicted = len(accepted); gold_matches = sum(bool(values) for values in gold.values())
    precision = correct / predicted if predicted else None
    recall = correct / gold_matches if gold_matches else None
    report = {
        "schema_version": SCORE_SCHEMA,
        "status": "COMPLETE_ONE_SHOT_TRUTH_FIREWALLED_BLIND_HYBRID_EVALUATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_rule": rule,
        "counts": {
            "blind_rows": len(gold), "gold_match_rows": gold_matches,
            "ce_overlap_rows": len(ce_groups),
            "forced_abstain_missing_ce": len(gold) - len(ce_groups),
            "accepted": predicted, "correct_exact": correct,
        },
        "metrics": {
            "exact_precision": precision, "exact_recall": recall,
            "precision_wilson_95": wilson(correct, predicted),
            "coverage_all_blind_rows": predicted / len(gold),
            "coverage_ce_overlap_rows": predicted / len(ce_groups),
            "coverage_gold_match_rows": correct / gold_matches if gold_matches else None,
        },
        "artifacts": {
            "implementation": artifact(Path(__file__)),
            "freeze": artifact(freeze_path), "typed_meta": artifact(typed_meta_path),
            "ce_meta": artifact(ce_meta_path), "ce_scores": artifact(ce_path),
            "blind_gold_source": artifact(blind_source),
        },
        "truth_firewall": {
            "typed_and_ce_inference_sealed_before_gold_open": True,
            "blind_gold_open_count": 1, "test_rows_read": 0,
            "threshold_or_model_selection_on_blind": False,
        },
    }
    write_json_new(output, report)
    print(json.dumps({"status": report["status"], "counts": report["counts"], "metrics": report["metrics"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--dev-gate", required=True); p.add_argument("--dev-gate-sha256", required=True)
    p.add_argument("--typed-blind", required=True); p.add_argument("--typed-blind-sha256", required=True)
    p.add_argument("--ce-blind-pairs", required=True); p.add_argument("--ce-blind-pairs-sha256", required=True)
    p.add_argument("--output-root", required=True)
    p = sub.add_parser("infer-typed")
    p.add_argument("--freeze", required=True); p.add_argument("--model", required=True)
    p.add_argument("--model-inventory", required=True); p.add_argument("--model-inventory-sha256", required=True)
    p.add_argument("--adapter", required=True); p.add_argument("--output-root", required=True)
    p.add_argument("--batch-size", type=int, default=128); p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=192); p.add_argument("--gpu-memory-utilization", type=float, default=.88)
    p.add_argument("--max-lora-rank", type=int, default=16); p.add_argument("--seed", type=int, default=94137)
    p = sub.add_parser("score")
    p.add_argument("--freeze", required=True); p.add_argument("--typed-meta", required=True)
    p.add_argument("--ce-scores", required=True); p.add_argument("--ce-meta", required=True)
    p.add_argument("--blind-gold-source", required=True); p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "freeze": freeze(args)
    elif args.command == "infer-typed": infer_typed(args)
    else: score(args)


if __name__ == "__main__":
    main()
