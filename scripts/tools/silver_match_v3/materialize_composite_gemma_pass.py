#!/usr/bin/env python3
"""Materialize and audit one complete OpenRouter + direct-vLLM Gemma pass.

The interrupted OpenRouter pass is immutable: completed chunks retain their
original raw payload and exact API transcript.  Only the frozen missing-chunk
frontier may be supplied by deterministic direct vLLM.  Direct rows are
accepted only after raw-response re-parsing and independent prompt rendering.
The script writes a new complete raw-label directory; it never edits the
original pack or its interrupted OpenRouter artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .adjudicate_gemma import (
    CONFIDENCES,
    DECISIONS,
    build_item_prompt,
    parse_response,
    prompt_sha256,
)
from .common import read_jsonl, sha256_file


SCHEMA = "silver-match-v3-composite-openrouter-direct-vllm-transcript-audit-v1"
PARTIAL_SCHEMA = "silver-match-v3-partial-openrouter-label-freeze-v1"
PLAN_SCHEMA = "silver-match-v3-missing-direct-vllm-plan-v1"


def _ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _matches(reference: dict[str, Any], path: Path) -> bool:
    path = path.resolve()
    return (
        path.is_file()
        and reference.get("sha256") == sha256_file(path)
        and (
            reference.get("bytes") is None
            or int(reference.get("bytes")) == path.stat().st_size
        )
    )


def _index_jsonl(path: Path, label: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = list(read_jsonl(path))
    indexed = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in indexed or len(rows) != len(indexed):
        raise ValueError(f"{label} has missing or duplicate norm UIDs: {path}")
    return rows, indexed


def _chunk_paths(validation: dict[str, Any], validation_path: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for raw_path, expected_sha in ((validation.get("outputs") or {}).get("chunks") or {}).items():
        path = Path(raw_path)
        path = path.resolve() if path.is_absolute() else (validation_path.parent / path).resolve()
        if not path.is_file() or sha256_file(path) != str(expected_sha):
            raise ValueError(f"pack chunk hash drift: {path}")
        if path.stem in output:
            raise ValueError(f"duplicate chunk stem: {path.stem}")
        output[path.stem] = path
    if not output:
        raise ValueError("pack has no chunks")
    return output


def _validate_raw_payload(
    payload: dict[str, Any],
    *,
    task: str,
    chunk_id: str,
    expected_uids: list[str],
    bank_ids: set[str],
) -> None:
    if payload.get("task") != task or payload.get("chunk_id") != chunk_id:
        raise ValueError(f"raw label task/chunk mismatch: {chunk_id}")
    labels = payload.get("labels")
    if not isinstance(labels, list) or len(labels) != len(expected_uids):
        raise ValueError(f"raw label count mismatch: {chunk_id}")
    observed = [str(row.get("norm_uid") or "") for row in labels]
    if len(set(observed)) != len(observed) or set(observed) != set(expected_uids):
        raise ValueError(f"raw label UID coverage mismatch: {chunk_id}")
    for row in labels:
        decision = str(row.get("decision") or "").upper()
        confidence = str(row.get("confidence") or "").lower()
        reason = str(row.get("reason") or "").strip()
        metric_id = row.get("metric_id")
        metric_id = None if metric_id is None else str(metric_id)
        if decision not in DECISIONS or confidence not in CONFIDENCES or not reason:
            raise ValueError(f"invalid raw decision/confidence/reason: {chunk_id}")
        if decision == "MATCH":
            if metric_id not in bank_ids:
                raise ValueError(f"raw MATCH metric absent from bank: {chunk_id}/{metric_id}")
        elif metric_id is not None:
            raise ValueError(f"raw abstention carries metric ID: {chunk_id}")


def _prompt_components(paths: list[Path]) -> tuple[str, str]:
    text = "\n\n".join(path.read_text(encoding="utf-8").rstrip() for path in paths) + "\n"
    return text, prompt_sha256(text)


def _candidate_uid_order_sha(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(row["norm_uid"]) for row in rows).encode("utf-8")
    ).hexdigest()


def _verify_partial_openrouter(
    *,
    freeze_path: Path,
    pass_name: str,
    pack_root: Path,
    validation_path: Path,
    bank_path: Path,
    items_path: Path,
    chunks: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], set[str], set[str]]:
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    contracts = frozen.get("contracts") or {}
    if (
        frozen.get("schema_version") != PARTIAL_SCHEMA
        or frozen.get("status")
        != "FROZEN_PARTIAL_TRANSCRIPT_AUDITED_INTERRUPTED_NO_PROMOTION"
        or frozen.get("task") != "notice-and-comment"
        or contracts.get("every_preserved_chunk_passes_exact_request_reconstruction")
        is not True
        or contracts.get("missing_frontier_is_explicit") is not True
        or contracts.get("partial_labels_are_not_a_complete_gold_pass") is not True
    ):
        raise ValueError("invalid interrupted OpenRouter freeze")
    pass_row = (frozen.get("passes") or {}).get(pass_name)
    if not isinstance(pass_row, dict):
        raise KeyError(f"pass absent from partial freeze: {pass_name}")
    if (
        Path(str(pass_row.get("root") or "")).resolve() != pack_root
        or pass_row.get("task") != "notice-and-comment"
        or pass_row.get("promoted") is not False
        or not _matches(pass_row.get("pack_validation") or {}, validation_path)
        or not _matches(pass_row.get("bank") or {}, bank_path)
        or not _matches(pass_row.get("items") or {}, items_path)
    ):
        raise ValueError(f"partial freeze pack binding drift: {pass_name}")

    completed_rows = {
        str(row.get("chunk") or ""): row
        for row in pass_row.get("completed_chunks") or []
    }
    completed, missing = set(completed_rows), {
        str(value) for value in pass_row.get("missing_chunks") or []
    }
    if (
        "" in completed
        or completed & missing
        or completed | missing != set(chunks)
        or int(pass_row.get("completed_chunk_count", -1)) != len(completed)
        or int(pass_row.get("expected_chunk_count", -1)) != len(chunks)
    ):
        raise ValueError(f"partial freeze frontier drift: {pass_name}")

    actual_raw = {path.stem for path in (pack_root / "raw_labels").glob("part-*.json")}
    actual_transcripts = {
        path.stem for path in (pack_root / "api_transcripts").glob("part-*.json")
    }
    if actual_raw != completed or actual_transcripts != completed:
        raise ValueError(f"interrupted OpenRouter artifact inventory drift: {pass_name}")
    strict = pass_row.get("strict_partial_audit") or {}
    strict_rows = {str(row.get("chunk") or ""): row for row in strict.get("chunks") or []}
    if set(strict_rows) != completed:
        raise ValueError(f"strict partial audit accepted another chunk set: {pass_name}")
    for chunk_id, row in completed_rows.items():
        raw_path = pack_root / "raw_labels" / f"{chunk_id}.json"
        transcript_path = pack_root / "api_transcripts" / f"{chunk_id}.json"
        strict_row = strict_rows[chunk_id]
        if (
            not _matches(row.get("chunk_input") or {}, chunks[chunk_id])
            or not _matches(row.get("raw_label") or {}, raw_path)
            or not _matches(row.get("api_transcript") or {}, transcript_path)
            or strict_row.get("chunk_sha256") != sha256_file(chunks[chunk_id])
            or strict_row.get("raw_label_sha256") != sha256_file(raw_path)
            or strict_row.get("transcript_sha256") != sha256_file(transcript_path)
            or int(strict_row.get("request_count", 0)) < 1
        ):
            raise ValueError(f"OpenRouter completed chunk drift: {pass_name}/{chunk_id}")
    return frozen, completed_rows, completed, missing


def _verify_direct_run(
    *,
    pass_name: str,
    plan_path: Path,
    candidates_path: Path,
    output_path: Path,
    meta_path: Path,
    runner_path: Path,
    runner_log_path: Path,
    prompt_paths: list[Path],
    pack_bank_path: Path,
    pack_items: dict[str, dict[str, Any]],
    pack_metrics: dict[str, dict[str, Any]],
    bank_order: list[str],
    missing_uids: set[str],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "FROZEN_BEFORE_DIRECT_BATCH_VLLM_INFERENCE"
        or plan.get("task") != "notice-and-comment"
        or int(plan.get("missing_vote_count", -1)) != 625
        or (plan.get("contracts") or {}).get("truth_hidden") is not True
        or (plan.get("contracts") or {}).get("only_exact_missing_chunk_frontier_inferred")
        is not True
    ):
        raise ValueError("invalid frozen direct-vLLM plan")
    continuation_rows = {
        str(row.get("name") or ""): row for row in plan.get("continuations") or []
    }
    continuation = continuation_rows.get(pass_name)
    if not isinstance(continuation, dict):
        raise KeyError(f"continuation absent from direct-vLLM plan: {pass_name}")
    candidate_rows, candidate_by_uid = _index_jsonl(candidates_path, "direct candidates")
    continuation_bank_ref = continuation.get("pack_bank") or {}
    if (
        not _matches(continuation.get("candidates") or {}, candidates_path)
        or int(continuation.get("count", -1)) != len(candidate_rows)
        or set(candidate_by_uid) != missing_uids
        or continuation.get("candidate_uid_order_sha256")
        != _candidate_uid_order_sha(candidate_rows)
        or continuation_bank_ref.get("sha256") != sha256_file(pack_bank_path)
        or int(continuation_bank_ref.get("bytes", -1)) != pack_bank_path.stat().st_size
    ):
        raise ValueError(f"direct continuation binding drift: {pass_name}")
    expected_bank_order_sha = hashlib.sha256("\n".join(bank_order).encode("utf-8")).hexdigest()
    if continuation.get("bank_order_sha256") != expected_bank_order_sha:
        raise ValueError(f"direct bank order drift: {pass_name}")
    for row in candidate_rows:
        ids = [str(value.get("metric_id") or "") for value in row.get("candidates") or []]
        if (
            row.get("task") != "notice-and-comment"
            or row.get("truth_hidden") is not True
            or row.get("prior_predictions_hidden") is not True
            or int(row.get("candidate_depth", -1)) != len(bank_order)
            or ids != bank_order
            or str(row.get("norm_uid")) not in pack_items
        ):
            raise ValueError(f"invalid direct candidate row: {row.get('norm_uid')}")

    system_prompt, combined_prompt_sha = _prompt_components(prompt_paths)
    planned_prompt_refs = (plan.get("inputs") or {}).get("prompts") or []
    if (
        combined_prompt_sha != (plan.get("prompt") or {}).get("combined_sha256")
        or len(planned_prompt_refs) != len(prompt_paths)
        or [row.get("sha256") for row in planned_prompt_refs]
        != [sha256_file(path) for path in prompt_paths]
        or ((plan.get("inputs") or {}).get("runner") or {}).get("sha256")
        != sha256_file(runner_path)
    ):
        raise ValueError("direct prompt/runner binding drift")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    runtime = plan.get("runtime") or {}
    expected_runtime = {
        "batch_size": int(runtime["batch_size"]),
        "gpu_memory_utilization": float(runtime["gpu_memory_utilization"]),
        "keep_raw": True,
        "max_model_len": int(runtime["max_model_len"]),
        "max_tokens": int(runtime["max_tokens"]),
        "resume": False,
        "seed": int(continuation["seed"]),
        "temperature": 0.0,
    }
    observed_runtime = meta.get("runtime") or {}
    if (
        not output_path.is_file()
        or meta.get("output_sha256") != sha256_file(output_path)
        or meta.get("input_candidates_sha256") != sha256_file(candidates_path)
        or int(meta.get("eligible_count", -1)) != len(candidate_rows)
        or int(meta.get("new_count", -1)) != len(candidate_rows)
        or int(meta.get("invalid_count", -1)) != 0
        or int(meta.get("max_candidates", -1)) != len(bank_order)
        or meta.get("order_mode") != "original"
        or meta.get("prompt_sha256") != combined_prompt_sha
        or meta.get("model") != runtime.get("model")
        or meta.get("python_executable") != runtime.get("python")
        or meta.get("prompt_rendering")
        != {
            "context_chars": int(runtime["context_chars"]),
            "description_chars": int(runtime["description_chars"]),
            "example_chars": int(runtime["example_chars"]),
            "max_examples": int(runtime["max_examples"]),
        }
        or any(observed_runtime.get(key) != value for key, value in expected_runtime.items())
    ):
        raise ValueError(f"direct run metadata drift: {pass_name}")
    component_hashes = {
        Path(str(path)).name: digest
        for path, digest in (meta.get("prompt_component_sha256") or {}).items()
    }
    expected_component_hashes = {
        path.name: sha256_file(path) for path in prompt_paths
    }
    if component_hashes != expected_component_hashes:
        raise ValueError(f"direct prompt component order/hash drift: {pass_name}")

    output_rows, output_by_uid = _index_jsonl(output_path, "direct output")
    if len(output_rows) != len(candidate_rows) or set(output_by_uid) != missing_uids:
        raise ValueError(f"direct output coverage drift: {pass_name}")
    prompt_render = meta["prompt_rendering"]
    for candidate in candidate_rows:
        uid = str(candidate["norm_uid"])
        row = output_by_uid[uid]
        item = pack_items[uid]
        rendered = build_item_prompt(
            system_prompt,
            item,
            list(candidate["candidates"]),
            pack_metrics,
            context_chars=int(prompt_render["context_chars"]),
            description_chars=int(prompt_render["description_chars"]),
            example_chars=int(prompt_render["example_chars"]),
            max_examples=int(prompt_render["max_examples"]),
        )
        ids = [str(value["metric_id"]) for value in candidate["candidates"]]
        parsed, parse_error = parse_response(str(row.get("raw_response") or ""), set(ids))
        if (
            parsed is None
            or parse_error is not None
            or row.get("parse_error") is not None
            or row.get("decision") != parsed.get("decision")
            or row.get("metric_id") != parsed.get("metric_id")
            or row.get("confidence") != parsed.get("confidence")
            or row.get("reason") != parsed.get("reason")
            or row.get("candidate_ids") != ids
            or row.get("candidate_bank_source_sha256")
            != candidate.get("bank_source_sha256")
            or row.get("prompt_sha256") != combined_prompt_sha
            or row.get("item_prompt_sha256") != prompt_sha256(rendered)
            or row.get("model") != runtime.get("model")
            or row.get("order_mode") != "original"
            or row.get("task") != item.get("task")
            or row.get("corpus") != item.get("corpus")
            or row.get("row") != item.get("row")
        ):
            raise ValueError(f"direct raw-response/prompt audit failed: {pass_name}/{uid}")
    prompt_hash_by_uid = {
        uid: str(row.get("item_prompt_sha256") or "") for uid, row in output_by_uid.items()
    }
    for uid, row in output_by_uid.items():
        representative = str(row.get("inference_representative_norm_uid") or "")
        if (
            representative not in output_by_uid
            or prompt_hash_by_uid[representative] != prompt_hash_by_uid[uid]
            or int(row.get("inference_equivalence_size", 0)) < 1
        ):
            raise ValueError(f"direct prompt-equivalence provenance failed: {pass_name}/{uid}")
    return plan, meta, output_rows, output_by_uid


def run(args: argparse.Namespace) -> dict[str, Any]:
    pack_root = Path(args.pack_root).resolve()
    pass_name = args.pass_name
    freeze_path = Path(args.partial_freeze).resolve()
    plan_path = Path(args.direct_plan).resolve()
    candidates_path = Path(args.direct_candidates).resolve()
    output_path = Path(args.direct_output).resolve()
    meta_path = Path(args.direct_meta).resolve()
    runner_path = Path(args.runner).resolve()
    runner_log_path = Path(args.runner_log).resolve()
    prompt_paths = [Path(value).resolve() for value in args.prompt]
    output_root = Path(args.output_root).resolve()
    audit_path = Path(args.output_audit).resolve()
    if output_root.exists() or audit_path.exists():
        raise FileExistsError("composite output root or audit already exists")
    validation_path = pack_root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("truth_hidden") is not True:
        raise ValueError("pack is not truth-hidden")
    task = str(validation.get("task") or "")
    if task != "notice-and-comment":
        raise ValueError(f"unexpected task: {task}")
    bank_path, items_path = pack_root / "bank.json", pack_root / "items.jsonl"
    if (
        sha256_file(bank_path) != validation["outputs"]["bank"]["sha256"]
        or sha256_file(items_path) != validation["outputs"]["items"]["sha256"]
    ):
        raise ValueError("pack bank/items hash drift")
    chunks = _chunk_paths(validation, validation_path)
    item_rows, item_by_uid = _index_jsonl(items_path, "pack items")
    chunk_uids = {
        chunk_id: [str(row["norm_uid"]) for row in read_jsonl(path)]
        for chunk_id, path in chunks.items()
    }
    if set(uid for values in chunk_uids.values() for uid in values) != set(item_by_uid):
        raise ValueError("pack chunks do not exactly partition items")
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    pack_metrics = {str(row["metric_id"]): row for row in bank.get("metrics") or []}
    bank_order = [str(row["metric_id"]) for row in bank.get("metrics") or []]
    if len(pack_metrics) != 88 or len(bank_order) != len(set(bank_order)):
        raise ValueError("expected an exact 88-card bank")

    frozen, completed_rows, completed, missing = _verify_partial_openrouter(
        freeze_path=freeze_path,
        pass_name=pass_name,
        pack_root=pack_root,
        validation_path=validation_path,
        bank_path=bank_path,
        items_path=items_path,
        chunks=chunks,
    )
    missing_uids = {uid for chunk_id in missing for uid in chunk_uids[chunk_id]}
    plan, meta, output_rows, output_by_uid = _verify_direct_run(
        pass_name=pass_name,
        plan_path=plan_path,
        candidates_path=candidates_path,
        output_path=output_path,
        meta_path=meta_path,
        runner_path=runner_path,
        runner_log_path=runner_log_path,
        prompt_paths=prompt_paths,
        pack_bank_path=bank_path,
        pack_items=item_by_uid,
        pack_metrics=pack_metrics,
        bank_order=bank_order,
        missing_uids=missing_uids,
    )
    if not runner_log_path.is_file():
        raise FileNotFoundError(runner_log_path)

    raw_root = output_root / "raw_labels"
    raw_root.mkdir(parents=True)
    audit_rows: list[dict[str, Any]] = []
    for chunk_id in sorted(chunks):
        destination = raw_root / f"{chunk_id}.json"
        expected_uids = chunk_uids[chunk_id]
        if chunk_id in completed:
            source = pack_root / "raw_labels" / f"{chunk_id}.json"
            shutil.copyfile(source, destination)
            strict_row = {
                str(row.get("chunk") or ""): row
                for row in ((frozen["passes"][pass_name]).get("strict_partial_audit") or {}).get("chunks") or []
            }[chunk_id]
            backend = "openrouter_api"
            event_count = int(strict_row["request_count"])
            backend_artifacts = {
                "api_transcript": _ref(pack_root / "api_transcripts" / f"{chunk_id}.json"),
                "partial_freeze": _ref(freeze_path),
            }
        else:
            labels = []
            for uid in expected_uids:
                row = output_by_uid[uid]
                labels.append(
                    {
                        "norm_uid": uid,
                        "decision": row["decision"],
                        "metric_id": row["metric_id"],
                        "confidence": row["confidence"],
                        "reason": row["reason"],
                    }
                )
            payload = {"task": task, "chunk_id": chunk_id, "labels": labels}
            destination.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            backend = "direct_vllm"
            event_count = len(labels)
            backend_artifacts = {
                "direct_output": _ref(output_path),
                "direct_meta": _ref(meta_path),
                "candidates": _ref(candidates_path),
                "runner": _ref(runner_path),
                "runner_log": _ref(runner_log_path),
                "frozen_plan": _ref(plan_path),
            }
        payload = json.loads(destination.read_text(encoding="utf-8"))
        _validate_raw_payload(
            payload,
            task=task,
            chunk_id=chunk_id,
            expected_uids=expected_uids,
            bank_ids=set(pack_metrics),
        )
        audit_rows.append(
            {
                "chunk": chunk_id,
                "chunk_sha256": sha256_file(chunks[chunk_id]),
                "raw_label_path": str(destination),
                "raw_label_sha256": sha256_file(destination),
                "backend": backend,
                "event_count": event_count,
                "backend_artifacts": backend_artifacts,
            }
        )

    backend_counts = Counter(row["backend"] for row in audit_rows)
    result = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "complete": True,
        "truth_hidden": True,
        "task": task,
        "pass_name": pass_name,
        "model": "google/gemma-4-31b-it",
        "pack_root": str(pack_root),
        "bank": _ref(bank_path),
        "items": _ref(items_path),
        "pack_validation": _ref(validation_path),
        "partial_openrouter_freeze": _ref(freeze_path),
        "direct_plan": _ref(plan_path),
        "direct_output": _ref(output_path),
        "direct_meta": _ref(meta_path),
        "expected_chunks": len(chunks),
        "audited_chunks": len(audit_rows),
        "event_count": sum(int(row["event_count"]) for row in audit_rows),
        "backend_counts": dict(sorted(backend_counts.items())),
        "direct_row_count": len(output_rows),
        "chunks": audit_rows,
        "violations": [],
        "contract": {
            "every_chunk_has_exactly_one_verified_backend": True,
            "openrouter_chunks_retain_exact_request_transcripts": True,
            "direct_vllm_rows_reparsed_from_raw_responses": True,
            "direct_vllm_item_prompts_independently_reconstructed": True,
            "only_frozen_missing_chunks_filled_by_direct_vllm": True,
            "sample_keys_predictions_proposals_mi_and_outcomes_absent": True,
            "original_interrupted_pack_unmodified": True,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**result, "output": str(audit_path), "output_sha256": sha256_file(audit_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass-name", required=True)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--partial-freeze", required=True)
    parser.add_argument("--direct-plan", required=True)
    parser.add_argument("--direct-candidates", required=True)
    parser.add_argument("--direct-output", required=True)
    parser.add_argument("--direct-meta", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--runner-log", required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-audit", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
