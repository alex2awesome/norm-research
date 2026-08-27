#!/usr/bin/env python3
"""Freeze and token-preflight missing blind-label rows for direct batch vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .adjudicate_gemma import build_item_prompt
from .common import read_jsonl, sha256_file


SCHEMA = "silver-match-v3-missing-direct-vllm-plan-v1"
CONTINUATION_SCHEMA = "silver-match-v3-missing-openrouter-gpu-continuation-v1"


def _ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _specs(values: list[str]) -> list[tuple[str, Path, Path, Path]]:
    output = []
    names: set[str] = set()
    for value in values:
        name, separator, rest = value.partition("=")
        parts = rest.split(",") if separator else []
        if not name or name in names or len(parts) != 3:
            raise ValueError("--continuation must be unique NAME=REPORT,CANDIDATES,PACK_ROOT")
        output.append((name, Path(parts[0]).resolve(), Path(parts[1]).resolve(), Path(parts[2]).resolve()))
        names.add(name)
    if not output:
        raise ValueError("at least one continuation is required")
    return output


def _index(path: Path, label: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = list(read_jsonl(path))
    index = {str(row.get("norm_uid") or ""): row for row in rows}
    if "" in index or len(index) != len(rows):
        raise ValueError(f"{label} contains missing or duplicate UIDs")
    return rows, index


def _combined_prompt(paths: list[Path]) -> tuple[str, str]:
    text = "\n\n".join(path.read_text(encoding="utf-8").rstrip() for path in paths) + "\n"
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation", action="append", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-inventory-audit", required=True)
    parser.add_argument("--runtime-dependency-audit", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--context-chars", type=int, default=1400)
    parser.add_argument("--description-chars", type=int, default=520)
    parser.add_argument("--example-chars", type=int, default=180)
    parser.add_argument("--max-examples", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--supersedes-plan")
    args = parser.parse_args()

    plan_path = Path(args.output).resolve()
    output_root = Path(args.output_root).resolve()
    if plan_path.exists() or output_root.exists():
        raise FileExistsError("plan or output root already exists")
    prompts = [Path(value).resolve() for value in args.prompt]
    runner = Path(args.runner).resolve()
    manifest_path = Path(args.manifest).resolve()
    model = Path(args.model).resolve()
    python = Path(args.python).resolve()
    runtime_root = Path(args.runtime_root).resolve()
    if not model.is_dir() or not python.is_file() or not runtime_root.is_dir():
        raise FileNotFoundError("model, python, or runtime root is absent")

    model_audit_path = Path(args.model_inventory_audit).resolve()
    model_audit = json.loads(model_audit_path.read_text(encoding="utf-8"))
    runtime_audit_path = Path(args.runtime_dependency_audit).resolve()
    runtime_audit = json.loads(runtime_audit_path.read_text(encoding="utf-8"))
    if (
        model_audit.get("status") != "EXACT_RECURSIVE_PYCACHE_FREE_INVENTORY_PASS"
        or Path(str(model_audit.get("root") or "")).resolve() != model
        or runtime_audit.get("status") != "EXACT_PYTHON_RUNTIME_DEPENDENCIES_PASS"
    ):
        raise ValueError("model/runtime inventory gate failed")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = "notice-and-comment"
    canonical_bank_path = Path(manifest["banks"][task]["path"]).resolve()
    canonical_bank = json.loads(canonical_bank_path.read_text(encoding="utf-8"))
    canonical_metrics = {str(row["metric_id"]): row for row in canonical_bank["metrics"]}
    canonical_norms: dict[str, dict[str, Any]] = {}
    for corpus, meta in manifest["corpora"].items():
        if meta.get("task") != task:
            continue
        for row in read_jsonl(Path(meta["path"])):
            canonical_norms[str(row["norm_uid"])] = row

    system_prompt, prompt_sha = _combined_prompt(prompts)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    continuations = []
    total = 0
    global_uids_by_view: dict[str, set[str]] = {}
    for name, report_path, candidate_path, pack_root in _specs(args.continuation):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validation_path = pack_root / "validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        rows, candidates = _index(candidate_path, f"{name} candidates")
        items, item_by_uid = _index(pack_root / "items.jsonl", f"{name} items")
        pack_bank_path = pack_root / "bank.json"
        pack_bank = json.loads(pack_bank_path.read_text(encoding="utf-8"))
        pack_metrics = {str(row["metric_id"]): row for row in pack_bank["metrics"]}
        if (
            report.get("schema_version") != CONTINUATION_SCHEMA
            or report.get("status") != "FROZEN_EXACT_MISSING_FRONTIER_BEFORE_DIRECT_BATCH_VLLM"
            or report.get("pass_name") != name
            or (report.get("output") or {}).get("sha256") != sha256_file(candidate_path)
            or validation.get("truth_hidden") is not True
            or pack_metrics != canonical_metrics
            or int(report.get("candidate_depth", -1)) != len(pack_metrics)
        ):
            raise ValueError(f"invalid continuation/bank binding: {name}")
        candidate_uids = [str(row["norm_uid"]) for row in rows]
        if not set(candidate_uids).issubset(item_by_uid):
            raise ValueError(f"continuation has UIDs outside pack: {name}")
        fields = ("task", "corpus", "row", "norm", "context", "aspect", "polarity")
        token_counts = []
        bank_order = [str(row["metric_id"]) for row in pack_bank["metrics"]]
        for row in rows:
            uid = str(row["norm_uid"])
            item = item_by_uid[uid]
            canonical = canonical_norms.get(uid)
            if canonical is None or any(item.get(field) != canonical.get(field) for field in fields):
                raise ValueError(f"canonical norm text/context drift: {name}/{uid}")
            ids = [str(card["metric_id"]) for card in row.get("candidates") or []]
            if ids != bank_order:
                raise ValueError(f"candidate order differs from frozen pack bank: {name}/{uid}")
            rendered = build_item_prompt(
                system_prompt,
                canonical,
                list(row["candidates"]),
                pack_metrics,
                context_chars=args.context_chars,
                description_chars=args.description_chars,
                example_chars=args.example_chars,
                max_examples=args.max_examples,
            )
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": rendered}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            if isinstance(tokens, dict):
                tokens = tokens.get("input_ids")
            if tokens and isinstance(tokens[0], list):
                if len(tokens) != 1:
                    raise ValueError("token preflight unexpectedly returned a multi-row batch")
                tokens = tokens[0]
            if not isinstance(tokens, list) or len(tokens) < 100:
                raise ValueError("token preflight did not return a plausible input-id sequence")
            token_counts.append(len(tokens))
        if max(token_counts) + args.max_tokens > args.max_model_len:
            raise ValueError(f"token preflight exceeds model length: {name}")
        seed = int(validation.get("seed"))
        output_path = output_root / f"{name}.missing.direct-vllm.jsonl"
        if output_path.exists() or output_path.with_suffix(output_path.suffix + ".meta.json").exists():
            raise FileExistsError(output_path)
        argv = [
            str(python), "-u", "-m", "scripts.tools.silver_match_v3.adjudicate_gemma",
            "--manifest", str(manifest_path), "--candidates", str(candidate_path),
            "--output", str(output_path), "--prompt", str(prompts[0]),
        ]
        for addon in prompts[1:]:
            argv.extend(["--prompt-addon", str(addon)])
        argv.extend([
            "--model", str(model), "--max-candidates", str(len(pack_metrics)),
            "--context-chars", str(args.context_chars),
            "--description-chars", str(args.description_chars),
            "--example-chars", str(args.example_chars),
            "--max-examples", str(args.max_examples),
            "--batch-size", str(args.batch_size),
            "--max-model-len", str(args.max_model_len),
            "--max-tokens", str(args.max_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--seed", str(seed), "--order-mode", "original", "--keep-raw",
        ])
        continuations.append({
            "name": name,
            "count": len(rows),
            "seed": seed,
            "pack_validation": _ref(validation_path),
            "pack_items": _ref(pack_root / "items.jsonl"),
            "pack_bank": _ref(pack_bank_path),
            "continuation_freeze": _ref(report_path),
            "candidates": _ref(candidate_path),
            "candidate_uid_order_sha256": hashlib.sha256("\n".join(candidate_uids).encode()).hexdigest(),
            "bank_order_sha256": hashlib.sha256("\n".join(bank_order).encode()).hexdigest(),
            "prompt_tokens": {"min": min(token_counts), "max": max(token_counts)},
            "output": str(output_path),
            "runner_log": str(output_root / f"{name}.runner.log"),
            "argv": argv,
        })
        total += len(rows)
        global_uids_by_view[name] = set(candidate_uids)
    if total != 625:
        raise ValueError(f"expected exact 625-row missing frontier, got {total}")

    supersedes = None
    if args.supersedes_plan:
        superseded_path = Path(args.supersedes_plan).resolve()
        superseded = json.loads(superseded_path.read_text(encoding="utf-8"))
        prior_outputs = [Path(str(row.get("output") or "")) for row in superseded.get("continuations") or []]
        if (
            superseded.get("schema_version") != SCHEMA
            or superseded.get("status") != "FROZEN_BEFORE_DIRECT_BATCH_VLLM_INFERENCE"
            or any(path.exists() for path in prior_outputs)
        ):
            raise ValueError("superseded plan is invalid or already produced inference")
        supersedes = {
            **_ref(superseded_path),
            "reason": "TOKEN_PREFLIGHT_RETURNED_MAPPING_LENGTH_INSTEAD_OF_INPUT_ID_LENGTH",
            "inference_rows_written": 0,
            "scientific_settings_changed": False,
        }
    output_root.mkdir(parents=True)
    payload = {
        "schema_version": SCHEMA,
        "status": "FROZEN_BEFORE_DIRECT_BATCH_VLLM_INFERENCE",
        "task": task,
        "missing_vote_count": total,
        "continuations": continuations,
        "inputs": {
            "manifest": _ref(manifest_path),
            "canonical_bank": _ref(canonical_bank_path),
            "runner": _ref(runner),
            "prompts": [_ref(path) for path in prompts],
            "model_inventory_audit": _ref(model_audit_path),
            "runtime_dependency_audit": _ref(runtime_audit_path),
        },
        "prompt": {"combined_sha256": prompt_sha},
        "supersedes_prelaunch_plan": supersedes,
        "runtime": {
            "python": str(python), "runtime_root": str(runtime_root), "model": str(model),
            "gpu": args.gpu, "temperature": 0.0, "order_mode": "original",
            "batch_size": args.batch_size, "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens, "context_chars": args.context_chars,
            "description_chars": args.description_chars,
            "example_chars": args.example_chars, "max_examples": args.max_examples,
            "gpu_memory_utilization": args.gpu_memory_utilization, "keep_raw": True,
        },
        "contracts": {
            "truth_hidden": True,
            "all_88_bank_cards_present_in_frozen_pack_order_per_row": True,
            "canonical_norm_text_context_and_bank_cards_exactly_revalidated": True,
            "partial_vote_content_not_read_for_selection_or_settings": True,
            "no_prompt_or_runtime_retuning_from_partial_votes": True,
            "only_exact_missing_chunk_frontier_inferred": True,
            "raw_responses_and_item_prompt_hashes_required": True,
            "no_scoring_before_complete_independent_consensus": True,
        },
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "count": total, "output": str(plan_path), "sha256": sha256_file(plan_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
