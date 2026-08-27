#!/usr/bin/env python3
"""Direct K-nearest-metric retrieval with a promoted Nemotron LoRA.

This is the production counterpart of ``evaluate_nemotron_adapter.py``.  It
uses the identical single evidence-query view, query instruction, metric-card
rendering, normalized embeddings, dot product, and stable metric-index tie
break used by the sealed external-dev gate.  No labels, lexical lane, fusion
weights, reranker, or test artifact are opened.

The output is append/resume safe.  A metadata sidecar is written only after
every canonical norm has exactly one K-deep row, so partial output can never be
mistaken for a sealed production retrieval.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import metric_card, norm_query, read_jsonl, sha256_file
from .retrieve import append_rows, existing_uids, top_indices
from .train_nemotron_lora import _encode, format_query, load_nemotron_adapter


def _resolve(raw: str | Path, anchor: Path) -> Path:
    value = Path(raw)
    return value if value.is_absolute() else (anchor.parent / value).resolve()


def _file_artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _adapter_hashes(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    hashes = {
        value.name: sha256_file(value)
        for value in sorted(path.iterdir())
        if value.is_file()
    }
    required = {"README.md", "adapter_config.json", "adapter_model.safetensors"}
    if set(hashes) != required:
        raise ValueError(f"adapter files differ from the sealed LoRA triplet: {path}")
    return hashes


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    adapter_path = Path(args.adapter).resolve()
    encoder_path = Path(args.encoder).resolve()
    selection_path = Path(args.selection).resolve()
    model_inventory_path = Path(args.model_inventory).resolve()
    if not encoder_path.is_dir():
        raise FileNotFoundError(encoder_path)
    for path in (manifest_path, selection_path, model_inventory_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.corpus not in manifest.get("corpora", {}):
        raise KeyError(args.corpus)
    corpus_meta = manifest["corpora"][args.corpus]
    task = str(corpus_meta["task"])
    if task != args.task:
        raise ValueError(f"corpus/task mismatch: {args.corpus}/{task} != {args.task}")
    bank_meta = manifest["banks"][task]
    corpus_path = _resolve(corpus_meta["path"], manifest_path)
    bank_path = _resolve(bank_meta["path"], manifest_path)
    norms = list(read_jsonl(corpus_path))
    metrics = json.loads(bank_path.read_text(encoding="utf-8"))["metrics"]
    if len(norms) != int(corpus_meta["count"]):
        raise ValueError("canonical norm count differs from manifest")
    if len(metrics) != int(bank_meta["count"]):
        raise ValueError("bank metric count differs from manifest")
    if not 1 <= args.output_k <= len(metrics):
        raise ValueError("output K is outside the task bank")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    chosen = selection.get("chosen") or {}
    if (
        selection.get("task") != task
        or selection.get("status") != "SELECTED_FOR_PRODUCTION_RETRIEVAL"
        or selection.get("selection_split") != "external_dev_only"
        or selection.get("frozen_external_test_consumed") is not False
        or chosen.get("kind") != "nemotron_lora_adapter"
        or chosen.get("retrieval_geometry")
        != "direct_dense_evidence_query_metric_card_nemotron_instruction_v1"
        or int(chosen.get("candidate_depth", -1)) != args.output_k
        or Path(str((chosen.get("adapter") or {}).get("path") or "")).resolve()
        != adapter_path
        or (chosen.get("adapter") or {}).get("files") != _adapter_hashes(adapter_path)
        or Path(str((chosen.get("base_model") or {}).get("path") or "")).resolve()
        != encoder_path
        or (chosen.get("base_model") or {}).get("inventory_sha256")
        != sha256_file(model_inventory_path)
    ):
        raise ValueError("retrieval inputs differ from the promoted selection")

    completed = existing_uids(output_path) if args.resume else set()
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --resume")
    canonical_uids = [str(row["norm_uid"]) for row in norms]
    if len(canonical_uids) != len(set(canonical_uids)):
        raise ValueError("canonical corpus has duplicate norm UIDs")
    if not completed <= set(canonical_uids):
        raise ValueError("partial output includes a UID outside the canonical corpus")
    pending = [row for row in norms if str(row["norm_uid"]) not in completed]

    # All cache paths are caller-frozen by the queue.  Keep trusted model code
    # offline and writable without touching the immutable model snapshot.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model = load_nemotron_adapter(
        encoder_path,
        adapter_path,
        device=args.device,
        attention=args.attention,
        max_seq_length=args.max_seq_length,
    )
    cards = [metric_card(metric) for metric in metrics]
    metric_ids = [str(metric["metric_id"]) for metric in metrics]
    metric_embeddings = _encode(model, cards, args.encoder_batch_size)

    started = time.time()
    written = 0
    for start in range(0, len(pending), args.query_batch_size):
        batch = pending[start : start + args.query_batch_size]
        queries = [format_query(norm_query(norm)) for norm in batch]
        query_embeddings = _encode(model, queries, args.encoder_batch_size)
        scores = np.asarray(query_embeddings) @ np.asarray(metric_embeddings).T
        ranked = top_indices(scores, args.output_k)
        rows = []
        for row_index, norm in enumerate(batch):
            candidates = [
                {
                    "metric_id": metric_ids[int(metric_index)],
                    "rank": rank,
                    "dense_score": float(scores[row_index, int(metric_index)]),
                    "dense_evidence_score": float(
                        scores[row_index, int(metric_index)]
                    ),
                    "retrieval_lane": "nemotron_lora_direct_dense",
                }
                for rank, metric_index in enumerate(ranked[row_index], 1)
            ]
            rows.append(
                {
                    "schema_version": manifest["schema_version"],
                    "norm_uid": str(norm["norm_uid"]),
                    "corpus": args.corpus,
                    "task": task,
                    "row": int(norm["row"]),
                    "bank_source_sha256": str(bank_meta["source_sha256"]),
                    "candidates": candidates,
                }
            )
        written += append_rows(output_path, rows)
        total_done = len(completed) + written
        elapsed = time.time() - started
        print(
            json.dumps(
                {
                    "corpus": args.corpus,
                    "completed": total_done,
                    "expected": len(norms),
                    "newly_written": written,
                    "elapsed_seconds": round(elapsed, 3),
                    "rows_per_second": round(written / elapsed, 3)
                    if elapsed > 0
                    else None,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    observed_uids: set[str] = set()
    observed_count = 0
    duplicate_uids = 0
    for row in read_jsonl(output_path):
        uid = str(row.get("norm_uid") or "")
        duplicate_uids += int(uid in observed_uids)
        observed_uids.add(uid)
        observed_count += 1
    if (
        observed_count != len(norms)
        or duplicate_uids
        or observed_uids != set(canonical_uids)
    ):
        raise ValueError("completed retrieval does not exactly cover the canonical corpus")

    elapsed = time.time() - started
    meta = {
        "schema_version": manifest["schema_version"],
        "generator_schema_version": "silver-match-v3-nemotron-direct-k50-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "corpus": args.corpus,
        "task": task,
        "input_count": int(corpus_meta["count"]),
        "new_count": written,
        "observed_count": observed_count,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "output_k": args.output_k,
        "bank_source_sha256": str(bank_meta["source_sha256"]),
        "bank_artifact": _file_artifact(bank_path),
        "canonical_corpus": _file_artifact(corpus_path),
        "encoder": str(encoder_path),
        "model_inventory": _file_artifact(model_inventory_path),
        "adapter": str(adapter_path),
        "adapter_hashes": _adapter_hashes(adapter_path),
        "selection": _file_artifact(selection_path),
        "query_format": "nemotron",
        "dense_query_instruction": True,
        "query_views": "evidence",
        "retrieval_geometry": (
            "direct_dense_evidence_query_metric_card_nemotron_instruction_v1"
        ),
        "fusion_weights": None,
        "fusion_weights_sha256": None,
        "reranker": None,
        "attention": args.attention,
        "max_seq_length": args.max_seq_length,
        "query_batch_size": args.query_batch_size,
        "encoder_batch_size": args.encoder_batch_size,
        "elapsed_seconds": elapsed,
        "external_labels_opened": False,
        "external_test_consumed": False,
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if meta_path.exists():
        raise FileExistsError(f"refusing to overwrite sealed metadata: {meta_path}")
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, sort_keys=True), flush=True)
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--model-inventory", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-k", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", choices=("auto", "eager", "sdpa"), default="eager")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--query-batch-size", type=int, default=512)
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
