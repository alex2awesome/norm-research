#!/usr/bin/env python3
"""Retrieve wide candidate slates for a frozen all-task calibration sample."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from .common import (
    metric_card,
    norm_query,
    norm_statement_query,
    read_jsonl,
    sha256_file,
    write_jsonl,
)
from .config import BGE_ENCODER, DEFAULT_OUTPUT_ROOT
from .retrieve import (
    build_vectorizers,
    ranks_for,
    reciprocal_rank_union,
    top_indices,
    weighted_reciprocal_rank_union,
)


def uses_nemotron_query_format(
    requested: str, encoder: str, adapter: str | None
) -> bool:
    if requested not in {"auto", "raw", "nemotron"}:
        raise ValueError(f"unknown query format: {requested}")
    return requested == "nemotron" or (
        requested == "auto" and (bool(adapter) or "nemotron" in encoder.lower())
    )


def reserve_frozen_test(
    marker_path: Path,
    *,
    output_path: Path,
    manifest_path: Path,
    items_path: Path,
    selection_record_path: Path,
    adapter_path: Path | None,
    fusion_path: Path | None = None,
) -> dict:
    """Atomically consume a frozen-test run before model inference begins."""
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    completed_path = marker_path.with_name(marker_path.stem + ".completed.json")
    if marker_path.exists() or completed_path.exists():
        raise FileExistsError(f"frozen test was already consumed: {marker_path}")
    payload = {
        "status": "STARTED_TEST_INPUT_CONSUMED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "output": str(output_path),
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "items": {"path": str(items_path), "sha256": sha256_file(items_path)},
            "dev_selection_record": {
                "path": str(selection_record_path),
                "sha256": sha256_file(selection_record_path),
            },
            "adapter": (
                {
                    path.name: sha256_file(path)
                    for path in sorted(adapter_path.iterdir())
                    if path.is_file()
                }
                if adapter_path
                else None
            ),
            "fusion_weights": (
                {"path": str(fusion_path), "sha256": sha256_file(fusion_path)}
                if fusion_path
                else None
            ),
        },
    }
    with marker_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def hydrate_items(items: list[dict], manifest: dict) -> list[dict]:
    """Attach canonical norm/context fields when a teacher slate stores only UIDs."""
    missing = {row["norm_uid"] for row in items if not row.get("norm")}
    if not missing:
        return items
    canonical = {}
    for meta in manifest["corpora"].values():
        for row in read_jsonl(Path(meta["path"])):
            uid = row["norm_uid"]
            if uid in missing:
                canonical[uid] = row
        if len(canonical) == len(missing):
            break
    absent = sorted(missing - set(canonical))
    if absent:
        raise KeyError(f"calibration UIDs absent from manifest: {absent[:3]}")
    return [{**canonical.get(row["norm_uid"], {}), **row} for row in items]


def task_candidates(
    model,
    rows: list[dict],
    metrics: list[dict],
    *,
    component_k: int,
    output_k: int,
    query_formatter: Callable[[str], str] | None = None,
    multiview: bool = True,
    component_weights: dict[str, float] | None = None,
    rank_constant: float = 60.0,
) -> list[dict]:
    cards = [metric_card(metric) for metric in metrics]
    queries = [norm_query(row) for row in rows]
    statement_queries = [norm_statement_query(row) for row in rows]
    dense_queries = (
        [query_formatter(query) for query in queries]
        if query_formatter is not None
        else queries
    )
    dense_statement_queries = (
        [query_formatter(query) for query in statement_queries]
        if query_formatter is not None
        else statement_queries
    )
    metric_embeddings = np.asarray(
        model.encode(cards, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    )
    query_embeddings = np.asarray(
        model.encode(
            dense_queries,
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )
    statement_embeddings = (
        np.asarray(
            model.encode(
                dense_statement_queries,
                batch_size=256,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
        if multiview
        else query_embeddings
    )
    dense = query_embeddings @ metric_embeddings.T
    dense_statement = statement_embeddings @ metric_embeddings.T
    word, char, metric_word, metric_char = build_vectorizers(cards)
    word_scores = (word.transform(queries) @ metric_word.T).toarray()
    char_scores = (char.transform(queries) @ metric_char.T).toarray()
    word_statement_scores = (word.transform(statement_queries) @ metric_word.T).toarray()
    char_statement_scores = (char.transform(statement_queries) @ metric_char.T).toarray()
    k = min(component_k, len(metrics))
    dense_order, dense_statement_order, word_order, char_order, word_statement_order, char_statement_order = (
        top_indices(dense, k),
        top_indices(dense_statement, k),
        top_indices(word_scores, k),
        top_indices(char_scores, k),
        top_indices(word_statement_scores, k),
        top_indices(char_statement_scores, k),
    )
    output = []
    for i, row in enumerate(rows):
        lanes = (dense_order[i], word_order[i], char_order[i])
        if multiview:
            lanes = lanes + (
                dense_statement_order[i],
                word_statement_order[i],
                char_statement_order[i],
            )
        if component_weights is None:
            union = reciprocal_rank_union(
                lanes,
                rank_constant=rank_constant,
                limit=min(output_k, len(metrics)),
            )
        else:
            weighted_lanes = (
                (dense_order[i], component_weights["dense_rank"]),
                (
                    dense_statement_order[i],
                    component_weights["dense_statement_rank"] if multiview else 0.0,
                ),
                (word_order[i], component_weights["word_rank"]),
                (
                    word_statement_order[i],
                    component_weights["word_statement_rank"] if multiview else 0.0,
                ),
                (char_order[i], component_weights["char_rank"]),
                (
                    char_statement_order[i],
                    component_weights["char_statement_rank"] if multiview else 0.0,
                ),
            )
            union = weighted_reciprocal_rank_union(
                weighted_lanes,
                rank_constant=rank_constant,
                limit=min(output_k, len(metrics)),
            )
        rank_maps = {
            "dense": ranks_for(dense_order[i]),
            "dense_statement": ranks_for(dense_statement_order[i]),
            "word": ranks_for(word_order[i]),
            "char": ranks_for(char_order[i]),
            "word_statement": ranks_for(word_statement_order[i]),
            "char_statement": ranks_for(char_statement_order[i]),
        }
        candidates = []
        for rank, (metric_index, rrf_score) in enumerate(union, 1):
            candidates.append(
                {
                    "metric_id": metrics[metric_index]["metric_id"],
                    "metric_index": int(metric_index),
                    "rank": rank,
                    "rrf_score": float(rrf_score),
                    "dense_score": float(max(dense[i, metric_index], dense_statement[i, metric_index])),
                    "dense_evidence_score": float(dense[i, metric_index]),
                    "dense_statement_score": float(dense_statement[i, metric_index]),
                    "word_score": float(word_scores[i, metric_index]),
                    "char_score": float(char_scores[i, metric_index]),
                    "dense_rank": rank_maps["dense"].get(metric_index),
                    "dense_statement_rank": rank_maps["dense_statement"].get(metric_index),
                    "word_rank": rank_maps["word"].get(metric_index),
                    "char_rank": rank_maps["char"].get(metric_index),
                    "word_statement_rank": rank_maps["word_statement"].get(metric_index),
                    "char_statement_rank": rank_maps["char_statement"].get(metric_index),
                }
            )
        output.append(
            {
                "schema_version": row["schema_version"],
                "norm_uid": row["norm_uid"],
                "corpus": row["corpus"],
                "task": row["task"],
                "row": row["row"],
                "candidates": candidates,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifest.json"))
    parser.add_argument(
        "--items", default=str(DEFAULT_OUTPUT_ROOT / "alltask_calibration/items.jsonl")
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT_ROOT / "alltask_calibration/candidates_bge_base.jsonl")
    )
    parser.add_argument("--encoder", default=BGE_ENCODER)
    parser.add_argument("--adapter", help="optional task-specific Nemotron PEFT adapter")
    parser.add_argument(
        "--query-format",
        choices=("auto", "raw", "nemotron"),
        default="auto",
        help=(
            "Dense query template. auto applies Nemotron's documented instruction "
            "format when --adapter is present or the encoder path/name contains "
            "'nemotron'; metric cards remain unformatted."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", choices=("auto", "eager", "sdpa"), default="eager")
    parser.add_argument("--encoder-max-length", type=int, default=512)
    parser.add_argument("--single-view", action="store_true", help="disable statement/evidence RRF")
    parser.add_argument("--component-k", type=int, default=180)
    parser.add_argument("--output-k", type=int, default=180)
    parser.add_argument(
        "--fusion-weights",
        help="dev-selected optimize_retrieval_fusion report; never fit weights on test",
    )
    parser.add_argument("--rrf-constant", type=float, default=60.0)
    parser.add_argument(
        "--frozen-test-marker",
        help=(
            "Atomically mark this invocation as the one-shot frozen-test run. "
            "Requires --selection-record and refuses any prior marker."
        ),
    )
    parser.add_argument(
        "--selection-record",
        help="Dev-only frozen adapter/fusion selection artifact for a test invocation.",
    )
    args = parser.parse_args()

    manifest_path, items_path = Path(args.manifest).resolve(), Path(args.items).resolve()
    output_path = Path(args.output).resolve()
    output_meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or output_meta_path.exists():
        raise FileExistsError(f"refusing to overwrite retrieval output: {output_path}")
    if bool(args.frozen_test_marker) != bool(args.selection_record):
        raise ValueError("--frozen-test-marker and --selection-record must be supplied together")
    marker_path = Path(args.frozen_test_marker).resolve() if args.frozen_test_marker else None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.rrf_constant <= 0:
        raise ValueError("--rrf-constant must be positive")
    fusion_path = Path(args.fusion_weights).resolve() if args.fusion_weights else None
    component_weights = None
    if fusion_path:
        fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
        component_weights = (fusion.get("selected") or {}).get("component_weights")
        required = {
            "dense_rank",
            "dense_statement_rank",
            "word_rank",
            "word_statement_rank",
            "char_rank",
            "char_statement_rank",
        }
        if not isinstance(component_weights, dict) or set(component_weights) != required:
            raise ValueError(f"invalid component weights in {fusion_path}")
        if fusion.get("selection_split") != "dev":
            raise ValueError(f"fusion weights were not selected on dev: {fusion_path}")
        values = [float(component_weights[key]) for key in sorted(required)]
        if any(not np.isfinite(value) or value < 0 for value in values) or not any(
            value > 0 for value in values
        ):
            raise ValueError(f"fusion weights must be finite, nonnegative, and nonzero: {fusion_path}")
    if marker_path:
        selection_path = Path(args.selection_record).resolve()
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("selection_split") != "external_dev_only":
            raise ValueError("frozen test requires an external-dev-only selection record")
        chosen = selection.get("chosen") or {}
        chosen_fusion = chosen.get("fusion_report")
        if fusion_path is None or Path(str(chosen_fusion)).resolve() != fusion_path:
            raise ValueError("test fusion artifact differs from the dev-selected variant")
        if bool(args.adapter) != (chosen.get("kind") == "adapter"):
            raise ValueError("test adapter presence differs from the dev-selected variant")
        if selection.get("frozen_test_consumed") is not False:
            raise ValueError("selection record does not declare an unconsumed frozen test")
    # Consume the one-shot test only after every static input and selected
    # fusion artifact has passed validation, but before loading a model or
    # reading any test labels into retrieval computation.
    if marker_path:
        reserve_frozen_test(
            marker_path,
            output_path=output_path,
            manifest_path=manifest_path,
            items_path=items_path,
            selection_record_path=Path(args.selection_record).resolve(),
            adapter_path=Path(args.adapter).resolve() if args.adapter else None,
            fusion_path=fusion_path,
        )
    by_task: dict[str, list[dict]] = defaultdict(list)
    items = hydrate_items(list(read_jsonl(items_path)), manifest)
    for row in items:
        by_task[row["task"]].append(row)
    use_nemotron_format = uses_nemotron_query_format(
        args.query_format, str(args.encoder), args.adapter
    )
    # Dynamic ``trust_remote_code`` modules must be compiled into a writable
    # cache.  sk3's shared HF cache and account HOME are intentionally
    # read-only in batch jobs, so pin only the generated module cache locally.
    os.environ.setdefault(
        "HF_MODULES_CACHE", "/lfs/skampere3/0/alexspan/runtime_home/hf_modules"
    )
    Path(os.environ["HF_MODULES_CACHE"]).mkdir(parents=True, exist_ok=True)
    query_formatter = None
    if args.adapter:
        from .train_nemotron_lora import format_query, load_nemotron_adapter

        model = load_nemotron_adapter(
            args.encoder,
            args.adapter,
            device=args.device,
            attention=args.attention,
            max_seq_length=args.encoder_max_length,
        )
    else:
        from sentence_transformers import SentenceTransformer

        # Nemotron's cached SentenceTransformers snapshot contains its pooling
        # implementation as repository code.  Base-model evaluation must load
        # the same implementation and left-padding convention as adapter
        # evaluation; otherwise the supposedly frozen baseline cannot even be
        # instantiated (or can silently differ from the trainer).
        is_nemotron = "nemotron" in str(args.encoder).lower()
        kwargs = {"trust_remote_code": True} if is_nemotron else {}
        if is_nemotron:
            import torch

            kwargs["tokenizer_kwargs"] = {"padding_side": "left"}
            kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
            if args.attention != "auto":
                kwargs["model_kwargs"]["attn_implementation"] = args.attention
        model = SentenceTransformer(args.encoder, device=args.device, **kwargs)
        model.max_seq_length = args.encoder_max_length
    if use_nemotron_format:
        from .train_nemotron_lora import format_query

        query_formatter = format_query
    output = []
    for task, rows in sorted(by_task.items()):
        bank_path = Path(manifest["banks"][task]["path"])
        bank = json.loads(bank_path.read_text(encoding="utf-8"))["metrics"]
        task_rows = task_candidates(
            model,
            rows,
            bank,
            component_k=args.component_k,
            output_k=args.output_k,
            query_formatter=query_formatter,
            multiview=not args.single_view,
            component_weights=component_weights,
            rank_constant=args.rrf_constant,
        )
        bank_sha = manifest["banks"][task]["source_sha256"]
        for row in task_rows:
            row["bank_source_sha256"] = bank_sha
        output.extend(task_rows)
        print(f"{task}: {len(rows)}", flush=True)
    output.sort(key=lambda row: (row["task"], row["corpus"], row["norm_uid"]))
    write_jsonl(output_path, output)
    meta = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "items": str(items_path),
        "items_sha256": sha256_file(items_path),
        "encoder": args.encoder,
        "adapter": args.adapter,
        "adapter_hashes": (
            {
                path.name: sha256_file(path)
                for path in sorted(Path(args.adapter).iterdir())
                if path.is_file()
            }
            if args.adapter
            else None
        ),
        "query_format": "nemotron" if use_nemotron_format else "raw",
        "dense_query_instruction": use_nemotron_format,
        "query_views": "single" if args.single_view else "evidence+statement",
        "component_k": args.component_k,
        "output_k": args.output_k,
        "rrf_constant": args.rrf_constant,
        "fusion_weights": str(fusion_path) if fusion_path else None,
        "fusion_weights_sha256": sha256_file(fusion_path) if fusion_path else None,
        "component_weights": component_weights,
        "frozen_test": (
            {
                "started_marker": str(marker_path),
                "started_marker_sha256": sha256_file(marker_path),
                "dev_selection_record": str(Path(args.selection_record).resolve()),
                "dev_selection_record_sha256": sha256_file(
                    Path(args.selection_record).resolve()
                ),
            }
            if marker_path
            else None
        ),
        "count": len(output),
        "output_sha256": sha256_file(output_path),
    }
    output_meta_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if marker_path:
        completed_path = marker_path.with_name(marker_path.stem + ".completed.json")
        with completed_path.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "status": "COMPLETED",
                    "started_marker": str(marker_path),
                    "started_marker_sha256": sha256_file(marker_path),
                    "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
                    "meta": {"path": str(output_meta_path), "sha256": sha256_file(output_meta_path)},
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
    print(json.dumps(meta, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
