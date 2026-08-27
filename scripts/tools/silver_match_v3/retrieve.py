#!/usr/bin/env python3
"""Hybrid candidate retrieval over the frozen task-level metric banks.

This stage is deliberately label-free.  It combines dense BGE retrieval with
word- and character-level TF-IDF, then optionally applies the clean pretrained
BGE reranker.  It never uses the contaminated per-corpus cross-encoders.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import metric_card, norm_query, norm_statement_query, read_jsonl, sha256_file
from .config import BGE_ENCODER, BGE_RERANKER, DEFAULT_OUTPUT_ROOT


def stable_shard(norm_uid: str, num_shards: int) -> int:
    return int(norm_uid[:16], 16) % num_shards


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Deterministic descending top-k, breaking ties by metric index."""
    k = min(k, scores.shape[-1])
    # Stable mergesort retains ascending metric index for equal values.
    return np.argsort(-scores, kind="mergesort", axis=-1)[..., :k]


def reciprocal_rank_union(
    ranked_lists: Iterable[Iterable[int]],
    *,
    rank_constant: float = 60.0,
    limit: int | None = None,
) -> list[tuple[int, float]]:
    scores: defaultdict[int, float] = defaultdict(float)
    for ranking in ranked_lists:
        for rank, item in enumerate(ranking, 1):
            scores[int(item)] += 1.0 / (rank_constant + rank)
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return ordered if limit is None else ordered[:limit]


def weighted_reciprocal_rank_union(
    ranked_lists: Iterable[tuple[Iterable[int], float]],
    *,
    rank_constant: float = 60.0,
    limit: int | None = None,
) -> list[tuple[int, float]]:
    scores: defaultdict[int, float] = defaultdict(float)
    positive = 0
    for ranking, weight in ranked_lists:
        weight = float(weight)
        if weight < 0:
            raise ValueError("fusion weights must be nonnegative")
        if weight == 0:
            continue
        positive += 1
        for rank, item in enumerate(ranking, 1):
            scores[int(item)] += weight / (rank_constant + rank)
    if not positive:
        raise ValueError("at least one fusion weight must be positive")
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return ordered if limit is None else ordered[:limit]


def ranks_for(items: Iterable[int]) -> dict[int, int]:
    return {int(item): rank for rank, item in enumerate(items, 1)}


def existing_uids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["norm_uid"]) for row in read_jsonl(path)}


def uses_nemotron_query_format(
    requested: str, encoder: str, adapter: str | None
) -> bool:
    if requested not in {"auto", "raw", "nemotron"}:
        raise ValueError(f"unknown query format: {requested}")
    return requested == "nemotron" or (
        requested == "auto" and (bool(adapter) or "nemotron" in encoder.lower())
    )


def load_inputs(manifest_path: Path, corpus: str):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corpus not in manifest["corpora"]:
        raise KeyError(f"unknown corpus {corpus!r}")
    corpus_meta = manifest["corpora"][corpus]
    task = corpus_meta["task"]
    bank_meta = manifest["banks"][task]
    bank_payload = json.loads(Path(bank_meta["path"]).read_text(encoding="utf-8"))
    metrics = bank_payload["metrics"]
    norms = list(read_jsonl(Path(corpus_meta["path"])))
    return manifest, corpus_meta, bank_meta, task, metrics, norms


def build_vectorizers(cards: list[str]):
    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
    )
    char = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
    )
    return word, char, word.fit_transform(cards), char.fit_transform(cards)


def append_rows(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Non-interactive sk3 jobs inherit an AFS HOME that is intentionally
    # read-only.  Nemotron uses trusted remote code, so transformers needs a
    # writable dynamic-module cache even when the model snapshot itself is
    # fully local.  Set these before importing sentence-transformers below.
    os.environ.setdefault("HF_HOME", str(args.cache_dir))
    os.environ.setdefault(
        "HF_MODULES_CACHE", "/lfs/skampere3/0/alexspan/hf_modules_cache"
    )
    os.environ.setdefault("XDG_CACHE_HOME", "/lfs/skampere3/0/alexspan/.cache")

    manifest_path = Path(args.manifest)
    manifest, corpus_meta, bank_meta, task, metrics, norms = load_inputs(
        manifest_path, args.corpus
    )
    out_path = Path(args.output) if args.output else (
        manifest_path.parent
        / "candidates"
        / f"{args.corpus}.shard-{args.shard_id:03d}-of-{args.num_shards:03d}.jsonl"
    )
    completed = existing_uids(out_path) if args.resume else set()
    if out_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {out_path}; pass --resume")

    allowed_uids = None
    if args.uid_file:
        allowed_uids = {
            line.strip().split("\t", 1)[0]
            for line in Path(args.uid_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    norms = [
        row
        for row in norms
        if stable_shard(row["norm_uid"], args.num_shards) == args.shard_id
        and row["norm_uid"] not in completed
        and (allowed_uids is None or row["norm_uid"] in allowed_uids)
    ]
    if args.limit:
        norms = sorted(norms, key=lambda row: row["norm_uid"])[: args.limit]
    cards = [metric_card(metric) for metric in metrics]
    queries = [norm_query(norm) for norm in norms]
    statement_queries = [norm_statement_query(norm) for norm in norms]

    from sentence_transformers import CrossEncoder, SentenceTransformer

    use_nemotron_format = uses_nemotron_query_format(
        args.query_format, str(args.encoder), args.adapter
    )
    if args.adapter:
        from .train_nemotron_lora import format_query, load_nemotron_adapter

        encoder = load_nemotron_adapter(
            args.encoder,
            args.adapter,
            device=args.device,
            attention=args.attention,
            max_seq_length=args.encoder_max_length,
        )
        dense_queries = [format_query(query) for query in queries]
        dense_statement_queries = [format_query(query) for query in statement_queries]
    elif use_nemotron_format:
        import torch
        from .train_nemotron_lora import format_query

        model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        if args.attention != "auto":
            model_kwargs["attn_implementation"] = args.attention
        encoder = SentenceTransformer(
            args.encoder,
            cache_folder=args.cache_dir,
            device=args.device,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"padding_side": "left"},
        )
        encoder.max_seq_length = args.encoder_max_length
        dense_queries = [format_query(query) for query in queries]
        dense_statement_queries = [format_query(query) for query in statement_queries]
    else:
        encoder = SentenceTransformer(
            args.encoder, cache_folder=args.cache_dir, device=args.device
        )
        dense_queries = queries
        dense_statement_queries = statement_queries
    metric_embeddings = encoder.encode(
        cards,
        batch_size=args.metric_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    word, char, metric_word, metric_char = build_vectorizers(cards)
    fusion_path = Path(args.fusion_weights) if args.fusion_weights else None
    fusion_weights = None
    if fusion_path:
        payload = json.loads(fusion_path.read_text(encoding="utf-8"))
        fusion_weights = (payload.get("selected") or {}).get("component_weights")
        required = {
            "dense_rank",
            "dense_statement_rank",
            "word_rank",
            "word_statement_rank",
            "char_rank",
            "char_statement_rank",
        }
        if not isinstance(fusion_weights, dict) or set(fusion_weights) != required:
            raise ValueError(f"invalid task fusion weights in {fusion_path}")
    reranker = None
    if not args.no_reranker:
        reranker = CrossEncoder(
            args.reranker,
            num_labels=1,
            max_length=args.reranker_max_length,
            cache_folder=args.cache_dir,
        )

    started = time.time()
    written = 0
    for start in range(0, len(norms), args.query_batch_size):
        batch_norms = norms[start : start + args.query_batch_size]
        batch_queries = queries[start : start + args.query_batch_size]
        batch_dense_queries = dense_queries[start : start + args.query_batch_size]
        batch_statement_queries = statement_queries[start : start + args.query_batch_size]
        batch_dense_statement_queries = dense_statement_queries[
            start : start + args.query_batch_size
        ]
        dense_embeddings = encoder.encode(
            batch_dense_queries,
            batch_size=args.encoder_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        dense_scores = np.asarray(dense_embeddings) @ np.asarray(metric_embeddings).T
        if args.single_view:
            dense_statement_scores = dense_scores
        else:
            dense_statement_embeddings = encoder.encode(
                batch_dense_statement_queries,
                batch_size=args.encoder_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            dense_statement_scores = (
                np.asarray(dense_statement_embeddings) @ np.asarray(metric_embeddings).T
            )
        word_scores = (word.transform(batch_queries) @ metric_word.T).toarray()
        char_scores = (char.transform(batch_queries) @ metric_char.T).toarray()
        word_statement_scores = (
            word.transform(batch_statement_queries) @ metric_word.T
        ).toarray()
        char_statement_scores = (
            char.transform(batch_statement_queries) @ metric_char.T
        ).toarray()

        dense_ranked = top_indices(dense_scores, args.dense_k)
        dense_statement_ranked = top_indices(dense_statement_scores, args.dense_k)
        word_ranked = top_indices(word_scores, args.word_k)
        char_ranked = top_indices(char_scores, args.char_k)
        word_statement_ranked = top_indices(word_statement_scores, args.word_k)
        char_statement_ranked = top_indices(char_statement_scores, args.char_k)
        pools: list[list[int]] = []
        rrf_scores: list[dict[int, float]] = []
        component_ranks: list[dict[str, dict[int, int]]] = []
        for i in range(len(batch_norms)):
            if fusion_weights is None:
                lanes = (dense_ranked[i], word_ranked[i], char_ranked[i])
                if not args.single_view:
                    lanes = lanes + (
                        dense_statement_ranked[i],
                        word_statement_ranked[i],
                        char_statement_ranked[i],
                    )
                rrf = reciprocal_rank_union(
                    lanes,
                    rank_constant=args.rrf_constant,
                    limit=args.pre_rerank_k,
                )
            else:
                rrf = weighted_reciprocal_rank_union(
                    (
                        (dense_ranked[i], fusion_weights["dense_rank"]),
                        (
                            dense_statement_ranked[i],
                            fusion_weights["dense_statement_rank"],
                        ),
                        (word_ranked[i], fusion_weights["word_rank"]),
                        (
                            word_statement_ranked[i],
                            fusion_weights["word_statement_rank"],
                        ),
                        (char_ranked[i], fusion_weights["char_rank"]),
                        (
                            char_statement_ranked[i],
                            fusion_weights["char_statement_rank"],
                        ),
                    ),
                    rank_constant=args.rrf_constant,
                    limit=args.pre_rerank_k,
                )
            pools.append([item for item, _ in rrf])
            rrf_scores.append(dict(rrf))
            component_ranks.append(
                {
                    "dense": ranks_for(dense_ranked[i]),
                    "dense_statement": ranks_for(dense_statement_ranked[i]),
                    "word": ranks_for(word_ranked[i]),
                    "char": ranks_for(char_ranked[i]),
                    "word_statement": ranks_for(word_statement_ranked[i]),
                    "char_statement": ranks_for(char_statement_ranked[i]),
                }
            )

        rerank_scores: list[np.ndarray | None] = [None] * len(batch_norms)
        if reranker is not None:
            pairs = [
                [batch_queries[i], cards[metric_idx]]
                for i, pool in enumerate(pools)
                for metric_idx in pool
            ]
            raw = np.asarray(
                reranker.predict(
                    pairs,
                    batch_size=args.reranker_batch_size,
                    show_progress_bar=False,
                )
            ).reshape(-1)
            cursor = 0
            for i, pool in enumerate(pools):
                rerank_scores[i] = raw[cursor : cursor + len(pool)]
                cursor += len(pool)

        rows = []
        for i, norm in enumerate(batch_norms):
            pool = pools[i]
            if rerank_scores[i] is None:
                order = sorted(
                    range(len(pool)),
                    key=lambda j: (-rrf_scores[i][pool[j]], pool[j]),
                )
            else:
                order = sorted(
                    range(len(pool)),
                    key=lambda j: (-float(rerank_scores[i][j]), pool[j]),
                )
            order = order[: min(args.output_k, len(order))]
            candidates = []
            for final_rank, pool_pos in enumerate(order, 1):
                metric_idx = pool[pool_pos]
                ranks = component_ranks[i]
                candidates.append(
                    {
                        "metric_id": metrics[metric_idx]["metric_id"],
                        "rank": final_rank,
                        "reranker_score": (
                            None
                            if rerank_scores[i] is None
                            else float(rerank_scores[i][pool_pos])
                        ),
                        "rrf_score": float(rrf_scores[i][metric_idx]),
                        "dense_score": float(
                            max(
                                dense_scores[i, metric_idx],
                                dense_statement_scores[i, metric_idx],
                            )
                        ),
                        "dense_evidence_score": float(dense_scores[i, metric_idx]),
                        "dense_statement_score": float(
                            dense_statement_scores[i, metric_idx]
                        ),
                        "word_score": float(word_scores[i, metric_idx]),
                        "char_score": float(char_scores[i, metric_idx]),
                        "dense_rank": ranks["dense"].get(metric_idx),
                        "dense_statement_rank": ranks["dense_statement"].get(metric_idx),
                        "word_rank": ranks["word"].get(metric_idx),
                        "char_rank": ranks["char"].get(metric_idx),
                        "word_statement_rank": ranks["word_statement"].get(metric_idx),
                        "char_statement_rank": ranks["char_statement"].get(metric_idx),
                    }
                )
            rows.append(
                {
                    "schema_version": manifest["schema_version"],
                    "norm_uid": norm["norm_uid"],
                    "corpus": args.corpus,
                    "task": task,
                    "row": norm["row"],
                    "bank_source_sha256": bank_meta["source_sha256"],
                    "candidates": candidates,
                }
            )
        written += append_rows(out_path, rows)
        print(
            f"[{args.corpus}] {start + len(batch_norms)}/{len(norms)} "
            f"written={written} elapsed={time.time() - started:.0f}s",
            flush=True,
        )

    meta = {
        "schema_version": manifest["schema_version"],
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "corpus": args.corpus,
        "task": task,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "input_count": corpus_meta["count"],
        "new_count": written,
        "output_path": str(out_path),
        "output_sha256": sha256_file(out_path) if out_path.exists() else None,
        "bank_source_sha256": bank_meta["source_sha256"],
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
        "fusion_weights": str(fusion_path) if fusion_path else None,
        "fusion_weights_sha256": sha256_file(fusion_path) if fusion_path else None,
        "component_weights": fusion_weights,
        "reranker": None if args.no_reranker else args.reranker,
        "dense_k": args.dense_k,
        "word_k": args.word_k,
        "char_k": args.char_k,
        "pre_rerank_k": args.pre_rerank_k,
        "output_k": args.output_k,
        "elapsed_seconds": time.time() - started,
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument(
        "--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifest.json")
    )
    parser.add_argument("--output")
    parser.add_argument("--encoder", default=BGE_ENCODER)
    parser.add_argument("--adapter", help="optional task-specific Nemotron PEFT adapter")
    parser.add_argument(
        "--query-format",
        choices=("auto", "raw", "nemotron"),
        default="auto",
        help="apply Nemotron's documented query-only instruction format when appropriate",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", choices=("auto", "eager", "sdpa"), default="eager")
    parser.add_argument("--encoder-max-length", type=int, default=512)
    parser.add_argument("--single-view", action="store_true", help="disable statement/evidence RRF")
    parser.add_argument("--fusion-weights", help="task-specific dev-selected fusion report JSON")
    parser.add_argument("--reranker", default=BGE_RERANKER)
    parser.add_argument("--cache-dir", default="/lfs/skampere3/0/shared_hf_cache")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--dense-k", type=int, default=50)
    parser.add_argument("--word-k", type=int, default=30)
    parser.add_argument("--char-k", type=int, default=20)
    parser.add_argument("--pre-rerank-k", type=int, default=40)
    parser.add_argument("--output-k", type=int, default=16)
    parser.add_argument("--rrf-constant", type=float, default=60.0)
    parser.add_argument("--query-batch-size", type=int, default=2048)
    parser.add_argument("--encoder-batch-size", type=int, default=256)
    parser.add_argument("--metric-batch-size", type=int, default=128)
    parser.add_argument("--reranker-batch-size", type=int, default=512)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--uid-file", help="optional newline/TSV list of norm_uids to retrieve")
    parser.add_argument("--limit", type=int, default=0, help="deterministic UID-sorted debug limit")
    args = parser.parse_args()
    if not (0 <= args.shard_id < args.num_shards):
        parser.error("--shard-id must be in [0, --num-shards)")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
