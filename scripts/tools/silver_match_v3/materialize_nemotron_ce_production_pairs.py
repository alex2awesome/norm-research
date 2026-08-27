#!/usr/bin/env python3
"""Materialize unlabeled task-wide candidate pairs for Nemotron CE scoring.

The training pair builder deliberately requires truth.  Production inference
must not manufacture truth labels merely to reuse that builder, so this module
joins canonical norms, the frozen metric bank, and one audited diverse
retrieval union per corpus.  It emits exactly one unlabeled pair for every
retrieved ``(norm_uid, metric_id)`` and a one-row-per-norm universe for the
two-seed consensus aggregator.

All joins stream in canonical manifest order.  Candidate artifacts must be
complete, hash-bound outputs of at least two complete-bank retrieval lanes;
single-lane or diagnostic-subset artifacts fail closed.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator, Mapping, TextIO

from .common import (
    metric_card,
    norm_query,
    normalize_space,
    sha256_file,
)
from .train_nemotron_lora import source_group_key


PAIR_SCHEMA = "silver-match-v3-nemotron-ce-production-pair-v1"
UNIVERSE_SCHEMA = "silver-match-v3-nemotron-ce-production-universe-v1"
META_SCHEMA = "silver-match-v3-nemotron-ce-production-pairs-meta-v1"


def _resolve(raw: Any, anchor: Path) -> Path:
    value = Path(str(raw or ""))
    if not str(value):
        raise ValueError("empty artifact path")
    return value.resolve() if value.is_absolute() else (anchor.parent / value).resolve()


def _parse_candidates(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        corpus, separator, raw_path = value.partition("=")
        corpus = normalize_space(corpus)
        if not separator or not corpus or not raw_path or corpus in parsed:
            raise ValueError(f"invalid/duplicate --candidate binding: {value!r}")
        parsed[corpus] = Path(raw_path).resolve()
    if not parsed:
        raise ValueError("at least one --candidate CORPUS=PATH binding is required")
    return parsed


def _meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def _iter_handle(handle: TextIO, source: Path) -> Iterator[dict[str, Any]]:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            raise ValueError(f"blank JSONL row: {source}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL row: {source}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSONL row: {source}:{line_number}")
        yield row


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _load_scope(
    manifest_path: Path, task: str
) -> tuple[dict[str, Any], Path, str, dict[str, dict[str, Any]], list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    banks = manifest.get("banks") or {}
    corpora = manifest.get("corpora") or {}
    bank_meta = banks.get(task)
    if not isinstance(bank_meta, dict):
        raise ValueError(f"task is absent from manifest bank routing: {task}")
    bank_path = _resolve(bank_meta.get("path"), manifest_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    metrics = bank.get("metrics") or []
    bank_hash = normalize_space(bank_meta.get("source_sha256"))
    metric_ids = [normalize_space(row.get("metric_id")) for row in metrics]
    if (
        not bank_hash
        or not metric_ids
        or "" in metric_ids
        or len(metric_ids) != len(set(metric_ids))
        or len(metric_ids) != int(bank_meta.get("count", -1))
        or normalize_space(bank.get("source_sha256")) != bank_hash
    ):
        raise ValueError(f"manifest-derived bank contract failed: {task}")
    metric_by_id = dict(zip(metric_ids, metrics, strict=True))
    ordered_corpora = [
        corpus
        for corpus, meta in corpora.items()
        if isinstance(meta, dict) and meta.get("task") == task
    ]
    if not ordered_corpora:
        raise ValueError(f"task has no manifest corpora: {task}")
    return manifest, bank_path, bank_hash, metric_by_id, ordered_corpora


def _validate_candidate_meta(
    *,
    path: Path,
    manifest_path: Path,
    corpus: str,
    task: str,
    bank_hash: str,
    expected_count: int,
    expected_k: int,
) -> dict[str, Any]:
    meta_path = _meta_path(path)
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"candidate union or metadata is missing: {path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lanes = ((meta.get("union") or {}).get("lanes") or [])
    if (
        meta.get("output_sha256") != sha256_file(path)
        or meta.get("manifest_sha256") != sha256_file(manifest_path)
        or normalize_space(meta.get("corpus")) != corpus
        or normalize_space(meta.get("task")) != task
        or normalize_space(meta.get("bank_source_sha256")) != bank_hash
        or int(meta.get("input_count", -1)) != expected_count
        or int(meta.get("output_k", -1)) < expected_k
        or not isinstance(lanes, list)
        or len(lanes) < 2
    ):
        raise ValueError(f"candidate union contract failed: {corpus}")
    lane_names = [normalize_space(row.get("name")) for row in lanes if isinstance(row, dict)]
    if len(lane_names) != len(lanes) or "" in lane_names or len(set(lane_names)) != len(lanes):
        raise ValueError(f"candidate union has invalid lane identities: {corpus}")
    complete_bank_lane_names = [
        normalize_space(row.get("name"))
        for row in lanes
        if isinstance(row, dict) and normalize_space(row.get("kind")) == "complete-bank"
    ]
    if len(complete_bank_lane_names) < 2:
        raise ValueError(
            f"candidate union lacks two complete-bank lanes: {corpus}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "meta": str(meta_path),
        "meta_sha256": sha256_file(meta_path),
        "lane_names": lane_names,
        "complete_bank_lane_names": complete_bank_lane_names,
        "output_k": int(meta["output_k"]),
    }


def materialize(
    *,
    manifest_path: Path,
    task: str,
    candidates: Mapping[str, Path],
    output_path: Path,
    universe_path: Path,
    expected_k: int = 50,
    context_chars: int = 1400,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    output_path = output_path.resolve()
    universe_path = universe_path.resolve()
    meta_path = _meta_path(output_path)
    if expected_k < 1 or context_chars < 1:
        raise ValueError("expected_k and context_chars must be positive")
    if len({output_path, universe_path, meta_path}) != 3:
        raise ValueError("pair, universe, and metadata paths must be distinct")
    if any(path.exists() for path in (output_path, universe_path, meta_path)):
        raise FileExistsError("refusing to overwrite production pair artifacts")

    manifest, bank_path, bank_hash, metric_by_id, ordered_corpora = _load_scope(
        manifest_path, task
    )
    if set(candidates) != set(ordered_corpora):
        missing = sorted(set(ordered_corpora) - set(candidates))
        extra = sorted(set(candidates) - set(ordered_corpora))
        raise ValueError(f"candidate corpus bindings differ: missing={missing}, extra={extra}")
    if expected_k > len(metric_by_id):
        raise ValueError("expected_k exceeds the task bank size")
    bank_ids = frozenset(metric_by_id)
    metric_cards = {
        metric_id: metric_card(metric) for metric_id, metric in metric_by_id.items()
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    pair_temp = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    universe_temp = universe_path.with_suffix(universe_path.suffix + f".tmp.{os.getpid()}")
    pair_count = norm_count = 0
    corpus_reports: dict[str, Any] = {}
    seen_uids: set[str] = set()
    try:
        with pair_temp.open("x", encoding="utf-8") as pair_handle, universe_temp.open(
            "x", encoding="utf-8"
        ) as universe_handle:
            for corpus in ordered_corpora:
                corpus_meta = manifest["corpora"][corpus]
                canonical_path = _resolve(corpus_meta.get("path"), manifest_path)
                expected_count = int(corpus_meta.get("count", -1))
                candidate_path = candidates[corpus].resolve()
                candidate_ref = _validate_candidate_meta(
                    path=candidate_path,
                    manifest_path=manifest_path,
                    corpus=corpus,
                    task=task,
                    bank_hash=bank_hash,
                    expected_count=expected_count,
                    expected_k=expected_k,
                )
                source_k = int(candidate_ref["output_k"])
                if source_k > len(metric_by_id):
                    raise ValueError(f"candidate depth exceeds bank: {corpus}/{source_k}")
                observed = 0
                with canonical_path.open("r", encoding="utf-8") as canonical_handle, candidate_path.open(
                    "r", encoding="utf-8"
                ) as candidate_handle:
                    sentinel = object()
                    bundles = zip_longest(
                        _iter_handle(canonical_handle, canonical_path),
                        _iter_handle(candidate_handle, candidate_path),
                        fillvalue=sentinel,
                    )
                    for canonical, candidate in bundles:
                        if canonical is sentinel or candidate is sentinel:
                            raise ValueError(f"canonical/candidate length mismatch: {corpus}")
                        uid = normalize_space(canonical.get("norm_uid"))
                        canonical_row = int(canonical.get("row", observed))
                        if (
                            not uid
                            or uid in seen_uids
                            or canonical.get("task") != task
                            or canonical.get("corpus") != corpus
                            or normalize_space(candidate.get("norm_uid")) != uid
                            or candidate.get("task") != task
                            or candidate.get("corpus") != corpus
                            or int(candidate.get("row", -1)) != canonical_row
                            or normalize_space(candidate.get("bank_source_sha256")) != bank_hash
                        ):
                            raise ValueError(f"canonical/candidate routing mismatch: {corpus}/{uid}")
                        seen_uids.add(uid)
                        group = source_group_key(canonical)
                        query = norm_query(canonical, context_chars=context_chars)
                        if not group or not normalize_space(query):
                            raise ValueError(f"canonical norm is not renderable: {corpus}/{uid}")
                        values = candidate.get("candidates")
                        if not isinstance(values, list) or len(values) != source_k:
                            raise ValueError(f"candidate depth mismatch: {corpus}/{uid}")
                        all_ids = [normalize_space(value.get("metric_id")) for value in values]
                        ranks = [int(value.get("rank", -1)) for value in values]
                        if (
                            "" in all_ids
                            or len(all_ids) != len(set(all_ids))
                            or not set(all_ids) <= bank_ids
                            or ranks != list(range(1, source_k + 1))
                        ):
                            raise ValueError(f"candidate bank/rank contract failed: {corpus}/{uid}")
                        ids = all_ids[:expected_k]
                        universe_handle.write(
                            json.dumps(
                                {
                                    "schema_version": UNIVERSE_SCHEMA,
                                    "task": task,
                                    "corpus": corpus,
                                    "norm_uid": uid,
                                    "source_group": group,
                                    "split": "production",
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        for rank, metric_id in enumerate(ids, 1):
                            pair_handle.write(
                                json.dumps(
                                    {
                                        "schema_version": PAIR_SCHEMA,
                                        "task": task,
                                        "corpus": corpus,
                                        "norm_uid": uid,
                                        "source_group": group,
                                        "split": "production",
                                        "query": query,
                                        "metric_id": metric_id,
                                        "metric_card": metric_cards[metric_id],
                                        "candidate_rank": rank,
                                        "candidate_union_sha256": candidate_ref["sha256"],
                                        "current_bank_source_sha256": bank_hash,
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            pair_count += 1
                        observed += 1
                        norm_count += 1
                if observed != expected_count:
                    raise ValueError(
                        f"canonical corpus count mismatch: {corpus}/{observed}/{expected_count}"
                    )
                corpus_reports[corpus] = {
                    "canonical": {
                        "path": str(canonical_path),
                        "sha256": sha256_file(canonical_path),
                        "count": observed,
                    },
                    "candidate_union": candidate_ref,
                    "pair_count": observed * expected_k,
                }
            for handle in (pair_handle, universe_handle):
                handle.flush()
                os.fsync(handle.fileno())
        pair_temp.replace(output_path)
        universe_temp.replace(universe_path)
    except BaseException:
        pair_temp.unlink(missing_ok=True)
        universe_temp.unlink(missing_ok=True)
        raise

    expected_norms = sum(int(manifest["corpora"][corpus]["count"]) for corpus in ordered_corpora)
    if norm_count != expected_norms or pair_count != expected_norms * expected_k:
        output_path.unlink(missing_ok=True)
        universe_path.unlink(missing_ok=True)
        raise AssertionError("task-wide production pair coverage drift")
    report = {
        "schema_version": META_SCHEMA,
        "status": "FROZEN_COMPLETE_UNLABELED_PRODUCTION_PAIR_UNIVERSE",
        "task": task,
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "bank": {
            "path": str(bank_path),
            "sha256": sha256_file(bank_path),
            "source_sha256": bank_hash,
            "metric_count": len(metric_by_id),
        },
        "corpus_order": ordered_corpora,
        "corpora": corpus_reports,
        "norm_count": norm_count,
        "candidate_depth": expected_k,
        "pair_count": pair_count,
        "pairs": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "norm_universe": {
            "path": str(universe_path),
            "sha256": sha256_file(universe_path),
            "count": norm_count,
        },
        "labels_present": False,
        "single_lane_candidates_accepted": False,
        "diagnostic_subset_accepted": False,
        "release_ready": False,
    }
    try:
        _atomic_json(meta_path, report)
    except BaseException:
        # Publish the pair/universe/meta set atomically from the caller's
        # perspective.  A metadata failure must not strand apparently usable
        # unhashed production inputs that then block a clean rerun.
        output_path.unlink(missing_ok=True)
        universe_path.unlink(missing_ok=True)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidate", action="append", required=True, help="CORPUS=PATH")
    parser.add_argument("--output", required=True)
    parser.add_argument("--norm-universe", required=True)
    parser.add_argument("--expected-k", type=int, default=50)
    parser.add_argument("--context-chars", type=int, default=1400)
    args = parser.parse_args()
    report = materialize(
        manifest_path=Path(args.manifest),
        task=args.task,
        candidates=_parse_candidates(args.candidate),
        output_path=Path(args.output),
        universe_path=Path(args.norm_universe),
        expected_k=args.expected_k,
        context_chars=args.context_chars,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
