#!/usr/bin/env python3
"""Materialize a deterministic top-K union of complete retrieval lanes.

Every input lane must contain one complete-bank ranking for every norm in the
canonical corpus, in canonical row order.  The output is a weighted reciprocal
rank fusion (RRF) of those complete rankings.  This is intentionally a CPU-only
projection: encoder inference happens once per lane in ``retrieve.py`` and the
union never calls a model or an OpenAI-compatible server.

The implementation streams all lanes in lockstep, writes atomically, and fails
closed on any routing, hash, coverage, ordering, or bank-universe mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator, TextIO

from .common import sha256_file


PRESERVABLE_COMPONENTS = frozenset(
    {
        "rank",
        "dense_rank",
        "dense_statement_rank",
        "word_rank",
        "word_statement_rank",
        "char_rank",
        "char_statement_rank",
    }
)


def _resolve(path: str | Path, anchor: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (anchor.parent / value).resolve()


def _meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def _parse_lane(value: str) -> tuple[str, Path, float]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise ValueError("--lane must use NAME=PATH=WEIGHT")
    name, raw_path, raw_weight = parts
    if not name or not raw_path:
        raise ValueError(f"invalid lane: {value!r}")
    weight = float(raw_weight)
    if weight <= 0:
        raise ValueError(f"lane weight must be positive: {name}")
    return name, Path(raw_path).resolve(), weight


def _iter_jsonl_handle(handle: TextIO, *, source: Path) -> Iterator[dict[str, Any]]:
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


def _load_manifest_route(
    manifest_path: Path, corpus: str
) -> tuple[dict[str, Any], str, dict[str, Any], Path, list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_meta = (manifest.get("corpora") or {}).get(corpus)
    if not isinstance(corpus_meta, dict):
        raise KeyError(f"unknown corpus: {corpus}")
    task = str(corpus_meta.get("task") or "")
    bank_meta = (manifest.get("banks") or {}).get(task)
    if not isinstance(bank_meta, dict):
        raise KeyError(f"missing task bank: {task}")
    bank_path = _resolve(bank_meta["path"], manifest_path)
    bank_payload = json.loads(bank_path.read_text(encoding="utf-8"))
    metrics = bank_payload.get("metrics") or []
    bank_ids = [str(row.get("metric_id") or "") for row in metrics]
    if (
        not bank_ids
        or "" in bank_ids
        or len(set(bank_ids)) != len(bank_ids)
        or len(bank_ids) != int(bank_meta.get("count", -1))
    ):
        raise ValueError(f"invalid frozen bank: {bank_path}")
    canonical_path = _resolve(corpus_meta["path"], manifest_path)
    return corpus_meta, task, bank_meta, canonical_path, bank_ids


def _validate_lane_meta(
    *,
    name: str,
    path: Path,
    manifest_path: Path,
    corpus: str,
    task: str,
    bank_meta: dict[str, Any],
    bank_size: int,
    prefix_lane: bool,
) -> dict[str, Any]:
    meta_path = _meta_path(path)
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"lane artifact/metadata missing: {path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    actual_sha = sha256_file(path)
    expected_manifest_sha = sha256_file(manifest_path)
    if meta.get("output_sha256") != actual_sha:
        raise ValueError(f"lane output hash mismatch: {name}")
    if meta.get("manifest_sha256") != expected_manifest_sha:
        raise ValueError(f"lane manifest hash mismatch: {name}")
    if str(meta.get("corpus")) != corpus or str(meta.get("task")) != task:
        raise ValueError(f"lane route mismatch: {name}")
    if str(meta.get("bank_source_sha256")) != str(bank_meta["source_sha256"]):
        raise ValueError(f"lane bank hash mismatch: {name}")
    lane_k = int(meta.get("output_k", -1))
    if not 1 <= lane_k <= bank_size:
        raise ValueError(f"lane depth is outside the bank: {name}")
    if prefix_lane == (lane_k == bank_size):
        kind = "prefix" if prefix_lane else "complete-bank"
        raise ValueError(f"{kind} lane has incompatible depth: {name}/{lane_k}")
    if int(meta.get("input_count", -1)) < 0:
        raise ValueError(f"lane lacks frozen input count: {name}")
    return {
        "name": name,
        "path": str(path),
        "sha256": actual_sha,
        "meta": str(meta_path),
        "meta_sha256": sha256_file(meta_path),
        "retrieval_signature": {
            "encoder": meta.get("encoder"),
            "adapter": meta.get("adapter"),
            "query_format": meta.get("query_format"),
            "query_views": meta.get("query_views"),
            "fusion_weights_sha256": meta.get("fusion_weights_sha256"),
        },
        "kind": "preserved-prefix" if prefix_lane else "complete-bank",
        "output_k": lane_k,
    }


def materialize_union(
    *,
    manifest_path: Path,
    corpus: str,
    lanes: list[tuple[str, Path, float]],
    output_path: Path,
    output_k: int,
    rank_constant: float = 60.0,
    preserve_components: dict[str, list[str]] | None = None,
    preserve_k: int | None = None,
    prefix_lanes: set[str] | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    output_path = output_path.resolve()
    if len(lanes) < 2:
        raise ValueError("at least two independent retrieval lanes are required")
    names = [name for name, _, _ in lanes]
    if len(set(names)) != len(names):
        raise ValueError("retrieval lane names must be unique")
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    if output_k < 1:
        raise ValueError("output_k must be positive")
    if output_path.exists() or _meta_path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite retrieval union: {output_path}")

    corpus_meta, task, bank_meta, canonical_path, bank_ids = _load_manifest_route(
        manifest_path, corpus
    )
    bank_size = len(bank_ids)
    if output_k > bank_size:
        raise ValueError(f"output_k exceeds bank size: {output_k} > {bank_size}")
    preserve_components = preserve_components or {}
    prefix_lanes = prefix_lanes or set()
    if not prefix_lanes <= set(names):
        raise ValueError("prefix lane names are outside the supplied lanes")
    if preserve_components:
        if set(preserve_components) != set(names):
            raise ValueError("preserve-components must exactly cover every lane")
        if preserve_k is None or not 1 <= preserve_k <= bank_size:
            raise ValueError("preserve_k is required and must be inside the bank")
        if output_k != bank_size:
            raise ValueError("component-prefix preservation requires full-bank output")
        for name, components in preserve_components.items():
            if (
                not components
                or len(set(components)) != len(components)
                or not set(components) <= PRESERVABLE_COMPONENTS
            ):
                raise ValueError(f"invalid preserved components for lane: {name}")
            if name in prefix_lanes and components != ["rank"]:
                raise ValueError("prefix lanes may preserve only their final rank")
        algorithm = "coverage-preserving-component-prefix-rrf-v1"
    elif preserve_k is not None:
        raise ValueError("preserve_k was supplied without preserve-components")
    else:
        algorithm = "weighted-complete-bank-rrf-v1"
    lane_provenance = [
        {
            **_validate_lane_meta(
                name=name,
                path=path,
                manifest_path=manifest_path,
                corpus=corpus,
                task=task,
                bank_meta=bank_meta,
                bank_size=bank_size,
                prefix_lane=name in prefix_lanes,
            ),
            "weight": weight,
        }
        for name, path, weight in lanes
    ]
    if any(int(json.loads(Path(row["meta"]).read_text())["input_count"]) != int(corpus_meta["count"])
           for row in lane_provenance):
        raise ValueError("lane metadata count differs from canonical corpus")
    lane_depths = {row["name"]: int(row["output_k"]) for row in lane_provenance}
    if preserve_components and any(int(preserve_k) > lane_depths[name] for name in names):
        raise ValueError("preserve_k exceeds a retrieval lane depth")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    count = 0
    seen_uids: set[str] = set()
    sentinel = object()
    try:
        with ExitStack() as stack, temp_path.open("x", encoding="utf-8") as output:
            canonical_handle = stack.enter_context(canonical_path.open("r", encoding="utf-8"))
            lane_handles = [stack.enter_context(path.open("r", encoding="utf-8")) for _, path, _ in lanes]
            iterators = [
                _iter_jsonl_handle(canonical_handle, source=canonical_path),
                *[
                    _iter_jsonl_handle(handle, source=path)
                    for handle, (_, path, _) in zip(lane_handles, lanes, strict=True)
                ],
            ]
            for bundle in zip_longest(*iterators, fillvalue=sentinel):
                if sentinel in bundle:
                    raise ValueError("canonical corpus and retrieval lanes have different lengths")
                canonical = bundle[0]
                lane_rows = bundle[1:]
                uid = str(canonical.get("norm_uid") or "")
                canonical_row = int(canonical.get("row", -1))
                if not uid or uid in seen_uids:
                    raise ValueError(f"canonical row has missing/duplicate norm_uid: {uid!r}")
                seen_uids.add(uid)
                scores = {metric_id: 0.0 for metric_id in bank_ids}
                lane_ranks: dict[str, dict[str, int]] = {}
                component_lane_ranks: dict[str, dict[str, dict[str, int]]] = {}
                primary_payload: dict[str, dict[str, Any]] = {}
                preserved: set[str] = set()
                for (name, _, weight), row in zip(lanes, lane_rows, strict=True):
                    if (
                        str(row.get("norm_uid")) != uid
                        or str(row.get("corpus")) != corpus
                        or str(row.get("task")) != task
                        or int(row.get("row", -1)) != canonical_row
                        or str(row.get("bank_source_sha256"))
                        != str(bank_meta["source_sha256"])
                    ):
                        raise ValueError(f"lane routing/order mismatch: {name}/{uid}")
                    candidates = row.get("candidates")
                    lane_depth = lane_depths[name]
                    if not isinstance(candidates, list) or len(candidates) != lane_depth:
                        raise ValueError(f"lane depth mismatch: {name}/{uid}")
                    ids = [str(value.get("metric_id") or "") for value in candidates]
                    ranks = [int(value.get("rank", -1)) for value in candidates]
                    if (
                        len(set(ids)) != lane_depth
                        or not set(ids) <= set(bank_ids)
                        or (name not in prefix_lanes and set(ids) != set(bank_ids))
                    ):
                        raise ValueError(f"lane bank universe mismatch: {name}/{uid}")
                    if ranks != list(range(1, lane_depth + 1)):
                        raise ValueError(f"lane ranks are not contiguous: {name}/{uid}")
                    lane_ranks[name] = {}
                    component_lane_ranks[name] = {
                        component: {} for component in preserve_components.get(name, [])
                    }
                    for rank, value in enumerate(candidates, 1):
                        metric_id = str(value["metric_id"])
                        lane_ranks[name][metric_id] = rank
                        components = preserve_components.get(name)
                        if components:
                            for component in components:
                                component_rank = (
                                    rank
                                    if component == "rank"
                                    else int(value.get(component, -1))
                                )
                                if not 1 <= component_rank <= lane_depth:
                                    raise ValueError(
                                        f"invalid component rank: {name}/{component}/{uid}"
                                    )
                                component_lane_ranks[name][component][metric_id] = (
                                    component_rank
                                )
                                scores[metric_id] += weight / (
                                    rank_constant + component_rank
                                )
                                if component_rank <= int(preserve_k):
                                    preserved.add(metric_id)
                        else:
                            scores[metric_id] += weight / (rank_constant + rank)
                        primary_payload.setdefault(metric_id, dict(value))
                    for component, by_metric in component_lane_ranks[name].items():
                        if sorted(by_metric.values()) != list(range(1, lane_depth + 1)):
                            raise ValueError(
                                f"component ranks are not a permutation: {name}/{component}/{uid}"
                            )
                all_ordered = sorted(
                    bank_ids, key=lambda metric_id: (-scores[metric_id], metric_id)
                )
                if len(preserved) > output_k:
                    raise ValueError(
                        f"preserved prefix union exceeds output_k: {uid}/{len(preserved)}"
                    )
                ordered = [metric_id for metric_id in all_ordered if metric_id in preserved]
                ordered.extend(
                    metric_id for metric_id in all_ordered if metric_id not in preserved
                )
                ordered = ordered[:output_k]
                fused_candidates = []
                for rank, metric_id in enumerate(ordered, 1):
                    value = primary_payload[metric_id]
                    value.update(
                        {
                            "metric_id": metric_id,
                            "rank": rank,
                            "union_rrf_score": scores[metric_id],
                            "lane_ranks": {
                                name: lane_ranks[name].get(metric_id) for name in names
                            },
                            "component_lane_ranks": {
                                name: {
                                    component: component_lane_ranks[name][component][
                                        metric_id
                                    ]
                                    for component in preserve_components.get(name, [])
                                    if metric_id
                                    in component_lane_ranks[name][component]
                                }
                                for name in names
                                if preserve_components.get(name)
                            },
                            "preserved_prefix_member": metric_id in preserved,
                        }
                    )
                    fused_candidates.append(value)
                fused = {
                    "schema_version": canonical.get("schema_version", "silver-match-v3.0"),
                    "norm_uid": uid,
                    "corpus": corpus,
                    "task": task,
                    "row": canonical_row,
                    "bank_source_sha256": bank_meta["source_sha256"],
                    "preserved_prefix_union_size": len(preserved),
                    "candidates": fused_candidates,
                }
                output.write(json.dumps(fused, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
            output.flush()
            os.fsync(output.fileno())
        if count != int(corpus_meta["count"]):
            raise ValueError(f"union coverage mismatch: {count} != {corpus_meta['count']}")
        temp_path.replace(output_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    meta = {
        "schema_version": "silver-match-v3.0",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "corpus": corpus,
        "task": task,
        "input_count": count,
        "new_count": count,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "bank_source_sha256": bank_meta["source_sha256"],
        "encoder": "deterministic-multi-lane-union",
        "adapter": None,
        "query_format": algorithm,
        "dense_query_instruction": False,
        "query_views": "multiple-independent-retrieval-lanes",
        "fusion_weights": None,
        "fusion_weights_sha256": None,
        "component_weights": None,
        "reranker": None,
        "output_k": output_k,
        "union": {
            "algorithm": algorithm,
            "rank_constant": rank_constant,
            "preserve_k": preserve_k,
            "preserve_components": preserve_components,
            "prefix_lanes": sorted(prefix_lanes),
            "lanes": lane_provenance,
        },
    }
    meta_path = _meta_path(output_path)
    temp_meta = meta_path.with_suffix(meta_path.suffix + f".tmp.{os.getpid()}")
    temp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_meta.replace(meta_path)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--lane", action="append", required=True, help="NAME=PATH=WEIGHT")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-k", type=int, required=True)
    parser.add_argument("--rank-constant", type=float, default=60.0)
    parser.add_argument(
        "--preserve-component",
        action="append",
        default=[],
        help="LANE=RANK_FIELD; repeat to preserve component top-K unions",
    )
    parser.add_argument("--preserve-k", type=int)
    parser.add_argument(
        "--prefix-lane",
        action="append",
        default=[],
        help="lane name whose frozen input is an audited proper-prefix ranking",
    )
    args = parser.parse_args()
    lanes = [_parse_lane(value) for value in args.lane]
    preserve_components: dict[str, list[str]] = {}
    for value in args.preserve_component:
        if "=" not in value:
            raise ValueError("--preserve-component must use LANE=RANK_FIELD")
        name, component = value.split("=", 1)
        preserve_components.setdefault(name, []).append(component)
    result = materialize_union(
        manifest_path=Path(args.manifest),
        corpus=args.corpus,
        lanes=lanes,
        output_path=Path(args.output),
        output_k=args.output_k,
        rank_constant=args.rank_constant,
        preserve_components=preserve_components,
        preserve_k=args.preserve_k,
        prefix_lanes=set(args.prefix_lane),
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
