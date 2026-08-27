#!/usr/bin/env python3
"""Select task-specific evidence/statement hybrid-retrieval weights on dev."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .common import read_jsonl, sha256_file, write_jsonl


COMPONENTS = (
    "dense_rank",
    "dense_statement_rank",
    "word_rank",
    "word_statement_rank",
    "char_rank",
    "char_statement_rank",
)


def component_weights(
    evidence_share: float,
    dense_weight: float,
    word_weight: float,
    char_weight: float,
) -> np.ndarray:
    statement_share = 1.0 - evidence_share
    values = np.asarray(
        [
            dense_weight * evidence_share,
            dense_weight * statement_share,
            word_weight * evidence_share,
            word_weight * statement_share,
            char_weight * evidence_share,
            char_weight * statement_share,
        ],
        dtype=np.float64,
    )
    if not np.any(values > 0):
        raise ValueError("at least one fusion component must have positive weight")
    return values / np.sum(values)


def score_components(
    rank_tensor: np.ndarray, weights: np.ndarray, rank_constant: float
) -> np.ndarray:
    ranks = np.asarray(rank_tensor, dtype=np.float64)
    if ranks.ndim != 3 or ranks.shape[2] != len(COMPONENTS):
        raise ValueError("rank tensor must be [items, metrics, components]")
    if weights.shape != (len(COMPONENTS),):
        raise ValueError("weight vector shape mismatch")
    return np.sum(weights[None, None, :] / (rank_constant + ranks), axis=2)


def stable_ranks(scores: np.ndarray, gold_indices: np.ndarray) -> np.ndarray:
    output = []
    for row, gold in zip(scores, gold_indices):
        order = np.lexsort((np.arange(len(row)), -row))
        output.append(int(np.where(order == int(gold))[0][0]) + 1)
    return np.asarray(output, dtype=np.int64)


def summarize(
    ranks: np.ndarray, gold_ids: Sequence[str], bank_size: int
) -> dict[str, Any]:
    by_metric: dict[str, list[int]] = defaultdict(list)
    for rank, metric_id in zip(ranks.tolist(), gold_ids):
        by_metric[metric_id].append(rank)
    result: dict[str, Any] = {
        "n": len(ranks),
        "metrics": len(by_metric),
        "mrr": float(np.mean(1.0 / ranks)) if len(ranks) else None,
        "mean_rank": float(np.mean(ranks)) if len(ranks) else None,
        "median_rank": float(np.median(ranks)) if len(ranks) else None,
    }
    for k in (1, 5, 10, 16, 30, 50, 80, 120, 180):
        effective = min(k, bank_size)
        result[f"recall_at_{k}"] = (
            float(np.mean(ranks <= effective)) if len(ranks) else None
        )
        result[f"macro_recall_at_{k}"] = (
            float(
                np.mean(
                    [
                        np.mean(np.asarray(values) <= effective)
                        for values in by_metric.values()
                    ]
                )
            )
            if by_metric
            else None
        )
    return result


def select_weights(
    rank_tensor: np.ndarray,
    gold_indices: np.ndarray,
    gold_ids: Sequence[str],
    splits: Sequence[str],
    *,
    rank_constant: float,
    evidence_grid: Sequence[float],
    modality_grid: Sequence[float],
    primary_k: int = 16,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dev = np.asarray([split == "dev" for split in splits])
    if not np.any(dev):
        raise ValueError("fusion selection requires dev matches")
    trials = []
    for evidence, dense, word, char in itertools.product(
        evidence_grid, modality_grid, modality_grid, modality_grid
    ):
        if dense == word == char == 0:
            continue
        weights = component_weights(evidence, dense, word, char)
        scores = score_components(rank_tensor[dev], weights, rank_constant)
        ranks = stable_ranks(scores, gold_indices[dev])
        metrics = summarize(
            ranks, np.asarray(gold_ids)[dev].tolist(), rank_tensor.shape[1]
        )
        trials.append(
            {
                "evidence_share": evidence,
                "dense_weight": dense,
                "word_weight": word,
                "char_weight": char,
                "component_weights": dict(zip(COMPONENTS, weights.tolist())),
                "dev": metrics,
            }
        )
    if primary_k not in {1, 5, 10, 16, 30, 50, 80, 120, 180}:
        raise ValueError(f"unsupported primary_k: {primary_k}")
    secondary_k = 30 if primary_k >= 50 else 50
    best = max(
        trials,
        key=lambda row: (
            row["dev"][f"macro_recall_at_{primary_k}"]
            + row["dev"][f"recall_at_{primary_k}"],
            row["dev"][f"macro_recall_at_{secondary_k}"]
            + row["dev"][f"recall_at_{secondary_k}"],
            row["dev"]["mrr"],
            -abs(row["evidence_share"] - 0.5),
            row["dense_weight"],
        ),
    )
    return best, trials


def load_inputs(
    label_paths: Sequence[Path],
    candidate_paths: Sequence[Path],
    task: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    labels = []
    seen = set()
    for path in label_paths:
        for row in read_jsonl(path):
            if row.get("task") != task or row.get("decision") != "MATCH":
                continue
            uid = str(row["norm_uid"])
            if uid in seen:
                raise ValueError(f"duplicate MATCH label UID: {uid}")
            seen.add(uid)
            labels.append(row)
    candidates = {}
    for path in candidate_paths:
        for row in read_jsonl(path):
            if row.get("task") != task:
                continue
            uid = str(row["norm_uid"])
            if uid in candidates:
                raise ValueError(f"duplicate candidate UID: {uid}")
            candidates[uid] = row
    return labels, candidates


def tensorize(labels: Sequence[dict[str, Any]], candidates: dict[str, dict[str, Any]]):
    all_candidates = [
        candidate
        for label in labels
        for candidate in candidates[label["norm_uid"]]["candidates"]
    ]
    by_metric_index: dict[int, str] = {}
    for candidate in all_candidates:
        if candidate.get("metric_index") is None:
            continue
        index, metric_id = int(candidate["metric_index"]), str(candidate["metric_id"])
        prior = by_metric_index.get(index)
        if prior is not None and prior != metric_id:
            raise ValueError(
                f"metric_index {index} maps to both {prior} and {metric_id}"
            )
        by_metric_index[index] = metric_id
    unique_ids = {str(candidate["metric_id"]) for candidate in all_candidates}
    if by_metric_index and set(by_metric_index.values()) == unique_ids:
        metric_ids = [by_metric_index[index] for index in sorted(by_metric_index)]
    else:
        # Backward-compatible natural ordering for older a0/a1/... candidate
        # artifacts that predate explicit metric_index. This matches the bank
        # order used by production RRF tie-breaking, unlike lexicographic a10<a2.
        def natural_key(value: str):
            match = re.fullmatch(r"(.*?)(\d+)", value)
            return (match.group(1), int(match.group(2))) if match else (value, -1)

        metric_ids = sorted(unique_ids, key=natural_key)
    metric_index = {metric_id: idx for idx, metric_id in enumerate(metric_ids)}
    tensor = np.full(
        (len(labels), len(metric_ids), len(COMPONENTS)), np.inf, dtype=np.float64
    )
    gold_indices = []
    for row_idx, label in enumerate(labels):
        uid = label["norm_uid"]
        if uid not in candidates:
            raise KeyError(f"labeled UID missing candidates: {uid}")
        values = candidates[uid]["candidates"]
        if len(values) != len(metric_ids):
            raise ValueError(
                f"candidate slate is not full-bank for {uid}: {len(values)} != {len(metric_ids)}"
            )
        for candidate in values:
            idx = metric_index[str(candidate["metric_id"])]
            for component_idx, key in enumerate(COMPONENTS):
                rank = candidate.get(key)
                if rank is None:
                    raise ValueError(f"full-bank candidate lacks {key}: {uid}")
                tensor[row_idx, idx, component_idx] = float(rank)
        gold = str(label["metric_id"])
        if gold not in metric_index:
            raise KeyError(f"gold metric absent from full bank: {uid}/{gold}")
        gold_indices.append(metric_index[gold])
    return metric_ids, tensor, np.asarray(gold_indices, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--labels", action="append", required=True)
    parser.add_argument("--candidates", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trials-output")
    parser.add_argument("--rank-constant", type=float, default=60.0)
    parser.add_argument(
        "--primary-k",
        type=int,
        choices=(1, 5, 10, 16, 30, 50, 80, 120, 180),
        default=16,
        help="Dev-only recall depth used as the primary fusion objective.",
    )
    args = parser.parse_args()
    label_paths = [Path(path) for path in args.labels]
    candidate_paths = [Path(path) for path in args.candidates]
    labels, candidates = load_inputs(label_paths, candidate_paths, args.task)
    metric_ids, tensor, gold_indices = tensorize(labels, candidates)
    splits = [str(label.get("split") or "") for label in labels]
    gold_ids = [str(label["metric_id"]) for label in labels]
    best, trials = select_weights(
        tensor,
        gold_indices,
        gold_ids,
        splits,
        rank_constant=args.rank_constant,
        evidence_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
        modality_grid=(0.0, 0.5, 1.0, 2.0),
        primary_k=args.primary_k,
    )
    weights = np.asarray([best["component_weights"][key] for key in COMPONENTS])
    all_scores = score_components(tensor, weights, args.rank_constant)
    all_ranks = stable_ranks(all_scores, gold_indices)
    by_split = {}
    for split in ("train", "dev", "test"):
        mask = np.asarray([value == split for value in splits])
        ranks = stable_ranks(all_scores[mask], gold_indices[mask])
        by_split[split] = summarize(
            ranks, np.asarray(gold_ids)[mask].tolist(), len(metric_ids)
        )
    report = {
        "task": args.task,
        "selection_split": "dev",
        "count": len(labels),
        "split_counts": dict(sorted(Counter(splits).items())),
        "bank_size": len(metric_ids),
        "rank_constant": args.rank_constant,
        "selection_objective": {
            "split": "dev",
            "primary": f"macro_recall_at_{args.primary_k} + recall_at_{args.primary_k}",
            "primary_k": args.primary_k,
        },
        "selected": {key: value for key, value in best.items() if key != "dev"},
        "metrics": by_split,
        "label_inputs": {str(path): sha256_file(path) for path in label_paths},
        "candidate_inputs": {str(path): sha256_file(path) for path in candidate_paths},
        "items": [
            {
                "norm_uid": str(label["norm_uid"]),
                "split": str(label.get("split") or ""),
                "metric_id": str(label["metric_id"]),
                "exact_rank": int(rank),
            }
            for label, rank in zip(labels, all_ranks.tolist())
        ],
        "trial_count": len(trials),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.trials_output:
        write_jsonl(Path(args.trials_output), trials)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
