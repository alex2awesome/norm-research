#!/usr/bin/env python3
"""Apply dev-frozen retrieval fusion weights without any further selection."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import read_jsonl, sha256_file
from .optimize_retrieval_fusion import (
    COMPONENTS,
    load_inputs,
    score_components,
    stable_ranks,
    summarize,
    tensorize,
)


FINGERPRINT_KEYS = (
    "manifest_sha256",
    "encoder",
    "adapter_hashes",
    "query_format",
    "dense_query_instruction",
    "query_views",
    "component_k",
    "output_k",
)


def candidate_meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def model_fingerprint(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {key: meta.get(key) for key in FINGERPRINT_KEYS}


def validate_frozen_model(
    candidate_path: Path, fusion: Mapping[str, Any], *, require_test_marker: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    current_meta_path = candidate_meta_path(candidate_path)
    if not current_meta_path.exists():
        raise FileNotFoundError(f"candidate metadata missing: {current_meta_path}")
    current_meta = json.loads(current_meta_path.read_text(encoding="utf-8"))
    dev_paths = [Path(path) for path in (fusion.get("candidate_inputs") or {})]
    if not dev_paths:
        raise ValueError("fusion report has no dev candidate inputs")
    dev_fingerprints = []
    for path in dev_paths:
        meta_path = candidate_meta_path(path)
        if not meta_path.exists():
            raise FileNotFoundError(f"fusion dev candidate metadata missing: {meta_path}")
        dev_fingerprints.append(
            model_fingerprint(json.loads(meta_path.read_text(encoding="utf-8")))
        )
    if any(value != dev_fingerprints[0] for value in dev_fingerprints[1:]):
        raise ValueError("fusion report combines different encoder/adapter fingerprints")
    current_fingerprint = model_fingerprint(current_meta)
    if current_fingerprint != dev_fingerprints[0]:
        raise ValueError(
            f"test/dev retrieval fingerprint mismatch: {current_fingerprint} != "
            f"{dev_fingerprints[0]}"
        )
    if require_test_marker:
        frozen = current_meta.get("frozen_test")
        if not frozen:
            raise ValueError("test candidate metadata lacks one-shot frozen-test marker")
        marker = Path(frozen["started_marker"])
        completed = marker.with_name(marker.stem + ".completed.json")
        if not marker.exists() or not completed.exists():
            raise ValueError("frozen-test retrieval marker is incomplete")
        if sha256_file(marker) != frozen["started_marker_sha256"]:
            raise ValueError("frozen-test started-marker hash changed")
    return current_meta, current_fingerprint


def sliced_metrics(
    ranks: np.ndarray,
    labels: Sequence[Mapping[str, Any]],
    bank_size: int,
) -> dict[str, Any]:
    gold = [str(row["metric_id"]) for row in labels]
    output = {"all": summarize(ranks, gold, bank_size)}
    for field, key in (("human_panel", "by_human_panel"), ("corpus", "by_corpus")):
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(labels):
            grouped[str(row.get(field) or "UNSPECIFIED")].append(index)
        output[key] = {
            group: summarize(
                ranks[indices], [gold[index] for index in indices], bank_size
            )
            for group, indices in sorted(grouped.items())
        }
    output["items"] = [
        {
            "norm_uid": str(row["norm_uid"]),
            "corpus": str(row.get("corpus") or ""),
            "human_panel": str(row.get("human_panel") or "UNSPECIFIED"),
            "metric_id": str(row["metric_id"]),
            "exact_rank": int(rank),
        }
        for row, rank in zip(labels, ranks.tolist())
    ]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--fusion-report", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    label_path = Path(args.labels).resolve()
    candidate_path = Path(args.candidates).resolve()
    fusion_path = Path(args.fusion_report).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen fusion evaluation: {output_path}")
    fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
    if fusion.get("task") != args.task or fusion.get("selection_split") != "dev":
        raise ValueError("fusion report is not a dev-selected artifact for this task")
    labels, candidates = load_inputs([label_path], [candidate_path], args.task)
    labels = [row for row in labels if row.get("split") == args.split]
    if not labels:
        raise ValueError(f"no MATCH labels for {args.task}/{args.split}")
    metric_ids, tensor, gold_indices = tensorize(labels, candidates)
    weights = np.asarray(
        [fusion["selected"]["component_weights"][key] for key in COMPONENTS],
        dtype=np.float64,
    )
    scores = score_components(tensor, weights, float(fusion["rank_constant"]))
    ranks = stable_ranks(scores, gold_indices)
    candidate_meta, fingerprint = validate_frozen_model(
        candidate_path, fusion, require_test_marker=args.split == "test"
    )
    metrics = sliced_metrics(ranks, labels, len(metric_ids))
    report = {
        "task": args.task,
        "split": args.split,
        "role": "dev_reporting" if args.split == "dev" else "frozen_test_confirmation_only",
        "selection_performed": False,
        "fusion_weights": dict(zip(COMPONENTS, weights.tolist())),
        "rank_constant": fusion["rank_constant"],
        "model_fingerprint": fingerprint,
        "metrics": metrics,
        "input_hashes": {
            "labels": sha256_file(label_path),
            "candidates": sha256_file(candidate_path),
            "candidate_meta": sha256_file(candidate_meta_path(candidate_path)),
            "dev_fusion_report": sha256_file(fusion_path),
        },
        "candidate_frozen_test": candidate_meta.get("frozen_test"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
