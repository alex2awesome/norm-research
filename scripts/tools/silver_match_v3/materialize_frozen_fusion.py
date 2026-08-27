#!/usr/bin/env python3
"""Materialize a dev-selected full-bank fusion ranking at a fixed depth.

This is deliberately a transform, not a selector: it accepts one immutable
full-bank candidate artifact plus its already selected fusion report and emits
the deterministic weighted-RRF order.  It is useful for handing the deployed
retriever's exact top-k cards to the adjudicator without loading the encoder a
second time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import read_jsonl, sha256_file, write_jsonl
from .optimize_retrieval_fusion import COMPONENTS


def validate_fusion(
    candidate_path: Path, fusion_path: Path, *, require_frozen_test: bool = False
) -> tuple[dict[str, float], float, Mapping[str, Any], str]:
    fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
    if fusion.get("selection_split") != "dev":
        raise ValueError("fusion report must have been selected on dev")
    actual = sha256_file(candidate_path)
    expected = (fusion.get("candidate_inputs") or {}).get(str(candidate_path))
    provenance = "direct_dev_fusion_input"
    if expected is not None:
        if expected != actual:
            raise ValueError(f"candidate hash mismatch: {actual} != {expected}")
        if require_frozen_test:
            raise ValueError("frozen-test materialization cannot use the dev candidate input")
    else:
        meta_path = candidate_path.with_suffix(candidate_path.suffix + ".meta.json")
        if not meta_path.exists():
            raise ValueError("candidate artifact is not linked to the fusion report")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("output_sha256") != actual:
            raise ValueError("candidate metadata output hash mismatch")
        if meta.get("fusion_weights_sha256") != sha256_file(fusion_path):
            raise ValueError("candidate was not built with the supplied frozen fusion")
        provenance = "postselection_frozen_fusion_materialization"
        if require_frozen_test:
            frozen = meta.get("frozen_test")
            if not frozen:
                raise ValueError("candidate metadata lacks frozen-test provenance")
            marker = Path(frozen["started_marker"])
            completed = marker.with_name(marker.stem + ".completed.json")
            if not marker.exists() or not completed.exists():
                raise ValueError("frozen retriever test marker is incomplete")
            if sha256_file(marker) != frozen.get("started_marker_sha256"):
                raise ValueError("frozen retriever test marker hash changed")
    raw_weights = (fusion.get("selected") or {}).get("component_weights")
    if not isinstance(raw_weights, dict) or set(raw_weights) != set(COMPONENTS):
        raise ValueError("fusion report has invalid component weights")
    weights = {key: float(raw_weights[key]) for key in COMPONENTS}
    if any(value < 0 for value in weights.values()) or not any(
        value > 0 for value in weights.values()
    ):
        raise ValueError("fusion weights must be nonnegative and nonzero")
    rank_constant = float(fusion.get("rank_constant", 0))
    if rank_constant <= 0:
        raise ValueError("rank constant must be positive")
    return weights, rank_constant, fusion, provenance


def rerank_candidates(
    candidates: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
    rank_constant: float,
    limit: int,
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    if not candidates:
        raise ValueError("candidate slate is empty")
    scored = []
    seen_indices: set[int] = set()
    for fallback_index, value in enumerate(candidates):
        index = int(value.get("metric_index", fallback_index))
        if index in seen_indices:
            raise ValueError(f"duplicate metric index: {index}")
        seen_indices.add(index)
        score = 0.0
        for key in COMPONENTS:
            rank = value.get(key)
            if rank is None:
                if weights[key] > 0:
                    raise ValueError(f"positive-weight component is missing {key}")
                continue
            score += weights[key] / (rank_constant + float(rank))
        scored.append((score, index, value))
    ordered = sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]
    return [
        {
            **dict(value),
            "rank": rank,
            "rrf_score": float(score),
        }
        for rank, (score, _, value) in enumerate(ordered, 1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--fusion-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-k", type=int, default=50)
    parser.add_argument(
        "--require-frozen-test",
        action="store_true",
        help="require a completed one-shot frozen retriever-test provenance chain",
    )
    args = parser.parse_args()

    candidate_path = Path(args.candidates).resolve()
    fusion_path = Path(args.fusion_report).resolve()
    output_path = Path(args.output).resolve()
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen fusion output: {output_path}")
    weights, rank_constant, fusion, provenance = validate_fusion(
        candidate_path, fusion_path, require_frozen_test=args.require_frozen_test
    )
    rows = []
    for row in read_jsonl(candidate_path):
        if row.get("task") != fusion.get("task"):
            raise ValueError("candidate task differs from fusion task")
        rows.append(
            {
                **row,
                "candidates": rerank_candidates(
                    row.get("candidates") or [], weights, rank_constant, args.output_k
                ),
            }
        )
    write_jsonl(output_path, rows)
    meta = {
        "role": "frozen_fusion_materialization_no_selection",
        "selection_performed": False,
        "task": fusion.get("task"),
        "selection_split": fusion.get("selection_split"),
        "source_provenance": provenance,
        "source_is_frozen_test": args.require_frozen_test,
        "count": len(rows),
        "output_k": args.output_k,
        "rank_constant": rank_constant,
        "component_weights": weights,
        "input_candidates": str(candidate_path),
        "input_candidates_sha256": sha256_file(candidate_path),
        "dev_fusion_report": str(fusion_path),
        "dev_fusion_report_sha256": sha256_file(fusion_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
