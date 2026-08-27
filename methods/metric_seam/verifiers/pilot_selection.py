"""Freeze the bounded code-review verifier pilot before TRAIN execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .code_review_controls import CONTROLS


SCHEMA = "metric-seam.code-review-verifier-pilot-selection.v1"
DEFAULT_ASPECTS = ("a0", "a18", "a38", "a92")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def build_selection(
    *, join_path: Path, aspects: Sequence[str] = DEFAULT_ASPECTS
) -> dict:
    join = _load(join_path)
    if join.get("schema") != "metric-seam.code-review-cuf-join-candidates.v1":
        raise ValueError("unsupported CUF join schema")
    if len(set(aspects)) != len(aspects) or not aspects:
        raise ValueError("aspects must be unique and nonempty")
    snapshot_manifest_path = Path(join["source_snapshot"]["manifest_path"])
    snapshot = _load(snapshot_manifest_path)
    bank_path = Path(snapshot["snapshot"]["bank_path"])
    bank = {}
    for line in bank_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        bank[row["metric"]] = row

    rows_by_aspect: dict[str, list[Mapping]] = {}
    for row in join["rows"]:
        rows_by_aspect.setdefault(row["candidate_aspect_id"], []).append(row)
    selected = []
    for aspect in aspects:
        rows = rows_by_aspect.get(aspect)
        if not rows:
            raise ValueError(f"pilot aspect is absent from current cells: {aspect}")
        metric_summaries = {
            json.dumps(row.get("selected_bank_metric"), sort_keys=True) for row in rows
        }
        if len(metric_summaries) != 1 or rows[0].get("selected_bank_metric") is None:
            raise ValueError(f"pilot aspect lacks one auto-accepted CUF metric: {aspect}")
        metric_name = rows[0]["selected_bank_metric"]["metric"]
        metric = bank[metric_name]
        candidates = sorted(
            (
                unit for unit in metric["rows"]
                if unit.get("verdict") == "CERTIFIED-UNIT" and unit.get("level") == 1
            ),
            key=lambda unit: unit["node_id"],
        )
        if not candidates:
            raise ValueError(f"pilot metric has no level-1 certified CUF unit: {metric_name}")
        unit = candidates[0]
        selected.append({
            "pilot_unit_id": f"code-review:llama8b:k{metric['k']}:n{unit['node_id']}",
            "candidate_aspect_id": aspect,
            "metric_name": metric_name,
            "relation": unit["span"],
            "cuf": {
                "executor": "llama8b",
                "metric_k": metric["k"],
                "node_id": str(unit["node_id"]),
                "level": unit["level"],
                "verdict": unit["verdict"],
            },
            "hierarchy_cells": sorted(
                {row["cell_id"] for row in rows}
            ),
            "hierarchy_levels": sorted(
                {row["level"] for row in rows}
            ),
            "selection_rule": "first node_id among level-1 CERTIFIED-UNIT rows",
        })

    controls = [
        {
            "control_id": control.control_id,
            "construct": control.construct,
            "relation": control.relation,
            "expected": control.expected,
            "mutation_kind": control.mutation_kind,
        }
        for control in CONTROLS
    ]
    return {
        "schema": SCHEMA,
        "status": "candidate_selection_frozen_before_train_execution",
        "task": "code-review",
        "source_join": {"path": str(join_path), "sha256": _sha(join_path)},
        "source_cuf_snapshot": {
            "manifest_path": str(snapshot_manifest_path),
            "snapshot_id": snapshot["snapshot_id"],
            "executor": snapshot["executor"],
        },
        "selection_policy": {
            "real_unit_cap": 4,
            "one_unit_per_candidate_aspect": True,
            "semantic_join_queue_used": False,
            "heldout_items_or_outputs_loaded_by_builder": False,
            "prompt_or_model_outputs_loaded_by_builder": False,
            "selection_author_blinding_not_mechanically_established": True,
        },
        "real_units": selected,
        "controls": controls,
        "claim_limits": [
            "These are candidates, not certified code verifiers.",
            "CUF certification concerns prompt-addressable metric spans for llama8b; it is not code verification.",
            "Failed TRAIN discrimination is bounded corpus/implementation failure, not tacitness.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--join", type=Path, required=True)
    parser.add_argument("--aspects", nargs="+", default=list(DEFAULT_ASPECTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_selection(join_path=args.join, aspects=args.aspects)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"real_units": len(payload["real_units"]), "controls": len(payload["controls"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
