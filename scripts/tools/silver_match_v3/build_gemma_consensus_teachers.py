#!/usr/bin/env python3
"""Promote only order-stable, train-only Gemma MATCH labels to teachers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import normalize_space, read_jsonl, sha256_file, write_jsonl
from .make_calibration import split_for, split_group_for


CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def _resolve(path: str | Path, anchor: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else anchor.parent / value


def build_consensus(
    *,
    manifest_path: Path,
    candidates_path: Path,
    first_path: Path,
    second_path: Path,
    task: str,
    human_panel_paths: list[Path],
    min_confidence: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank_meta = manifest["banks"][task]
    bank_path = _resolve(bank_meta["path"], manifest_path)
    bank_payload = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_ids = {normalize_space(row.get("metric_id")) for row in bank_payload["metrics"]}
    bank_hash = normalize_space(
        bank_meta.get("source_sha256") or bank_payload.get("source_sha256")
    )
    norms: dict[str, dict[str, Any]] = {}
    for corpus, meta in sorted(manifest["corpora"].items()):
        if meta.get("task") != task:
            continue
        for row in read_jsonl(_resolve(meta["path"], manifest_path)):
            norms[row["norm_uid"]] = row

    frozen_groups: set[str] = set()
    for path in human_panel_paths:
        for row in read_jsonl(path):
            if row.get("task") != task:
                continue
            uid = normalize_space(row.get("norm_uid"))
            if uid not in norms:
                raise ValueError(f"human-panel UID absent from manifest: {uid}")
            frozen_groups.add(split_group_for(norms[uid]))

    candidates = {row["norm_uid"]: row for row in read_jsonl(candidates_path)}
    first = {row["norm_uid"]: row for row in read_jsonl(first_path)}
    second = {row["norm_uid"]: row for row in read_jsonl(second_path)}
    if not candidates or set(candidates) != set(first) or set(first) != set(second):
        raise ValueError("candidate and adjudication UID sets must be identical and non-empty")
    first_orders = {normalize_space(row.get("order_mode")) for row in first.values()}
    second_orders = {normalize_space(row.get("order_mode")) for row in second.values()}
    if len(first_orders) != 1 or len(second_orders) != 1 or first_orders == second_orders:
        raise ValueError(f"adjudications need distinct single order modes: {first_orders}, {second_orders}")

    threshold = CONFIDENCE_RANK[min_confidence]
    output = []
    counts = Counter()
    selected_groups: set[str] = set()
    for uid in sorted(candidates):
        candidate = candidates[uid]
        left, right = first[uid], second[uid]
        if candidate.get("task") != task or left.get("task") != task or right.get("task") != task:
            counts["other_task"] += 1
            continue
        norm = norms.get(uid)
        if norm is None:
            raise ValueError(f"candidate UID absent from manifest: {uid}")
        group = split_group_for(norm)
        if split_for(group) != "train":
            raise ValueError(f"distillation candidate is not predeclared train split: {uid}")
        if group in frozen_groups:
            raise ValueError(f"distillation candidate overlaps calibration source group: {uid}")
        candidate_hash = normalize_space(candidate.get("bank_source_sha256"))
        pass_hashes = {
            normalize_space(left.get("candidate_bank_source_sha256")),
            normalize_space(right.get("candidate_bank_source_sha256")),
        }
        if candidate_hash != bank_hash or pass_hashes != {bank_hash}:
            raise ValueError(f"bank provenance mismatch for {uid}")
        if left.get("model") != right.get("model"):
            raise ValueError(f"model mismatch for {uid}")
        if left.get("prompt_sha256") != right.get("prompt_sha256"):
            raise ValueError(f"prompt mismatch for {uid}")
        counts[f"pair:{left.get('decision')}|{right.get('decision')}"] += 1
        if left.get("decision") != "MATCH" or right.get("decision") != "MATCH":
            counts["excluded_nonmatch_pair"] += 1
            continue
        metric_id = normalize_space(left.get("metric_id"))
        if not metric_id or metric_id != normalize_space(right.get("metric_id")):
            counts["excluded_metric_disagreement"] += 1
            continue
        if metric_id not in bank_ids:
            raise ValueError(f"unknown consensus metric for {uid}: {metric_id}")
        left_conf = normalize_space(left.get("confidence"))
        right_conf = normalize_space(right.get("confidence"))
        if min(CONFIDENCE_RANK.get(left_conf, -1), CONFIDENCE_RANK.get(right_conf, -1)) < threshold:
            counts["excluded_low_confidence"] += 1
            continue
        selected_groups.add(group)
        output.append(
            {
                "schema_version": manifest["schema_version"],
                "norm_uid": uid,
                "corpus": norm["corpus"],
                "task": task,
                "row": norm["row"],
                "split": "train",
                "source_group": group,
                "decision": "MATCH",
                "metric_id": metric_id,
                "confidence": min((left_conf, right_conf), key=CONFIDENCE_RANK.get),
                "label_source": "gemma4_gepa_order_consensus",
                "supervision_strength": "distilled_order_consensus",
                "current_bank_source_sha256": bank_hash,
                "gemma_model": left["model"],
                "gemma_prompt_sha256": left["prompt_sha256"],
                "gemma_order_modes": sorted(first_orders | second_orders),
                "gemma_reasons": [left.get("reason"), right.get("reason")],
            }
        )
    report = {
        "task": task,
        "candidate_rows": len(candidates),
        "selected_rows": len(output),
        "selected_source_groups": len(selected_groups),
        "selected_metric_coverage": len({row["metric_id"] for row in output}),
        "selected_by_corpus": dict(sorted(Counter(row["corpus"] for row in output).items())),
        "min_confidence": min_confidence,
        "order_modes": sorted(first_orders | second_orders),
        "frozen_human_source_groups": len(frozen_groups),
        "source_group_overlap_with_human_panels": len(selected_groups & frozen_groups),
        "counts": dict(sorted(counts.items())),
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--human-panel", action="append", default=[])
    parser.add_argument("--min-confidence", choices=("medium", "high"), default="high")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "manifest": args.manifest,
            "candidates": args.candidates,
            "first": args.first,
            "second": args.second,
        }.items()
    }
    panels = [Path(path).resolve() for path in args.human_panel]
    output_path = Path(args.output).resolve()
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError(f"immutable consensus output exists: {output_path}")
    rows, report = build_consensus(
        manifest_path=paths["manifest"],
        candidates_path=paths["candidates"],
        first_path=paths["first"],
        second_path=paths["second"],
        task=args.task,
        human_panel_paths=panels,
        min_confidence=args.min_confidence,
    )
    if not rows:
        raise ValueError("no order-consensus exact teachers survived")
    write_jsonl(output_path, rows)
    meta = {
        "schema_version": "silver-match-v3-gemma-consensus-teachers-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report": report,
        "input_hashes": {
            **{name: sha256_file(path) for name, path in paths.items()},
            "human_panels": {str(path): sha256_file(path) for path in panels},
        },
        "output_sha256": sha256_file(output_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
