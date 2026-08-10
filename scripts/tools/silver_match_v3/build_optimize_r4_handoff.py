#!/usr/bin/env python3
"""Freeze a select-firewalled rich error handoff for an unexposed R4 author."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    output = {str(row.get("norm_uid") or ""): row for row in rows}
    if "" in output or len(output) != len(rows):
        raise ValueError(f"missing or duplicate norm_uid: {path}")
    return output


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def build(
    *,
    pack_root: Path,
    truth_path: Path,
    original_path: Path,
    hashed_path: Path,
    score_path: Path,
    prompt_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pack_root = pack_root.resolve()
    inputs = {
        "truth": truth_path.resolve(),
        "original": original_path.resolve(),
        "hashed": hashed_path.resolve(),
        "score": score_path.resolve(),
        "prompt": prompt_path.resolve(),
        "items": pack_root / "items.jsonl",
        "bank": pack_root / "bank.json",
        "candidates": pack_root / "candidates.frozen-k50.jsonl",
        "pack_validation": pack_root / "validation.json",
    }
    if any("select" in str(path).lower() for path in inputs.values()):
        raise ValueError("select-path input is forbidden in optimize R4 handoff")
    validation = json.loads(inputs["pack_validation"].read_text(encoding="utf-8"))
    if validation.get("truth_hidden") is not True or validation.get("task") != "press-releases":
        raise ValueError("source pack is not the frozen truth-hidden PR optimize pack")
    items = _index(inputs["items"])
    truth = _index(inputs["truth"])
    original = _index(inputs["original"])
    hashed = _index(inputs["hashed"])
    candidates = _index(inputs["candidates"])
    uids = set(items)
    if any(set(value) != uids for value in (truth, original, hashed, candidates)):
        raise ValueError("handoff inputs lack exact optimize-panel coverage")
    if any(
        row.get("gepa_role") != "optimize"
        or row.get("prompt_gradient_eligible") is not True
        or row.get("evaluation_only") is not False
        for row in truth.values()
    ):
        raise ValueError("truth is not wholly optimize-only gradient evidence")
    bank_payload = json.loads(inputs["bank"].read_text(encoding="utf-8"))
    bank = {str(row["metric_id"]): row for row in bank_payload["metrics"]}

    errors: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    for uid in sorted(uids):
        gold, left, right = truth[uid], original[uid], hashed[uid]
        gold_key = (gold["decision"], gold.get("metric_id"))
        left_key = (left["decision"], left.get("metric_id"))
        right_key = (right["decision"], right.get("metric_id"))
        candidate_rows = candidates[uid]["candidates"]
        candidate_ids = [str(row["metric_id"]) for row in candidate_rows]
        gold_in_k50 = gold.get("decision") == "MATCH" and str(gold.get("metric_id")) in candidate_ids
        if gold.get("decision") == "MATCH" and not gold_in_k50:
            misses.append(
                {
                    "norm_uid": uid,
                    "task": "press-releases",
                    "corpus": items[uid].get("corpus"),
                    "norm": items[uid].get("norm"),
                    "context": items[uid].get("context"),
                    "gold_metric_id": gold.get("metric_id"),
                    "candidate_ids": candidate_ids,
                    "event": "truth_exact_leaf_absent_from_frozen_k50",
                }
            )
        if left_key == gold_key and right_key == gold_key:
            continue
        enriched_candidates = []
        for row in candidate_rows:
            metric = bank[str(row["metric_id"])]
            enriched_candidates.append(
                {
                    "rank": row.get("rank"),
                    "metric_id": row["metric_id"],
                    "name": metric.get("name"),
                    "description": metric.get("description"),
                }
            )
        errors.append(
            {
                "schema_version": "silver-match-v3-pr-r4-optimize-error-handoff-v1",
                "norm_uid": uid,
                "task": "press-releases",
                "corpus": items[uid].get("corpus"),
                "norm": items[uid].get("norm"),
                "context": items[uid].get("context"),
                "polarity": items[uid].get("polarity"),
                "gold": {
                    "decision": gold["decision"],
                    "metric_id": gold.get("metric_id"),
                    "metric": bank.get(str(gold.get("metric_id"))) if gold.get("decision") == "MATCH" else None,
                },
                "original": {
                    "decision": left["decision"],
                    "metric_id": left.get("metric_id"),
                    "reason": left.get("reason"),
                },
                "hashed": {
                    "decision": right["decision"],
                    "metric_id": right.get("metric_id"),
                    "reason": right.get("reason"),
                },
                "error_kind": "stable_wrong" if left_key == right_key else "order_unstable",
                "gold_in_frozen_k50": gold_in_k50,
                "candidates": enriched_candidates,
            }
        )
    report = {
        "schema_version": "silver-match-v3-pr-r4-optimize-handoff-freeze-v1",
        "status": "FROZEN_FOR_FRESH_SELECT_UNEXPOSED_AGENT",
        "task": "press-releases",
        "role": "optimize_prompt_gradient_only",
        "source_pack_truth_hidden": True,
        "optimize_gold_in_gradient_errors": True,
        "select_inputs_consumed": False,
        "select_paths_permitted": False,
        "production_promotion_from_this_exposed_thread_permitted": False,
        "error_count": len(errors),
        "truth_match_count": sum(row["decision"] == "MATCH" for row in truth.values()),
        "retriever_miss_at_50_count": len(misses),
        "inputs": {key: _artifact(path) for key, path in inputs.items()},
        "instructions": [
            "Use only these optimize artifacts to author and freeze R4 before opening any PR select artifact.",
            "Correct overmatching and MATCH-versus-NO_EXPLICIT precedence; preserve every typed state.",
            "Require a criterion/evidence-span test and a contrastive exact-leaf justification.",
            "Do not treat the six K50 misses as adjudicator errors; route them to retrieval/full-bank rescue.",
        ],
    }
    return errors, misses, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--hashed", required=True)
    parser.add_argument("--score", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    errors_path = output_root / "gradient_errors.jsonl"
    misses_path = output_root / "retriever_misses_at_50.jsonl"
    report_path = output_root / "HANDOFF_FREEZE.json"
    if any(path.exists() for path in (errors_path, misses_path, report_path)):
        raise FileExistsError("refusing to overwrite R4 handoff")
    errors, misses, report = build(
        pack_root=Path(args.pack_root),
        truth_path=Path(args.truth),
        original_path=Path(args.original),
        hashed_path=Path(args.hashed),
        score_path=Path(args.score),
        prompt_path=Path(args.prompt),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(errors_path, errors)
    write_jsonl(misses_path, misses)
    report["outputs"] = {
        "gradient_errors": _artifact(errors_path),
        "retriever_misses_at_50": _artifact(misses_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "handoff_freeze_sha256": sha256_file(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
