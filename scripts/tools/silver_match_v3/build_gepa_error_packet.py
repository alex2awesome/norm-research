#!/usr/bin/env python3
"""Build a truth-revealed GEPA error packet from a predeclared train panel only."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl


def _unique(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in read_jsonl(path):
        uid = str(row.get("norm_uid") or "")
        if not uid or uid in rows:
            raise ValueError(f"missing/duplicate norm_uid in {path}: {uid!r}")
        rows[uid] = row
    return rows


def build_packet(
    *,
    manifest_path: Path,
    task: str,
    truth_path: Path,
    original_path: Path,
    hashed_path: Path,
    candidates_path: Path,
) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if task not in manifest.get("banks", {}):
        raise KeyError(task)
    bank_path = Path(manifest["banks"][task]["path"])
    if not bank_path.is_absolute():
        bank_path = manifest_path.parent / bank_path
    bank_payload = json.loads(bank_path.read_text(encoding="utf-8"))
    bank = {str(row["metric_id"]): row for row in bank_payload["metrics"]}

    truth = _unique(truth_path)
    original = _unique(original_path)
    hashed = _unique(hashed_path)
    candidates = _unique(candidates_path)
    if not truth or not (truth.keys() == original.keys() == hashed.keys() == candidates.keys()):
        raise ValueError("truth/prediction/candidate UID universes differ")
    if any(row.get("task") != task or row.get("split") != "train" for row in truth.values()):
        raise ValueError("GEPA error analysis is restricted to task-matched train rows")

    wanted = set(truth)
    norms: dict[str, dict] = {}
    for corpus, meta in manifest["corpora"].items():
        if meta["task"] != task:
            continue
        path = Path(meta["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        for row in read_jsonl(path):
            uid = str(row.get("norm_uid") or "")
            if uid in wanted:
                if uid in norms:
                    raise ValueError(f"duplicate canonical norm_uid: {uid}")
                norms[uid] = row
    if norms.keys() != truth.keys():
        raise ValueError("train panel UIDs are not exactly recoverable from canonical norms")

    packet = []
    confusion = Counter()
    order_disagreements = 0
    strict_confirmed = 0
    strict_correct = 0
    for uid in sorted(truth):
        gold = truth[uid]
        if gold.get("decision") != "MATCH":
            raise ValueError("current exact-leaf error packet requires MATCH truth rows")
        gold_id = str(gold.get("metric_id") or "")
        first, second = original[uid], hashed[uid]
        first_match = first.get("decision") == "MATCH"
        second_match = second.get("decision") == "MATCH"
        first_id = str(first.get("metric_id") or "") if first_match else None
        second_id = str(second.get("metric_id") or "") if second_match else None
        consensus = first_match and second_match and first_id == second_id
        if not consensus:
            order_disagreements += 1
            continue
        strict_confirmed += 1
        if first_id == gold_id:
            strict_correct += 1
            continue
        confusion[(gold_id, str(first_id))] += 1
        rank_by_id = {
            str(row["metric_id"]): int(row["rank"])
            for row in candidates[uid].get("candidates") or []
        }
        norm = norms[uid]
        packet.append(
            {
                "norm_uid": uid,
                "task": task,
                "corpus": gold.get("corpus"),
                "row": gold.get("row"),
                "norm": norm.get("norm"),
                "context": norm.get("context"),
                "gold_metric_id": gold_id,
                "gold_metric": bank[gold_id],
                "gold_candidate_rank": rank_by_id.get(gold_id),
                "predicted_metric_id": first_id,
                "predicted_metric": bank[str(first_id)],
                "predicted_candidate_rank": rank_by_id.get(str(first_id)),
                "original_reason": first.get("reason"),
                "hashed_reason": second.get("reason"),
                "truth_reason": gold.get("reason"),
                "taxonomy": None,
            }
        )
    summary = {
        "schema_version": "silver-match-v3-gepa-train-error-packet-v1",
        "task": task,
        "panel_role": "predeclared_train_only",
        "n": len(truth),
        "strict_confirmed": strict_confirmed,
        "strict_correct": strict_correct,
        "strict_wrong": len(packet),
        "order_unstable_or_abstained": order_disagreements,
        "strict_exact_precision": (
            strict_correct / strict_confirmed if strict_confirmed else None
        ),
        "confusions": [
            {"gold_metric_id": gold, "predicted_metric_id": pred, "count": count}
            for (gold, pred), count in sorted(
                confusion.items(), key=lambda value: (-value[1], value[0])
            )
        ],
        "inputs": {
            str(path): sha256_file(path)
            for path in (
                manifest_path,
                bank_path,
                truth_path,
                original_path,
                hashed_path,
                candidates_path,
            )
        },
    }
    return packet, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--hashed", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    if output.exists() or meta_path.exists():
        raise FileExistsError(output)
    packet, summary = build_packet(
        manifest_path=Path(args.manifest).resolve(),
        task=args.task,
        truth_path=Path(args.truth).resolve(),
        original_path=Path(args.original).resolve(),
        hashed_path=Path(args.hashed).resolve(),
        candidates_path=Path(args.candidates).resolve(),
    )
    write_jsonl(output, packet)
    summary["output"] = {"path": str(output), "sha256": sha256_file(output)}
    meta_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
