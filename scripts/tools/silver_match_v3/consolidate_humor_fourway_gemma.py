#!/usr/bin/env python3
"""Strictly consolidate paired Humor Gemma shards before full-bank rescue.

CE scores are absent from the decision rule.  A provisional MATCH requires
the base and typed-LoRA arms to agree on the same leaf in both original and
hashed candidate orders, with no low-confidence arm.  Every other outcome is
an abstention/rescue route, never a forced metric.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .adjudicate_gemma import CONFIDENCES, DECISIONS, ordered_candidates
from .common import normalize_space, read_jsonl, sha256_file
from .package_humor_ce_reducer_shard import SCHEMA as CANDIDATE_SCHEMA
from .run_paired_gemma_lora_batch import SCHEMA as PAIRED_SCHEMA


SCHEMA = "silver-match-v3-humor-fourway-gemma-primary-v1"
REPORT_SCHEMA = "silver-match-v3-humor-fourway-gemma-primary-report-v1"
META_SCHEMA = "silver-match-v3-paired-gemma4-lora-inference-meta-v1"


def _write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _valid(payload: Mapping[str, Any], candidate_ids: set[str]) -> bool:
    decision = normalize_space(payload.get("decision"))
    metric_id = payload.get("metric_id")
    return bool(
        decision in DECISIONS
        and payload.get("confidence") in CONFIDENCES
        and normalize_space(payload.get("reason"))
        and payload.get("parse_error") is None
        and (
            (decision == "MATCH" and str(metric_id or "") in candidate_ids)
            or (decision != "MATCH" and metric_id is None)
        )
    )


def _confidence_min(payloads: Sequence[Mapping[str, Any]]) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    return min((str(row.get("confidence") or "low") for row in payloads), key=order.__getitem__)


def consolidate(args: argparse.Namespace) -> dict[str, Any]:
    candidate_paths = [Path(value).resolve() for value in args.candidate]
    roots = [Path(value).resolve() for value in args.paired_root]
    output = Path(args.output).resolve()
    report_output = Path(args.report_output).resolve()
    if len(candidate_paths) != len(roots) or not candidate_paths:
        raise ValueError("candidate and paired-root bindings must be non-empty and aligned")
    if output.exists() or report_output.exists():
        raise FileExistsError("refusing to overwrite four-way consolidation")

    counts: Counter[str] = Counter()
    artifact_rows: list[dict[str, Any]] = []

    def rows() -> Iterable[dict[str, Any]]:
        seen: set[str] = set()
        for candidate_path, root in zip(candidate_paths, roots, strict=True):
            original_path = root / "paired.original.jsonl"
            hashed_path = root / "paired.hashed.jsonl"
            meta_path = root / "paired_inference.meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            outputs = meta.get("outputs") or {}
            if (
                meta.get("schema_version") != META_SCHEMA
                or meta.get("status") != "COMPLETE_TRUTH_BLIND_PAIRED_INFERENCE"
                or meta.get("truth_read") is not False
                or (outputs.get("original") or {}).get("sha256") != sha256_file(original_path)
                or (outputs.get("hashed") or {}).get("sha256") != sha256_file(hashed_path)
            ):
                raise ValueError(f"paired inference metadata is incomplete: {root}")
            artifact_rows.append(
                {
                    "candidates": _artifact(candidate_path),
                    "paired_original": _artifact(original_path),
                    "paired_hashed": _artifact(hashed_path),
                    "paired_meta": _artifact(meta_path),
                }
            )
            streams = zip_longest(
                read_jsonl(candidate_path),
                read_jsonl(original_path),
                read_jsonl(hashed_path),
                fillvalue=None,
            )
            for candidate, original, hashed in streams:
                if candidate is None or original is None or hashed is None:
                    raise ValueError(f"paired/candidate length mismatch: {root}")
                uid = normalize_space(candidate.get("norm_uid"))
                cards = list(candidate.get("candidates") or [])
                candidate_ids = [normalize_space(row.get("metric_id")) for row in cards]
                expected_hashed = [
                    row["metric_id"] for row in ordered_candidates(cards, "hashed", uid)
                ]
                if (
                    candidate.get("schema_version") != CANDIDATE_SCHEMA
                    or not uid
                    or uid in seen
                    or original.get("schema_version") != PAIRED_SCHEMA
                    or hashed.get("schema_version") != PAIRED_SCHEMA
                    or original.get("norm_uid") != uid
                    or hashed.get("norm_uid") != uid
                    or original.get("order_mode") != "original"
                    or hashed.get("order_mode") != "hashed"
                    or list(original.get("candidate_ids") or []) != candidate_ids
                    or list(hashed.get("candidate_ids") or []) != expected_hashed
                    or original.get("base_item_prompt_sha256")
                    != original.get("lora_item_prompt_sha256")
                    or hashed.get("base_item_prompt_sha256")
                    != hashed.get("lora_item_prompt_sha256")
                ):
                    raise ValueError(f"paired four-way routing/order contract differs: {uid}")
                seen.add(uid)
                candidate_set = set(candidate_ids)
                payloads = [
                    original.get("base") or {},
                    original.get("lora") or {},
                    hashed.get("base") or {},
                    hashed.get("lora") or {},
                ]
                valid = all(_valid(row, candidate_set) for row in payloads)
                votes = [
                    (normalize_space(row.get("decision")), row.get("metric_id"))
                    for row in payloads
                ]
                stable = valid and len(set(votes)) == 1
                confidence = _confidence_min(payloads)
                if stable and votes[0][0] == "MATCH" and confidence != "low":
                    decision = "MATCH"
                    metric_id = str(votes[0][1])
                    route = "PROVISIONAL_FOURWAY_MATCH"
                elif stable and votes[0][0] != "MATCH":
                    decision = votes[0][0]
                    metric_id = None
                    route = "FULL285_RESCUE_TYPED_ABSTENTION"
                elif valid:
                    decision = "UNSTABLE_MATCH"
                    metric_id = None
                    route = "FULL285_RESCUE_DISAGREEMENT_OR_LOW_MATCH"
                else:
                    decision = "INVALID_OUTPUT"
                    metric_id = None
                    route = "FULL285_RESCUE_INVALID_OUTPUT"
                if stable and votes[0][0] == "MATCH" and confidence == "low":
                    decision = "UNSTABLE_MATCH"
                    metric_id = None
                    route = "FULL285_RESCUE_DISAGREEMENT_OR_LOW_MATCH"
                counts[route] += 1
                counts[f"decision:{decision}"] += 1
                counts["norms"] += 1
                yield {
                    "schema_version": SCHEMA,
                    "task": candidate["task"],
                    "corpus": candidate["corpus"],
                    "row": int(candidate.get("row", -1)),
                    "norm_uid": uid,
                    "decision": decision,
                    "metric_id": metric_id,
                    "confidence": confidence,
                    "reason": (
                        "Four-way base/LoRA original/hashed consensus."
                        if stable
                        else "Four-way base/LoRA original/hashed outputs did not yield a stable valid decision."
                    ),
                    "candidate_ids": candidate_ids,
                    "routing_category": route,
                    "ce_only_acceptance": False,
                    "fourway_votes": [
                        {
                            "arm": arm,
                            "decision": row.get("decision"),
                            "metric_id": row.get("metric_id"),
                            "confidence": row.get("confidence"),
                            "reason": row.get("reason"),
                            "parse_error": row.get("parse_error"),
                        }
                        for arm, row in zip(
                            ("base_original", "lora_original", "base_hashed", "lora_hashed"),
                            payloads,
                            strict=True,
                        )
                    ],
                    "full285_rescue_required": decision != "MATCH",
                }

    count = _write_jsonl_new(output, rows())
    if count != 77378 or counts["norms"] != count:
        raise ValueError(f"Humor four-way coverage mismatch: {count}")
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "COMPLETE_PRE_FULL285_RESCUE",
        "output": {**_artifact(output), "count": count},
        "counts": dict(sorted(counts.items())),
        "shards": artifact_rows,
        "policy": {
            "ce_only_acceptance_forbidden": True,
            "provisional_match_requires_base_and_lora_original_and_hashed_same_leaf": True,
            "low_confidence_match_requires_full285_rescue": True,
            "every_nonmatch_requires_full285_rescue": True,
            "final_release_ready": False,
        },
    }
    _write_json_new(report_output, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--paired-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-output", required=True)
    return parser.parse_args(argv)


def main() -> None:
    print(json.dumps(consolidate(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
