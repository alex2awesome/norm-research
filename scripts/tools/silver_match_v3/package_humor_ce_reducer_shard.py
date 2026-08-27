#!/usr/bin/env python3
"""Package one Humor CE score shard as non-authoritative Gemma candidates.

Every norm is emitted.  The slate is the stable union of the CE top eight and
the original diverse-retrieval top eight.  CE thresholds are recorded only as
diagnostics and never produce a match decision.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import read_jsonl, sha256_file
from .run_nemotron_ce import SCORE_META_SCHEMA, SCORE_SCHEMA


SCHEMA = "silver-match-v3-humor-ce-candidate-reducer-v1"
REPORT_SCHEMA = "silver-match-v3-humor-ce-candidate-reducer-report-v1"


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


def _groups(rows: Iterable[dict[str, Any]]) -> Iterable[list[dict[str, Any]]]:
    group: list[dict[str, Any]] = []
    uid: str | None = None
    for row in rows:
        current = str(row.get("norm_uid") or "")
        if not current:
            raise ValueError("score row lacks norm_uid")
        if uid is not None and current != uid:
            yield group
            group = []
        uid = current
        group.append(row)
    if group:
        yield group


def package(args: argparse.Namespace) -> dict[str, Any]:
    scores = Path(args.scores).resolve()
    meta_path = scores.with_suffix(scores.suffix + ".meta.json")
    manifest_path = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    report_path = Path(args.report_output).resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite packaged candidate shard")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        meta.get("schema_version") != SCORE_META_SCHEMA
        or meta.get("output_sha256") != sha256_file(scores)
        or meta.get("input_pairs_sha256") != args.expected_pairs_sha256
        or meta.get("classification_mode") != "binary"
        or list(meta.get("score_labels") or []) != ["REJECT", "EXACT"]
    ):
        raise ValueError("score shard metadata is not the frozen binary K200 reducer")
    contract = meta.get("checkpoint_contract") or {}
    threshold = float(contract.get("score_threshold", float("nan")))
    if (
        contract.get("threshold_provenance") != "checkpoint.dev"
        or contract.get("checkpoint_metadata_sha256")
        != args.expected_checkpoint_metadata_sha256
    ):
        raise ValueError("score shard checkpoint/dev-threshold contract differs")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_meta = (manifest.get("corpora") or {}).get(args.corpus) or {}
    bank_meta = (manifest.get("banks") or {}).get(args.task) or {}
    if corpus_meta.get("task") != args.task:
        raise ValueError("relocated manifest lacks Humor routing")
    canonical = {
        str(row["norm_uid"]): row for row in read_jsonl(Path(corpus_meta["path"]))
    }
    bank = json.loads(Path(bank_meta["path"]).read_text(encoding="utf-8"))
    metric_ids = {str(row["metric_id"]) for row in bank.get("metrics") or []}
    if len(canonical) != 77378 or len(metric_ids) != 285:
        raise ValueError("Humor canonical or bank cardinality differs")
    bank_source_sha = str(bank_meta.get("source_sha256") or bank.get("source_sha256") or "")
    excluded_truth: set[str] = set()
    excluded_truth_ref: dict[str, Any] | None = None
    if args.excluded_truth:
        excluded_truth_path = Path(args.excluded_truth).resolve()
        excluded_truth_sha = sha256_file(excluded_truth_path)
        if excluded_truth_sha != args.expected_excluded_truth_sha256:
            raise ValueError("excluded joined-truth SHA differs")
        for row in read_jsonl(excluded_truth_path):
            uid = str(row.get("norm_uid") or "")
            if not uid or uid in excluded_truth:
                raise ValueError(f"excluded truth has missing/duplicate UID: {uid!r}")
            excluded_truth.add(uid)
        if len(excluded_truth) != args.expected_excluded_uids:
            raise ValueError("excluded joined-truth UID count differs")
        excluded_truth_ref = {
            "path": str(excluded_truth_path),
            "sha256": excluded_truth_sha,
            "norm_uids": len(excluded_truth),
        }

    diagnostics: Counter[str] = Counter()
    score_rows = read_jsonl(scores)

    def output_rows() -> Iterable[dict[str, Any]]:
        for group in _groups(score_rows):
            uid = str(group[0]["norm_uid"])
            if len(group) != args.expected_depth or uid not in canonical:
                raise ValueError(f"K200 group/routing mismatch: {uid}/{len(group)}")
            if uid in excluded_truth:
                raise ValueError(f"excluded joined-truth UID reached Gemma packaging: {uid}")
            ids = [str(row.get("metric_id") or "") for row in group]
            if "" in ids or len(ids) != len(set(ids)) or not set(ids) <= metric_ids:
                raise ValueError(f"invalid K200 metric identities: {uid}")
            if any(
                row.get("schema_version") != SCORE_SCHEMA
                or str(row.get("norm_uid")) != uid
                for row in group
            ):
                raise ValueError(f"score schema/group mismatch: {uid}")
            ranked = sorted(
                enumerate(group),
                key=lambda value: (
                    -float((value[1].get("probabilities") or {}).get("EXACT", -1.0)),
                    value[0],
                ),
            )
            selected: list[tuple[int, dict[str, Any]]] = []
            seen: set[str] = set()
            for index, row in [*ranked[: args.ce_top], *list(enumerate(group[: args.retrieval_top]))]:
                metric_id = str(row["metric_id"])
                if metric_id not in seen:
                    selected.append((index, row))
                    seen.add(metric_id)
            for index, row in ranked:
                if len(selected) >= args.max_candidates:
                    break
                metric_id = str(row["metric_id"])
                if metric_id not in seen:
                    selected.append((index, row))
                    seen.add(metric_id)
            selected = selected[: args.max_candidates]
            above = sum(
                float((row.get("probabilities") or {}).get("EXACT", -1.0)) >= threshold
                for row in group
            )
            route = "zero" if above == 0 else "one" if above == 1 else "multiple"
            diagnostics[route] += 1
            diagnostics["norms"] += 1
            norm = canonical[uid]
            yield {
                "schema_version": SCHEMA,
                "task": args.task,
                "corpus": args.corpus,
                "row": int(norm.get("row", -1)),
                "norm_uid": uid,
                "bank_source_sha256": bank_source_sha,
                "candidates": [
                    {
                        "metric_id": str(row["metric_id"]),
                        "ce_exact_probability": float(row["probabilities"]["EXACT"]),
                        "retrieval_rank": index + 1,
                        "candidate_reduction_only": True,
                    }
                    for index, row in selected
                ],
                "ce_diagnostic": {
                    "above_frozen_threshold_count": above,
                    "routing_category": route,
                    "score_threshold": threshold,
                    "threshold_provenance": "checkpoint.dev",
                    "automatic_acceptance_allowed": False,
                },
                "production_policy": {
                    "requires_typed_gemma_or_strong_llm": True,
                    "ce_only_acceptance_forbidden": True,
                    "full285_rescue_on_llm_abstention_or_candidate_miss": True,
                },
            }

    count = _write_jsonl_new(output, output_rows())
    if count != int(meta.get("norm_group_count", -1)) or diagnostics["norms"] != count:
        raise ValueError("packaged norm count differs from score metadata")
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "COMPLETE_NON_AUTHORITATIVE_CANDIDATE_REDUCTION",
        "task": args.task,
        "corpus": args.corpus,
        "scores": {"path": str(scores), "sha256": sha256_file(scores)},
        "scores_meta": {"path": str(meta_path), "sha256": sha256_file(meta_path)},
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "excluded_joined_truth": excluded_truth_ref,
        "output": {"path": str(output), "sha256": sha256_file(output), "count": count},
        "candidate_depth": args.max_candidates,
        "candidate_recipe": {
            "ce_top": args.ce_top,
            "retrieval_top": args.retrieval_top,
            "deduplicated_stable_union": True,
        },
        "ce_routing_diagnostics": dict(sorted(diagnostics.items())),
        "safety": {
            "every_scored_norm_emitted_once": True,
            "ce_only_acceptance_forbidden": True,
            "every_norm_requires_typed_gemma_or_strong_llm": True,
            "full285_rescue_preserved": True,
            "joined_truth_uids_excluded_from_gemma": bool(excluded_truth_ref),
        },
    }
    _write_json_new(report_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--expected-pairs-sha256", required=True)
    parser.add_argument("--expected-checkpoint-metadata-sha256", required=True)
    parser.add_argument("--excluded-truth")
    parser.add_argument("--expected-excluded-truth-sha256")
    parser.add_argument("--expected-excluded-uids", type=int, default=0)
    parser.add_argument("--task", default="humor")
    parser.add_argument("--corpus", default="humor_multi")
    parser.add_argument("--expected-depth", type=int, default=200)
    parser.add_argument("--ce-top", type=int, default=8)
    parser.add_argument("--retrieval-top", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=16)
    args = parser.parse_args(argv)
    if min(args.expected_depth, args.ce_top, args.retrieval_top, args.max_candidates) < 1:
        parser.error("depths must be positive")
    if bool(args.excluded_truth) != bool(args.expected_excluded_truth_sha256):
        parser.error("excluded truth and its expected SHA must be supplied together")
    if args.excluded_truth and args.expected_excluded_uids < 1:
        parser.error("expected excluded UID count must be positive")
    return args


def main() -> None:
    print(json.dumps(package(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
