#!/usr/bin/env python3
"""Freeze the clean, task-specific Humor MATCH teacher for Nemotron LoRA v1.

The current optimize panel remains train-only.  Fully consumed historical
select panels are re-declared as retriever supervision and split by immutable
source-group hashing.  Fresh select identities are an evaluation firewall and
may not overlap any teacher UID or source group.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .train_nemotron_lora import source_group_key, split_source_group


TASK = "humor"
BANK_SOURCE_SHA256 = (
    "1b4a29d34b4ef4d999e0cb0b2d1125286372349ff6dfa21a6adc5bc8e76f0de9"
)


def _ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _unique_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    uids = [str(row.get("norm_uid") or "") for row in rows]
    if not rows or "" in uids or len(uids) != len(set(uids)):
        raise ValueError(f"empty, missing, or duplicate norm_uid: {path}")
    return rows


def _verify_bridge(
    path: Path,
    report_path: Path,
    *,
    expected_count: int,
    expected_schema: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _unique_rows(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_ref = report.get("output") or {}
    if (
        report.get("schema_version") != expected_schema
        or report.get("task") != TASK
        or report.get("bank_source_sha256") != BANK_SOURCE_SHA256
        or int(report.get("count", -1)) != expected_count
        or len(rows) != expected_count
        or Path(str(output_ref.get("path") or "")).resolve() != path
        or output_ref.get("sha256") != sha256_file(path)
    ):
        raise ValueError(f"bridge/report binding failure: {path}")
    return rows, report


def _exact_consensus_support(row: dict[str, Any]) -> int:
    decision = str(row.get("decision") or "")
    metric_id = str(row.get("metric_id") or "")
    sources = set(str(value) for value in row.get("agreement_sources") or [])
    predictions = row.get("source_predictions") or {}
    agreeing = {
        str(name)
        for name, prediction in predictions.items()
        if str(prediction.get("decision") or "") == decision
        and str(prediction.get("metric_id") or "") == metric_id
    }
    if len(sources) < 2 or not sources <= agreeing:
        raise ValueError(
            f"row lacks two-source exact consensus: {row.get('norm_uid')} "
            f"declared={sorted(sources)} agreeing={sorted(agreeing)}"
        )
    return len(sources)


def freeze(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = Path(args.manifest).resolve()
    optimize_path = Path(args.optimize_bridge).resolve()
    optimize_report_path = Path(args.optimize_report).resolve()
    historical_path = Path(args.historical_bridge).resolve()
    historical_report_path = Path(args.historical_report).resolve()
    fresh_identities_path = Path(args.fresh_identities).resolve()
    fresh_freeze_path = Path(args.fresh_freeze).resolve()
    candidates_path = Path(args.candidates).resolve()
    candidates_meta_path = Path(args.candidates_meta).resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank = (manifest.get("banks") or {}).get(TASK) or {}
    if bank.get("source_sha256") != BANK_SOURCE_SHA256:
        raise ValueError("unexpected Humor bank source identity")
    corpus_meta = (manifest.get("corpora") or {}).get("humor_multi") or {}
    norms_path = Path(str(corpus_meta.get("path") or "")).resolve()
    if (
        corpus_meta.get("task") != TASK
        or corpus_meta.get("sha256") != sha256_file(norms_path)
    ):
        raise ValueError("strict task-local manifest does not bind merged norms")
    norms = {str(row["norm_uid"]): row for row in _unique_rows(norms_path)}

    optimize, optimize_report = _verify_bridge(
        optimize_path,
        optimize_report_path,
        expected_count=296,
        expected_schema="silver-match-v3-ce-optimize-truth-bridge-report-v1",
    )
    historical, historical_report = _verify_bridge(
        historical_path,
        historical_report_path,
        expected_count=600,
        expected_schema="silver-match-v3-ce-historical-train-bridge-report-v1",
    )
    if optimize_report.get("status") != "COMPLETE":
        raise ValueError("optimize truth bridge is not complete")
    if historical_report.get("status") != (
        "FROZEN_HISTORICAL_TRAIN_AFTER_NEW_SELECT_FREEZE"
    ):
        raise ValueError("historical truth bridge lacks frozen role transition")

    fresh = _unique_rows(fresh_identities_path)
    fresh_freeze = json.loads(fresh_freeze_path.read_text(encoding="utf-8"))
    fresh_ref = (fresh_freeze.get("outputs") or {}).get("identities") or {}
    if (
        fresh_freeze.get("status") != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or int(fresh_freeze.get("selected_count", -1)) != 300
        or len(fresh) != 300
        or fresh_ref.get("sha256") != sha256_file(fresh_identities_path)
    ):
        raise ValueError("fresh select freeze/identity binding failure")

    all_rows = optimize + historical
    all_uids = [str(row["norm_uid"]) for row in all_rows]
    all_groups = [str(row.get("source_group") or "") for row in all_rows]
    if (
        len(all_uids) != 896
        or len(set(all_uids)) != 896
        or "" in all_groups
        or len(set(all_groups)) != 896
    ):
        raise ValueError("clean truth universe is not exactly 896 source-disjoint rows")
    if set(all_uids) - set(norms):
        raise ValueError("clean truth UID absent from strict task-local manifest")
    fresh_uids = {str(row["norm_uid"]) for row in fresh}
    fresh_groups = {str(row["source_group"]) for row in fresh}
    if set(all_uids) & fresh_uids or set(all_groups) & fresh_groups:
        raise ValueError("fresh select overlaps the clean retriever truth universe")

    candidate_meta = json.loads(candidates_meta_path.read_text(encoding="utf-8"))
    if (
        candidate_meta.get("task") != TASK
        or int((candidate_meta.get("output") or {}).get("count", -1)) != 896
        or (candidate_meta.get("output") or {}).get("sha256")
        != sha256_file(candidates_path)
        or int(candidate_meta.get("expected_k", -1)) != 50
    ):
        raise ValueError("frozen K50 candidate binding failure")
    candidate_uids = {str(row["norm_uid"]) for row in _unique_rows(candidates_path)}
    if candidate_uids != set(all_uids):
        raise ValueError("K50 candidates do not exactly cover all 896 clean-truth rows")

    teachers: list[dict[str, Any]] = []
    truth_decisions: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    split_metrics: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    support_counts: Counter[int] = Counter()
    prior_permanent_exclusions = 0
    for generation, rows in (("current_optimize", optimize), ("historical", historical)):
        for row in rows:
            uid = str(row["norm_uid"])
            truth_decisions[str(row.get("decision") or "MISSING")] += 1
            if row.get("decision") != "MATCH":
                continue
            metric_id = str(row.get("metric_id") or "")
            if not metric_id:
                raise ValueError(f"MATCH row lacks metric_id: {uid}")
            support_counts[_exact_consensus_support(row)] += 1
            norm = norms[uid]
            canonical_group = source_group_key(norm)
            if generation == "current_optimize":
                split = "train"
            else:
                split = split_source_group(
                    canonical_group,
                    seed=args.split_seed,
                    train_percent=80,
                    dev_percent=10,
                )
            prior_excluded = norm.get("permanently_excluded_from_retriever_gradients") is True
            prior_permanent_exclusions += int(prior_excluded)
            teachers.append(
                {
                    "schema_version": "silver-match-v3-humor-clean-nemotron-teacher-v1",
                    "task": TASK,
                    "corpus": str(norm["corpus"]),
                    "norm_uid": uid,
                    "source_group": canonical_group,
                    "decision": "MATCH",
                    "metric_id": metric_id,
                    "acceptable_metric_ids": [metric_id],
                    "current_bank_source_sha256": BANK_SOURCE_SHA256,
                    "split": split,
                    "supervision_strength": "strong",
                    "label_source": "exact_multi_pass_clean_truth_retriever_bridge_v1",
                    "gradient_eligible": True,
                    "retriever_training_eligible": True,
                    "ce_training_eligible": True,
                    "truth_generation": generation,
                    "source_truth_sha256": str(row.get("source_truth_sha256") or ""),
                    "agreement_sources": list(row.get("agreement_sources") or []),
                    "exact_consensus_support": len(row.get("agreement_sources") or []),
                    "prior_permanent_retriever_exclusion": prior_excluded,
                    "role_transition_authority": (
                        "root_directive_2026-07-13_clean_truth_task_specific_humor_"
                        "nemotron_lora_v1"
                    ),
                }
            )
            split_counts[split] += 1
            split_metrics[split].add(metric_id)

    teachers.sort(key=lambda row: str(row["norm_uid"]))
    if len(teachers) != 388:
        raise ValueError(f"expected 388 exact MATCH teachers, got {len(teachers)}")
    if set(split_counts) != {"train", "dev", "test"} or min(split_counts.values()) < 1:
        raise ValueError(f"explicit teacher split is empty: {dict(split_counts)}")
    split_groups: dict[str, set[str]] = {
        split: {str(row["source_group"]) for row in teachers if row["split"] == split}
        for split in ("train", "dev", "test")
    }
    if any(
        split_groups[a] & split_groups[b]
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))
    ):
        raise ValueError("source group appears in more than one explicit split")

    report = {
        "schema_version": "silver-match-v3-humor-clean-nemotron-teacher-report-v1",
        "status": "FROZEN_CLEAN_TRUTH_MATCH_ONLY_READY",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "task": TASK,
        "bank_source_sha256": BANK_SOURCE_SHA256,
        "split_seed": args.split_seed,
        "split_policy": {
            "current_optimize": "train_only",
            "historical": "source_group_sha256_80_10_10",
            "fresh_select": "forbidden_evaluation_firewall",
        },
        "truth_universe_rows": 896,
        "truth_decision_counts": dict(sorted(truth_decisions.items())),
        "teacher_rows": len(teachers),
        "teacher_split_counts": dict(sorted(split_counts.items())),
        "teacher_metric_coverage": {
            split: len(split_metrics[split]) for split in ("train", "dev", "test")
        },
        "exact_consensus_support_counts": {
            str(key): value for key, value in sorted(support_counts.items())
        },
        "weak_or_forced_positive_rows": 0,
        "fresh_select_uid_overlap": 0,
        "fresh_select_source_group_overlap": 0,
        "cross_split_source_group_overlap": 0,
        "prior_permanent_retriever_exclusion_rows_explicitly_reauthorized": (
            prior_permanent_exclusions
        ),
        "role_transition": {
            "append_only_teacher_bridge": True,
            "source_truth_labels_changed": False,
            "source_norm_rows_changed": False,
            "authority": (
                "root directive: validated 896-row clean truth is sufficient for "
                "task-specific Humor Nemotron LoRA v1"
            ),
        },
        "inputs": {
            "manifest": _ref(manifest_path),
            "merged_norms": _ref(norms_path),
            "optimize_bridge": _ref(optimize_path),
            "optimize_bridge_report": _ref(optimize_report_path),
            "historical_bridge": _ref(historical_path),
            "historical_bridge_report": _ref(historical_report_path),
            "fresh_select_identities": _ref(fresh_identities_path),
            "fresh_select_freeze": _ref(fresh_freeze_path),
            "frozen_k50_candidates": _ref(candidates_path),
            "frozen_k50_candidates_meta": _ref(candidates_meta_path),
        },
    }
    return teachers, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--optimize-bridge", required=True)
    parser.add_argument("--optimize-report", required=True)
    parser.add_argument("--historical-bridge", required=True)
    parser.add_argument("--historical-report", required=True)
    parser.add_argument("--fresh-identities", required=True)
    parser.add_argument("--fresh-freeze", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidates-meta", required=True)
    parser.add_argument("--split-seed", type=int, default=874192)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite frozen Nemotron teacher")
    rows, report = freeze(args)
    write_jsonl(output, rows)
    report["output"] = {**_ref(output), "count": len(rows)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {**report, "report_sha256": sha256_file(report_path)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
