#!/usr/bin/env python3
"""Score retrieval/adjudication rounds against the frozen teacher slate."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import read_jsonl, write_jsonl


def safe_rate(num: int, den: int) -> float | None:
    return num / den if den else None


def load_unique(paths: list[Path], key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            value = str(row[key])
            if value in rows:
                raise ValueError(f"duplicate {key}={value} across inputs")
            rows[value] = row
    return rows


def teacher_kind(row: dict[str, Any]) -> str:
    return str(row["decision"])


def adjudication_score(teacher: dict[str, Any], predicted: dict[str, Any]) -> float:
    truth, decision = teacher_kind(teacher), predicted["decision"]
    if truth == "MATCH":
        return float(decision == "MATCH" and predicted.get("metric_id") == teacher.get("metric_id"))
    if truth == "NOISE":
        return float(decision == "NOISE")
    if truth == "BANK_GAP":
        return float(decision == "NO_CANDIDATE_FITS")
    if truth == "ABSTAIN_LEGACY":
        return float(decision not in {"MATCH", "INVALID_OUTPUT"})
    if truth in {
        "MATCH_FAMILY_ONLY",
        "NO_EXPLICIT_CRITERION",
        "CONTEXT_NEEDED",
        "GENERIC_VERDICT",
        "NO_CANDIDATE_FITS",
        "NOISE",
    }:
        return float(decision == truth)
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teachers", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--adjudications", nargs="*")
    parser.add_argument("--split", choices=("all", "train", "dev", "test"), default="all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--errors-output")
    args = parser.parse_args()

    teachers = {
        row["norm_uid"]: row
        for row in read_jsonl(Path(args.teachers))
        if args.split == "all" or row.get("split") == args.split
    }
    candidates = load_unique([Path(path) for path in args.candidates], "norm_uid")
    adjudications = (
        load_unique([Path(path) for path in args.adjudications], "norm_uid")
        if args.adjudications
        else {}
    )

    match_rows = [row for row in teachers.values() if row["decision"] == "MATCH"]
    recall = {}
    for k in (1, 3, 5, 10, 16, 30, 50, 80, 120, 180):
        hit = 0
        for teacher in match_rows:
            candidate_ids = [
                row["metric_id"] for row in candidates.get(teacher["norm_uid"], {}).get("candidates", [])
            ]
            hit += teacher.get("metric_id") in candidate_ids[:k]
        recall[f"recall_at_{k}"] = safe_rate(hit, len(match_rows))

    by_corpus: dict[str, dict[str, Any]] = {}
    for corpus in sorted({row["corpus"] for row in teachers.values()}):
        subset = [row for row in match_rows if row["corpus"] == corpus]
        by_corpus[corpus] = {
            "n_match": len(subset),
            "recall_at_16": safe_rate(
                sum(
                    row.get("metric_id")
                    in [
                        value["metric_id"]
                        for value in candidates.get(row["norm_uid"], {}).get("candidates", [])[:16]
                    ]
                    for row in subset
                ),
                len(subset),
            ),
        }

    report: dict[str, Any] = {
        "split": args.split,
        "n_teachers": len(teachers),
        "n_candidates": sum(uid in candidates for uid in teachers),
        "retrieval": {"n_match": len(match_rows), **recall, "by_corpus": by_corpus},
    }
    errors = []
    if adjudications:
        joined = [
            (teacher, adjudications[uid])
            for uid, teacher in teachers.items()
            if uid in adjudications
        ]
        scores = [adjudication_score(teacher, pred) for teacher, pred in joined]
        truth_counts = Counter(teacher_kind(teacher) for teacher, _ in joined)
        pred_counts = Counter(pred["decision"] for _, pred in joined)
        exact_match_rows = [
            (teacher, pred)
            for teacher, pred in joined
            if teacher["decision"] == "MATCH"
            and teacher.get("metric_id")
            in pred.get("candidate_ids", [])
        ]
        predicted_matches = [(teacher, pred) for teacher, pred in joined if pred["decision"] == "MATCH"]
        true_match_predictions = [
            (teacher, pred)
            for teacher, pred in joined
            if teacher["decision"] == "MATCH" and pred["decision"] == "MATCH"
        ]
        exact = sum(
            pred.get("metric_id") == teacher.get("metric_id")
            for teacher, pred in exact_match_rows
        )
        report["adjudication"] = {
            "n_joined": len(joined),
            "mean_teacher_score": float(np.mean(scores)) if scores else None,
            "truth_counts": dict(sorted(truth_counts.items())),
            "prediction_counts": dict(sorted(pred_counts.items())),
            "candidate_present_match_accuracy": safe_rate(exact, len(exact_match_rows)),
            "binary_match_precision": safe_rate(
                sum(teacher["decision"] == "MATCH" for teacher, _ in predicted_matches),
                len(predicted_matches),
            ),
            "binary_match_recall": safe_rate(
                len(true_match_predictions),
                truth_counts.get("MATCH", 0),
            ),
            "noise_recall": safe_rate(
                sum(
                    teacher["decision"] == "NOISE" and pred["decision"] == "NOISE"
                    for teacher, pred in joined
                ),
                truth_counts.get("NOISE", 0),
            ),
            "bank_gap_no_candidate_recall": safe_rate(
                sum(
                    teacher["decision"] == "BANK_GAP"
                    and pred["decision"] == "NO_CANDIDATE_FITS"
                    for teacher, pred in joined
                ),
                truth_counts.get("BANK_GAP", 0),
            ),
            "legacy_abstain_nonmatch_recall": safe_rate(
                sum(
                    teacher["decision"] == "ABSTAIN_LEGACY"
                    and pred["decision"] not in {"MATCH", "INVALID_OUTPUT"}
                    for teacher, pred in joined
                ),
                truth_counts.get("ABSTAIN_LEGACY", 0),
            ),
            "by_task": {
                task: {
                    "n": sum(teacher["task"] == task for teacher, _ in joined),
                    "mean_teacher_score": float(
                        np.mean(
                            [
                                adjudication_score(teacher, pred)
                                for teacher, pred in joined
                                if teacher["task"] == task
                            ]
                        )
                    ),
                    "match_exact_accuracy": safe_rate(
                        sum(
                            teacher["decision"] == "MATCH"
                            and pred["decision"] == "MATCH"
                            and teacher.get("metric_id") == pred.get("metric_id")
                            for teacher, pred in joined
                            if teacher["task"] == task
                        ),
                        sum(
                            teacher["decision"] == "MATCH"
                            for teacher, _ in joined
                            if teacher["task"] == task
                        ),
                    ),
                }
                for task in sorted({teacher["task"] for teacher, _ in joined})
            },
        }
        for teacher, pred in joined:
            if adjudication_score(teacher, pred) < 1.0:
                errors.append(
                    {
                        "norm_uid": teacher["norm_uid"],
                        "corpus": teacher["corpus"],
                        "split": teacher.get("split"),
                        "teacher_decision": teacher["decision"],
                        "teacher_metric_id": teacher.get("metric_id"),
                        "prediction_decision": pred["decision"],
                        "prediction_metric_id": pred.get("metric_id"),
                        "prediction_confidence": pred.get("confidence"),
                        "prediction_reason": pred.get("reason"),
                        "candidate_ids": pred.get("candidate_ids"),
                        "label_source": teacher.get("label_source"),
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.errors_output:
        write_jsonl(Path(args.errors_output), errors)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
