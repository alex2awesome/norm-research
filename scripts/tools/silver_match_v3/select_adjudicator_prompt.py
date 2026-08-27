#!/usr/bin/env python3
"""Freeze a task adjudicator prompt from original/hashed human-dev runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="NAME:ORIGINAL_JSONL:ORIGINAL_SCORE:HASHED_JSONL:HASHED_SCORE",
    )
    parser.add_argument("--candidate-depth", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    variants = []
    for spec in args.variant:
        name, original_raw, original_score_raw, hashed_raw, hashed_score_raw = spec.split(":", 4)
        original, hashed = Path(original_raw).resolve(), Path(hashed_raw).resolve()
        score_paths = [Path(original_score_raw).resolve(), Path(hashed_score_raw).resolve()]
        predictions = [
            {str(row["norm_uid"]): row for row in read_jsonl(path)}
            for path in (original, hashed)
        ]
        if set(predictions[0]) != set(predictions[1]):
            raise ValueError(f"order runs have different UID coverage: {name}")
        uids = sorted(predictions[0])
        decision_agree = sum(
            predictions[0][uid]["decision"] == predictions[1][uid]["decision"]
            for uid in uids
        )
        exact_agree = sum(
            (
                predictions[0][uid]["decision"],
                predictions[0][uid].get("metric_id"),
            )
            == (
                predictions[1][uid]["decision"],
                predictions[1][uid].get("metric_id"),
            )
            for uid in uids
        )
        scores = [json.loads(path.read_text(encoding="utf-8")) for path in score_paths]
        metas = [
            json.loads(path.with_suffix(path.suffix + ".meta.json").read_text(encoding="utf-8"))
            for path in (original, hashed)
        ]
        if metas[0]["prompt_sha256"] != metas[1]["prompt_sha256"]:
            raise ValueError(f"order runs use different prompts: {name}")
        if metas[0]["input_candidates_sha256"] != metas[1]["input_candidates_sha256"]:
            raise ValueError(f"order runs use different candidates: {name}")
        if metas[0]["max_candidates"] != args.candidate_depth or metas[1][
            "max_candidates"
        ] != args.candidate_depth:
            raise ValueError(f"variant was not evaluated at requested depth: {name}")
        adjudications = [value["adjudication"] for value in scores]
        teacher_scores = [float(value["mean_teacher_score"]) for value in adjudications]
        exact_scores = [
            float(value["by_task"][args.task]["match_exact_accuracy"])
            for value in adjudications
        ]
        variants.append(
            {
                "name": name,
                "count": len(uids),
                "prompt": metas[0]["prompt"],
                "prompt_addons": metas[0].get("prompt_addons") or [],
                "prompt_sha256": metas[0]["prompt_sha256"],
                "prompt_component_sha256": metas[0].get("prompt_component_sha256"),
                "candidate_sha256": metas[0]["input_candidates_sha256"],
                "order_exact_agreement": exact_agree / len(uids),
                "order_decision_agreement": decision_agree / len(uids),
                "dev": {
                    "original": adjudications[0],
                    "hashed": adjudications[1],
                    "min_teacher_score": min(teacher_scores),
                    "mean_teacher_score": sum(teacher_scores) / 2,
                    "min_match_exact_accuracy": min(exact_scores),
                    "mean_match_exact_accuracy": sum(exact_scores) / 2,
                },
                "inputs": {
                    "original": {"path": str(original), "sha256": sha256_file(original)},
                    "original_score": {
                        "path": str(score_paths[0]),
                        "sha256": sha256_file(score_paths[0]),
                    },
                    "hashed": {"path": str(hashed), "sha256": sha256_file(hashed)},
                    "hashed_score": {
                        "path": str(score_paths[1]),
                        "sha256": sha256_file(score_paths[1]),
                    },
                },
            }
        )
    chosen = max(
        variants,
        key=lambda row: (
            row["dev"]["min_teacher_score"],
            row["dev"]["mean_teacher_score"],
            row["dev"]["min_match_exact_accuracy"],
            row["order_exact_agreement"],
            row["name"],
        ),
    )
    report = {
        "schema_version": "silver-match-v3-adjudicator-gepa-selection-v1",
        "task": args.task,
        "selection_split": "external_dev_only",
        "candidate_depth": args.candidate_depth,
        "objective": (
            "maximize worst-order teacher score, then mean score, exact-ID accuracy, "
            "and order stability"
        ),
        "order_policy": "production MATCH requires exact original/hashed agreement",
        "chosen": chosen,
        "variants": variants,
        "adjudicator_test_consumed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
