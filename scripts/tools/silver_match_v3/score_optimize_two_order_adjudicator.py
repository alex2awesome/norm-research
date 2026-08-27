#!/usr/bin/env python3
"""Score two-order adjudication strictly on an optimize-role GEPA panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .score_two_order_adjudicator import summarize


def score(
    truth_path: Path, original_path: Path, hashed_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = {
        "truth": truth_path.resolve(),
        "original": original_path.resolve(),
        "hashed": hashed_path.resolve(),
    }
    truth = list(read_jsonl(paths["truth"]))
    original = {str(row["norm_uid"]): row for row in read_jsonl(paths["original"])}
    hashed = {str(row["norm_uid"]): row for row in read_jsonl(paths["hashed"])}
    uids = {str(row.get("norm_uid") or "") for row in truth}
    if "" in uids or len(uids) != len(truth) or set(original) != uids or set(hashed) != uids:
        raise ValueError("optimize score inputs do not have exact paired coverage")
    if any(
        row.get("gepa_role") != "optimize"
        or row.get("split") != "train"
        or row.get("prompt_gradient_eligible") is not True
        or row.get("evaluation_only") is not False
        for row in truth
    ):
        raise ValueError("truth is not wholly optimize-role prompt-gradient evidence")
    tasks = {str(row.get("task") or "") for row in truth}
    banks = {str(row.get("current_bank_source_sha256") or "") for row in truth}
    prompt_hashes = {
        str(row.get("prompt_sha256") or "")
        for row in [*original.values(), *hashed.values()]
    }
    if len(tasks) != 1 or "" in tasks or len(banks) != 1 or "" in banks:
        raise ValueError("truth does not bind exactly one task and bank")
    if len(prompt_hashes) != 1 or "" in prompt_hashes:
        raise ValueError("two order outputs do not bind exactly one prompt")
    task, bank_hash, prompt_hash = next(iter(tasks)), next(iter(banks)), next(iter(prompt_hashes))
    for mode, predictions in (("original", original), ("hashed", hashed)):
        for uid, row in predictions.items():
            if (
                row.get("task") != task
                or row.get("candidate_bank_source_sha256") != bank_hash
            ):
                raise ValueError(f"{mode} task/bank mismatch: {uid}")

    errors: list[dict[str, Any]] = []
    for row in truth:
        uid = str(row["norm_uid"])
        left, right = original[uid], hashed[uid]
        truth_key = (row["decision"], row.get("metric_id"))
        left_key = (left["decision"], left.get("metric_id"))
        right_key = (right["decision"], right.get("metric_id"))
        if left_key != truth_key or right_key != truth_key:
            errors.append(
                {
                    "norm_uid": uid,
                    "task": task,
                    "corpus": row.get("corpus"),
                    "truth": {
                        "decision": row["decision"],
                        "metric_id": row.get("metric_id"),
                        "reason": row.get("reason"),
                    },
                    "original": {
                        "decision": left["decision"],
                        "metric_id": left.get("metric_id"),
                        "reason": left.get("reason"),
                        "candidate_ids": left.get("candidate_ids"),
                    },
                    "hashed": {
                        "decision": right["decision"],
                        "metric_id": right.get("metric_id"),
                        "reason": right.get("reason"),
                        "candidate_ids": right.get("candidate_ids"),
                    },
                    "order_stable": left_key == right_key,
                    "error_kind": (
                        "stable_wrong" if left_key == right_key else "order_unstable"
                    ),
                }
            )
    report = {
        "schema_version": "silver-match-v3-optimize-two-order-adjudicator-score-v1",
        "role": "optimize_prompt_gradient_only",
        "task": task,
        "bank_source_sha256": bank_hash,
        "prompt_sha256": prompt_hash,
        "prompt_gradient_evidence_allowed": True,
        "prompt_or_model_selection_performed": False,
        "scientific_evaluation_claim_allowed": False,
        "metrics": summarize(truth, original, hashed),
        "error_count": len(errors),
        "inputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
    }
    return report, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--hashed", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--errors-output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    errors_output = Path(args.errors_output).resolve()
    if output.exists() or errors_output.exists():
        raise FileExistsError("refusing to overwrite optimize score artifacts")
    report, errors = score(Path(args.truth), Path(args.original), Path(args.hashed))
    write_jsonl(errors_output, errors)
    report["errors"] = {
        "path": str(errors_output),
        "sha256": sha256_file(errors_output),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "output_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
