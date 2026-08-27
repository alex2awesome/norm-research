#!/usr/bin/env python3
"""Freeze a high-precision Gemma verifier prompt using human dev only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import sha256_file
from .score_verifier_calibration import wilson_interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="NAME:/path/dev-score.json:/path/verification.jsonl",
    )
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--min-wrong-rejection", type=float, default=0.90)
    parser.add_argument(
        "--min-retained-count",
        type=int,
        default=5,
        help="minimum dev retains needed before calibration is called adequately powered",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    variants = []
    for spec in args.variant:
        name, raw_score, raw_verification = spec.split(":", 2)
        score_path, verification_path = Path(raw_score).resolve(), Path(raw_verification).resolve()
        score = json.loads(score_path.read_text())
        meta_path = verification_path.with_suffix(verification_path.suffix + ".meta.json")
        meta = json.loads(meta_path.read_text())
        eligible = (
            (score.get("retained_precision") or 0.0) >= args.min_precision
            and (score.get("wrong_proposal_rejection_rate") or 0.0)
            >= args.min_wrong_rejection
        )
        retained_count = int(score.get("counts", {}).get("retained", 0))
        retained_true = int(score.get("counts", {}).get("retained_true", 0))
        precision_interval = score.get("retained_precision_wilson_95") or wilson_interval(
            retained_true, retained_count
        )
        statistically_supported = retained_count >= args.min_retained_count
        variants.append(
            {
                "name": name,
                "eligible": eligible,
                "statistically_supported": statistically_supported,
                "retained_count": retained_count,
                "retained_precision_wilson_95": precision_interval,
                "dev_score": score,
                "score_path": str(score_path),
                "score_sha256": sha256_file(score_path),
                "verification_path": str(verification_path),
                "verification_sha256": sha256_file(verification_path),
                "prompt": meta["prompt"],
                "prompt_addons": meta.get("prompt_addons") or [],
                "prompt_sha256": meta["prompt_sha256"],
                "prompt_component_sha256": meta.get("prompt_component_sha256"),
            }
        )
    eligible = [row for row in variants if row["eligible"]]
    if not eligible:
        raise ValueError("no verifier prompt clears the high-precision dev gate")
    chosen = max(
        eligible,
        key=lambda row: (
            row["dev_score"]["retained_precision"],
            row["dev_score"]["retained_recall_of_correct_proposals"],
            row["dev_score"]["wrong_proposal_rejection_rate"],
            row["dev_score"]["conflict_exact_correction_rate"],
            row["name"],
        ),
    )
    payload = {
        "schema_version": "silver-match-v3-verifier-gepa-selection-v2",
        "task": args.task,
        "selection_split": "dev",
        "objective": "maximize retained exact-label precision before recall",
        "minimum_retained_precision": args.min_precision,
        "minimum_wrong_proposal_rejection_rate": args.min_wrong_rejection,
        "minimum_retained_count_for_power": args.min_retained_count,
        "calibration_power_status": (
            "supported" if chosen["statistically_supported"] else "underpowered"
        ),
        "requires_independent_audit_before_gradient_use": not chosen[
            "statistically_supported"
        ],
        "chosen": chosen,
        "variants": variants,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
