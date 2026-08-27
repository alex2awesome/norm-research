#!/usr/bin/env python3
"""Select a verifier prompt from strict two-order human-dev scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--variant", action="append", required=True,
        help="NAME:TWO_ORDER_SCORE:ORIGINAL_JSONL:HASHED_JSONL",
    )
    parser.add_argument("--min-point-precision", type=float, default=0.90)
    parser.add_argument("--min-wilson-lower", type=float, default=0.80)
    parser.add_argument("--min-retained", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    variants = []
    for raw in args.variant:
        name, raw_score, raw_original, raw_hashed = raw.split(":", 3)
        paths = {
            "score": Path(raw_score).resolve(),
            "original": Path(raw_original).resolve(),
            "hashed": Path(raw_hashed).resolve(),
        }
        score = json.loads(paths["score"].read_text(encoding="utf-8"))
        if score.get("selection_split") != "dev":
            raise ValueError(f"{name} score is not dev-only")
        metas = {
            order: json.loads(paths[order].with_suffix(paths[order].suffix + ".meta.json").read_text())
            for order in ("original", "hashed")
        }
        if metas["original"].get("order_mode") != "original" or metas["hashed"].get("order_mode") != "hashed":
            raise ValueError(f"{name} does not contain original+hashed orders")
        if metas["original"].get("prompt_sha256") != metas["hashed"].get("prompt_sha256"):
            raise ValueError(f"{name} verifier orders use different prompts")
        if any(metas[order].get("output_sha256") != sha256_file(paths[order]) for order in metas):
            raise ValueError(f"{name} verifier output/meta hash mismatch")
        policy = score["policies"]["high_only"]
        interval = policy.get("retained_precision_wilson_95") or [0.0, 0.0]
        eligible = (
            int(policy.get("retained") or 0) >= args.min_retained
            and float(policy.get("retained_precision") or 0.0) >= args.min_point_precision
            and float(interval[0]) >= args.min_wilson_lower
        )
        variants.append(
            {
                "name": name,
                "eligible": eligible,
                "statistically_supported": eligible,
                "retained_count": int(policy.get("retained") or 0),
                "retained_precision_wilson_95": interval,
                "dev_score": score,
                "strict_policy": policy,
                "score_path": str(paths["score"]),
                "score_sha256": sha256_file(paths["score"]),
                "verification_paths": {order: str(paths[order]) for order in paths if order != "score"},
                "verification_sha256": {order: sha256_file(paths[order]) for order in paths if order != "score"},
                "prompt": metas["original"]["prompt"],
                "prompt_addons": metas["original"].get("prompt_addons") or [],
                "prompt_sha256": metas["original"]["prompt_sha256"],
                "prompt_component_sha256": metas["original"].get("prompt_component_sha256") or {},
            }
        )
    eligible = [row for row in variants if row["eligible"]]
    if not eligible:
        raise ValueError("no strict two-order verifier prompt clears the dev support gate")
    chosen = max(
        eligible,
        key=lambda row: (
            row["strict_policy"]["retained_precision_wilson_95"][0],
            row["strict_policy"]["retained_precision"],
            row["strict_policy"]["retained_recall_of_correct_proposals"],
            row["strict_policy"]["retained"],
            row["name"],
        ),
    )
    payload = {
        "schema_version": "silver-match-v3-verifier-gepa-selection-v2",
        "task": args.task,
        "selection_split": "dev",
        "objective": "maximize strict two-order high-confidence retained precision, Wilson support, then recall",
        "minimum_retained_precision": args.min_point_precision,
        "minimum_retained_precision_wilson_lower": args.min_wilson_lower,
        "minimum_retained_count_for_power": args.min_retained,
        "calibration_power_status": "supported",
        "requires_independent_audit_before_gradient_use": True,
        "chosen": chosen,
        "variants": variants,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "output_sha256": sha256_file(output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
