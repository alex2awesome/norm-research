#!/usr/bin/env python3
"""Seal the exact-leaf GEPA gate before any r5 production labeling."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .common import sha256_file
from .score_verifier_calibration import wilson_interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument("--panel-split", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="NAME:/path/prompt-train-score.json:/path/prompt-dev-score.json",
    )
    parser.add_argument("--min-exact-precision", type=float, default=0.90)
    parser.add_argument("--min-exact-ci-lower", type=float, default=0.80)
    parser.add_argument("--min-confirmed-support", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = Path(args.manifest).resolve()
    candidates = Path(args.candidate_pool).resolve()
    panel = Path(args.panel_split).resolve()
    variants = []
    for spec in args.variant:
        name, raw_train, raw_dev = spec.split(":", 2)
        train_path, dev_path = Path(raw_train).resolve(), Path(raw_dev).resolve()
        train = json.loads(train_path.read_text(encoding="utf-8"))
        dev = json.loads(dev_path.read_text(encoding="utf-8"))
        if train.get("panel_role") != "prompt_train" or dev.get("panel_role") != "prompt_dev":
            raise ValueError(f"variant does not use train/dev prompt panels: {name}")
        if train.get("selection_universe") != "predeclared_train_only" or dev.get(
            "selection_universe"
        ) != "predeclared_train_only":
            raise ValueError(f"variant GEPA universe is not train-only: {name}")
        if train.get("prompt_sha256") != dev.get("prompt_sha256"):
            raise ValueError(f"train/dev prompt mismatch: {name}")
        strict = (dev.get("metrics") or {}).get("strict_consensus") or {}
        precision = strict.get("exact_id_precision")
        support = int(strict.get("confirmed_match_count") or 0)
        correct = int(strict.get("correct_exact_id_count") or 0)
        precision_interval = wilson_interval(correct, support)
        eligible = (
            precision is not None
            and float(precision) >= args.min_exact_precision
            and precision_interval is not None
            and precision_interval[0] >= args.min_exact_ci_lower
            and support >= args.min_confirmed_support
        )
        variants.append(
            {
                "name": name,
                "prompt_sha256": dev["prompt_sha256"],
                "eligible": eligible,
                "exact_leaf_dev": strict,
                "exact_leaf_precision_wilson_95": precision_interval,
                "train_score": {"path": str(train_path), "sha256": sha256_file(train_path)},
                "dev_score": {"path": str(dev_path), "sha256": sha256_file(dev_path)},
            }
        )
    eligible = [row for row in variants if row["eligible"]]
    chosen = (
        max(
            eligible,
            key=lambda row: (
                row["exact_leaf_dev"]["exact_id_precision"],
                row["exact_leaf_dev"]["confirmed_match_count"],
                row["exact_leaf_dev"]["exact_id_recall_of_truth_matches"],
                row["name"],
            ),
        )
        if eligible
        else None
    )
    payload = {
        "schema_version": "silver-match-v3-r5-gepa-exact-leaf-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "status": (
            "ELIGIBLE_FOR_INDEPENDENT_PRODUCTION_CONSENSUS"
            if chosen
            else "REJECTED_NO_EXACT_PROMPT_GATE"
        ),
        "target": "exact_frozen_bank_leaf_id",
        "family_equivalence_is_primary": False,
        "thresholds": {
            "minimum_exact_id_precision": args.min_exact_precision,
            "minimum_exact_id_precision_ci_lower": args.min_exact_ci_lower,
            "minimum_confirmed_match_support": args.min_confirmed_support,
        },
        "chosen": chosen,
        "variants": variants,
        "inputs": {
            "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
            "candidate_pool": {
                "path": str(candidates),
                "sha256": sha256_file(candidates),
            },
            "train_only_panel_split": {"path": str(panel), "sha256": sha256_file(panel)},
        },
        "gradient_authorized": False,
        "next_step_if_rejected": (
            "retain promoted r4; any new r5 attempt requires fresh independent exact-leaf "
            "labels and a new blind precision audit"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
