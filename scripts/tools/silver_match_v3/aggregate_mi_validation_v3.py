#!/usr/bin/env python3
"""Aggregate frozen task-level MI↔silver correlations with heterogeneity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

from .common import sha256_file


def _meta(rows: list[dict]) -> dict | None:
    usable = [row for row in rows if row["rho"] is not None and row["n_metrics"] > 3]
    if len(usable) < 2:
        return None
    z = np.array([np.arctanh(np.clip(row["rho"], -.999999, .999999)) for row in usable])
    variance = np.array([1.0 / (row["n_metrics"] - 3) for row in usable])
    fixed_weights = 1.0 / variance
    fixed = float(np.sum(fixed_weights * z) / np.sum(fixed_weights))
    q = float(np.sum(fixed_weights * (z - fixed) ** 2))
    df = len(usable) - 1
    c = float(np.sum(fixed_weights) - np.sum(fixed_weights ** 2) / np.sum(fixed_weights))
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    random_weights = 1.0 / (variance + tau2)
    random_z = float(np.sum(random_weights * z) / np.sum(random_weights))
    random_se = math.sqrt(1.0 / float(np.sum(random_weights)))
    critical = float(norm.ppf(.975))
    return {
        "tasks": len(usable),
        "task_names": [row["task"] for row in usable],
        "equal_task_mean_rho": float(np.mean([row["rho"] for row in usable])),
        "task_rho_range": [
            float(min(row["rho"] for row in usable)),
            float(max(row["rho"] for row in usable)),
        ],
        "positive_tasks": sum(row["rho"] > 0 for row in usable),
        "fixed_effect_fisher_rho": float(np.tanh(fixed)),
        "random_effect_fisher_rho": float(np.tanh(random_z)),
        "random_effect_rho_95": [
            float(np.tanh(random_z - critical * random_se)),
            float(np.tanh(random_z + critical * random_se)),
        ],
        "tau_squared_fisher_z": tau2,
        "cochran_q": q,
        "q_df": df,
        "i_squared": max(0.0, (q - df) / q) if q > 0 else 0.0,
        "caveat": "Fisher-z variance treats metric-level ranks as approximately independent; task source-group bootstrap intervals remain primary.",
    }


def aggregate(paths: list[Path]) -> dict:
    seen = set()
    rows = []
    artifacts = {}
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        task = str(report.get("task") or "")
        if report.get("status") != "TASK_FROZEN_ANALYSIS" or not task or task in seen:
            raise ValueError(f"invalid/duplicate task report: {path}")
        seen.add(task)
        primary = report["results"]["source_presence"]["OPT"]
        rows.append(
            {
                "task": task,
                "rho": primary["spearman_rho"],
                "rho_source_group_bootstrap_95": primary.get(
                    "source_group_bootstrap_rho_95"
                ),
                "permutation_p_two_sided": primary["permutation_p_two_sided"],
                "partial_rho": primary[
                    "partial_rho_given_log_leaf_count_and_HM"
                ],
                "n_metrics": primary["n_metrics"],
                "certificate_bank_coverage": report["certificate"]["bank_coverage"],
                "exact_matches": report["exact_matches"],
                "exact_match_rate": report["exact_match_rate"],
                "precision_claim_supported": report["precision_claim_supported"],
                "false_abstention_claim_supported": report[
                    "false_abstention_claim_supported"
                ],
            }
        )
        artifacts[task] = {"path": str(path), "sha256": sha256_file(path)}
    supported = [
        row
        for row in rows
        if row["precision_claim_supported"]
        and row["false_abstention_claim_supported"]
    ]
    return {
        "schema_version": "silver-match-v3-mi-validation-meta-v1",
        "estimand": "source_presence.OPT.spearman_rho",
        "task_reports": artifacts,
        "tasks": sorted(rows, key=lambda row: row["task"]),
        "all_released_tasks_meta": _meta(rows),
        "both_blind_claims_supported_meta": _meta(supported),
        "both_blind_claims_supported_tasks": sorted(row["task"] for row in supported),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = aggregate([Path(path).resolve() for path in args.report])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "tasks": len(result["tasks"]),
        "output": str(output),
        "output_sha256": sha256_file(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
