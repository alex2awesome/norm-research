#!/usr/bin/env python
"""Summarize whether rule recipes move the direct policy-isomorphism frontier on both folds."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from methods.codability.experiments.build_fresh_item_partitions import sha256_file
from methods.codability.experiments.compile_residual_isomorphism_bank import BEST_SOURCE


FOLD_SUFFIX = re.compile(r"_from_(?:prompt_selection|unit_certification)$")
METRICS = ("mae_tvd", "spearman", "binary_flip_rate", "absolute_bias")
IDENTITY_EQUIVALENCE_MARGINS = {
    "mae_tvd": 0.0,
    "spearman": 0.05,
    "binary_flip_rate": 0.02,
    "absolute_bias": 0.02,
}


def recipe_id(arm_id: str) -> str:
    return FOLD_SUFFIX.sub("_from_crossfit_source", arm_id)


def _metrics(row: dict) -> dict:
    return {key: row["certificate"]["point"]["candidate_robust"][key] for key in METRICS}


def _target_metrics(row: dict) -> dict:
    return {key: row["certificate"]["point"]["target_self_robust"][key] for key in METRICS}


def identity_excess(values: dict, target: dict) -> dict:
    return {
        "mae_tvd": max(values["mae_tvd"] - target["mae_tvd"], 0.0),
        "spearman": max(target["spearman"] - values["spearman"], 0.0),
        "binary_flip_rate": max(values["binary_flip_rate"] - target["binary_flip_rate"], 0.0),
        "absolute_bias": max(values["absolute_bias"] - target["absolute_bias"], 0.0),
    }


def _pareto(challenger: dict, incumbent: dict, *, tolerance: float = 1e-12) -> bool:
    return bool(all(challenger[key] <= incumbent[key] + tolerance for key in METRICS)
                and any(challenger[key] < incumbent[key] - tolerance for key in METRICS))


def _margin_frontier(challenger: dict, incumbent: dict, *,
                     challenger_excess: dict | None = None,
                     incumbent_excess: dict | None = None) -> bool:
    """Primary MAE must improve; secondary identity deficits may vary only within frozen margins."""
    challenger_secondary = challenger_excess or challenger
    incumbent_secondary = incumbent_excess or incumbent
    return bool(
        challenger["mae_tvd"] < incumbent["mae_tvd"]
        and challenger_secondary["spearman"] <= (
            incumbent_secondary["spearman"]
            + IDENTITY_EQUIVALENCE_MARGINS["spearman"])
        and challenger_secondary["binary_flip_rate"] <= (
            incumbent_secondary["binary_flip_rate"]
            + IDENTITY_EQUIVALENCE_MARGINS["binary_flip_rate"])
        and challenger_secondary["absolute_bias"] <= (
            incumbent_secondary["absolute_bias"]
            + IDENTITY_EQUIVALENCE_MARGINS["absolute_bias"])
    )


def summarize(*, candidate_reports: list[dict], incumbent_reports: list[dict],
              candidate_paths: list[str], incumbent_paths: list[str],
              arm_bank_path: str) -> dict:
    if len(candidate_reports) != 2 or len(incumbent_reports) != 2:
        raise ValueError("exactly two candidate and two incumbent fold reports are required")
    fold_ids = [report["partition"] for report in candidate_reports]
    if set(fold_ids) != {report["partition"] for report in incumbent_reports}:
        raise ValueError("candidate/incumbent partitions do not match")
    incumbents_by_fold = {}
    candidates_by_fold = {}
    domain_by_cell = {cell["cell_id"]: cell["domain"]
                      for cell in candidate_reports[0]["cells"]}
    for report in incumbent_reports:
        incumbents_by_fold[report["partition"]] = {
            cell["cell_id"]: next(row for row in cell["rows"]
                                  if row["arm_id"] == BEST_SOURCE[cell["cell_id"]])
            for cell in report["cells"]
        }
    for report in candidate_reports:
        candidates_by_fold[report["partition"]] = {
            cell["cell_id"]: {recipe_id(row["arm_id"]): row for row in cell["rows"]}
            for cell in report["cells"]
        }
    cells = []
    cell_ids = sorted(set.intersection(*(
        set(values) for values in candidates_by_fold.values())))
    for cell_id in cell_ids:
        recipe_sets = [set(candidates_by_fold[fold][cell_id]) for fold in fold_ids]
        common_recipes = sorted(set.intersection(*recipe_sets))
        rows = []
        for recipe in common_recipes:
            folds = []
            for fold in fold_ids:
                challenger_row = candidates_by_fold[fold][cell_id][recipe]
                incumbent_row = incumbents_by_fold[fold][cell_id]
                challenger = _metrics(challenger_row)
                incumbent = _metrics(incumbent_row)
                target = _target_metrics(challenger_row)
                challenger_excess = identity_excess(challenger, target)
                incumbent_excess = identity_excess(incumbent, target)
                folds.append({
                    "partition": fold,
                    "arm_id": challenger_row["arm_id"],
                    "incumbent_arm_id": incumbent_row["arm_id"],
                    "challenger": challenger,
                    "incumbent": incumbent,
                    "target_self": target,
                    "identity_excess": challenger_excess,
                    "incumbent_identity_excess": incumbent_excess,
                    "mae_gain_over_incumbent": incumbent["mae_tvd"] - challenger["mae_tvd"],
                    "rho_gain_over_incumbent": challenger["spearman"] - incumbent["spearman"],
                    "identity_pareto_improves": _pareto(challenger_excess, incumbent_excess),
                    "identity_margin_frontier_improves": _margin_frontier(
                        challenger, incumbent, challenger_excess=challenger_excess,
                        incumbent_excess=incumbent_excess),
                    "target_identity_valid": challenger_row["certificate"]["gates"][
                        "target_identity_valid"],
                    "policy_isomorphic": challenger_row["certificate"]["policy_isomorphic"],
                })
            target_valid = all(row["target_identity_valid"] for row in folds)
            rows.append({
                "recipe_id": recipe,
                "folds": folds,
                "target_identity_valid_on_both_folds": target_valid,
                "stable_identity_pareto_improvement": bool(
                    target_valid and all(row["identity_pareto_improves"] for row in folds)),
                "stable_identity_margin_frontier_improvement": bool(
                    target_valid and all(row["identity_margin_frontier_improves"]
                                         for row in folds)),
                "stable_point_mae_improvement": all(
                    row["mae_gain_over_incumbent"] > 0 for row in folds),
                "isomorphic_on_both_folds": all(row["policy_isomorphic"] for row in folds),
                "worst_mae_tvd": max(row["challenger"]["mae_tvd"] for row in folds),
                "worst_mae_excess": max(row["identity_excess"]["mae_tvd"] for row in folds),
                "worst_rho_deficit": max(row["identity_excess"]["spearman"] for row in folds),
            })
        rows.sort(key=lambda row: (
            not row["isomorphic_on_both_folds"],
            not row["stable_identity_margin_frontier_improvement"],
            not row["stable_identity_pareto_improvement"],
            not row["stable_point_mae_improvement"],
            row["worst_mae_excess"], row["worst_rho_deficit"], row["recipe_id"],
        ))
        cells.append({
            "cell_id": cell_id,
            "domain": domain_by_cell[cell_id],
            "incumbent_arm_id": BEST_SOURCE[cell_id],
            "n_recipes_on_both_folds": len(rows),
            "n_stable_identity_pareto_improvements": sum(
                row["stable_identity_pareto_improvement"] for row in rows),
            "n_stable_identity_margin_frontier_improvements": sum(
                row["stable_identity_margin_frontier_improvement"] for row in rows),
            "n_stable_point_mae_improvements": sum(
                row["stable_point_mae_improvement"] for row in rows),
            "n_isomorphic_on_both_folds": sum(row["isomorphic_on_both_folds"] for row in rows),
            "ranked_recipes": rows,
        })
    return {
        "schema": "crossfit_policy_rule_frontier/v1",
        "estimand": "opposite-fold direct 3B reconstruction of the 8B sparse policy",
        "folds": fold_ids,
        "candidate_reports": [{"path": path, "sha256": sha256_file(path)}
                              for path in candidate_paths],
        "incumbent_reports": [{"path": path, "sha256": sha256_file(path)}
                              for path in incumbent_paths],
        "arm_bank": {"path": arm_bank_path, "sha256": sha256_file(arm_bank_path)},
        "cells": cells,
        "summary": {
            "n_cells": len(cells),
            "n_stable_identity_pareto_improvements": sum(
                cell["n_stable_identity_pareto_improvements"] for cell in cells),
            "n_stable_identity_margin_frontier_improvements": sum(
                cell["n_stable_identity_margin_frontier_improvements"] for cell in cells),
            "n_stable_point_mae_improvements": sum(
                cell["n_stable_point_mae_improvements"] for cell in cells),
            "n_isomorphic_on_both_folds": sum(
                cell["n_isomorphic_on_both_folds"] for cell in cells),
        },
        "identity_equivalence_margins": IDENTITY_EQUIVALENCE_MARGINS,
        "promotion_rule": ("A new recipe may displace the intact-text incumbent only after "
                           "primary MAE improves on both public folds while secondary identity "
                           "deficits are non-worse under the certificate's already-frozen "
                           "equivalence margins; paired uncertainty and semantic/provenance "
                           "audits remain subsequent gates. Strict Pareto status is retained "
                           "separately."),
        "claim_boundary": ("Stable point improvement is candidate generation evidence, not a "
                           "confirmatory claim and not lockbox authorization."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate-report", action="append", required=True)
    parser.add_argument("--incumbent-report", action="append", required=True)
    parser.add_argument("--arm-bank", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    candidate_reports = [json.loads(Path(path).read_text())
                         for path in args.candidate_report]
    incumbent_reports = [json.loads(Path(path).read_text())
                         for path in args.incumbent_report]
    result = summarize(
        candidate_reports=candidate_reports, incumbent_reports=incumbent_reports,
        candidate_paths=args.candidate_report, incumbent_paths=args.incumbent_report,
        arm_bank_path=args.arm_bank)
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(out), **result["summary"]}, indent=1))


if __name__ == "__main__":
    main()
