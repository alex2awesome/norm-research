#!/usr/bin/env python
"""Evaluate rank-articulation selection algorithms from teaching fold to opposite fold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from methods.codability.experiments.build_fresh_item_partitions import sha256_file
from methods.codability.experiments.compile_residual_isomorphism_bank import BEST_SOURCE
from methods.codability.experiments.synthesize_residual_policy_revisions import identity_loss


METRICS = ("mae_tvd", "spearman", "binary_flip_rate", "absolute_bias")


def _point(row: dict) -> dict:
    return row["certificate"]["point"]


def _robust(row: dict) -> dict:
    point = _point(row)["candidate_robust"]
    return {key: point[key] for key in METRICS}


def choose(rows: list[dict], selector: str, *, mae_window: float = 0.03) -> dict:
    if selector == "minimum_identity_loss":
        return min(rows, key=lambda row: (identity_loss(_point(row)), row["arm_id"]))
    if selector == "minimum_mae":
        return min(rows, key=lambda row: (
            _point(row)["candidate_robust"]["mae_tvd"], row["arm_id"]))
    if selector in {"maximum_rank", "maximum_rank_within_mae_window"}:
        pool = rows
        if selector.endswith("within_mae_window"):
            best = min(_point(row)["candidate_robust"]["mae_tvd"] for row in rows)
            pool = [row for row in rows
                    if _point(row)["candidate_robust"]["mae_tvd"] <= best + mae_window]
        return max(pool, key=lambda row: (
            _point(row)["candidate_robust"]["spearman"],
            -_point(row)["candidate_robust"]["mae_tvd"], row["arm_id"]))
    raise ValueError(f"unknown selector {selector}")


def summarize(*, candidate_paths: list[str], incumbent_paths: list[str],
              arm_bank_path: str, cell_id: str = "N_humor_49") -> dict:
    if len(candidate_paths) != 2 or len(incumbent_paths) != 2:
        raise ValueError("exactly two candidate and incumbent reports are required")
    candidates = {report["partition"]: report for report in
                  (json.loads(Path(path).read_text()) for path in candidate_paths)}
    incumbents = {report["partition"]: report for report in
                  (json.loads(Path(path).read_text()) for path in incumbent_paths)}
    if set(candidates) != set(incumbents) or len(candidates) != 2:
        raise ValueError("candidate and incumbent fold sets differ")
    bank = json.loads(Path(arm_bank_path).read_text())
    bank_cell = next(cell for cell in bank["cells"] if cell["id"] == cell_id)
    specs = {arm["id"]: arm for arm in bank_cell["arms"]}
    candidate_rows, incumbent_rows = {}, {}
    for partition, report in candidates.items():
        cell = next(value for value in report["cells"] if value["cell_id"] == cell_id)
        candidate_rows[partition] = {row["arm_id"]: row for row in cell["rows"]}
    for partition, report in incumbents.items():
        cell = next(value for value in report["cells"] if value["cell_id"] == cell_id)
        incumbent_rows[partition] = next(
            row for row in cell["rows"] if row["arm_id"] == BEST_SOURCE[cell_id])

    selectors = (
        "minimum_identity_loss", "minimum_mae",
        "maximum_rank_within_mae_window", "maximum_rank",
    )
    output = []
    partitions = sorted(candidates)
    for selector in selectors:
        directions = []
        for train in partitions:
            test = next(partition for partition in partitions if partition != train)
            training_pool = [
                row for arm_id, row in candidate_rows[train].items()
                if specs[arm_id].get("source_partition") == train]
            selected = choose(training_pool, selector)
            arm_id = selected["arm_id"]
            evaluation = candidate_rows[test][arm_id]
            train_metrics = _robust(selected)
            test_metrics = _robust(evaluation)
            incumbent_metrics = _robust(incumbent_rows[test])
            directions.append({
                "train_partition": train, "test_partition": test,
                "selected_arm_id": arm_id,
                "training": train_metrics,
                "opposite_fold": test_metrics,
                "opposite_fold_incumbent": incumbent_metrics,
                "mae_gain_over_incumbent": (incumbent_metrics["mae_tvd"]
                                            - test_metrics["mae_tvd"]),
                "rho_gain_over_incumbent": (test_metrics["spearman"]
                                            - incumbent_metrics["spearman"]),
                "opposite_fold_policy_isomorphic": evaluation["certificate"][
                    "policy_isomorphic"],
            })
        output.append({
            "selector": selector,
            "directions": directions,
            "improves_mae_both_directions": all(
                row["mae_gain_over_incumbent"] > 0.0 for row in directions),
            "improves_rho_both_directions": all(
                row["rho_gain_over_incumbent"] > 0.0 for row in directions),
            "isomorphic_both_directions": all(
                row["opposite_fold_policy_isomorphic"] for row in directions),
        })
    return {
        "schema": "rank_contrast_algorithm_crossfit/v1",
        "estimand": "teaching-fold selection algorithm evaluated on the opposite public fold",
        "cell_id": cell_id,
        "candidate_reports": [{"path": path, "sha256": sha256_file(path)}
                              for path in candidate_paths],
        "incumbent_reports": [{"path": path, "sha256": sha256_file(path)}
                              for path in incumbent_paths],
        "arm_bank": {"path": arm_bank_path, "sha256": sha256_file(arm_bank_path)},
        "selectors": output,
        "summary": {
            "n_selectors": len(output),
            "n_improve_mae_both_directions": sum(
                row["improves_mae_both_directions"] for row in output),
            "n_improve_rho_both_directions": sum(
                row["improves_rho_both_directions"] for row in output),
            "n_isomorphic_both_directions": sum(
                row["isomorphic_both_directions"] for row in output),
        },
        "claim_boundary": (
            "Each direction is leakage-free conditional on its selector, but this selector menu was "
            "audited after G7 scores existed. It is exploratory algorithm diagnosis, not a frozen "
            "promotion test or lockbox authorization."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate-report", action="append", required=True)
    parser.add_argument("--incumbent-report", action="append", required=True)
    parser.add_argument("--arm-bank", required=True)
    parser.add_argument("--cell-id", default="N_humor_49")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = summarize(candidate_paths=args.candidate_report,
                       incumbent_paths=args.incumbent_report,
                       arm_bank_path=args.arm_bank, cell_id=args.cell_id)
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(out), **result["summary"]}, indent=1))


if __name__ == "__main__":
    main()
