#!/usr/bin/env python
"""Run and aggregate the frozen fixed-target name-surface atlas manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from methods.codability.experiments.fixed_target_surface import (
    build_fixed_target_surface,
    load_surface,
    save_surface,
)
from methods.codability.experiments.surface_comparison import compare_surfaces
from methods.codability.name_sufficiency import DATA


MANIFEST_PATH = Path(__file__).with_name("name_surface_atlas_manifest_v1.json")


def load_atlas_manifest(path: str | Path = MANIFEST_PATH) -> dict:
    return json.loads(Path(path).read_text())


def manifest_sha256(path: str | Path = MANIFEST_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _by_id(rows: list[dict]) -> dict[str, dict]:
    out = {row["id"]: row for row in rows}
    if len(out) != len(rows):
        raise ValueError("atlas manifest contains duplicate ids")
    return out


def surface_path(out_dir: str | Path, surface_id: str) -> Path:
    return Path(out_dir) / "surfaces" / f"{surface_id}.npz"


def comparison_path(out_dir: str | Path, comparison_id: str) -> Path:
    return Path(out_dir) / "comparisons" / f"{comparison_id}.json"


def run_surface(surface_id: str, *, data_dir: str, out_dir: str,
                manifest: dict, n_boot: int | None = None) -> dict:
    specs = _by_id(manifest["surfaces"])
    if surface_id not in specs:
        raise ValueError(f"unknown surface id {surface_id!r}")
    spec = specs[surface_id]
    defaults = manifest["default_analysis"]
    bundle = build_fixed_target_surface(
        data_dir=data_dir, domains=spec["domains"], executor_tag=spec["executor_tag"],
        target_tag=spec["target_tag"],
        executor_grid_template=spec.get("executor_grid_template"),
        target_grid_template=spec.get("target_grid_template"),
        messages_grid_template=spec.get("messages_grid_template"),
        divergence=defaults["divergence"],
        min_target_information=defaults["min_target_information"],
        train_frac=defaults["train_frac"], n_boot=n_boot or defaults["n_boot"],
        seed=defaults["seed"])
    bundle["report"]["atlas"] = {"surface_id": surface_id,
                                  "manifest_sha256": manifest_sha256()}
    path, report = save_surface(bundle, surface_path(out_dir, surface_id))
    return {"id": surface_id, "surface": str(path), "report": str(report),
            "n_metric_cells": bundle["report"]["n_metric_cells"],
            "n_ineligible_metrics": bundle["report"]["n_ineligible_metrics"],
            "n_errors": bundle["report"]["n_errors"]}


def run_comparison(comparison_id: str, *, out_dir: str, manifest: dict) -> dict:
    specs = _by_id(manifest["comparisons"])
    if comparison_id not in specs:
        raise ValueError(f"unknown comparison id {comparison_id!r}")
    spec = specs[comparison_id]
    defaults = manifest["default_analysis"]
    small_path = surface_path(out_dir, spec["small"])
    big_path = surface_path(out_dir, spec["big"])
    result = compare_surfaces(
        load_surface(small_path), load_surface(big_path),
        gap_delta=defaults["gap_delta"], equivalence_delta=defaults["equivalence_delta"],
        min_signature_rho=defaults["min_signature_rho"],
        signature_equivalence_delta=defaults["signature_equivalence_delta"])
    result["atlas"] = {"comparison_id": comparison_id, "ladder": spec["ladder"],
                       "manifest_sha256": manifest_sha256()}
    out = comparison_path(out_dir, comparison_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    return {"id": comparison_id, "path": str(out),
            "n_evaluable": result["pooled"]["n_evaluable"],
            "n_confirmed_gaps": result["pooled"]["baseline_gap_confirmed"],
            "n_substitutions": result["pooled"][
                "methodological_substitution_among_confirmed_gaps"]["success"]}


def aggregate_atlas(*, out_dir: str, manifest: dict) -> dict:
    surface_rows, comparison_rows = [], []
    for spec in manifest["surfaces"]:
        path = surface_path(out_dir, spec["id"])
        if not path.exists():
            surface_rows.append({"id": spec["id"], "status": "missing"})
            continue
        surface = load_surface(path)
        report = surface["report"]
        surface_rows.append({"id": spec["id"], "status": "complete",
                             "sha256": surface["sha256"],
                             "n_metric_cells": report["n_metric_cells"],
                             "n_ineligible_metrics": report["n_ineligible_metrics"],
                             "n_errors": report["n_errors"]})
    for spec in manifest["comparisons"]:
        path = comparison_path(out_dir, spec["id"])
        if not path.exists():
            comparison_rows.append({"id": spec["id"], "ladder": spec["ladder"],
                                    "status": "missing"})
            continue
        result = json.loads(path.read_text())
        summary = result["pooled"]
        valid_rows = [row for domain in result["by_domain"].values()
                      for row in domain["per_metric"] if row.get("heldout", {}).get("valid")]
        gap_rows = [row for row in valid_rows
                    if row["heldout"]["gates"]["baseline_gap_confirmed"]]
        comparison_rows.append({
            "id": spec["id"], "ladder": spec["ladder"], "status": "complete",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_evaluable": summary["n_evaluable"],
            "n_confirmed_gaps": summary["baseline_gap_confirmed"],
            "n_no_confirmed_gap": len(valid_rows) - len(gap_rows),
            "n_information_improved": sum(
                row["heldout"]["gates"]["articulation_improvement_confirmed"]
                for row in gap_rows),
            "n_information_noninferior": sum(
                row["heldout"]["gates"]["noninferior_to_big_sparse"]
                for row in gap_rows),
            "n_signature_improved": sum(
                row["heldout"]["gates"]["signature_improved"]
                for row in gap_rows),
            "n_signature_noninferior": sum(
                row["heldout"]["gates"]["signature_noninferior_to_big"]
                for row in gap_rows),
            "n_substitutions": summary[
                "methodological_substitution_among_confirmed_gaps"]["success"],
            "n_familywise_substitutions": result.get("familywise_substitution_count", 0),
        })
    complete_comparisons = [row for row in comparison_rows if row["status"] == "complete"]
    atlas = {
        "schema": "name_surface_atlas/v1", "manifest_sha256": manifest_sha256(),
        "surfaces": surface_rows, "comparisons": comparison_rows,
        "coverage": {"surfaces_complete": sum(row["status"] == "complete" for row in surface_rows),
                     "surfaces_total": len(surface_rows),
                     "comparisons_complete": len(complete_comparisons),
                     "comparisons_total": len(comparison_rows)},
        "totals_over_comparisons_not_unique_metrics": {
            "evaluable": sum(row["n_evaluable"] for row in complete_comparisons),
            "confirmed_gaps": sum(row["n_confirmed_gaps"] for row in complete_comparisons),
            "cellwise_substitutions": sum(
                row["n_substitutions"] for row in complete_comparisons),
            "familywise_substitutions": sum(
                row["n_familywise_substitutions"] for row in complete_comparisons)},
        "ladder_caveats": manifest["ladder_caveats"],
        "potential_status": ("Not evaluated from legacy word costs; requires finite debts and "
                             "composable CUF-certified articulation units."),
        "paper_grade_claim_eligible": False,
    }
    out = Path(out_dir) / "atlas_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(atlas, indent=1))
    return atlas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["surface", "comparison", "aggregate"])
    parser.add_argument("--ids", default="", help="comma-separated ids; default all for mode")
    parser.add_argument("--data-dir", default=DATA)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--n-boot", type=int, default=None,
                        help="surface smoke override; omit for frozen manifest value")
    args = parser.parse_args()
    manifest = load_atlas_manifest(args.manifest)
    requested = [value for value in args.ids.split(",") if value]
    if args.mode == "surface":
        ids = requested or [row["id"] for row in manifest["surfaces"]]
        for surface_id in ids:
            print(json.dumps(run_surface(surface_id, data_dir=args.data_dir,
                                         out_dir=args.out_dir, manifest=manifest,
                                         n_boot=args.n_boot)))
    elif args.mode == "comparison":
        ids = requested or [row["id"] for row in manifest["comparisons"]]
        for comparison_id in ids:
            print(json.dumps(run_comparison(comparison_id, out_dir=args.out_dir,
                                            manifest=manifest)))
    else:
        print(json.dumps(aggregate_atlas(out_dir=args.out_dir, manifest=manifest), indent=1))


if __name__ == "__main__":
    main()
