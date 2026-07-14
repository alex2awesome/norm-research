#!/usr/bin/env python
"""Run reciprocal cross-family fixed-target surfaces and comparisons."""
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


MANIFEST_PATH = Path(__file__).with_name("cross_family_surface_manifest_v1.json")


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict:
    return json.loads(Path(path).read_text())


def manifest_sha256(path: str | Path = MANIFEST_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _map(rows: list[dict]) -> dict[str, dict]:
    out = {row["id"]: row for row in rows}
    if len(out) != len(rows):
        raise ValueError("duplicate manifest id")
    return out


def _surface_path(surface_id: str, out_dir: str | Path, within_dir: str | Path,
                  manifest: dict) -> Path:
    external = _map(manifest["existing_surfaces"])
    if surface_id in external:
        return Path(within_dir) / external[surface_id]["path"]
    return Path(out_dir) / "surfaces" / f"{surface_id}.npz"


def run_surface(surface_id: str, *, data_dir: str, out_dir: str, manifest: dict) -> dict:
    spec = _map(manifest["surfaces"])[surface_id]
    d = manifest["default_analysis"]
    bundle = build_fixed_target_surface(
        data_dir=data_dir, domains=spec["domains"], executor_tag=spec["executor_tag"],
        target_tag=spec["target_tag"],
        executor_grid_template=spec.get("executor_grid_template"),
        target_grid_template=spec.get("target_grid_template"),
        divergence=d["divergence"], min_target_information=d["min_target_information"],
        train_frac=d["train_frac"], n_boot=d["n_boot"], seed=d["seed"])
    bundle["report"]["cross_family_atlas"] = {"surface_id": surface_id,
                                               "manifest_sha256": manifest_sha256()}
    path, report = save_surface(bundle, Path(out_dir) / "surfaces" / f"{surface_id}.npz")
    return {"id": surface_id, "path": str(path), "report": str(report),
            "n_metric_cells": bundle["report"]["n_metric_cells"],
            "n_ineligible": bundle["report"]["n_ineligible_metrics"],
            "n_errors": bundle["report"]["n_errors"]}


def run_comparison(comparison_id: str, *, out_dir: str, within_dir: str,
                   manifest: dict) -> dict:
    spec = _map(manifest["comparisons"])[comparison_id]
    d = manifest["default_analysis"]
    small = load_surface(_surface_path(spec["small"], out_dir, within_dir, manifest))
    big = load_surface(_surface_path(spec["big"], out_dir, within_dir, manifest))
    result = compare_surfaces(
        small, big, gap_delta=d["gap_delta"], equivalence_delta=d["equivalence_delta"],
        min_signature_rho=d["min_signature_rho"],
        signature_equivalence_delta=d["signature_equivalence_delta"])
    result["cross_family_atlas"] = {"comparison_id": comparison_id,
                                    "direction": spec["direction"],
                                    "manifest_sha256": manifest_sha256()}
    out = Path(out_dir) / "comparisons" / f"{comparison_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    summary = result["pooled"]
    return {"id": comparison_id, "path": str(out), "n_evaluable": summary["n_evaluable"],
            "n_gaps": summary["baseline_gap_confirmed"],
            "n_substitutions": summary[
                "methodological_substitution_among_confirmed_gaps"]["success"]}


def aggregate(*, out_dir: str, within_dir: str, manifest: dict) -> dict:
    surfaces = []
    for spec in manifest["surfaces"]:
        path = _surface_path(spec["id"], out_dir, within_dir, manifest)
        if path.exists():
            s = load_surface(path)
            surfaces.append({"id": spec["id"], "status": "complete", "sha256": s["sha256"],
                             "n_metric_cells": s["report"]["n_metric_cells"],
                             "n_ineligible": s["report"]["n_ineligible_metrics"],
                             "n_errors": s["report"]["n_errors"]})
        else:
            surfaces.append({"id": spec["id"], "status": "missing"})
    comparisons = []
    for spec in manifest["comparisons"]:
        path = Path(out_dir) / "comparisons" / f"{spec['id']}.json"
        if not path.exists():
            comparisons.append({"id": spec["id"], "direction": spec["direction"],
                                "status": "missing"})
            continue
        result = json.loads(path.read_text())
        valid = [row for domain in result["by_domain"].values() for row in domain["per_metric"]
                 if row.get("heldout", {}).get("valid")]
        gaps = [row for row in valid if row["heldout"]["gates"]["baseline_gap_confirmed"]]
        comparisons.append({
            "id": spec["id"], "direction": spec["direction"], "status": "complete",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_evaluable": len(valid), "n_gaps": len(gaps),
            "n_information_improved": sum(
                row["heldout"]["gates"]["articulation_improvement_confirmed"] for row in gaps),
            "n_information_noninferior": sum(
                row["heldout"]["gates"]["noninferior_to_big_sparse"] for row in gaps),
            "n_signature_improved": sum(
                row["heldout"]["gates"]["signature_improved"] for row in gaps),
            "n_signature_noninferior": sum(
                row["heldout"]["gates"]["signature_noninferior_to_big"] for row in gaps),
            "n_substitutions": sum(row["heldout"]["methodological_substitution"] for row in gaps),
            "n_familywise_substitutions": result.get("familywise_substitution_count", 0),
        })
    complete_comparisons = [row for row in comparisons if row["status"] == "complete"]
    atlas = {"schema": "cross_family_surface_atlas/v1",
             "manifest_sha256": manifest_sha256(), "surfaces": surfaces,
             "comparisons": comparisons, "interpretive_rule": manifest["interpretive_rule"],
             "coverage": {"surfaces_complete": sum(r["status"] == "complete" for r in surfaces),
                           "surfaces_total": len(surfaces),
                          "comparisons_complete": len(complete_comparisons),
                          "comparisons_total": len(comparisons)},
             "totals_over_comparisons_not_unique_metrics": {
                 "evaluable": sum(row["n_evaluable"] for row in complete_comparisons),
                 "confirmed_gaps": sum(row["n_gaps"] for row in complete_comparisons),
                 "cellwise_substitutions": sum(
                     row["n_substitutions"] for row in complete_comparisons),
                 "familywise_substitutions": sum(
                     row["n_familywise_substitutions"] for row in complete_comparisons)},
             "potential_status": ("Not evaluated from legacy word costs; requires finite debts "
                                  "and composable CUF-certified articulation units."),
             "paper_grade_claim_eligible": False}
    out = Path(out_dir) / "atlas_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(atlas, indent=1))
    return atlas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["surface", "comparison", "aggregate"])
    parser.add_argument("--ids", default="")
    parser.add_argument("--data-dir", default=DATA)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--within-dir", required=True)
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    requested = [value for value in args.ids.split(",") if value]
    if args.mode == "surface":
        ids = requested or [row["id"] for row in manifest["surfaces"]]
        for surface_id in ids:
            print(json.dumps(run_surface(surface_id, data_dir=args.data_dir,
                                         out_dir=args.out_dir, manifest=manifest)))
    elif args.mode == "comparison":
        ids = requested or [row["id"] for row in manifest["comparisons"]]
        for comparison_id in ids:
            print(json.dumps(run_comparison(comparison_id, out_dir=args.out_dir,
                                            within_dir=args.within_dir, manifest=manifest)))
    else:
        print(json.dumps(aggregate(out_dir=args.out_dir, within_dir=args.within_dir,
                                   manifest=manifest), indent=1))


if __name__ == "__main__":
    main()
