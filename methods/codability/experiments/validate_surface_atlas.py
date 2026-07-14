#!/usr/bin/env python
"""Validate persisted fixed-target surface atlases and their cross-file joins."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from methods.codability.experiments.fixed_target_surface import load_surface


ATLAS_MANIFESTS = {
    "name_surface_atlas/v1": "name_surface_atlas_manifest_v1.json",
    "cross_family_surface_atlas/v1": "cross_family_surface_manifest_v1.json",
}
COMPARISON_SCHEMA = "fixed_target_surface_comparison/v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_reference(raw_path: str, atlas_dir: Path) -> Path | None:
    path = Path(raw_path)
    if path.is_absolute():
        return path if path.exists() else None
    for base in (Path.cwd(), atlas_dir, *atlas_dir.parents):
        candidate = base / path
        if candidate.exists():
            return candidate
    return None


def _comparison_rows(result: dict) -> list[dict]:
    return [row for domain in result.get("by_domain", {}).values()
            for row in domain.get("per_metric", [])]


def _check_equal(errors: list[str], label: str, observed, expected) -> None:
    if observed != expected:
        errors.append(f"{label}: observed {observed!r}, expected {expected!r}")


def validate_atlas(atlas_dir: str | Path, *, check_manifest: bool = True) -> dict:
    """Return a machine-readable integrity certificate for one atlas directory."""
    atlas_dir = Path(atlas_dir)
    errors: list[str] = []
    warnings: list[str] = []
    summary_path = atlas_dir / "atlas_summary.json"
    if not summary_path.exists():
        return {"valid": False, "atlas_dir": str(atlas_dir),
                "errors": [f"missing {summary_path}"], "warnings": []}
    summary = json.loads(summary_path.read_text())
    schema = summary.get("schema")
    if schema not in ATLAS_MANIFESTS:
        errors.append(f"unsupported atlas schema {schema!r}")

    if check_manifest and schema in ATLAS_MANIFESTS:
        manifest_name = ATLAS_MANIFESTS[schema]
        manifest = _resolve_reference(
            f"methods/codability/experiments/{manifest_name}", atlas_dir)
        if manifest is None:
            warnings.append(f"could not locate {manifest_name}")
        else:
            _check_equal(errors, "atlas manifest SHA-256",
                         summary.get("manifest_sha256"), _sha256(manifest))

    surface_results = []
    for entry in summary.get("surfaces", []):
        surface_id = entry["id"]
        npz_path = atlas_dir / "surfaces" / f"{surface_id}.npz"
        json_path = npz_path.with_suffix(".json")
        local_errors = []
        if entry.get("status") != "complete":
            local_errors.append(f"summary status is {entry.get('status')!r}")
        if not npz_path.exists() or not json_path.exists():
            local_errors.append("missing NPZ or JSON sidecar")
        else:
            try:
                surface = load_surface(npz_path)
                report = surface["report"]
                arrays = surface["arrays"]
                n_rows = len(surface["meta"])
                if json.loads(json_path.read_text()) != report:
                    local_errors.append("JSON sidecar differs from embedded report")
                if entry.get("sha256") != _sha256(npz_path):
                    local_errors.append("summary SHA-256 differs from NPZ")
                if report.get("n_arm_rows") != n_rows:
                    local_errors.append("report n_arm_rows differs from metadata")
                for key, values in arrays.items():
                    if values.ndim == 0 or values.shape[0] != n_rows:
                        local_errors.append(f"array {key} is not row-aligned")
                    if key.endswith("_draws"):
                        n_boot = report.get("config", {}).get("n_boot")
                        if values.ndim != 2 or values.shape[1] != n_boot:
                            local_errors.append(f"array {key} has wrong bootstrap width")
                _check_equal(local_errors, "surface metric count",
                             entry.get("n_metric_cells"), report.get("n_metric_cells"))
                entry_ineligible = entry.get(
                    "n_ineligible_metrics", entry.get("n_ineligible"))
                _check_equal(local_errors, "surface ineligible count", entry_ineligible,
                             report.get("n_ineligible_metrics"))
                _check_equal(local_errors, "surface error count", entry.get("n_errors"),
                             report.get("n_errors"))
                if report.get("n_errors"):
                    local_errors.append("surface contains runtime errors")
            except (KeyError, TypeError, ValueError) as exc:
                local_errors.append(f"surface load failed: {exc}")
        errors.extend(f"surface {surface_id}: {value}" for value in local_errors)
        surface_results.append({"id": surface_id, "valid": not local_errors})

    comparison_results = []
    total_evaluable = total_gaps = total_cellwise = total_familywise = 0
    for entry in summary.get("comparisons", []):
        comparison_id = entry["id"]
        path = atlas_dir / "comparisons" / f"{comparison_id}.json"
        local_errors = []
        if entry.get("status") != "complete":
            local_errors.append(f"summary status is {entry.get('status')!r}")
        if not path.exists():
            local_errors.append("missing comparison JSON")
        else:
            result = json.loads(path.read_text())
            rows = _comparison_rows(result)
            valid = [row for row in rows if row.get("heldout", {}).get("valid")]
            gaps = [row for row in valid
                    if row["heldout"]["gates"]["baseline_gap_confirmed"]]
            cellwise = sum(bool(row["heldout"].get("methodological_substitution"))
                           for row in gaps)
            familywise = sum(bool(row["heldout"].get("familywise", {}).get(
                "methodological_substitution")) for row in gaps)
            total_evaluable += len(valid)
            total_gaps += len(gaps)
            total_cellwise += cellwise
            total_familywise += familywise
            if result.get("schema") != COMPARISON_SCHEMA:
                local_errors.append(f"unsupported schema {result.get('schema')!r}")
            if not result.get("validation", {}).get("valid"):
                local_errors.append("surface-pair validation is not valid")
            if entry.get("sha256") != _sha256(path):
                local_errors.append("summary SHA-256 differs from comparison JSON")
            _check_equal(local_errors, "comparison family size",
                         result.get("config", {}).get("family_size"), len(rows))
            _check_equal(local_errors, "comparison evaluable count",
                         entry.get("n_evaluable"), len(valid))
            _check_equal(local_errors, "comparison gap count",
                         entry.get("n_confirmed_gaps", entry.get("n_gaps")), len(gaps))
            _check_equal(local_errors, "comparison cellwise count",
                         entry.get("n_substitutions"), cellwise)
            _check_equal(local_errors, "comparison familywise count",
                         entry.get("n_familywise_substitutions"), familywise)
            _check_equal(local_errors, "embedded familywise count",
                         result.get("familywise_substitution_count"), familywise)
            for side in ("small_surface", "big_surface"):
                reference = result.get(side, {})
                referenced_path = _resolve_reference(reference.get("path", ""), atlas_dir)
                if referenced_path is None:
                    local_errors.append(f"cannot resolve {side} path")
                elif reference.get("sha256") != _sha256(referenced_path):
                    local_errors.append(f"{side} SHA-256 differs from referenced surface")
        errors.extend(f"comparison {comparison_id}: {value}" for value in local_errors)
        comparison_results.append({"id": comparison_id, "valid": not local_errors})

    coverage = summary.get("coverage", {})
    _check_equal(errors, "surface coverage total", coverage.get("surfaces_total"),
                 len(surface_results))
    _check_equal(errors, "surface coverage complete", coverage.get("surfaces_complete"),
                 sum(row["valid"] for row in surface_results))
    _check_equal(errors, "comparison coverage total", coverage.get("comparisons_total"),
                 len(comparison_results))
    _check_equal(errors, "comparison coverage complete", coverage.get("comparisons_complete"),
                 sum(row["valid"] for row in comparison_results))
    totals = summary.get("totals_over_comparisons_not_unique_metrics", {})
    _check_equal(errors, "aggregate evaluable count", totals.get("evaluable"), total_evaluable)
    _check_equal(errors, "aggregate gap count", totals.get("confirmed_gaps"), total_gaps)
    _check_equal(errors, "aggregate cellwise count",
                 totals.get("cellwise_substitutions"), total_cellwise)
    _check_equal(errors, "aggregate familywise count",
                 totals.get("familywise_substitutions"), total_familywise)

    return {
        "schema": "surface_atlas_integrity/v1", "valid": not errors,
        "atlas_dir": str(atlas_dir), "atlas_schema": schema,
        "summary_sha256": _sha256(summary_path),
        "counts": {"surfaces": len(surface_results),
                   "comparisons": len(comparison_results),
                   "evaluable_comparison_rows": total_evaluable,
                   "confirmed_gaps": total_gaps,
                   "cellwise_substitutions": total_cellwise,
                   "familywise_substitutions": total_familywise},
        "errors": errors, "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("atlas_dirs", nargs="+")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    reports = [validate_atlas(path) for path in args.atlas_dirs]
    result = {"schema": "surface_atlas_integrity_batch/v1",
              "valid": all(report["valid"] for report in reports),
              "atlases": reports}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
