"""Freeze the shared v13.1 Tier-A upgrades from consolidated Tier-B lanes."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from .cr3_sampled_value_certify import (
    VALUE_BOUND_RELEASE,
    VALUE_BOUND_RESULTS_SCHEMA,
    _file_sha256,
)
from .run_v13_value_campaign import (
    METRICS_MANIFEST_SCHEMA,
    _atomic_json,
    _materialize_entry_paths,
    load_metrics_manifest,
    select_metric_entries,
    select_tier_a_upgrades,
)


def freeze_tier_a_upgrades(
    *, results_path: str | Path, metrics_manifest_path: str | Path,
    out_root: str | Path,
) -> dict:
    """Select ten upgrades once, after all Tier-B constructors are present."""
    results_path = Path(results_path).resolve()
    metrics_manifest_path = Path(metrics_manifest_path).resolve()
    out_root = Path(out_root).resolve()
    frame = pd.read_parquet(results_path)
    if set(frame.get("schema", [])) != {VALUE_BOUND_RESULTS_SCHEMA}:
        raise ValueError("consolidated results use an unexpected schema")
    if set(map(str, frame["tier"].unique())) != {"B"}:
        raise ValueError("upgrade selection requires Tier-B-only results")
    if set(map(str, frame["channel"].unique())) != {"mcq", "behavioral"}:
        raise ValueError("upgrade selection requires both declared channels")

    metrics_manifest, base = load_metrics_manifest(metrics_manifest_path)
    entries = select_metric_entries(metrics_manifest, base)
    expected_metrics = {
        (str(entry["task"]), str(entry["level"]), str(entry["metric_key"]))
        for entry in entries
    }
    observed_metrics = {
        tuple(map(str, values))
        for values in frame[["task", "level", "metric_key"]]
        .drop_duplicates().itertuples(index=False, name=None)
    }
    if observed_metrics != expected_metrics:
        raise RuntimeError("consolidated results do not match the frozen Tier-B metric set")
    constructors = set(map(str, frame["constructor"].unique()))
    expected_rows = len(entries) * len(constructors) * 2
    cell_key = ["task", "level", "metric_key", "constructor", "channel"]
    if len(frame) != expected_rows or frame.duplicated(cell_key).any():
        raise RuntimeError("consolidated Tier-B model/metric/channel matrix is incomplete")

    chosen, selection = select_tier_a_upgrades(frame.to_dict("records"), entries)
    if len(chosen) != 10:
        raise RuntimeError("Tier-A upgrade selection did not yield ten unique metrics")
    out_root.mkdir(parents=True, exist_ok=True)
    upgrade_manifest = {
        "schema": METRICS_MANIFEST_SCHEMA,
        "release": VALUE_BOUND_RELEASE,
        "auto_upgrade_tier_a": False,
        "selection_provenance": selection,
        "source_tier_b_results": str(results_path),
        "source_tier_b_results_sha256": _file_sha256(results_path),
        "metrics": [_materialize_entry_paths(entry, base) for entry in chosen],
    }
    manifest_path = out_root / "metrics_manifest.json"
    selection_path = out_root / "selection.json"
    _atomic_json(manifest_path, upgrade_manifest)
    _atomic_json(selection_path, selection)
    report = {
        "release": VALUE_BOUND_RELEASE,
        "n_selected": len(chosen),
        "constructors": sorted(constructors),
        "metrics_manifest_path": str(manifest_path),
        "selection_path": str(selection_path),
        "source_metrics_manifest_path": str(metrics_manifest_path),
        "source_metrics_manifest_sha256": _file_sha256(metrics_manifest_path),
        "source_results_path": str(results_path),
        "source_results_sha256": _file_sha256(results_path),
        "selection": selection,
    }
    _atomic_json(out_root / "upgrade_report.json", report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--metrics-manifest", required=True)
    parser.add_argument("--out-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    freeze_tier_a_upgrades(
        results_path=args.results,
        metrics_manifest_path=args.metrics_manifest,
        out_root=args.out_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
