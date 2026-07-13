"""Assemble consolidated v13.1 stages into one campaign-wide result table."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .cr3_sampled_value_certify import (
    VALUE_BOUND_RELEASE,
    VALUE_BOUND_RESULTS_SCHEMA,
    _file_sha256,
    _payload_sha256,
)
from .run_v13_value_campaign import _atomic_json, _atomic_parquet


RELEASE_RESULTS_SCHEMA = "cr3-value-bound-release-results-v13.1"


def assemble_release_results(
    stages: Mapping[str, str | Path], *, out_root: str | Path,
    expected_results: int | None = None,
) -> pd.DataFrame:
    if not stages:
        raise ValueError("at least one consolidated campaign stage is required")
    frames = []
    sources = []
    for stage, source in stages.items():
        stage = str(stage).strip()
        if not stage:
            raise ValueError("campaign stage names must be nonempty")
        path = Path(source).resolve()
        if path.is_dir():
            path = path / "results.parquet"
        frame = pd.read_parquet(path)
        if set(frame.get("schema", [])) != {VALUE_BOUND_RESULTS_SCHEMA}:
            raise ValueError(f"stage {stage} uses an unexpected result schema")
        frame = frame.copy()
        frame.insert(0, "campaign_stage", stage)
        frame["source_results_path"] = str(path)
        key = [
            "tier", "task", "level", "metric_key", "constructor", "channel",
        ]
        if frame.duplicated(key).any():
            raise RuntimeError(f"stage {stage} contains duplicate result cells")
        frames.append(frame)
        sources.append({
            "campaign_stage": stage, "results_path": str(path),
            "results_sha256": _file_sha256(path), "n_results": int(len(frame)),
        })
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if expected_results is not None and len(combined) != int(expected_results):
        raise RuntimeError(
            f"release table has {len(combined)} rows, expected {int(expected_results)}"
        )
    order = [
        "campaign_stage", "tier", "task", "level", "metric_key", "constructor",
        "channel",
    ]
    combined = combined.sort_values(order, kind="mergesort").reset_index(drop=True)
    destination = Path(out_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results_path = destination / "results.parquet"
    _atomic_parquet(results_path, combined)
    manifest = {
        "schema": RELEASE_RESULTS_SCHEMA,
        "release": VALUE_BOUND_RELEASE,
        "sources": sources,
        "n_results": int(len(combined)),
        "results_path": str(results_path),
        "mcq_and_behavioral_values_numerically_combined": False,
    }
    manifest["release_results_sha256"] = _payload_sha256(manifest)
    _atomic_json(destination / "campaign_manifest.json", manifest)
    return combined


def _parse_stages(values: Sequence[str]) -> dict[str, str]:
    stages = {}
    for value in values:
        stage, separator, path = str(value).partition("=")
        if not separator or not stage or not path:
            raise ValueError("--stage values must have the form NAME=RESULTS_OR_DIRECTORY")
        if stage in stages:
            raise ValueError(f"duplicate campaign stage {stage!r}")
        stages[stage] = path
    return stages


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--expected-results", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    assemble_release_results(
        _parse_stages(args.stage), out_root=args.out_root,
        expected_results=args.expected_results,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
