"""Consolidate independently scheduled v13.1 model lanes without copying GPU artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from .cr3_sampled_value_certify import (
    VALUE_BOUND_RELEASE,
    VALUE_BOUND_RESULTS_SCHEMA,
    _file_sha256,
    _payload_sha256,
)
from .run_v13_value_campaign import (
    CAMPAIGN_MANIFEST_SCHEMA,
    _atomic_json,
    _atomic_parquet,
)


LANE_CONSOLIDATION_SCHEMA = "cr3-value-bound-lane-consolidation-v13.1"
CELL_KEY = ["tier", "task", "level", "metric_key", "constructor", "channel"]


def consolidate_lane_roots(
    lane_roots: Sequence[str | Path], *, out_root: str | Path,
    expected_constructors: Sequence[str] | None = None,
    expected_channels: Sequence[str] = ("mcq", "behavioral"),
) -> pd.DataFrame:
    roots = [Path(root).resolve() for root in lane_roots]
    if not roots:
        raise ValueError("at least one lane root is required")
    frames = []
    sources = []
    for root in roots:
        result_path = root / "results.parquet"
        manifest_path = root / "campaign_manifest.json"
        if not result_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"lane {root} lacks results or campaign manifest")
        campaign = json.loads(manifest_path.read_text(encoding="utf-8"))
        if campaign.get("schema") != CAMPAIGN_MANIFEST_SCHEMA:
            raise ValueError(f"lane {root} uses an unexpected campaign schema")
        frame = pd.read_parquet(result_path)
        if set(frame.get("schema", [])) != {VALUE_BOUND_RESULTS_SCHEMA}:
            raise ValueError(f"lane {root} uses an unexpected result schema")
        frame = frame.copy()
        frame["artifact_root"] = str(root)
        frames.append(frame)
        sources.append({
            "root": str(root),
            "results_sha256": _file_sha256(result_path),
            "campaign_manifest_sha256": _file_sha256(manifest_path),
            "n_results": int(len(frame)),
        })
    combined = pd.concat(frames, ignore_index=True)
    missing = sorted(set(CELL_KEY).difference(combined.columns))
    if missing:
        raise ValueError(f"lane result tables lack cell keys: {missing}")
    duplicated = combined.duplicated(CELL_KEY, keep=False)
    if duplicated.any():
        cells = combined.loc[duplicated, CELL_KEY].drop_duplicates().to_dict("records")
        raise ValueError(f"duplicate model/metric/channel cells across lanes: {cells[:5]}")

    channels = set(map(str, expected_channels))
    if set(map(str, combined["channel"].unique())) != channels:
        raise ValueError("consolidated channels differ from the declared channel set")
    constructors = (
        set(map(str, expected_constructors))
        if expected_constructors is not None
        else set(map(str, combined["constructor"].unique()))
    )
    if set(map(str, combined["constructor"].unique())) != constructors:
        raise ValueError("consolidated constructors differ from the declared model matrix")
    for tier, tier_frame in combined.groupby("tier", dropna=False):
        metrics = tier_frame[["task", "level", "metric_key"]].drop_duplicates()
        expected = len(metrics) * len(constructors) * len(channels)
        if len(tier_frame) != expected:
            raise RuntimeError(
                f"tier {tier} is incomplete: observed {len(tier_frame)}, expected {expected}"
            )

    combined = combined.sort_values(CELL_KEY, kind="mergesort").reset_index(drop=True)
    destination = Path(out_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(destination / "results.parquet", combined)
    manifest = {
        "schema": LANE_CONSOLIDATION_SCHEMA,
        "release": VALUE_BOUND_RELEASE,
        "source_lanes": sources,
        "constructors": sorted(constructors),
        "channels": sorted(channels),
        "n_results": int(len(combined)),
        "artifact_roots_are_referenced_not_copied": True,
        "results_path": str(destination / "results.parquet"),
    }
    manifest["consolidation_sha256"] = _payload_sha256(manifest)
    _atomic_json(destination / "campaign_manifest.json", manifest)
    return combined


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-roots", nargs="+", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--expected-constructors", nargs="+")
    parser.add_argument(
        "--expected-channels", nargs="+", choices=["mcq", "behavioral"],
        default=["mcq", "behavioral"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    consolidate_lane_roots(
        args.lane_roots, out_root=args.out_root,
        expected_constructors=args.expected_constructors,
        expected_channels=args.expected_channels,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
