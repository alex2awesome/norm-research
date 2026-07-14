#!/usr/bin/env python
"""Aggregate rung/channel articulation gains from persisted fixed-target reader surfaces."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from methods.codability.experiments.fixed_target_surface import load_surface
from methods.codability.experiments.name_surface_atlas import (
    load_atlas_manifest,
    surface_path,
)


def _ci(values, confidence=0.95):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    tail = (1.0 - confidence) / 2.0
    return [float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))]


def _median(values):
    values = [value for value in values if np.isfinite(value)]
    return None if not values else float(np.median(values))


def analyze_surface(surface: dict, spec: dict) -> dict:
    groups = defaultdict(dict)
    for index, row in enumerate(surface["meta"]):
        groups[(row["domain"], int(row["gi"]))][row["rung"]] = index
    arm_rows, selected_rows = [], []
    arrays = surface["arrays"]
    for (domain, gi), arms in sorted(groups.items()):
        if "name" not in arms:
            continue
        ni = arms["name"]
        candidates = []
        for rung, ri in arms.items():
            if rung == "name":
                continue
            score_gain = float(arrays["heldout_score"][ri] - arrays["heldout_score"][ni])
            rho_gain = float(arrays["heldout_rho"][ri] - arrays["heldout_rho"][ni])
            score_ci = _ci(arrays["score_draws"][ri] - arrays["score_draws"][ni])
            rho_ci = _ci(arrays["rho_draws"][ri] - arrays["rho_draws"][ni])
            meta = surface["meta"][ri]
            row = {"domain": domain, "gi": gi, "name": meta.get("metric_name"),
                   "rung": rung, "channel": meta["dose"]["channel"],
                   "word_count": meta["dose"].get("word_count"),
                   "development_score": float(arrays["dev_score"][ri]),
                   "score_gain": score_gain, "score_gain_CI": score_ci,
                   "score_gain_confirmed": bool(score_ci and score_ci[0] > 0.0),
                   "rho_gain": rho_gain, "rho_gain_CI": rho_ci,
                   "rho_gain_confirmed": bool(rho_ci and rho_ci[0] > 0.0)}
            arm_rows.append(row)
            candidates.append((ri, row))
        if candidates:
            # Development-only arm choice; held-out gain below is selection-honest.
            ri, row = max(candidates, key=lambda pair: arrays["dev_score"][pair[0]])
            selected_rows.append(row)

    summaries = []
    for (domain, rung), rows in sorted(
            defaultdict(list, {key: [row for row in arm_rows
                                     if (row["domain"], row["rung"]) == key]
                               for key in {(row["domain"], row["rung"]) for row in arm_rows}}).items()):
        summaries.append({"domain": domain, "rung": rung, "channel": rows[0]["channel"],
                          "n": len(rows),
                          "median_score_gain": _median([row["score_gain"] for row in rows]),
                          "confirmed_score_gain": sum(row["score_gain_confirmed"] for row in rows),
                          "median_rho_gain": _median([row["rho_gain"] for row in rows]),
                          "confirmed_rho_gain": sum(row["rho_gain_confirmed"] for row in rows)})
    return {
        "surface_id": spec["id"], "family": spec.get("family"),
        "nominal_scale_b": spec.get("nominal_scale_b"),
        "target_tag": spec["target_tag"], "n_metrics": len(groups),
        "development_selected": {
            "n": len(selected_rows),
            "confirmed_score_gain": sum(row["score_gain_confirmed"] for row in selected_rows),
            "confirmed_rho_gain": sum(row["rho_gain_confirmed"] for row in selected_rows),
            "selected_rungs": dict(sorted(Counter(row["rung"] for row in selected_rows).items())),
            "selected_channels": dict(sorted(Counter(row["channel"]
                                                      for row in selected_rows).items())),
        },
        "by_domain_rung": summaries,
        "per_metric_arm": arm_rows,
    }


def build_frontier_report(*, atlas_dir: str, manifest: dict) -> dict:
    surfaces = []
    for spec in manifest["surfaces"]:
        path = surface_path(atlas_dir, spec["id"])
        if path.exists():
            surfaces.append(analyze_surface(load_surface(path), spec))
    groups = defaultdict(list)
    for row in surfaces:
        prefix = row["surface_id"].split("_")[0]
        groups[prefix].append({"surface_id": row["surface_id"],
                               "family": row["family"],
                               "nominal_scale_b": row["nominal_scale_b"],
                               **row["development_selected"]})
    return {"schema": "surface_frontier_report/v1", "surfaces": surfaces,
            "scale_profiles": {key: sorted(rows, key=lambda row: row["nominal_scale_b"])
                               for key, rows in sorted(groups.items())},
            "scope": ("Per-arm and development-selected held-out gains against each surface's "
                      "small-reader name baseline. This is articulation usefulness, not scale "
                      "substitution or a certified unit cost.")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--atlas-dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = load_atlas_manifest(args.manifest) if args.manifest else load_atlas_manifest()
    report = build_frontier_report(atlas_dir=args.atlas_dir, manifest=manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

