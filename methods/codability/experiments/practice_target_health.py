#!/usr/bin/env python
"""Summarize sealed archival practice targets without exposing item labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from methods.codability.experiments.build_fresh_item_partitions import (
    MANIFEST_PATH,
    load_manifest,
    sha256_file,
)
from methods.metric_implementer.vinfo import fixed_target_channel_certificate


def summarize_target(values) -> dict:
    values = np.asarray(values, float)
    if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("practice target must be a finite vector with at least four values")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("practice targets must lie in [0,1]")
    cert = fixed_target_channel_certificate(values, values)
    return {"n": len(values), "mean": float(values.mean()),
            "binary_positive_rate": float((values > 0.5).mean()),
            "T_tvd": float(cert["tvd"]["T_target"]),
            "T_shannon": float(cert["shannon"]["T_target"]),
            "n_unique": int(len(np.unique(values)))}


def _read_targets(path: Path) -> list[float]:
    return [float(json.loads(line)["practice_target"])
            for line in path.read_text().splitlines() if line.strip()]


def build_report(packet_manifest_path: str | Path, *,
                 protocol_path: str | Path = MANIFEST_PATH) -> dict:
    packet_manifest_path = Path(packet_manifest_path)
    packet = json.loads(packet_manifest_path.read_text())
    protocol = load_manifest(protocol_path)
    domains = []
    for domain in packet["domains"]:
        spec = protocol["domains"][domain["domain"]]
        values, partitions = [], {}
        for partition in domain["partitions"]:
            path = Path(partition["targets_path"])
            if not path.exists():
                # Project-relative paths are expected in the frozen packet.
                path = Path.cwd() / partition["targets_path"]
            if sha256_file(path) != partition["targets_sha256"]:
                raise ValueError(f"target hash mismatch for {domain['domain']}/{partition['id']}")
            target = _read_targets(path)
            if len(target) != partition["n"]:
                raise ValueError(f"target count mismatch for {domain['domain']}/{partition['id']}")
            partitions[partition["id"]] = summarize_target(target)
            values.extend(target)
        domains.append({"domain": domain["domain"], "label_description": spec["label"],
                        "holdout_grade": spec["holdout_grade"],
                        "overall": summarize_target(values), "by_partition": partitions,
                        "claim_grade": ("archival-community-preference-proxy"
                                        if domain["domain"] in {"humor", "cw", "math"}
                                        else "external-outcome-proxy-not-community-rating")})
    return {"schema": "practice_target_health/v1",
            "packet_manifest_sha256": sha256_file(packet_manifest_path),
            "protocol_manifest_sha256": sha256_file(protocol_path), "domains": domains,
            "claim_boundary": ("P targets are archival proxies with their stated provenance. "
                               "They are not model targets, construct definitions, or truth; the "
                               "press-release pickup outcome is not a professional rating.")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--protocol", default=str(MANIFEST_PATH))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = build_report(args.packet_manifest, protocol_path=args.protocol)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"out": str(out), "domains": [row["domain"]
                                                    for row in report["domains"]]}, indent=1))


if __name__ == "__main__":
    main()
