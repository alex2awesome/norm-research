#!/usr/bin/env python3
"""Freeze a stratified optimization/selection split for verifier GEPA.

The split is grouped by the verifier target (correct proposal, wrong proposal,
or typed abstention) and assigned by a salted UID hash.  All output hashes are
recorded before row-level GEPA error inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def _key(group: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{group}".encode()).hexdigest()


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    indexed = {str(row["norm_uid"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate norm_uid in {path}")
    return indexed


def _stratum(truth: dict[str, Any], primary: dict[str, Any]) -> str:
    if truth.get("decision") != "MATCH":
        return f"typed:{truth.get('decision')}"
    return (
        "match:proposal_correct"
        if str(truth.get("metric_id")) == str(primary.get("metric_id"))
        else "match:proposal_wrong"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--optimize-fraction", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=1129)
    args = parser.parse_args()
    if not 0.0 < args.optimize_fraction < 1.0:
        parser.error("--optimize-fraction must be between zero and one")

    paths = {
        name: Path(getattr(args, name)).resolve()
        for name in ("truth", "primary", "candidates")
    }
    rows = {name: _index(path) for name, path in paths.items()}
    uid_set = set(rows["truth"])
    if any(set(indexed) != uid_set for indexed in rows.values()):
        raise ValueError("truth, primary, and candidates lack exact UID coverage")

    groups: dict[str, list[str]] = defaultdict(list)
    for uid in uid_set:
        values = {
            str(indexed[uid]["source_group"])
            for indexed in rows.values()
            if indexed[uid].get("source_group")
        }
        if len(values) > 1:
            raise ValueError(f"source_group mismatch for {uid}: {sorted(values)}")
        groups[values.pop() if values else uid].append(uid)
    strata: dict[str, list[str]] = defaultdict(list)
    for group, uids in groups.items():
        signature = sorted(
            _stratum(rows["truth"][uid], rows["primary"][uid]) for uid in uids
        )
        strata[json.dumps(signature, separators=(",", ":"))].append(group)

    roles: dict[str, str] = {}
    allocation: dict[str, dict[str, int]] = {}
    group_roles: dict[str, str] = {}
    for stratum, group_ids in sorted(strata.items()):
        ordered = sorted(group_ids, key=lambda group: (_key(group, args.seed), group))
        n_optimize = round(len(ordered) * args.optimize_fraction)
        if len(ordered) > 1:
            n_optimize = min(max(n_optimize, 1), len(ordered) - 1)
        for group in ordered[:n_optimize]:
            group_roles[group] = "optimize"
        for group in ordered[n_optimize:]:
            group_roles[group] = "select"
        allocation[stratum] = {
            "total": len(ordered),
            "optimize": n_optimize,
            "select": len(ordered) - n_optimize,
        }

    for group, uids in groups.items():
        for uid in uids:
            roles[uid] = group_roles[group]

    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    output_hashes: dict[str, dict[str, str]] = {}
    for role in ("optimize", "select"):
        role_dir = output / role
        role_dir.mkdir()
        output_hashes[role] = {}
        role_uids = sorted(uid for uid, value in roles.items() if value == role)
        for name, indexed in rows.items():
            path = role_dir / f"{name}.jsonl"
            write_jsonl(path, [indexed[uid] for uid in role_uids])
            output_hashes[role][name] = sha256_file(path)

    payload = {
        "schema_version": "silver-match-v3-verifier-gepa-split-v1",
        "seed": args.seed,
        "optimize_fraction": args.optimize_fraction,
        "allocation": allocation,
        "role_counts": {
            role: sum(value == role for value in roles.values())
            for role in ("optimize", "select")
        },
        "source_group_counts": {
            role: sum(value == role for value in group_roles.values())
            for role in ("optimize", "select")
        },
        "source_group_disjoint": not (
            {group for group, role in group_roles.items() if role == "optimize"}
            & {group for group, role in group_roles.items() if role == "select"}
        ),
        "role_uid_sha256": {
            role: hashlib.sha256(
                "\n".join(sorted(uid for uid, value in roles.items() if value == role)).encode()
            ).hexdigest()
            for role in ("optimize", "select")
        },
        "input_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "output_hashes": output_hashes,
    }
    freeze = output / "FREEZE.json"
    freeze.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**payload, "freeze_sha256": sha256_file(freeze)}, sort_keys=True))


if __name__ == "__main__":
    main()
