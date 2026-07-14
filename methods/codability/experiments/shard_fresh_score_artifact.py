#!/usr/bin/env python
"""Shard immutable fresh score matrices by declared item partition for lockbox isolation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from methods.codability.experiments.build_fresh_item_partitions import sha256_file
from methods.codability.experiments.common_target_ladder import (
    policy_cell_identity,
    validate_policy_cell_panel,
)
from methods.codability.experiments.policy_data import (
    atomic_savez_compressed,
    atomic_write_text,
    validate_frozen_implementation,
)


def _artifact_scope(arrays: dict, path: Path) -> str | None:
    """Return a stable filename scope when several jobs emit the same basename.

    Target jobs using the same checkpoint, domain, and repetition intentionally have identical
    raw filenames (for example the Llama-70B N and G targets).  A flat partition directory would
    therefore silently let the later target view overwrite the earlier one.  ``model_job_id`` is
    part of every fresh score artifact and is the appropriate namespace boundary.
    """
    if "model_job_id" not in arrays:
        return None
    value = str(np.asarray(arrays["model_job_id"]).item())
    scope = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return scope or None


def _score_cell_identities(arrays: dict, path: Path) -> list[dict]:
    """Collapse repeated score rows to authenticated, non-colliding cell identities."""
    if "meta" not in arrays:
        return []
    raw = np.asarray(arrays["meta"])
    if raw.ndim != 1:
        raise ValueError(f"score metadata is not one-dimensional in {path}")
    rows = [
        value if isinstance(value, dict) else json.loads(str(value))
        for value in raw
    ]
    with_ids = [row for row in rows if row.get("cell_id") is not None]
    if not with_ids:
        return []
    if len(with_ids) != len(rows):
        raise ValueError(f"score metadata mixes rows with and without cell_id in {path}")
    by_cell: dict[str, dict] = {}
    for row in rows:
        identity = policy_cell_identity(row, context=f"score shard {path}")
        cell_id = identity["cell_id"]
        previous = by_cell.setdefault(cell_id, identity)
        if previous != identity:
            changed = sorted(
                key for key in set(previous) | set(identity)
                if previous.get(key) != identity.get(key)
            )
            raise ValueError(
                f"score rows disagree on cell identity for {cell_id!r} in {path}; "
                f"changed={changed}"
            )
    identities = [by_cell[cell_id] for cell_id in sorted(by_cell)]
    validate_policy_cell_panel(identities, context=f"score shard {path}")
    return identities


def shard_artifact(
    path: str | Path,
    out_dir: str | Path,
    *,
    execution_manifest_path: str | Path | None = None,
) -> list[dict]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    execution_manifest_sha256 = None
    if execution_manifest_path is not None:
        manifest_path = Path(execution_manifest_path)
        manifest = json.loads(manifest_path.read_text())
        validate_frozen_implementation(
            manifest,
            manifest_path=manifest_path,
            section="scoring",
        )
        execution_manifest_sha256 = sha256_file(manifest_path)
        observed = (
            str(arrays["execution_manifest_sha256"])
            if "execution_manifest_sha256" in arrays else None
        )
        if observed != execution_manifest_sha256:
            raise ValueError(
                "raw score artifact does not belong to the frozen execution manifest"
            )
    scores = np.asarray(arrays["scores"], float)
    cell_identities = _score_cell_identities(arrays, path)
    if "meta" in arrays and len(np.asarray(arrays["meta"])) != scores.shape[0]:
        raise ValueError(f"score rows and metadata are unaligned in {path}")
    cell_identity_json = json.dumps(
        cell_identities, sort_keys=True, separators=(",", ":")
    )
    cell_identity_sha256 = hashlib.sha256(cell_identity_json.encode()).hexdigest()
    hashes = np.asarray(arrays["probe_sha256"])
    partitions = np.asarray(arrays["probe_partition"])
    if scores.shape[1] != len(hashes) or len(partitions) != len(hashes):
        raise ValueError(f"unaligned score artifact {path}")
    source_hash = sha256_file(path)
    scope = _artifact_scope(arrays, path)
    outputs = []
    for partition in sorted({str(value) for value in partitions}):
        mask = partitions.astype(str) == partition
        partition_hashes = hashes[mask]
        partition_set_hash = hashlib.sha256(
            "\n".join(str(value) for value in partition_hashes).encode()).hexdigest()
        # Keep job/view provenance in the path.  Without this namespace, N and G artifacts from
        # the same model/domain/repetition have the same basename and one silently overwrites the
        # other.  Legacy/synthetic artifacts without model_job_id retain the old flat layout.
        out = Path(out_dir) / partition
        if scope:
            out = out / scope
        out = out / path.name
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(arrays)
        payload["scores"] = scores[:, mask]
        payload["probe_sha256"] = partition_hashes
        payload["probe_partition"] = partitions[mask]
        payload["probe_set_sha256"] = np.asarray(partition_set_hash)
        payload["source_artifact_sha256"] = np.asarray(source_hash)
        payload["isolated_partition"] = np.asarray(partition)
        payload["cell_identity_manifest"] = np.asarray(cell_identity_json)
        payload["cell_identity_sha256"] = np.asarray(cell_identity_sha256)
        atomic_savez_compressed(out, **payload)
        report = {"schema": "partition_sharded_score/v2_job_scoped", "partition": partition,
                  "n_items": int(mask.sum()), "n_score_rows": int(scores.shape[0]),
                  "artifact_scope": scope,
                  "source_path": str(path), "source_artifact_sha256": source_hash,
                  "execution_manifest_sha256": execution_manifest_sha256,
                  "n_cells": len(cell_identities),
                  "cell_identities": cell_identities,
                  "cell_identity_sha256": cell_identity_sha256,
                  "probe_set_sha256": partition_set_hash, "path": str(out),
                  "sha256": sha256_file(out)}
        atomic_write_text(
            out.with_suffix(".json"), json.dumps(report, indent=1) + "\n")
        outputs.append(report)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--execution-manifest", default=None)
    args = parser.parse_args()
    reports = [
        row
        for path in args.paths
        for row in shard_artifact(
            path,
            args.out_dir,
            execution_manifest_path=args.execution_manifest,
        )
    ]
    print(json.dumps({"schema": "partition_shard_batch/v2_job_scoped", "n_sources": len(args.paths),
                      "n_shards": len(reports), "shards": reports}, indent=1))


if __name__ == "__main__":
    main()
