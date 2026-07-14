#!/usr/bin/env python
"""Validate and summarize informativeness, form reliability, and test-retest of fresh targets."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from methods.codability.grid_auc_report import spearman
from methods.codability.experiments.build_fresh_item_partitions import sha256_file
from methods.metric_implementer.vinfo import fixed_target_channel_certificate


def _load(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        scores = np.asarray(z["scores"], float)
        meta = [json.loads(str(value)) for value in z["meta"]]
        hashes = [str(value) for value in z["probe_sha256"]]
        partitions = [str(value) for value in z["probe_partition"]]
        scalars = {key: str(z[key]) for key in (
            "probe_set_sha256", "reader", "model_job_id", "prompt_manifest_sha256",
            "packet_manifest_sha256", "readout_template_sha256")}
        repetition = int(z["repetition"])
    if scores.shape != (len(meta), len(hashes)) or len(partitions) != len(hashes):
        raise ValueError(f"unaligned target score artifact {path}")
    expected_set_hash = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
    if scalars["probe_set_sha256"] != expected_set_hash:
        raise ValueError(f"probe-set hash mismatch in {path}")
    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.exists():
        raise ValueError(f"missing sidecar for {path}")
    sidecar = json.loads(sidecar_path.read_text())
    for key in ("probe_set_sha256", "prompt_manifest_sha256", "packet_manifest_sha256",
                "readout_template_sha256"):
        if sidecar.get(key) != scalars[key]:
            raise ValueError(f"sidecar {key} mismatch in {path}")
    return {"path": str(path), "sha256": sha256_file(path), "scores": scores,
            "meta": meta, "hashes": hashes, "partitions": partitions,
            "repetition": repetition, **scalars, "sidecar": sidecar}


def target_summary(values: np.ndarray) -> dict:
    """Aggregate a form-by-item target orbit without publishing item verdicts."""
    values = np.asarray(values, float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 4:
        raise ValueError("target orbit must be form-by-item with at least four items")
    if not np.isfinite(values).all():
        return {"valid": False, "error": "nonfinite_target_scores",
                "nan_rate": float(np.isnan(values).mean())}
    target = values.mean(axis=0)
    cert = fixed_target_channel_certificate(target, target)
    bin_target = target > 0.5
    form_flips = [float(((row > 0.5) != bin_target).mean()) for row in values]
    form_mae = [float(np.abs(row - target).mean()) for row in values]
    return {
        "valid": bool(cert.get("valid")), "n_items": int(values.shape[1]),
        "n_forms": int(values.shape[0]), "mean_target": float(target.mean()),
        "binary_positive_rate": float(bin_target.mean()),
        "T_tvd": float(cert["tvd"]["T_target"]) if cert.get("valid") else None,
        "T_shannon": float(cert["shannon"]["T_target"]) if cert.get("valid") else None,
        "mean_form_variance": float(np.var(values, axis=0).mean()),
        "mean_form_MAE": float(np.mean(form_mae)), "max_form_MAE": float(np.max(form_mae)),
        "mean_form_flip_rate": float(np.mean(form_flips)),
        "max_form_flip_rate": float(np.max(form_flips)),
    }


def analyze_artifact(artifact: dict) -> dict:
    grouped = defaultdict(list)
    for index, meta in enumerate(artifact["meta"]):
        grouped[meta["cell_id"]].append(index)
    cells = []
    partitions = np.asarray(artifact["partitions"])
    for cell_id, indices in sorted(grouped.items()):
        forms = artifact["scores"][indices]
        meta = artifact["meta"][indices[0]]
        by_partition = {}
        for partition in sorted(set(partitions)):
            mask = partitions == partition
            by_partition[partition] = target_summary(forms[:, mask])
        cells.append({"cell_id": cell_id, "view": meta["view"],
                      "domain": meta["domain"], "forms": [artifact["meta"][i]["form"]
                                                           for i in indices],
                      "overall": target_summary(forms), "by_partition": by_partition})
    return {"path": artifact["path"], "sha256": artifact["sha256"],
            "model_job_id": artifact["model_job_id"], "reader": artifact["reader"],
            "repetition": artifact["repetition"], "n_items": len(artifact["hashes"]),
            "cells": cells}


def _retest(left: dict, right: dict) -> list[dict]:
    if left["hashes"] != right["hashes"] or left["meta"] != right["meta"]:
        raise ValueError("test-retest artifacts are not aligned")
    grouped = defaultdict(list)
    for index, meta in enumerate(left["meta"]):
        grouped[meta["cell_id"]].append(index)
    rows = []
    for cell_id, indices in sorted(grouped.items()):
        a = left["scores"][indices].mean(axis=0)
        b = right["scores"][indices].mean(axis=0)
        rho = spearman(a, b)
        rows.append({"cell_id": cell_id, "repetitions": [left["repetition"], right["repetition"]],
                     "mean_absolute_difference": float(np.abs(a - b).mean()),
                     "binary_flip_rate": float(((a > 0.5) != (b > 0.5)).mean()),
                     "spearman": None if rho is None else float(rho)})
    return rows


def build_report(scores_dir: str | Path) -> dict:
    artifacts = [_load(path) for path in sorted(Path(scores_dir).glob("**/*.npz"))]
    reports = [analyze_artifact(artifact) for artifact in artifacts]
    groups = defaultdict(list)
    for artifact in artifacts:
        groups[(artifact["model_job_id"], artifact["sidecar"]["domain"])].append(artifact)
    retest = []
    for (model_job_id, domain), rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["repetition"])
        for left, right in zip(rows, rows[1:]):
            retest.append({"model_job_id": model_job_id, "domain": domain,
                           "cells": _retest(left, right)})
    return {"schema": "fresh_target_score_report/v1", "n_artifacts": len(artifacts),
            "artifacts": reports, "test_retest": retest,
            "claim_scope": ("Aggregate target health only. Practice labels remain a separate P "
                            "target; no item-level target scores are emitted.")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = build_report(args.scores_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"out": str(out), "n_artifacts": report["n_artifacts"],
                      "n_test_retest": len(report["test_retest"])}, indent=1))


if __name__ == "__main__":
    main()
