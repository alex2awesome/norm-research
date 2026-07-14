#!/usr/bin/env python
"""Persist one executor's complete articulation surface against one fixed target.

Pairwise runners are convenient for a single hypothesis but wasteful and potentially inconsistent
for a scale/family atlas. This module evaluates every declared articulation arm once on one common
development/held-out split and stores the raw paired bootstrap draws. Surfaces sharing a target,
split seed, and manifest can then be compared without re-executing prompts or changing resamples.

Legacy grids still lack cryptographic probe IDs and certified articulation units. Their surfaces are
therefore diagnostic, even though the split and bootstrap algebra are held out and reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from methods.codability.experiments.fixed_target_name_substitution import (
    DATA,
    DEFAULT_RUNGS,
    _legacy_dose,
    _slice_orbit,
    alignment_report,
    find_grid,
    load_grid_orbits,
    resolve_grid_dir,
    sha256_file,
    soft_stratified_split,
)
from methods.codability.experiments.target_articulation_frontier import (
    SCORE_KEY,
    bootstrap_orbit_values,
    manifest_sha256,
    orbit_recovery,
    target_orbit_mean,
)


SCHEMA = "fixed_target_reader_surface/v1"
DEFAULT_SURFACE_RUNGS = ["name", *DEFAULT_RUNGS]


def _domain_seed(seed: int, domain: str, gi: int) -> int:
    digest = int(hashlib.sha256(domain.encode()).hexdigest()[:8], 16)
    return int(seed + digest % 10_000_000 + gi)


def _pad_draws(values: Sequence[float], n_boot: int) -> np.ndarray:
    out = np.full(n_boot, np.nan, float)
    values = np.asarray(values, float)[:n_boot]
    out[:len(values)] = values
    return out


def _surface_row(*, domain: str, gi: int, metric_name: str | None, rung: str,
                 dose: dict, dev: dict, heldout: dict, draws: dict[str, np.ndarray],
                 target_id: str, metric_seed: int, n_original: int, n_excluded: int,
                 n_development: int, n_heldout: int, n_boot: int) -> tuple[dict, dict]:
    robust_dev, robust_test = dev["robust"], heldout["robust"]
    meta = {
        "domain": domain, "gi": int(gi), "metric_name": metric_name, "rung": rung,
        "dose": dose, "target_id": target_id, "metric_seed": int(metric_seed),
        "n_original": int(n_original), "n_excluded": int(n_excluded),
        "n_development": int(n_development), "n_heldout": int(n_heldout),
        "n_boot_valid": int(len(draws[SCORE_KEY])),
    }
    arrays = {
        "dev_score": float(robust_dev[SCORE_KEY]),
        "dev_rho": np.nan if robust_dev["spearman"] is None else float(robust_dev["spearman"]),
        "dev_positive_polarity": bool(robust_dev["all_positive_polarity"]),
        "heldout_score": float(robust_test[SCORE_KEY]),
        "heldout_rho": (np.nan if robust_test["spearman"] is None
                        else float(robust_test["spearman"])),
        "heldout_mae": float(robust_test["mean_absolute_error"]),
        "positive_polarity": bool(robust_test["all_positive_polarity"]),
        "score_draws": _pad_draws(draws[SCORE_KEY], n_boot),
        "rho_draws": _pad_draws(draws["spearman"], n_boot),
        "mae_draws": _pad_draws(draws["mean_absolute_error"], n_boot),
    }
    return meta, arrays


def build_fixed_target_surface(*, data_dir: str, domains: Sequence[str], executor_tag: str,
                               target_tag: str, rungs: Sequence[str] | None = None,
                               executor_grid_template: str | None = None,
                               target_grid_template: str | None = None,
                               messages_grid_template: str | None = None,
                               grid_glob: str = "grid_*.npz", sparse: str = "name",
                               divergence: str = "tvd", min_target_information: float = 0.01,
                               train_frac: float = 0.5, n_boot: int = 500,
                               seed: int = 1207) -> dict:
    """Build an in-memory surface bundle; call :func:`save_surface` to persist it."""
    if n_boot <= 0:
        raise ValueError("n_boot must be positive for a comparable persisted surface")
    rungs = list(DEFAULT_SURFACE_RUNGS if rungs is None else rungs)
    if sparse not in rungs:
        rungs.insert(0, sparse)
    meta_rows, array_rows, ineligible, errors, inputs = [], [], [], [], {}

    for domain in domains:
        try:
            executor_dir = resolve_grid_dir(data_dir, domain, executor_grid_template)
            target_dir = resolve_grid_dir(data_dir, domain, target_grid_template)
            messages_dir = resolve_grid_dir(data_dir, domain, messages_grid_template)
            executor_path = find_grid(executor_dir, executor_tag, grid_glob)
            target_path = find_grid(target_dir, target_tag, grid_glob)
            executor = load_grid_orbits(executor_path)
            target_grid = (executor if target_path == executor_path
                           else load_grid_orbits(target_path))
            alignment = alignment_report(executor, target_grid)
            if not alignment["shape_equal"] or not alignment["row_metadata_equal"]:
                raise ValueError(f"executor/target grids are structurally unaligned: {alignment}")
            messages_path = messages_dir / "messages.json"
            messages = json.loads(messages_path.read_text())
            inputs[domain] = {
                "executor_grid": {"path": executor["path"], "sha256": executor["sha256"],
                                  "reader": executor["reader"]},
                "target_grid": {"path": target_grid["path"], "sha256": target_grid["sha256"],
                                "reader": target_grid["reader"]},
                "messages": {"path": str(messages_path), "sha256": sha256_file(messages_path)},
                "alignment": alignment,
            }
        except (FileNotFoundError, ValueError) as exc:
            errors.append({"domain": domain, "error": str(exc)})
            continue

        common = sorted(set(executor["orbits"]) & set(target_grid["orbits"]))
        for gi in common:
            message = messages.get(str(gi))
            if message is None:
                errors.append({"domain": domain, "gi": gi, "error": "message metadata missing"})
                continue
            try:
                executor_rungs = executor["orbits"][gi]
                target_rungs = target_grid["orbits"][gi]
                if sparse not in executor_rungs or sparse not in target_rungs:
                    raise ValueError(f"missing sparse rung {sparse!r}")
                n_items = len(next(iter(target_rungs[sparse].values())))
                exemplar_idx = message.get("exemplar_idx") or {}
                excluded = sorted(set((exemplar_idx.get("pos") or [])
                                      + (exemplar_idx.get("neg") or [])))
                excluded = [int(idx) for idx in excluded if 0 <= int(idx) < n_items]
                keep = np.ones(n_items, bool)
                keep[excluded] = False
                kept = np.flatnonzero(keep)
                target_orbit = _slice_orbit(target_rungs[sparse], kept)
                target = target_orbit_mean(target_orbit)
                metric_seed = _domain_seed(seed, domain, gi)
                dev_idx, heldout_idx = soft_stratified_split(
                    target, train_frac=train_frac, seed=metric_seed)
                q_dev, q_test = target[dev_idx], target[heldout_idx]
                target_dev = orbit_recovery(
                    q_dev, _slice_orbit(target_orbit, dev_idx), divergence=divergence,
                    min_target_information=min_target_information)
                target_test = orbit_recovery(
                    q_test, _slice_orbit(target_orbit, heldout_idx), divergence=divergence,
                    min_target_information=min_target_information)
                if not target_dev.get("valid") or not target_test.get("valid"):
                    reasons = sorted({row.get("error", "invalid_target")
                                      for row in (target_dev, target_test)
                                      if not row.get("valid")})
                    ineligible.append({"domain": domain, "gi": gi,
                                       "name": message.get("name"),
                                       "reason": "target_uninformative:" + ",".join(reasons)})
                    continue
                rng = np.random.default_rng(metric_seed + 10_000)
                samples = rng.integers(0, len(q_test), size=(n_boot, len(q_test)))
                found = 0
                for rung in rungs:
                    if rung not in executor_rungs:
                        continue
                    orbit = _slice_orbit(executor_rungs[rung], kept)
                    dev = orbit_recovery(
                        q_dev, _slice_orbit(orbit, dev_idx), divergence=divergence,
                        min_target_information=min_target_information)
                    heldout = orbit_recovery(
                        q_test, _slice_orbit(orbit, heldout_idx), divergence=divergence,
                        min_target_information=min_target_information)
                    if not dev.get("valid") or not heldout.get("valid"):
                        continue
                    draws = bootstrap_orbit_values(
                        q_test, _slice_orbit(orbit, heldout_idx), samples,
                        divergence=divergence,
                        min_target_information=min_target_information)
                    meta, arrays = _surface_row(
                        domain=domain, gi=gi, metric_name=message.get("name"), rung=rung,
                        dose=_legacy_dose(message, rung), dev=dev, heldout=heldout, draws=draws,
                        target_id=f"name:{domain}:{gi}:{target_grid['reader']}",
                        metric_seed=metric_seed, n_original=n_items, n_excluded=len(excluded),
                        n_development=len(dev_idx), n_heldout=len(heldout_idx), n_boot=n_boot)
                    meta_rows.append(meta)
                    array_rows.append(arrays)
                    found += 1
                if found == 0:
                    ineligible.append({"domain": domain, "gi": gi,
                                       "name": message.get("name"),
                                       "reason": "no_valid_executor_arms"})
            except ValueError as exc:
                errors.append({"domain": domain, "gi": gi,
                               "name": message.get("name"), "error": str(exc)})

    if not array_rows:
        raise ValueError("surface contains no evaluable arm rows")
    arrays = {
        key: np.stack([row[key] for row in array_rows]) if key.endswith("_draws")
        else np.asarray([row[key] for row in array_rows])
        for key in array_rows[0]
    }
    metric_cells = {(row["domain"], row["gi"]) for row in meta_rows}
    report = {
        "schema": SCHEMA,
        "analysis_status": "retrospective_heldout_surface",
        "config": {"data_dir": data_dir, "domains": list(domains),
                   "executor_tag": executor_tag, "target_tag": target_tag,
                   "rungs": rungs, "executor_grid_template": executor_grid_template,
                   "target_grid_template": target_grid_template,
                   "messages_grid_template": messages_grid_template,
                   "grid_glob": grid_glob, "sparse": sparse, "divergence": divergence,
                   "min_target_information": min_target_information,
                   "train_frac": train_frac, "n_boot": n_boot, "seed": seed},
        "manifest_sha256": manifest_sha256(),
        "n_arm_rows": len(meta_rows), "n_metric_cells": len(metric_cells),
        "metric_cells_by_domain": dict(sorted(Counter(row["domain"] for row in meta_rows).items())),
        "n_ineligible_metrics": len(ineligible), "ineligible": ineligible,
        "n_errors": len(errors), "errors": errors, "inputs": inputs,
        "claim_scope": ("Diagnostic legacy surface. Arms are held out but probe hashes, certified "
                        "units, matched controls, and prospective preregistration are absent."),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    # The counter above counts arm rows; expose the actual per-domain metric count separately.
    report["metric_cells_by_domain"] = {
        domain: len({row["gi"] for row in meta_rows if row["domain"] == domain})
        for domain in sorted({row["domain"] for row in meta_rows})}
    return {"report": report, "meta": meta_rows, "arrays": arrays}


def save_surface(bundle: Mapping, path: str | Path) -> tuple[Path, Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = bundle["arrays"]
    np.savez_compressed(path, meta=np.asarray([json.dumps(row, sort_keys=True)
                                               for row in bundle["meta"]], dtype=object),
                        report_json=np.asarray(json.dumps(bundle["report"], sort_keys=True)),
                        **arrays)
    report_path = path.with_suffix(".json")
    report_path.write_text(json.dumps(bundle["report"], indent=1))
    return path, report_path


def load_surface(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        report = json.loads(str(z["report_json"].item()))
        meta = [json.loads(str(value)) for value in z["meta"]]
        arrays = {key: np.asarray(z[key]) for key in z.files
                  if key not in {"report_json", "meta"}}
    if report.get("schema") != SCHEMA:
        raise ValueError(f"unsupported surface schema {report.get('schema')!r}")
    if len(meta) != len(arrays["heldout_score"]):
        raise ValueError("surface metadata/array rows are unaligned")
    return {"report": report, "meta": meta, "arrays": arrays,
            "path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=DATA)
    parser.add_argument("--domains", required=True)
    parser.add_argument("--executor-tag", required=True)
    parser.add_argument("--target-tag", required=True)
    parser.add_argument("--rungs", default=",".join(DEFAULT_SURFACE_RUNGS))
    parser.add_argument("--executor-grid-template", default=None)
    parser.add_argument("--target-grid-template", default=None)
    parser.add_argument("--messages-grid-template", default=None)
    parser.add_argument("--grid-glob", default="grid_*.npz")
    parser.add_argument("--sparse", default="name")
    parser.add_argument("--divergence", choices=["tvd", "shannon"], default="tvd")
    parser.add_argument("--min-target-information", type=float, default=0.01)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1207)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    bundle = build_fixed_target_surface(
        data_dir=args.data_dir, domains=[value for value in args.domains.split(",") if value],
        executor_tag=args.executor_tag, target_tag=args.target_tag,
        rungs=[value for value in args.rungs.split(",") if value],
        executor_grid_template=args.executor_grid_template,
        target_grid_template=args.target_grid_template,
        messages_grid_template=args.messages_grid_template,
        grid_glob=args.grid_glob, sparse=args.sparse, divergence=args.divergence,
        min_target_information=args.min_target_information, train_frac=args.train_frac,
        n_boot=args.n_boot, seed=args.seed)
    surface, report = save_surface(bundle, args.out)
    print(f"-> {surface}")
    print(f"-> {report}")


if __name__ == "__main__":
    main()
