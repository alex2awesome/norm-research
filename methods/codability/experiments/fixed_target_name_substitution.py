#!/usr/bin/env python
"""Experiment N: can explicit articulation recover a larger reader's name-only policy?

The fixed target is the larger reader's name-only soft behavior, averaged over its declared prompt
form orbit.  Candidate definitions, explanations, rules, examples, and dossiers are executed by a
smaller reader.  Candidate selection occurs on development probes; the selected arm is tested once
on held-out probes with the target-indexed frontier estimator.

Legacy ``grid_*_v1`` inputs support a retrospective method-validation analysis.  They do not carry
cryptographic probe identities, matched inert controls, certified CUF units, or a preregistered
target/channel manifest, so this runner never upgrades those results to paper-grade evidence.
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

from methods.codability.experiments.target_articulation_frontier import (
    SCORE_KEY,
    dose_record,
    load_manifest,
    manifest_sha256,
    monotone_frontier,
    orbit_recovery,
    paired_substitution_test,
    select_minimal_cost,
    target_orbit_mean,
)
from methods.codability.name_sufficiency import DATA, DOMAINS


SCHEMA = "fixed_target_name_substitution/v2"
DEFAULT_RUNGS = ["definition", "explanation", "full_rubric", "exemplars", "dossier",
                 "dossier_v2"]


class MetricIneligible(ValueError):
    """A scientifically undefined cell (usually an uninformative fixed target), not a run error."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _scalar(z, key: str, default=None):
    return z[key].item() if key in z.files and np.asarray(z[key]).shape == () else default


def _grid_identity(path: str | Path) -> tuple[str, str]:
    with np.load(path, allow_pickle=True) as z:
        reader = str(_scalar(z, "reader", ""))
    file_tag = Path(path).stem.removeprefix("grid_")
    return reader, file_tag


def find_grid(grid_dir: str | Path, reader_tag: str, grid_glob: str = "grid_*.npz") -> Path:
    matches = []
    for path in sorted(Path(grid_dir).glob(grid_glob)):
        reader, file_tag = _grid_identity(path)
        if reader_tag in {reader, reader.rstrip("/").split("/")[-1], file_tag}:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"reader tag {reader_tag!r} matched {len(matches)} files in {grid_dir}")
    return matches[0]


def resolve_grid_dir(data_dir: str | Path, domain: str,
                     template: str | None = None) -> Path:
    """Resolve a reader-specific grid directory without relocating source artifacts.

    Templates are relative to ``data_dir`` unless absolute and may contain ``{domain}`` and
    ``{data_dir}``.  This is the cross-family seam: target, larger baseline, and smaller executor
    may come from different directories while retaining their original provenance.
    """
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; choose from {sorted(DOMAINS)}")
    if template is None:
        return Path(data_dir) / DOMAINS[domain][0]
    rendered = template.format(domain=domain, data_dir=str(data_dir))
    path = Path(rendered)
    return path if path.is_absolute() else Path(data_dir) / path


def load_grid_orbits(path: str | Path) -> dict:
    """Load raw score rows without averaging away their form identifiers."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        if "scores" not in z.files or "meta" not in z.files:
            raise ValueError(f"{path} lacks scores/meta")
        scores = np.asarray(z["scores"], float)
        meta = [json.loads(str(value)) for value in z["meta"]]
        if scores.ndim != 2 or scores.shape[0] != len(meta):
            raise ValueError(f"{path} has unaligned scores/meta")
        reader = str(_scalar(z, "reader", path.stem))
        protocol = _scalar(z, "protocol_schema")
        probe_set_hash = _scalar(z, "probe_set_sha256")
        probe_hashes = ([str(value) for value in z["probe_sha256"]]
                        if "probe_sha256" in z.files else None)
        ref_dir = str(_scalar(z, "ref_dir", ""))

    grouped: dict[int, dict[str, dict[str, list[np.ndarray]]]] = {}
    nan_count = int(np.isnan(scores).sum())
    for row, desc in zip(scores, meta):
        gi, rung = int(desc["gi"]), str(desc["rung"])
        form = str(desc.get("form", "canonical"))
        grouped.setdefault(gi, {}).setdefault(rung, {}).setdefault(form, []).append(row)
    orbits = {}
    for gi, rungs in grouped.items():
        orbits[gi] = {}
        for rung, forms in rungs.items():
            orbits[gi][rung] = {}
            for form, rows in forms.items():
                with np.errstate(invalid="ignore"):
                    mean = np.nanmean(np.stack(rows), axis=0)
                orbits[gi][rung][form] = np.nan_to_num(mean, nan=0.5)

    meta_keys = sorted((int(desc["gi"]), str(desc["rung"]),
                        str(desc.get("form", "canonical"))) for desc in meta)
    return {
        "path": str(path), "sha256": sha256_file(path), "reader": reader,
        "protocol_schema": None if protocol is None else str(protocol),
        "probe_set_sha256": None if probe_set_hash is None else str(probe_set_hash),
        "probe_sha256": probe_hashes, "ref_dir": ref_dir,
        "n_items": int(scores.shape[1]), "n_rows": int(scores.shape[0]),
        "nan_score_count": nan_count, "meta_keys": meta_keys, "orbits": orbits,
    }


def alignment_report(small: dict, big: dict) -> dict:
    shape_equal = small["n_items"] == big["n_items"]
    metadata_equal = small["meta_keys"] == big["meta_keys"]
    probe_set_equal = bool(small["probe_set_sha256"] and big["probe_set_sha256"]
                           and small["probe_set_sha256"] == big["probe_set_sha256"])
    item_hashes_equal = bool(small["probe_sha256"] and big["probe_sha256"]
                             and small["probe_sha256"] == big["probe_sha256"])
    verified = bool(shape_equal and metadata_equal and (probe_set_equal or item_hashes_equal))
    return {
        "shape_equal": shape_equal, "row_metadata_equal": metadata_equal,
        "probe_set_hash_equal": probe_set_equal, "per_item_hashes_equal": item_hashes_equal,
        "cryptographically_verified": verified,
        "status": ("verified" if verified else
                   "row_metadata_and_shape_only" if shape_equal and metadata_equal else
                   "unaligned"),
    }


def soft_stratified_split(target: Sequence[float], *, train_frac: float = 0.5,
                          seed: int = 0, n_strata: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Random split within target-rank strata, preserving soft-target heterogeneity."""
    q = np.asarray(target, float)
    if q.ndim != 1 or len(q) < 12 or not np.isfinite(q).all():
        raise ValueError("soft target must contain at least 12 finite items")
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    order = np.argsort(q, kind="mergesort")
    train, test = [], []
    for stratum in np.array_split(order, min(n_strata, len(q) // 2)):
        stratum = stratum.copy()
        rng.shuffle(stratum)
        cut = min(max(int(round(len(stratum) * train_frac)), 1), len(stratum) - 1)
        train.extend(stratum[:cut])
        test.extend(stratum[cut:])
    return np.asarray(sorted(train), int), np.asarray(sorted(test), int)


def _slice_orbit(orbit: Mapping[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {form: np.asarray(values, float)[idx] for form, values in orbit.items()}


def _word_count(message: dict, rung: str) -> int:
    stored = message.get("word_len", {}).get(rung)
    if stored is not None:
        return int(stored)
    text = str(message.get("rungs", {}).get(rung, ""))
    return len(text.split())


def _legacy_dose(message: dict, rung: str, *, control: bool = False) -> dict:
    mapping = load_manifest()["legacy_rung_mapping"].get(rung, {})
    words = _word_count(message, rung)
    return dose_record(
        rung,
        "control" if control else mapping.get("channel", "composed"),
        word_count=words,
        certified_unit_count=None,
        interaction_degree=mapping.get("interaction_degree"),
        scalar_cost=float(words),
        cost_basis="legacy_message_word_count_not_certified_units",
    )


def _select_control(control_rows: list[dict], selected_cost: float) -> dict | None:
    if not control_rows:
        return None
    return min(control_rows, key=lambda row: (abs(row["dose"]["scalar_cost"] - selected_cost),
                                              row["dose"]["candidate_id"]))


def analyze_metric(*, domain: str, gi: int, small_rungs: dict, big_rungs: dict,
                   target_rungs: dict, message: dict, small_reader: str, big_reader: str,
                   target_reader: str,
                   rungs: list[str], control_rungs: list[str], sparse: str,
                   divergence: str, min_target_information: float, train_frac: float,
                   gap_delta: float, equivalence_delta: float, min_signature_rho: float,
                   signature_equivalence_delta: float, n_boot: int, seed: int) -> dict:
    if sparse not in small_rungs or sparse not in big_rungs or sparse not in target_rungs:
        raise ValueError(f"missing sparse rung {sparse!r}")
    n_items = len(next(iter(target_rungs[sparse].values())))
    all_rungs = list(small_rungs.values()) + list(big_rungs.values()) + list(target_rungs.values())
    if any(len(values) != n_items for orbit in all_rungs
           for values in orbit.values()):
        raise ValueError("rung/form item counts differ")

    keep = np.ones(n_items, bool)
    exemplar_idx = message.get("exemplar_idx") or {}
    excluded = sorted(set((exemplar_idx.get("pos") or []) + (exemplar_idx.get("neg") or [])))
    excluded = [idx for idx in excluded if 0 <= int(idx) < n_items]
    keep[excluded] = False
    kept_idx = np.flatnonzero(keep)
    target_orbit = _slice_orbit(target_rungs[sparse], kept_idx)
    target = target_orbit_mean(target_orbit)
    dev, test = soft_stratified_split(target, train_frac=train_frac, seed=seed)
    q_dev, q_test = target[dev], target[test]

    big_sparse_kept = _slice_orbit(big_rungs[sparse], kept_idx)
    big_dev = orbit_recovery(q_dev, _slice_orbit(big_sparse_kept, dev),
                             divergence=divergence,
                             min_target_information=min_target_information)
    small_sparse_kept = _slice_orbit(small_rungs[sparse], kept_idx)
    small_dev = orbit_recovery(q_dev, _slice_orbit(small_sparse_kept, dev),
                               divergence=divergence,
                               min_target_information=min_target_information)
    if not big_dev.get("valid") or not small_dev.get("valid"):
        reasons = sorted({row.get("error", "invalid_sparse_channel")
                          for row in (big_dev, small_dev) if not row.get("valid")})
        raise MetricIneligible("development_sparse_unmeasurable:" + ",".join(reasons))

    candidate_rows = []
    for rung in rungs:
        if rung == sparse or rung not in small_rungs:
            continue
        orbit = _slice_orbit(small_rungs[rung], kept_idx)
        candidate_rows.append({
            "dose": _legacy_dose(message, rung),
            "recovery": orbit_recovery(q_dev, _slice_orbit(orbit, dev),
                                       divergence=divergence,
                                       min_target_information=min_target_information),
        })
    target_score = float(big_dev["robust"][SCORE_KEY] - equivalence_delta)
    try:
        selected = select_minimal_cost(candidate_rows, target_score=target_score,
                                       min_signature_rho=min_signature_rho)
    except ValueError as exc:
        raise MetricIneligible(f"no_valid_articulation_candidate:{exc}") from exc
    selected_rung = selected["candidate_id"]

    control_rows = []
    for rung in control_rungs:
        if rung not in small_rungs:
            continue
        orbit = _slice_orbit(small_rungs[rung], kept_idx)
        control_rows.append({
            "dose": _legacy_dose(message, rung, control=True),
            "recovery": orbit_recovery(q_dev, _slice_orbit(orbit, dev),
                                       divergence=divergence,
                                       min_target_information=min_target_information),
        })
    selected_control = _select_control(control_rows, selected["dose"]["scalar_cost"])
    control_orbit = None
    if selected_control is not None:
        control_orbit = _slice_orbit(
            _slice_orbit(small_rungs[selected_control["dose"]["candidate_id"]], kept_idx), test)

    heldout = paired_substitution_test(
        q_test,
        small_sparse_orbit=_slice_orbit(small_sparse_kept, test),
        big_sparse_orbit=_slice_orbit(big_sparse_kept, test),
        articulated_orbit=_slice_orbit(_slice_orbit(small_rungs[selected_rung], kept_idx), test),
        control_orbit=control_orbit,
        divergence=divergence,
        min_target_information=min_target_information,
        gap_delta=gap_delta,
        equivalence_delta=equivalence_delta,
        min_signature_rho=min_signature_rho,
        signature_equivalence_delta=signature_equivalence_delta,
        n_boot=n_boot,
        seed=seed + 10_000,
    )
    if not heldout.get("valid"):
        raise MetricIneligible(f"heldout_unmeasurable:{heldout.get('error', 'invalid_arm')}")

    return {
        "domain": domain, "gi": int(gi), "name": message.get("name"),
        "target": {
            "target_id": f"name:{domain}:{gi}:{target_reader}",
            "target_view": "name", "community_or_frame": domain,
            "informant_or_source": target_reader, "probe_set_id": "legacy-grid-columns",
            "frozen_before_candidate_evaluation": True,
            "construction": "mean of larger-reader name-only prompt-form orbit",
            "n_target_forms": len(target_orbit),
        },
        "readers": {"small": small_reader, "big": big_reader, "target": target_reader},
        "probe_split": {"seed": int(seed), "train_frac": float(train_frac),
                        "n_original": int(n_items), "n_exemplar_excluded": len(excluded),
                        "n_development": int(len(dev)), "n_heldout": int(len(test)),
                        "stratification": "random within five target-rank strata"},
        "development": {
            "small_sparse": small_dev, "big_sparse": big_dev,
            "candidate_frontier": monotone_frontier(candidate_rows),
            "candidate_rows": candidate_rows,
            "selection": selected,
            "control_rows": control_rows,
            "selected_control": (None if selected_control is None
                                 else selected_control["dose"]["candidate_id"]),
        },
        "heldout": heldout,
        "selected_rung": selected_rung,
        "selected_channel": selected["dose"]["channel"],
        "selected_legacy_word_cost": selected["dose"]["word_count"],
        "claim_grade": "diagnostic_reanalysis_of_legacy_artifacts",
        "claim_limitations": [
            "legacy prompt arms and margins were not preregistered under this target view",
            "legacy word count is not a certified articulation-unit scale",
            "paper-grade status additionally depends on cryptographic probe alignment",
            "articulation-specific status requires a matched inert or wrong-construct control",
        ],
    }


def summarize_metrics(rows: list[dict]) -> dict:
    evaluable = [row for row in rows if "error" not in row and "ineligible" not in row
                 and row.get("heldout", {}).get("valid")]
    ineligible = [row for row in rows if "ineligible" in row]
    errors = [row for row in rows if "error" in row]
    selected = Counter(row["selected_rung"] for row in evaluable)
    channels = Counter(row["selected_channel"] for row in evaluable)
    gaps = [row for row in evaluable
            if row["heldout"]["gates"]["baseline_gap_confirmed"]]
    methodological = [row for row in gaps if row["heldout"]["methodological_substitution"]]
    equivalent = [row for row in gaps
                  if row["heldout"]["equivalent_methodological_substitution"]]
    controlled = [row for row in gaps
                  if row["heldout"]["articulation_specific_substitution"] is not None]
    specific = [row for row in controlled
                if row["heldout"]["articulation_specific_substitution"] is True]
    sensitivity = []
    for recovery_margin in (0.02, 0.05, 0.10, 0.20):
        for signature_margin in (0.05, 0.10, 0.20):
            for rho_floor in (0.3, 0.5):
                successes = 0
                for row in gaps:
                    heldout, gates = row["heldout"], row["heldout"]["gates"]
                    match_ci = heldout["articulated_minus_big"]["CI"]
                    rho_ci = heldout["articulated_signature_CI"]
                    rho_match_ci = heldout["signature_articulated_minus_big"]["CI"]
                    success = bool(
                        gates["baseline_gap_confirmed"]
                        and gates["articulation_improvement_confirmed"]
                        and gates["positive_polarity"]
                        and gates["signature_improved"]
                        and match_ci and match_ci[0] >= -recovery_margin
                        and rho_ci and rho_ci[0] >= rho_floor
                        and rho_match_ci and rho_match_ci[0] >= -signature_margin)
                    successes += int(success)
                sensitivity.append({
                    "recovery_noninferiority_margin": recovery_margin,
                    "signature_noninferiority_margin": signature_margin,
                    "minimum_signature_rho": rho_floor,
                    "success": successes,
                    "n_confirmed_gaps": len(gaps),
                    "rate": None if not gaps else successes / len(gaps),
                })
    return {
        "n_rows": len(rows), "n_evaluable": len(evaluable),
        "n_ineligible": len(ineligible), "n_errors": len(errors),
        "ineligible_reasons": dict(sorted(Counter(
            row["ineligible"] for row in ineligible).items())),
        "baseline_gap_confirmed": len(gaps),
        "methodological_substitution_among_confirmed_gaps": {
            "success": len(methodological), "n": len(gaps),
            "rate": None if not gaps else len(methodological) / len(gaps),
        },
        "equivalent_methodological_substitution_among_confirmed_gaps": {
            "success": len(equivalent), "n": len(gaps),
            "rate": None if not gaps else len(equivalent) / len(gaps),
        },
        "articulation_specific_substitution_among_confirmed_gaps": {
            "success": len(specific), "n_available": len(controlled),
            "n_unavailable": len(gaps) - len(controlled),
            "rate": None if not controlled else len(specific) / len(controlled),
            "note": "Uncontrolled rows are unavailable, not failures.",
        },
        "development_target_attained": sum(
            row["development"]["selection"]["target_attained"] for row in evaluable),
        "articulation_debt": {
            "finite": len(methodological),
            "right_censored_within_legacy_bank": len(gaps) - len(methodological),
            "eligible_baseline_gaps": len(gaps),
            "finite_word_costs": [row["selected_legacy_word_cost"] for row in methodological],
            "cost_basis": "legacy message words; not certified articulation units",
        },
        "posthoc_margin_sensitivity": {
            "status": "diagnostic_only_not_preregistered",
            "grid": sensitivity,
        },
        "selected_rungs": dict(sorted(selected.items())),
        "selected_channels": dict(sorted(channels.items())),
    }


def analyze_domain(*, data_dir: str, domain: str, small_tag: str, big_tag: str,
                   target_tag: str | None,
                   rungs: list[str], control_rungs: list[str], sparse: str,
                   divergence: str, min_target_information: float, train_frac: float,
                   gap_delta: float, equivalence_delta: float, min_signature_rho: float,
                   signature_equivalence_delta: float, n_boot: int, seed: int,
                   grid_glob: str = "grid_*.npz",
                   small_grid_template: str | None = None,
                   big_grid_template: str | None = None,
                   target_grid_template: str | None = None,
                   messages_grid_template: str | None = None) -> dict:
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; choose from {sorted(DOMAINS)}")
    grid_dir = resolve_grid_dir(data_dir, domain)
    small_grid_dir = resolve_grid_dir(data_dir, domain, small_grid_template)
    big_grid_dir = resolve_grid_dir(data_dir, domain, big_grid_template)
    target_grid_dir = resolve_grid_dir(data_dir, domain, target_grid_template)
    messages_grid_dir = resolve_grid_dir(data_dir, domain, messages_grid_template)
    target_tag = big_tag if target_tag is None else target_tag
    small_path = find_grid(small_grid_dir, small_tag, grid_glob)
    big_path = find_grid(big_grid_dir, big_tag, grid_glob)
    target_path = find_grid(target_grid_dir, target_tag, grid_glob)
    small, big = load_grid_orbits(small_path), load_grid_orbits(big_path)
    target = big if target_path == big_path else load_grid_orbits(target_path)
    alignment = {"small_vs_big": alignment_report(small, big),
                 "small_vs_target": alignment_report(small, target),
                 "big_vs_target": alignment_report(big, target)}
    if any(not row["shape_equal"] or not row["row_metadata_equal"]
           for row in alignment.values()):
        raise ValueError(f"reader/target grids are not structurally aligned: {alignment}")
    messages_path = messages_grid_dir / "messages.json"
    messages = json.loads(messages_path.read_text())

    rows = []
    common = sorted(set(small["orbits"]) & set(big["orbits"]) & set(target["orbits"]))
    for gi in common:
        if str(gi) not in messages:
            rows.append({"domain": domain, "gi": gi, "error": "message metadata missing"})
            continue
        try:
            row = analyze_metric(
                domain=domain, gi=gi, small_rungs=small["orbits"][gi],
                big_rungs=big["orbits"][gi], target_rungs=target["orbits"][gi],
                message=messages[str(gi)], small_reader=small["reader"],
                big_reader=big["reader"], target_reader=target["reader"], rungs=rungs,
                control_rungs=control_rungs, sparse=sparse, divergence=divergence,
                min_target_information=min_target_information, train_frac=train_frac,
                gap_delta=gap_delta, equivalence_delta=equivalence_delta,
                min_signature_rho=min_signature_rho,
                signature_equivalence_delta=signature_equivalence_delta,
                n_boot=n_boot, seed=seed + gi)
            rows.append(row)
        except MetricIneligible as exc:
            rows.append({"domain": domain, "gi": gi,
                         "name": messages[str(gi)].get("name"), "ineligible": str(exc)})
        except ValueError as exc:
            rows.append({"domain": domain, "gi": gi,
                         "name": messages[str(gi)].get("name"), "error": str(exc)})

    return {
        "domain": domain, "summary": summarize_metrics(rows), "per_metric": rows,
        "alignment": alignment,
        "grid_directories": {"canonical": str(grid_dir), "small": str(small_grid_dir),
                             "big": str(big_grid_dir), "target": str(target_grid_dir),
                             "messages": str(messages_grid_dir)},
        "inputs": {
            "small_grid": {key: small[key] for key in ("path", "sha256", "reader",
                                                        "protocol_schema", "n_items", "n_rows",
                                                        "nan_score_count")},
            "big_grid": {key: big[key] for key in ("path", "sha256", "reader",
                                                    "protocol_schema", "n_items", "n_rows",
                                                    "nan_score_count")},
            "target_grid": {key: target[key] for key in ("path", "sha256", "reader",
                                                          "protocol_schema", "n_items", "n_rows",
                                                          "nan_score_count")},
            "messages": {"path": str(messages_path), "sha256": sha256_file(messages_path)},
        },
    }


def analyze_name_substitution(*, data_dir: str, domains: list[str], small_tag: str,
                              big_tag: str, target_tag: str | None = None,
                              rungs: list[str] | None = None,
                              control_rungs: list[str] | None = None, sparse: str = "name",
                              divergence: str = "tvd", min_target_information: float = 0.01,
                              train_frac: float = 0.5, gap_delta: float = 0.02,
                              equivalence_delta: float = 0.02,
                              min_signature_rho: float = 0.5,
                              signature_equivalence_delta: float = 0.05,
                              n_boot: int = 1000, seed: int = 0,
                              grid_glob: str = "grid_*.npz",
                              small_grid_template: str | None = None,
                              big_grid_template: str | None = None,
                              target_grid_template: str | None = None,
                              messages_grid_template: str | None = None) -> dict:
    rungs = list(DEFAULT_RUNGS if rungs is None else rungs)
    control_rungs = list(control_rungs or [])
    by_domain, pooled_rows = {}, []
    for offset, domain in enumerate(domains):
        try:
            result = analyze_domain(
                data_dir=data_dir, domain=domain, small_tag=small_tag, big_tag=big_tag,
                target_tag=target_tag,
                rungs=rungs, control_rungs=control_rungs, sparse=sparse,
                divergence=divergence, min_target_information=min_target_information,
                train_frac=train_frac, gap_delta=gap_delta,
                equivalence_delta=equivalence_delta, min_signature_rho=min_signature_rho,
                signature_equivalence_delta=signature_equivalence_delta,
                n_boot=n_boot, seed=seed + 100_000 * offset, grid_glob=grid_glob,
                small_grid_template=small_grid_template,
                big_grid_template=big_grid_template,
                target_grid_template=target_grid_template,
                messages_grid_template=messages_grid_template)
            by_domain[domain] = result
            pooled_rows.extend(result["per_metric"])
        except (FileNotFoundError, ValueError) as exc:
            by_domain[domain] = {"domain": domain, "error": str(exc)}
    return {
        "schema": SCHEMA,
        "experiment": "N_lexical_fixed_target",
        "analysis_status": "retrospective_heldout_method_validation",
        "target_view": "larger-reader name-only form-quotient behavior",
        "claim_scope": ("Diagnostic reanalysis. The fixed-target estimator and held-out selection "
                        "are valid for these arrays; legacy provenance/alignment/control gaps "
                        "prevent a paper-grade substitution claim."),
        "paper_grade_claim_eligible": False,
        "manifest": {"path": str(Path(__file__).with_name(
            "target_articulation_manifest_v1.json")), "sha256": manifest_sha256()},
        "config": {"data_dir": data_dir, "domains": domains, "small_tag": small_tag,
                   "big_tag": big_tag, "target_tag": (big_tag if target_tag is None else target_tag),
                   "rungs": rungs, "control_rungs": control_rungs,
                   "sparse": sparse, "divergence": divergence,
                   "min_target_information": min_target_information,
                   "train_frac": train_frac, "gap_delta": gap_delta,
                   "equivalence_delta": equivalence_delta,
                   "min_signature_rho": min_signature_rho,
                   "signature_equivalence_delta": signature_equivalence_delta,
                   "n_boot": n_boot, "seed": seed, "grid_glob": grid_glob},
        "grid_templates": {"small": small_grid_template, "big": big_grid_template,
                           "target": target_grid_template, "messages": messages_grid_template},
        "pooled": summarize_metrics(pooled_rows),
        "by_domain": by_domain,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=DATA)
    parser.add_argument("--domains", default="humor,math")
    parser.add_argument("--small-tag", default="Llama-3.2-3B-Instruct")
    parser.add_argument("--big-tag", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--target-tag", default=None,
                        help="fixed name-policy informant; defaults to --big-tag")
    parser.add_argument("--rungs", default=",".join(DEFAULT_RUNGS))
    parser.add_argument("--control-rungs", default="")
    parser.add_argument("--sparse", default="name")
    parser.add_argument("--divergence", choices=["tvd", "shannon"], default="tvd")
    parser.add_argument("--min-target-information", type=float, default=0.01)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--gap-delta", type=float, default=0.02)
    parser.add_argument("--equivalence-delta", type=float, default=0.02)
    parser.add_argument("--min-signature-rho", type=float, default=0.5)
    parser.add_argument("--signature-equivalence-delta", type=float, default=0.05)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-glob", default="grid_*.npz")
    parser.add_argument("--small-grid-template", default=None,
                        help="relative/absolute template; supports {domain} and {data_dir}")
    parser.add_argument("--big-grid-template", default=None)
    parser.add_argument("--target-grid-template", default=None)
    parser.add_argument("--messages-grid-template", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = analyze_name_substitution(
        data_dir=args.data_dir,
        domains=[value for value in args.domains.split(",") if value],
        small_tag=args.small_tag, big_tag=args.big_tag, target_tag=args.target_tag,
        rungs=[value for value in args.rungs.split(",") if value],
        control_rungs=[value for value in args.control_rungs.split(",") if value],
        sparse=args.sparse, divergence=args.divergence,
        min_target_information=args.min_target_information, train_frac=args.train_frac,
        gap_delta=args.gap_delta, equivalence_delta=args.equivalence_delta,
        min_signature_rho=args.min_signature_rho,
        signature_equivalence_delta=args.signature_equivalence_delta,
        n_boot=args.n_boot, seed=args.seed, grid_glob=args.grid_glob,
        small_grid_template=args.small_grid_template,
        big_grid_template=args.big_grid_template,
        target_grid_template=args.target_grid_template,
        messages_grid_template=args.messages_grid_template)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
