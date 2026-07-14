#!/usr/bin/env python
"""Derive held-out scale--articulation substitutions from aligned reader surfaces."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

from methods.codability.experiments.fixed_target_name_substitution import summarize_metrics
from methods.codability.experiments.fixed_target_surface import load_surface


SCHEMA = "fixed_target_surface_comparison/v2"


def _finite(value) -> bool:
    return value is not None and np.isfinite(value)


def _ci(values: np.ndarray, confidence: float) -> list[float] | None:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    tail = (1.0 - confidence) / 2.0
    return [float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))]


def _index(surface: Mapping) -> dict[tuple[str, int, str], int]:
    out = {}
    for index, row in enumerate(surface["meta"]):
        key = (row["domain"], int(row["gi"]), row["rung"])
        if key in out:
            raise ValueError(f"duplicate surface row {key}")
        out[key] = index
    return out


def validate_surface_pair(small: Mapping, big: Mapping) -> dict:
    sc, bc = small["report"]["config"], big["report"]["config"]
    fields = ("target_tag", "rungs", "sparse", "divergence",
              "min_target_information", "train_frac", "n_boot", "seed")
    mismatches = {field: [sc.get(field), bc.get(field)] for field in fields
                  if sc.get(field) != bc.get(field)}
    if mismatches:
        raise ValueError(f"surface configs are not comparable: {mismatches}")
    if small["report"].get("manifest_sha256") != big["report"].get("manifest_sha256"):
        raise ValueError("surface articulation manifests differ")
    domains = set(sc["domains"]) & set(bc["domains"])
    if not domains:
        raise ValueError("surfaces have no common domains")
    target_hashes = {}
    for domain in domains:
        si = small["report"]["inputs"].get(domain)
        bi = big["report"]["inputs"].get(domain)
        if si is None or bi is None:
            continue
        sh, bh = si["target_grid"]["sha256"], bi["target_grid"]["sha256"]
        if sh != bh:
            raise ValueError(f"target grid hash differs for {domain}")
        target_hashes[domain] = sh
    sidx, bidx = _index(small), _index(big)
    common_names = sorted(set(sidx) & set(bidx))
    if not common_names:
        raise ValueError("surfaces share no arm rows")
    for key in common_names:
        sm, bm = small["meta"][sidx[key]], big["meta"][bidx[key]]
        for field in ("target_id", "metric_seed", "n_heldout"):
            if sm[field] != bm[field]:
                raise ValueError(f"surface row {key} differs on {field}")
    return {"valid": True, "common_arm_rows": len(common_names),
            "common_domains": sorted(domains),
            "target_sha256_by_domain": target_hashes,
            "small_executor": sc["executor_tag"], "big_executor": bc["executor_tag"],
            "target": sc["target_tag"]}


def _difference(arrays_left: Mapping, left: int, arrays_right: Mapping, right: int,
                *, point_key: str, draws_key: str, confidence: float) -> dict:
    point = float(arrays_left[point_key][left] - arrays_right[point_key][right])
    draws = arrays_left[draws_key][left] - arrays_right[draws_key][right]
    return {"point": point, "CI": _ci(draws, confidence)}


def _select_candidate(small: Mapping, big: Mapping, sidx: Mapping, bidx: Mapping, *,
                      domain: str, gi: int, sparse: str, equivalence_delta: float,
                      min_signature_rho: float) -> dict:
    big_sparse = bidx[(domain, gi, sparse)]
    target_score = float(big["arrays"]["dev_score"][big_sparse] - equivalence_delta)
    candidates = []
    for key, index in sidx.items():
        if key[:2] != (domain, gi) or key[2] == sparse:
            continue
        meta = small["meta"][index]
        score = float(small["arrays"]["dev_score"][index])
        rho = float(small["arrays"]["dev_rho"][index])
        cost = meta["dose"].get("scalar_cost")
        if (cost is not None and _finite(score) and _finite(rho)
                and bool(small["arrays"]["dev_positive_polarity"][index])):
            candidates.append((index, meta, score, rho, float(cost)))
    if not candidates:
        raise ValueError("no valid development articulation candidates")
    attained = [row for row in candidates if row[2] >= target_score
                and row[3] >= min_signature_rho]
    if attained:
        chosen = min(attained, key=lambda row: (row[4], row[1]["rung"]))
        rule = "minimal scalar cost reaching target and signature floor"
    else:
        chosen = max(candidates, key=lambda row: (row[2], -row[4]))
        rule = "best oriented recovery; target not attained"
    index, meta, score, rho, _ = chosen
    return {"index": index, "candidate_id": meta["rung"], "target_attained": bool(attained),
            "target_score": target_score, "selected_score": score,
            "selected_signature_rho": rho, "dose": meta["dose"], "selection_rule": rule}


def compare_surfaces(small: Mapping, big: Mapping, *, gap_delta: float = 0.02,
                     equivalence_delta: float = 0.02, min_signature_rho: float = 0.5,
                     signature_equivalence_delta: float = 0.05,
                     confidence: float = 0.95, familywise_alpha: float = 0.05) -> dict:
    validation = validate_surface_pair(small, big)
    sidx, bidx = _index(small), _index(big)
    sparse = small["report"]["config"]["sparse"]
    s_cells = {(domain, gi) for domain, gi, rung in sidx if rung == sparse}
    b_cells = {(domain, gi) for domain, gi, rung in bidx if rung == sparse}
    common_cells = sorted(s_cells & b_cells)
    familywise_confidence = 1.0 - familywise_alpha / max(len(common_cells), 1)
    rows = []
    for domain, gi in common_cells:
        try:
            si, bi = sidx[(domain, gi, sparse)], bidx[(domain, gi, sparse)]
            selection = _select_candidate(
                small, big, sidx, bidx, domain=domain, gi=gi, sparse=sparse,
                equivalence_delta=equivalence_delta, min_signature_rho=min_signature_rho)
            ai = selection["index"]
            baseline = _difference(big["arrays"], bi, small["arrays"], si,
                                   point_key="heldout_score", draws_key="score_draws",
                                   confidence=confidence)
            improvement = _difference(small["arrays"], ai, small["arrays"], si,
                                      point_key="heldout_score", draws_key="score_draws",
                                      confidence=confidence)
            match = _difference(small["arrays"], ai, big["arrays"], bi,
                                point_key="heldout_score", draws_key="score_draws",
                                confidence=confidence)
            rho_gain = _difference(small["arrays"], ai, small["arrays"], si,
                                   point_key="heldout_rho", draws_key="rho_draws",
                                   confidence=confidence)
            rho_match = _difference(small["arrays"], ai, big["arrays"], bi,
                                    point_key="heldout_rho", draws_key="rho_draws",
                                    confidence=confidence)
            rho_ci = _ci(small["arrays"]["rho_draws"][ai], confidence)
            baseline_confirmed = bool(baseline["CI"] and baseline["CI"][0] > gap_delta)
            improved = bool(improvement["CI"] and improvement["CI"][0] > 0.0)
            noninferior = bool(match["CI"] and match["CI"][0] >= -equivalence_delta)
            equivalent = bool(match["CI"] and match["CI"][0] >= -equivalence_delta
                              and match["CI"][1] <= equivalence_delta)
            polarity = bool(small["arrays"]["positive_polarity"][ai])
            signature_floor = bool(rho_ci and rho_ci[0] >= min_signature_rho)
            signature_improved = bool(rho_gain["CI"] and rho_gain["CI"][0] > 0.0)
            signature_noninferior = bool(
                rho_match["CI"] and rho_match["CI"][0] >= -signature_equivalence_delta)
            methodological = bool(baseline_confirmed and improved and noninferior and polarity
                                  and signature_floor and signature_improved
                                  and signature_noninferior)

            # A separate finite-bank read: Bonferroni simultaneous intervals over every declared
            # common metric cell in this comparison. This is deliberately not substituted for the
            # primary cellwise read, and it does not control across multiple atlas comparisons.
            fw_baseline = _difference(big["arrays"], bi, small["arrays"], si,
                                      point_key="heldout_score", draws_key="score_draws",
                                      confidence=familywise_confidence)
            fw_improvement = _difference(small["arrays"], ai, small["arrays"], si,
                                         point_key="heldout_score", draws_key="score_draws",
                                         confidence=familywise_confidence)
            fw_match = _difference(small["arrays"], ai, big["arrays"], bi,
                                   point_key="heldout_score", draws_key="score_draws",
                                   confidence=familywise_confidence)
            fw_rho_gain = _difference(small["arrays"], ai, small["arrays"], si,
                                      point_key="heldout_rho", draws_key="rho_draws",
                                      confidence=familywise_confidence)
            fw_rho_match = _difference(small["arrays"], ai, big["arrays"], bi,
                                       point_key="heldout_rho", draws_key="rho_draws",
                                       confidence=familywise_confidence)
            fw_rho_ci = _ci(small["arrays"]["rho_draws"][ai], familywise_confidence)
            fw_gates = {
                "baseline_gap_confirmed": bool(fw_baseline["CI"]
                                               and fw_baseline["CI"][0] > gap_delta),
                "articulation_improvement_confirmed": bool(
                    fw_improvement["CI"] and fw_improvement["CI"][0] > 0.0),
                "noninferior_to_big_sparse": bool(
                    fw_match["CI"] and fw_match["CI"][0] >= -equivalence_delta),
                "equivalent_to_big_sparse": bool(
                    fw_match["CI"] and fw_match["CI"][0] >= -equivalence_delta
                    and fw_match["CI"][1] <= equivalence_delta),
                "positive_polarity": polarity,
                "signature_floor": bool(fw_rho_ci and fw_rho_ci[0] >= min_signature_rho),
                "signature_improved": bool(fw_rho_gain["CI"] and fw_rho_gain["CI"][0] > 0.0),
                "signature_noninferior_to_big": bool(
                    fw_rho_match["CI"]
                    and fw_rho_match["CI"][0] >= -signature_equivalence_delta),
            }
            fw_methodological = bool(all(fw_gates[key] for key in (
                "baseline_gap_confirmed", "articulation_improvement_confirmed",
                "noninferior_to_big_sparse", "positive_polarity", "signature_floor",
                "signature_improved", "signature_noninferior_to_big")))
            meta = small["meta"][ai]
            heldout = {
                "valid": True,
                "baseline_gap_big_minus_small": baseline,
                "articulation_gain_over_small": improvement,
                "articulated_minus_big": match,
                "signature_gain_over_small": rho_gain,
                "signature_articulated_minus_big": rho_match,
                "articulated_signature_CI": rho_ci,
                "gates": {"baseline_gap_confirmed": baseline_confirmed,
                          "articulation_improvement_confirmed": improved,
                          "noninferior_to_big_sparse": noninferior,
                          "equivalent_to_big_sparse": equivalent,
                          "positive_polarity": polarity,
                          "signature_floor": signature_floor,
                          "signature_improved": signature_improved,
                          "signature_noninferior_to_big": signature_noninferior,
                          "articulation_specificity": None},
                "methodological_substitution": methodological,
                "equivalent_methodological_substitution": bool(methodological and equivalent),
                "articulation_specific_substitution": None,
                "paper_grade_substitution": False,
                "familywise": {
                    "method": "Bonferroni simultaneous percentile intervals",
                    "family_size": len(common_cells), "alpha": familywise_alpha,
                    "confidence_per_interval": familywise_confidence,
                    "gates": fw_gates,
                    "methodological_substitution": fw_methodological,
                    "equivalent_methodological_substitution": bool(
                        fw_methodological and fw_gates["equivalent_to_big_sparse"]),
                },
            }
            rows.append({
                "domain": domain, "gi": gi, "name": meta.get("metric_name"),
                "target": {"target_id": meta["target_id"], "target_view": "name",
                           "informant_or_source": small["report"]["config"]["target_tag"]},
                "readers": {"small": small["report"]["config"]["executor_tag"],
                            "big": big["report"]["config"]["executor_tag"],
                            "target": small["report"]["config"]["target_tag"]},
                "development": {"selection": {key: value for key, value in selection.items()
                                                if key != "index"}},
                "heldout": heldout,
                "selected_rung": selection["candidate_id"],
                "selected_channel": selection["dose"]["channel"],
                "selected_legacy_word_cost": selection["dose"].get("word_count"),
                "claim_grade": "diagnostic_surface_comparison",
            })
        except ValueError as exc:
            rows.append({"domain": domain, "gi": gi,
                         "ineligible": f"no_valid_development_articulation:{exc}"})

    by_domain = {}
    for domain in sorted({row["domain"] for row in rows}):
        domain_rows = [row for row in rows if row["domain"] == domain]
        by_domain[domain] = {"summary": summarize_metrics(domain_rows),
                             "per_metric": domain_rows}
    return {
        "schema": SCHEMA,
        "analysis_status": "retrospective_heldout_surface_comparison",
        "validation": validation,
        "config": {"gap_delta": gap_delta, "equivalence_delta": equivalence_delta,
                   "min_signature_rho": min_signature_rho,
                   "signature_equivalence_delta": signature_equivalence_delta,
                   "confidence": confidence, "familywise_alpha": familywise_alpha,
                   "familywise_confidence": familywise_confidence,
                   "family_size": len(common_cells)},
        "small_surface": {"path": small.get("path"), "sha256": small.get("sha256"),
                          "executor": small["report"]["config"]["executor_tag"]},
        "big_surface": {"path": big.get("path"), "sha256": big.get("sha256"),
                        "executor": big["report"]["config"]["executor_tag"]},
        "pooled": summarize_metrics(rows), "by_domain": by_domain,
        "familywise_substitution_count": sum(
            row.get("heldout", {}).get("familywise", {}).get("methodological_substitution", False)
            for row in rows),
        "paper_grade_claim_eligible": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--small-surface", required=True)
    parser.add_argument("--big-surface", required=True)
    parser.add_argument("--gap-delta", type=float, default=0.02)
    parser.add_argument("--equivalence-delta", type=float, default=0.02)
    parser.add_argument("--min-signature-rho", type=float, default=0.5)
    parser.add_argument("--signature-equivalence-delta", type=float, default=0.05)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = compare_surfaces(
        load_surface(args.small_surface), load_surface(args.big_surface),
        gap_delta=args.gap_delta, equivalence_delta=args.equivalence_delta,
        min_signature_rho=args.min_signature_rho,
        signature_equivalence_delta=args.signature_equivalence_delta,
        confidence=args.confidence, familywise_alpha=args.familywise_alpha)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
