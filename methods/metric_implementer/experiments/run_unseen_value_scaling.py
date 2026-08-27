"""Compute Heaps/Herdan scaling-law fits on REAL discovery + value curves and the adaptive-bias
diagnostic, emit a results JSON and plot-ready arrays. Zero-GPU; consumes existing frozen samples.

Curves produced (all from data already on disk):
  1. DISCOVERY (count) rarefaction  E[S(m)]  from an alpha-probe frozen sample -> power vs saturating.
  2. VALUE rarefaction  E[V(m)]  from the same sample's per-criterion additive value -> power vs sat.
  3. ADAPTIVE-BIAS diagnostic: raw DRAW-ORDER accumulation alpha  vs  rarefied (exchangeable) alpha on
     the SAME final pool. Their gap is a data-only measure of adaptive front-loading (no propensity
     model needed). Plus Good-Turing missing-mass stability across the 3 independent proposer runs
     (glm_a/b/c) = an exchangeability check that validates the i.i.d.-audit assumption.
  4. REGISTRY accumulation curves (real GEPA draw order) for every task with a thick-enough lineage.

  python -m methods.metric_implementer.experiments.run_unseen_value_scaling \
      --npz notebooks/data/prompt_optimality_20260703/aligned_npz/creative-writing_R3_metric40_sigs.npz \
      --out notebooks/data/unseen_value_scaling_20260717
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from . import unseen_value_scaling as uvs
from .alpha_probe import chao1, collide, rarefaction, spectrum, missing_mass_bound
from .value_census import additive_value, submodular_value, value_rarefaction
from .. import vinfo


def _fit_block(m, y, label):
    cmp = uvs.compare_scaling_forms(m, y, n_boot=500)
    p = cmp["power"]
    print(f"  [{label}]  alpha={p.get('alpha'):.3f} CI{[round(x,3) for x in p.get('alpha_ci',[])]}  "
          f"R2_lin={p.get('r2_linear'):.4f}  verdict={cmp['verdict']} ({cmp['verdict_note']})")
    return {"m": np.asarray(m).tolist(), "y": np.asarray(y).tolist(), "fit": cmp}


def discovery_and_value(npz_path, do_redundancy=True):
    z = np.load(npz_path, allow_pickle=True)
    if "M_i" not in z:
        return None
    sigs = np.asarray(z["sigs"], float)                          # (n_criteria, n_probes)
    tags = np.asarray(z["tags"])
    tau = float(z["tau"]) if "tau" in z else 0.02
    M_i = (np.asarray(z["M_i"], float) > 0.5).astype(int)        # metric's own verdict, anchor-free
    n_crit = sigs.shape[0]
    labels = collide(sigs, tau)                                  # paraphrase-collapse -> species
    f = spectrum(labels, tags)["f"]
    print(f"\n== {os.path.basename(npz_path)}  ({n_crit} criteria, {len(set(labels))} species, tau={tau}) ==")

    # 1. discovery (count) rarefaction
    ms_s, S = rarefaction(f, n_crit, n_points=40)
    disc = _fit_block(ms_s, S, "discovery E[S(m)]")

    # 2. value rarefaction (ADDITIVE / redundancy-blind — see caveat below)
    species, n_s, v_s = additive_value(sigs, labels, M_i)
    ms_v, V = value_rarefaction(n_s, v_s, n_crit, n_points=40)
    val = _fit_block(ms_v, V, "additive value E[V(m)]")

    # 2b. value-coverage curve  f_{p_seen}(value)  + its derived exponent (curve #3).
    #     Coverage x-axis: OBSERVED richness D (fraction of observed) — NOT Chao1, which explodes
    #     when doubletons are scarce (f2 small) and makes the absolute '% covered' unreliable.
    D = len(set(labels.tolist()))
    f2 = f.get(2, 0)
    richness_chao1 = chao1(f, D)
    chao1_reliable = f2 >= 5                                # crude guard: Chao1 needs enough doubletons
    S_on_v = S if np.array_equal(ms_s, ms_v) else np.interp(ms_v, ms_s, S)
    vc = uvs.value_coverage_curve(ms_v, S_on_v, V, richness=D)   # observed-richness coverage
    # NOTE: do NOT feed the fit_power_law alpha CIs into the distinguishability test — they are
    # ci_kind="fit_stability" (~70x too narrow, see module docstring) and would spuriously declare a
    # near-zero exponent "distinguishable from 0". Pass None => distinguishable_from_zero=None (honestly
    # untested); the regime label still flags |exponent|<=0.02 as indistinguishable-from-neutral.
    vc_exp = uvs.value_coverage_exponent(alpha_count=disc["fit"]["power"].get("alpha"),
                                         alpha_value=val["fit"]["power"].get("alpha"))
    print(f"  [value-coverage] coverage=S/D(observed); Chao1={richness_chao1:.0f} "
          f"({'usable' if chao1_reliable else 'UNRELIABLE f2=%d' % f2}); "
          f"exponent (a_V-a_S)={vc_exp['value_coverage_exponent']:+.3f} -> {vc_exp['regime'].split(':')[0]}")

    # 2c. REDUNDANCY of the recoverable signal (the S3 audit fix). The ORDER-INDEPENDENT evidence is
    #     the headline: on the SAME greedy-selected species, summed-standalone value (additive) vs the
    #     joint conditional-MI value (R_full) — their gap is pure double-counting, and the greedy stop
    #     point is a minimum sufficient-set size. We DELIBERATELY do NOT compare the Heaps alpha of the
    #     random-order additive rarefaction (40 pts, ~linear) to the alpha of the greedy joint curve:
    #     greedy best-first saturates fast even at ZERO redundancy (verified: 30 non-overlapping sources
    #     give greedy alpha 0.17 vs random 1.00), so that gap is an ORDERING artifact, not redundancy.
    #     For a clean redundancy-on-shape read we instead hold the GREEDY ORDER FIXED and vary only
    #     additive-vs-joint: cumulative additive along the greedy order vs V_best (joint) along it.
    redundancy = {"ran": False}
    try:
        if not do_redundancy:
            raise RuntimeError("skipped (cross-metric sweep)")
        sm = submodular_value(sigs, labels, M_i, top_k=120)
        vbest = np.asarray(sm["V_best"], float)             # cumulative joint (redundancy-discounted)
        chosen = list(sm.get("order", []))
        if len(vbest) >= 5 and vbest[-1] > 0 and chosen:
            add_greedy = np.cumsum(np.asarray(v_s, float)[chosen])       # additive, SAME greedy order
            joint_greedy = vbest[1:]                                     # joint, same greedy order
            mm_g = np.arange(1, len(joint_greedy) + 1, dtype=float)
            a_add_g = uvs.fit_power_law(mm_g, add_greedy, n_boot=0).get("alpha")
            a_joint_g = uvs.fit_power_law(mm_g, joint_greedy, n_boot=0).get("alpha")
            # The CORRECT value scaling estimand post-audit: value does NOT follow a Heaps law —
            # fit the SATURATING form on the joint curve and report (y_inf ceiling, tau timescale),
            # normalized by H(M_i) so the ceiling is on an interpretable 0-1 scale.
            h_mi = float(vinfo._h_bits(float(M_i.mean())))
            sat_fit = uvs.fit_saturating(mm_g, joint_greedy, n_boot=200)
            redundancy = {
                "ran": True,
                # order-independent HEADLINE (trustworthy):
                "H_Mi_bits": h_mi,
                "additive_sum_selected": sm.get("additive_sum_selected"),
                "R_full_joint": sm["R_full"],
                "joint_over_H": (sm["R_full"] / h_mi if h_mi > 0 else float("nan")),
                "redundancy_gap_selected": sm.get("redundancy_gap_selected"),
                "n_selected": sm.get("n_selected"),
                "V_best": sm["V_best"],
                # the value "scaling law" = saturation parameters, not a power exponent:
                "saturating_fit": {k: sat_fit.get(k) for k in
                                   ("ok", "y_inf", "tau", "y_inf_ci", "tau_ci", "r2_linear", "ci_kind")},
                "y_inf_over_H": (sat_fit.get("y_inf", float("nan")) / h_mi
                                 if (sat_fit.get("ok") and h_mi > 0) else float("nan")),
                # clean same-order shape read (redundancy only, ordering held fixed) — few greedy pts,
                # descriptive, NOT a scaling law:
                "alpha_additive_greedy_order": a_add_g,
                "alpha_joint_greedy_order": a_joint_g,
                "shape_note": "both fit on greedy-ordered points; descriptive curve shape, not a Heaps law",
                # explicitly retired confound (kept as a labelled diagnostic, not a redundancy claim):
                "alpha_additive_random_order_DIAGNOSTIC": val["fit"]["power"].get("alpha"),
            }
            gap = redundancy["redundancy_gap_selected"]
            sat_s = (f"y_inf/H={redundancy['y_inf_over_H']:.2f} tau={sat_fit['tau']:.1f}"
                     if sat_fit.get("ok") else "sat-fit failed")
            print(f"  [redundancy] {sm.get('n_selected')} greedy species: additive-sum "
                  f"{sm.get('additive_sum_selected'):.3f} vs JOINT {sm['R_full']:.3f} bits "
                  f"(gap={gap:+.3f} = {100*gap/max(sm.get('additive_sum_selected',1e-9),1e-9):.0f}% double-counted); "
                  f"joint/H={redundancy['joint_over_H']:.2f}; {sat_s}")
    except Exception as e:
        redundancy = {"ran": False, "reason": str(e)[:120]}

    # 3a. exchangeability check across the 3 independent proposer runs (glm_a/b/c). NOTE: this only
    #     tests that the three proposer samples are MUTUALLY exchangeable — a PREREQUISITE for treating
    #     any one as an i.i.d. audit stream. It is NOT itself an audit-vs-adaptive-mining de-bias.
    run_mm = {}
    for run in ("glm_a", "glm_b", "glm_c"):
        mask = tags == run
        if mask.sum() >= 10:
            lab_r = collide(sigs[mask], tau)
            fr = spectrum(lab_r, tags[mask])["f"]
            mm = missing_mass_bound(fr, int(mask.sum()))
            run_mm[run] = {"n": int(mask.sum()), "m0_hat": mm["m0_hat"], "m0_ci_hi": mm["m0_ci_hi"]}
    if run_mm:
        vals = [v["m0_hat"] for v in run_mm.values()]
        print(f"  [exchangeability] GT M0 across 3 proposer runs {[round(v,3) for v in vals]} "
              f"spread={max(vals)-min(vals):.3f}; M0~{np.mean(vals):.2f} is HIGH but partly a tau={tau} "
              f"artifact (collide leaves {len(set(labels))}/{n_crit} own species)")

    return {"npz": os.path.basename(npz_path), "n_criteria": n_crit, "n_species": len(set(labels.tolist())),
            "discovery": disc, "value": val,
            "value_coverage": vc, "value_coverage_exponent": vc_exp,
            "redundancy_check": redundancy,
            "chao1": {"richness": richness_chao1, "f2": int(f2), "reliable": bool(chao1_reliable)},
            "exchangeability_missing_mass": run_mm,
            "alpha_value_vs_count": {"alpha_value": val["fit"]["power"].get("alpha"),
                                     "alpha_count": disc["fit"]["power"].get("alpha")}}


def _norm_body(body):
    """Normalize a prose criterion body for paraphrase-tolerant dedup: lowercase alnum, first 120
    chars. Two version bodies that differ only in wording collapse; genuinely new criteria don't."""
    return re.sub(r"[^a-z0-9 ]", "", (body or "").lower())[:120].strip()


def registry_curves():
    """Discovery curve over the REAL GEPA draw order: accumulate DISTINCT criterion formulations
    (paraphrase-normalized bodies) across every metric-version in a task, ordered by created_at
    wall-clock. Registry bodies are one prose criterion each, so distinct-body accumulation over time
    IS the adaptive metric-proposal trajectory (with true timestamps)."""
    tasks = [os.path.basename(os.path.dirname(p)) for p in
             glob.glob("outputs/metric_implementer/*/registry")]
    out = {}
    for task in sorted(set(tasks)):
        vs = glob.glob(
            f"outputs/metric_implementer/{task}/registry/metrics/*/versions/v*__prompt.json")
        recs = []
        for vf in vs:
            try:
                d = json.load(open(vf))
                recs.append((d.get("created_at", ""), _norm_body(d.get("body", ""))))
            except Exception:
                continue
        recs = [r for r in recs if r[1]]
        recs.sort(key=lambda r: r[0])                       # wall-clock GEPA draw order
        if len(recs) < 8:
            print(f"\n== registry [{task}] too thin ({len(recs)} timestamped bodies) — skipped ==")
            continue
        seen, curve = set(), []
        for _, nb in recs:
            seen.add(nb); curve.append(len(seen))
        m = np.arange(1, len(curve) + 1, dtype=float)
        y = np.asarray(curve, float)
        if curve[-1] < 4:                                   # degenerate: nearly one criterion
            print(f"\n== registry [{task}] degenerate ({curve[-1]} distinct) — skipped ==")
            continue
        cmp = uvs.compare_scaling_forms(m, y, n_boot=300)
        a = cmp["power"].get("alpha")
        # REAL sampling CI (moving-block unit bootstrap of the discovery increments), not the
        # fit-stability CI — the fit-stability CI is ~70x too narrow on cumulative curves.
        inc = np.diff(np.concatenate([[0], curve]))
        ub = uvs.heaps_unit_bootstrap_ci(inc.astype(int), n_boot=500)
        tail_new = curve[-1] - curve[int(0.8 * len(curve))]
        fit_ci = cmp["power"].get("alpha_ci")
        print(f"\n== registry [{task}] {len(recs)} versions, {curve[-1]} distinct criteria ==")
        print(f"  alpha={a:.3f}  fit-stability CI={[round(x,2) for x in fit_ci]} (too narrow)  "
              f"UNIT-boot CI={[round(x,2) for x in ub['alpha_ci']] if ub.get('ok') else 'na'}")
        print(f"  verdict={cmp['verdict']}  last-20%-added={tail_new} "
              f"({'saturating' if tail_new <= 0.1*curve[-1] else 'still discovering'})")
        out[task] = {"n_versions": len(recs), "curve": curve, "fit": cmp,
                     "unit_bootstrap_ci": (ub.get("alpha_ci") if ub.get("ok") else None),
                     "tail_new_last20pct": int(tail_new)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="notebooks/data/prompt_optimality_20260703/aligned_npz/"
                                      "creative-writing_R3_metric40_sigs.npz")
    ap.add_argument("--glob", default="notebooks/data/prompt_optimality_20260703/aligned_npz/*_sigs.npz",
                    help="sweep ALL matching npz for the cross-metric exponent distribution")
    ap.add_argument("--extra-globs", nargs="*", default=[
        "notebooks/data/two_faces_20260702/r3_humor/llama8b_glm/*_sigs.npz",
        "notebooks/data/two_faces_20260702/r3_cw/llama8b_glm/*_sigs.npz"],
                    help="additional sigs npz (other domains — cross-domain replication)")
    ap.add_argument("--sweep-redundancy", action="store_true",
                    help="run the greedy redundancy check for EVERY sweep metric (slow, CPU-only)")
    ap.add_argument("--out", default="notebooks/data/unseen_value_scaling_20260717")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # detailed example metric (full curves for plotting)
    results = {"example": discovery_and_value(a.npz), "registry": {}, "cross_metric": []}

    # cross-metric sweep: the DISTRIBUTION of (alpha_S, alpha_V, exponent, M0) is the real result.
    # Domain = the npz basename prefix (creative-writing / humor / ...), so two-faces files widen scope.
    paths = sorted(set(glob.glob(a.glob)) | {q for g in a.extra_globs for q in glob.glob(g)})
    print(f"\n===== cross-metric sweep ({len(paths)} npz, redundancy={'ON' if a.sweep_redundancy else 'off'}) =====")
    for p in paths:
        try:
            dv = discovery_and_value(p, do_redundancy=a.sweep_redundancy)
        except Exception as e:
            print(f"  SKIP {os.path.basename(p)}: {e}")
            continue
        if not dv:
            continue
        run_mm = dv["exchangeability_missing_mass"]
        m0 = float(np.mean([v["m0_hat"] for v in run_mm.values()])) if run_mm else float("nan")
        rc = dv.get("redundancy_check") or {}
        results["cross_metric"].append({
            "npz": dv["npz"], "domain": dv["npz"].split("_R")[0],
            "n_criteria": dv["n_criteria"], "n_species": dv["n_species"],
            "alpha_count": dv["discovery"]["fit"]["power"].get("alpha"),
            "alpha_value": dv["value"]["fit"]["power"].get("alpha"),
            "value_coverage_exponent": dv["value_coverage_exponent"]["value_coverage_exponent"],
            "regime": dv["value_coverage_exponent"]["regime"],
            "missing_mass_mean": m0,
            "discovery_verdict": dv["discovery"]["fit"]["verdict"],
            "value_verdict": dv["value"]["fit"]["verdict"],
            "redundancy": ({k: rc.get(k) for k in
                            ("H_Mi_bits", "n_selected", "additive_sum_selected", "R_full_joint",
                             "joint_over_H", "redundancy_gap_selected", "y_inf_over_H",
                             "saturating_fit", "V_best")} if rc.get("ran") else None)})

    results["registry"] = registry_curves()

    cm = results["cross_metric"]
    if cm:
        exps = np.array([c["value_coverage_exponent"] for c in cm], float)
        mms = np.array([c["missing_mass_mean"] for c in cm], float)
        print(f"\n  {len(cm)} metrics: value-coverage exponent (alpha_V - alpha_S) "
              f"mean={np.nanmean(exps):+.3f} median={np.nanmedian(exps):+.3f} "
              f"[{np.nanmin(exps):+.3f},{np.nanmax(exps):+.3f}]")
        print(f"  missing mass (Good-Turing, i.i.d. runs) mean={np.nanmean(mms):.3f} "
              f"-> pools are {'far from covered' if np.nanmean(mms) > 0.5 else 'near-covered'}")
        n_dim = sum(1 for c in cm if c["value_coverage_exponent"] < -0.02)
        n_tail = sum(1 for c in cm if c["value_coverage_exponent"] > 0.02)
        print(f"  regimes: {n_dim} diminishing / {len(cm)-n_dim-n_tail} neutral / {n_tail} tail-gems")
        # per-domain redundancy/saturation summary (the corrected value estimand)
        doms = sorted(set(c["domain"] for c in cm))
        for d in doms:
            rows = [c["redundancy"] for c in cm if c["domain"] == d and c.get("redundancy")]
            if not rows:
                continue
            nsel = [r["n_selected"] for r in rows]
            jh = [r["joint_over_H"] for r in rows]
            gp = [100 * r["redundancy_gap_selected"] / max(r["additive_sum_selected"], 1e-9) for r in rows]
            print(f"  [{d}] {len(rows)} metrics: saturates in {min(nsel)}-{max(nsel)} criteria; "
                  f"joint/H {min(jh):.2f}-{max(jh):.2f} (median {np.median(jh):.2f}); "
                  f"redundancy gap {min(gp):.0f}-{max(gp):.0f}%")

    out_json = os.path.join(a.out, "scaling_results.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
