"""Run the EXCHANGEABLE joint-value rarefaction on the real per-metric pools — the rigor-repair pass
for the value-saturation finding. Zero-GPU, CPU logistic OOF only.

ASSUMPTION LEDGER this run addresses (what "no Heaps law for value" rests on, and which legs can be
made design-valid):
  [FIXED by construction] exchangeability of the value curve's x-axis: uniform random m-subsets of
      the frozen pool (sampling w/o replacement) replace the greedy optimizer trajectory. E[V(m)] is
      a well-defined finite-population estimand — same maneuver as classical rarefaction.
  [FIXED] sampling uncertainty: probe bootstrap (resample probes WITH replacement, GROUP-disjoint
      folds so duplicates cannot leak) + fresh subset redraws per replicate -> real percentile CIs on
      y_inf/H, tau, and the front-loading statistic D. Conditional on the mined pool.
  [FIXED] the saturates-faster-than-discovery claim: paired same-subset curves + distribution-free
      D = mean[V/V_max - S/S_max]; bootstrap CI -> error-controlled regime call (replaces the
      heuristic value_coverage_exponent label that was honestly 'untested').
  [PARTIAL] surrogate linearity: joint value = logistic V-information = LOWER bound on Shannon joint
      MI; ceiling bracketed [y_inf, H(M_i)]. Interaction-augmented family check on full pool bounds
      how much pairwise synergy the linear family leaves behind.
  [NOT FIXABLE by rebalancing] extrapolation beyond the mined pool: adaptive mining means the pool
      itself is not an i.i.d. draw from "all minable criteria"; IW de-bias is provably invalid for
      missing mass (ESS collapse + unseen types carry no weight). Requires an i.i.d. audit stream.

  python -m methods.metric_implementer.experiments.run_exchangeable_value \
      --out notebooks/data/unseen_value_scaling_20260717
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from . import unseen_value_scaling as uvs
from .alpha_probe import collide
from .value_census import (_species_bin_signatures, additive_value, exchangeable_joint_value,
                           joint_value_bits)


def _synergy_check(sp_bin, labels, M_i, v_add, *, top: int = 10):
    """Interaction-augmented family on the FULL pool: add pairwise AND features of the top-`top`
    additive species to the logistic family. The gain over the linear family bounds (empirically,
    for pairwise synergy) how much the linear V-information lower bound leaves behind."""
    y = np.asarray(M_i, int)
    lin = joint_value_bits(y, sp_bin)
    top_idx = list(np.argsort(-np.asarray(v_add))[:top])
    prods = [sp_bin[:, i] * sp_bin[:, j] for a, i in enumerate(top_idx) for j in top_idx[a + 1:]]
    X_aug = np.hstack([sp_bin, np.array(prods).T]) if prods else sp_bin
    aug = joint_value_bits(y, X_aug)
    return {"joint_linear": lin, "joint_pairwise_aug": aug, "pairwise_synergy_gain": aug - lin}


def analyze(npz_path, *, n_points=10, n_subsets=30, n_boot=0, boot_subsets=10, seed=0):
    z = np.load(npz_path, allow_pickle=True)
    if "M_i" not in z:
        return None
    sigs = np.asarray(z["sigs"], float)
    tau = float(z["tau"]) if "tau" in z else 0.02
    M_i = (np.asarray(z["M_i"], float) > 0.5).astype(int)
    labels = collide(sigs, tau)
    _, _, sp_bin = _species_bin_signatures(sigs, labels)
    _, _, v_add = additive_value(sigs, labels, M_i)
    name = os.path.basename(npz_path)
    print(f"\n== {name}  ({sigs.shape[0]} criteria, {len(set(labels.tolist()))} species) ==", flush=True)

    point = exchangeable_joint_value(sp_bin, labels, M_i, n_points=n_points,
                                     n_subsets=n_subsets, seed=seed)
    m = np.asarray(point["m"], float)
    V = np.asarray(point["V_mean"])
    S = np.asarray(point["S_mean"])
    H = point["H_Mi_bits"]

    # form verdict ON THE DESIGN-VALID CURVE (the re-test of "no Heaps law for value")
    cmp_v = uvs.compare_scaling_forms(m, V, n_boot=0)
    cmp_s = uvs.compare_scaling_forms(m, S, n_boot=0)
    sat = cmp_v["saturating"]
    fs = uvs.value_frontloading_stat(m, V, S)

    # form-free saturation reads (no exponential assumed)
    def _frac_at(frac):
        i = int(np.searchsorted(m, frac * m[-1]))
        i = min(i, len(m) - 1)
        return float(V[i] / V[-1]) if V[-1] > 1e-9 else float("nan")

    out = {"npz": name, "domain": name.split("_R")[0], "n_criteria": int(sigs.shape[0]),
           "n_species": int(len(set(labels.tolist()))), "H_Mi_bits": H,
           "m": m.tolist(), "V_mean": V.tolist(), "S_mean": S.tolist(),
           "value_verdict": cmp_v["verdict"], "discovery_verdict": cmp_s["verdict"],
           "alpha_S_exchangeable": cmp_s["power"].get("alpha"),
           "sat_fit": ({"y_inf": sat.get("y_inf"), "tau": sat.get("tau"),
                        "y_inf_over_H": (sat.get("y_inf", np.nan) / H if (sat.get("ok") and H > 0)
                                         else float("nan")),
                        "tau_frac_of_pool": (sat.get("tau", np.nan) / m[-1] if sat.get("ok")
                                             else float("nan"))} if sat.get("ok") else None),
           "frontloading_D": (fs.get("D") if fs.get("ok") else None),
           "V_frac_at_10pct_draws": _frac_at(0.10), "V_frac_at_25pct_draws": _frac_at(0.25),
           "estimand_note": point["estimand_note"]}
    sat_s = (f"y_inf/H={out['sat_fit']['y_inf_over_H']:.2f} tau={out['sat_fit']['tau']:.0f} draws"
             if out["sat_fit"] else "sat-fit failed")
    print(f"  [exchangeable] value verdict={cmp_v['verdict']} vs discovery={cmp_s['verdict']} "
          f"(alpha_S={out['alpha_S_exchangeable']:.2f}); {sat_s}; D={out['frontloading_D'] if fs.get('ok') else 'na'}"
          f"; V@10%draws={out['V_frac_at_10pct_draws']:.2f} of full", flush=True)

    if n_boot > 0:
        rng = np.random.default_rng(seed + 1)
        n_probes = len(M_i)
        yh, tauh, Dh, n_skip = [], [], [], 0
        for b in range(n_boot):
            bi = rng.integers(0, n_probes, n_probes)
            if np.unique(M_i[bi]).size < 2:
                n_skip += 1
                continue
            rb = exchangeable_joint_value(sp_bin, labels, M_i, n_points=n_points,
                                          n_subsets=boot_subsets, seed=1000 + b, probe_idx=bi)
            Vb = np.asarray(rb["V_mean"])
            Hb = rb["H_Mi_bits"]
            fb = uvs.fit_saturating(m, Vb, n_boot=0)
            if fb.get("ok") and Hb > 0:
                yh.append(fb["y_inf"] / Hb)
                tauh.append(fb["tau"])
            db = uvs.value_frontloading_stat(m, Vb, np.asarray(rb["S_mean"]))
            if db.get("ok"):
                Dh.append(db["D"])
        ci = lambda v: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                        if len(v) >= 10 else None)
        out["bootstrap"] = {"n_boot": n_boot, "n_skipped_degenerate": n_skip,
                            "ci_kind": "probe_bootstrap_group_folds_plus_subset_redraw",
                            "y_inf_over_H_ci": ci(yh), "tau_ci": ci(tauh), "D_ci": ci(Dh),
                            "D_excludes_zero": (bool(ci(Dh)[0] > 0 or ci(Dh)[1] < 0)
                                                if ci(Dh) else None)}
        syn = _synergy_check(sp_bin, labels, M_i, v_add)
        out["synergy_check"] = syn
        print(f"  [bootstrap n={n_boot}] y_inf/H CI={out['bootstrap']['y_inf_over_H_ci']} "
              f"tau CI={out['bootstrap']['tau_ci']} D CI={out['bootstrap']['D_ci']} "
              f"(excludes 0: {out['bootstrap']['D_excludes_zero']})", flush=True)
        print(f"  [synergy] linear {syn['joint_linear']:.3f} vs +pairwise {syn['joint_pairwise_aug']:.3f} "
              f"bits (gain {syn['pairwise_synergy_gain']:+.3f})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="notebooks/data/prompt_optimality_20260703/aligned_npz/*_sigs.npz")
    ap.add_argument("--extra-globs", nargs="*", default=[
        "notebooks/data/two_faces_20260702/r3_humor/llama8b_glm/*_sigs.npz",
        "notebooks/data/two_faces_20260702/r3_cw/llama8b_glm/*_sigs.npz"])
    ap.add_argument("--boot-metrics", nargs="*", default=[
        "creative-writing_R3_metric40_sigs.npz", "creative-writing_R3_metric23_sigs.npz",
        "humor_R3_metric34_sigs.npz"],
        help="basenames that get the full probe bootstrap + synergy check (slow)")
    ap.add_argument("--n-subsets", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=50)
    ap.add_argument("--out", default="notebooks/data/unseen_value_scaling_20260717")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    paths = sorted(set(glob.glob(a.glob)) | {q for g in a.extra_globs for q in glob.glob(g)})
    print(f"===== exchangeable joint-value run: {len(paths)} npz, boot on {a.boot_metrics} =====",
          flush=True)
    rows = []
    for p in paths:
        nb = a.n_boot if os.path.basename(p) in a.boot_metrics else 0
        try:
            r = analyze(p, n_subsets=a.n_subsets, n_boot=nb)
        except Exception as e:
            print(f"  SKIP {os.path.basename(p)}: {e}", flush=True)
            continue
        if r:
            rows.append(r)

    # cross-metric summary: does the design-valid curve REPLICATE the saturation finding?
    ok = [r for r in rows if r.get("sat_fit")]
    n_sat = sum(1 for r in rows if r["value_verdict"] == "saturating")
    n_disc_pow = sum(1 for r in rows if r["discovery_verdict"] == "power")
    Ds = [r["frontloading_D"] for r in rows if r["frontloading_D"] is not None]
    print(f"\n===== SUMMARY ({len(rows)} metrics) =====")
    print(f"  value verdict saturating: {n_sat}/{len(rows)}; discovery verdict power: "
          f"{n_disc_pow}/{len(rows)}")
    if ok:
        yy = [r["sat_fit"]["y_inf_over_H"] for r in ok]
        print(f"  y_inf/H: median {np.median(yy):.2f} range [{min(yy):.2f},{max(yy):.2f}]")
    if Ds:
        print(f"  front-loading D: median {np.median(Ds):+.3f} range [{min(Ds):+.3f},{max(Ds):+.3f}]")
    out_json = os.path.join(a.out, "exchangeable_value.json")
    with open(out_json, "w") as fh:
        json.dump({"what": ("Design-valid (exchangeable random-subset) joint-value rarefaction; "
                            "repairs the greedy-order curve's missing sampling interpretation. "
                            "All numbers conditional on the mined pool; joint value is a logistic "
                            "V-information LOWER bound on Shannon joint MI."),
                   "rows": rows}, fh, indent=2, default=float)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
