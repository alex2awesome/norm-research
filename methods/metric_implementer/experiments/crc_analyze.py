"""Capture-recapture analysis: per-metric upper/lower bounds + validity gates on a frozen signature
checkpoint dir, and cross-executor scaling collation. Pure CPU post-hoc.

For each ``<task>_<level>_metric{gi}_sigs.npz`` it runs ``alpha_probe.conditional_crc_report``:
conditional (non-overlapping) species -> B_E (two-list Lincoln-Petersen + bootstrap = UPPER bound on
the reachable space), D_obs (LOWER bound = observed species), coverage, within-sample Chao1, the
Ω-order permutation gate (``order_stability``), and the two-method (Chao1 ≈ Lincoln-Petersen)
saturation check.

Examples (sk3, CPU):
    python -m methods.metric_implementer.experiments.crc_analyze \
        --dir /lfs/.../outputs/crc_scaling/llama8b --cmi-thresh 0.15
    # scaling table across executors (name:path,name:path,...):
    python -m methods.metric_implementer.experiments.crc_analyze \
        --scaling llama8b:/lfs/.../llama8b,llama70b:/lfs/.../llama70b,qwen122b:/lfs/.../qwen122b
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from . import alpha_probe as ap


def _probe_chao_stats(S, tags, cmi_thresh=0.15, n=5, frac=0.8, seed=0):
    """Probe-SUBSAMPLE variance component: resample probe columns n times (each 80% of probes),
    recompute Chao1; returns std + [min, max]. The third B_E variance source alongside the
    criteria-split bootstrap (``B_E_upper_std``) and the Omega-order permutation (``order_stability``).
    B_E is reported as a point with all three plus the ADVERSE interval (§12.6.2: partition statistics
    are quoted at the adverse end of the perturbation orbit), never a bare scalar."""
    rng = np.random.default_rng(seed)
    m = S.shape[1]
    chs = []
    for _ in range(n):
        idx = sorted(rng.choice(m, max(2, int(frac * m)), replace=False))
        lab = ap.conditional_species(S[:, idx], cmi_thresh=cmi_thresh)
        v = lab >= 0
        sp = ap.spectrum(lab[v], [tags[i] for i in range(len(lab)) if v[i]])
        chs.append(ap.chao1(sp["f"], sp["D"]))
    if len(chs) < 2:
        return {"std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"std": float(np.std(chs)), "min": float(np.min(chs)), "max": float(np.max(chs))}


def _analyze_dir(d, cmi_thresh, n_orders, n_splits, seed):
    files = sorted(glob.glob(os.path.join(d, "*_metric*_sigs.npz")))
    rows = []
    for f in files:
        try:
            z = np.load(f, allow_pickle=True)
            sigs = np.asarray(z["sigs"], float); tags = list(z["tags"])
            name = str(z["name"]) if "name" in z.files else os.path.basename(f)
        except Exception as e:
            print(f"  skip {os.path.basename(f)}: {e}"); continue
        r = ap.conditional_crc_report(sigs, tags, cmi_thresh=cmi_thresh, n_orders=n_orders,
                                      n_splits=n_splits, seed=seed)
        diag = ap.signature_diagnostics(sigs)                       # inter-sig L1 to contextualize form drift
        fi_path = f.replace("_sigs.npz", "_forminv.json")
        fi = json.load(open(fi_path)) if os.path.exists(fi_path) else {}
        drift = fi.get("median_drift", float("nan"))
        pcs = _probe_chao_stats(np.asarray(z["sigs"], float), tags, cmi_thresh)
        ost = r["order_stability"]
        # ADVERSE interval (§12.6.2): [min, max] over the sampled perturbation orbit (Ω-order perms ∪
        # probe subsamples). Certificates must hold at the adverse end; the point is for intuition.
        adverse = {"chao1_adverse_lo": float(np.nanmin([ost["chao1_min"], pcs["min"]])),
                   "chao1_adverse_hi": float(np.nanmax([ost["chao1_max"], pcs["max"]])),
                   "D_adverse_lo": ost["D_min"], "D_adverse_hi": ost["D_max"]}
        rows.append({"file": os.path.basename(f),
                     "gi": int(re.search(r"metric(\d+)_sigs", f).group(1)) if re.search(r"metric(\d+)_sigs", f) else -1,
                     "name": name, "mean_between_sig_l1": diag["mean_between_sig_l1"],
                     "form_drift": drift,
                     "drift_over_inter_sig": float(drift / diag["mean_between_sig_l1"])
                       if diag["mean_between_sig_l1"] > 1e-9 and drift == drift else float("nan"),
                     "form_binary_flip": fi.get("binary_flip_rate", float("nan")),
                     "probe_chao_std": pcs["std"], **adverse, **r})
    return rows


def _agg(rows):
    if not rows:
        return None
    be = np.array([r["B_E_upper"] for r in rows if np.isfinite(r["B_E_upper"])])
    cv = np.array([r["coverage"] for r in rows if np.isfinite(r["coverage"])])
    ch = np.array([r["chao1_within"] for r in rows if np.isfinite(r["chao1_within"])])
    bes = np.array([r["B_E_upper_std"] for r in rows if np.isfinite(r["B_E_upper_std"])])
    fd = np.array([r.get("form_drift", np.nan) for r in rows if np.isfinite(r.get("form_drift", np.nan))])
    doi = np.array([r.get("drift_over_inter_sig", np.nan) for r in rows
                    if np.isfinite(r.get("drift_over_inter_sig", np.nan))])
    bf = np.array([r.get("form_binary_flip", np.nan) for r in rows
                   if np.isfinite(r.get("form_binary_flip", np.nan))])
    isl = np.array([r.get("mean_between_sig_l1", np.nan) for r in rows
                    if np.isfinite(r.get("mean_between_sig_l1", np.nan))])
    agree = sum(1 for r in rows if r.get("two_method_agree"))
    return {"n": len(rows),
            "B_E_median": float(np.median(be)) if be.size else float("nan"),
            "B_E_mean": float(np.mean(be)) if be.size else float("nan"),
            "B_E_min": float(np.min(be)) if be.size else float("nan"),
            "B_E_max": float(np.max(be)) if be.size else float("nan"),
            "B_E_within_std_median": float(np.median(bes)) if bes.size else float("nan"),
            "chao1_median": float(np.median(ch)) if ch.size else float("nan"),
            "coverage_median": float(np.median(cv)) if cv.size else float("nan"),
            "coverage_frac_gt08": float(np.mean(cv > 0.8)) if cv.size else float("nan"),
            "two_method_agree_frac": float(agree / max(len(rows), 1)),
            "inter_sig_l1_median": float(np.median(isl)) if isl.size else float("nan"),
            "form_drift_median": float(np.median(fd)) if fd.size else float("nan"),
            "drift_over_inter_sig_median": float(np.median(doi)) if doi.size else float("nan"),
            "form_binary_flip_median": float(np.median(bf)) if bf.size else float("nan")}


def main(argv=None):
    ap_ = argparse.ArgumentParser(prog="crc_analyze", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--dir", default=None, help="single executor checkpoint dir (per-metric detail)")
    ap_.add_argument("--scaling", default=None,
                     help="name:path,name:path,... — collate B_E/coverage vs executor (scaling law)")
    ap_.add_argument("--cmi-thresh", type=float, default=0.15)
    ap_.add_argument("--n-orders", type=int, default=8, help="Ω-order permutations (validity gate)")
    ap_.add_argument("--n-splits", type=int, default=15, help="A/B splits for B_E bootstrap")
    ap_.add_argument("--seed", type=int, default=0)
    ap_.add_argument("--json", default=None, help="write per-metric rows to this JSON")
    a = ap_.parse_args(argv)

    if a.scaling:
        specs = [s.split(":", 1) for s in a.scaling.split(",") if ":" in s]
        print("=== capture-recapture scaling: B_E (upper) / D_obs (lower) / coverage / form-robustness vs executor ===")
        print("%-12s %3s %7s %7s %7s %6s %6s %6s %8s %7s" % (
            "executor", "n", "B_E_med", "B_Emin", "B_Emax", "cov", ">0.8", "agree", "drft/isig", "flip%"))
        print("-" * 86)
        all_rows = {}
        for name, d in specs:
            rows = _analyze_dir(d, a.cmi_thresh, a.n_orders, a.n_splits, a.seed)
            all_rows[name] = rows
            g = _agg(rows)
            if g:
                print("%-12s %3d %7.1f %7.1f %7.1f %6.3f %6.2f %6.2f %8.2f %7.1f" % (
                    name[:12], g["n"], g["B_E_median"], g["B_E_min"], g["B_E_max"],
                    g["coverage_median"], g["coverage_frac_gt08"], g["two_method_agree_frac"],
                    g["drift_over_inter_sig_median"], 100 * g["form_binary_flip_median"]))
        if a.json:
            json.dump({k: [_no_np(v) for v in vs] for k, vs in all_rows.items()}, open(a.json, "w"),
                      indent=2)
        return

    if not a.dir:
        ap_.error("need --dir or --scaling")
    rows = _analyze_dir(a.dir, a.cmi_thresh, a.n_orders, a.n_splits, a.seed)
    g = _agg(rows)
    print("=== %s : %d metrics (cmi_thresh=%.2f, %d orders, %d splits) ===" % (
        a.dir, len(rows), a.cmi_thresh, a.n_orders, a.n_splits))
    if g:
        print("B_E upper: median=%.1f mean=%.1f [%.0f,%.0f] (within-metric std median=%.1f) | "
              "Chao1 med=%.1f | coverage med=%.3f (>0.8: %.0f%%) | two-method agree %.0f%%" % (
              g["B_E_median"], g["B_E_mean"], g["B_E_min"], g["B_E_max"],
              g["B_E_within_std_median"], g["chao1_median"], g["coverage_median"],
              100 * g["coverage_frac_gt08"], 100 * g["two_method_agree_frac"]))
    print("\n%-30s %5s %7s %5s %5s %5s %6s %5s %5s %5s" % (
        "metric", "iid", "B_E", "boot", "probe", "OmOrd", "Dobs", "cov", "f1/N", "agree"))
    for r in sorted(rows, key=lambda x: x["B_E_upper"]):
        print("%-30s %5d %7.1f %5.1f %5.1f %5.1f %6.0f %5.3f %5.2f %5s" % (
            r["name"][:30], r["iid_n"], r["B_E_upper"], r["B_E_upper_std"],
            r.get("probe_chao_std", float("nan")), r["order_stability"]["D_std"],
            r["D_obs_lower"], r["coverage"], r.get("f1_over_N", float("nan")),
            "Y" if r.get("two_method_agree") else "n"))
    if a.json:
        json.dump([_no_np(r) for r in rows], open(a.json, "w"), indent=2)


def _no_np(o):
    if isinstance(o, dict):
        return {k: _no_np(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_no_np(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _no_np(o.tolist())
    return o


if __name__ == "__main__":
    main()
