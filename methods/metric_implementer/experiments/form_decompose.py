"""Form-effect decomposition: MAIN effect (per-form strictness shift, calibratable away) vs
item-x-form INTERACTION (genuine entanglement -- the linguistic quantity).

Reads a checkpoint dir and, per metric, decomposes form sensitivity at both seats
(notes/2026-07-01__form-effects-control-plan.md §4):

  TARGET seat  (needs `M_i_forms` saved by rescore_executor --orbit-target):
    raw flip rate of each form vs binarized m_bar, then per-form QUANTILE calibration
    (match each form's YES-rate to m_bar's, i.e. remove the strictness shift), re-flip.
    flip_cal << flip_raw  => instrument artifact;  flip_cal ~ flip_raw  => entanglement.
    Plus boundary concentration: |m_bar - 0.5| among flipped vs unflipped probes.

  INSTRUMENT seat  (needs `pairs` in *_forminv.json, alpha_probe >= 2026-07-01):
    per (criterion, form) records carry drift = mean|sigma - sigma'| and bias = mean(sigma' - sigma);
    |bias|/drift is the share of drift a uniform shift explains. Aggregated per form kind =
    WHICH transformation (question / boilerplate / suffix / reorder) breaks readouts, and how much
    of that is calibratable.

CPU-only, post-hoc. Usage:
    python -m methods.metric_implementer.experiments.form_decompose --dir <ckpt_dir> [--json OUT]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def target_seat(z) -> dict | None:
    """Raw-vs-calibrated flip decomposition of the orbit target. None if M_i_forms absent."""
    if "M_i_forms" not in z.files:
        return None
    mat = np.asarray(z["M_i_forms"], float)                    # (n_forms, n_probes)
    names = [str(x) for x in z["M_i_form_names"]] if "M_i_form_names" in z.files else \
            [f"form{r}" for r in range(mat.shape[0])]
    with np.errstate(invalid="ignore"):
        m_bar = np.nanmean(mat, axis=0)
    b_ref = np.nan_to_num(m_bar, nan=0.5) > 0.5
    yes_ref = float(b_ref.mean())
    per_form, flipped_mask = {}, np.zeros(mat.shape[1], bool)
    for r, nm in enumerate(names):
        row = np.nan_to_num(mat[r], nan=0.5)
        raw = row > 0.5
        thr = np.quantile(row, 1.0 - yes_ref) if 0.0 < yes_ref < 1.0 else 0.5
        cal = row > thr                                        # strictness shift removed
        per_form[nm] = {"flip_raw": float((raw != b_ref).mean()),
                        "flip_cal": float((cal != b_ref).mean()),
                        "bias": float((row - np.nan_to_num(m_bar, nan=0.5)).mean())}
        flipped_mask |= raw != b_ref
    dist = np.abs(np.nan_to_num(m_bar, nan=0.5) - 0.5)
    return {"per_form": per_form,
            "flip_raw": float(np.mean([v["flip_raw"] for v in per_form.values()])),
            "flip_cal": float(np.mean([v["flip_cal"] for v in per_form.values()])),
            "boundary_dist_flipped": float(np.median(dist[flipped_mask])) if flipped_mask.any() else float("nan"),
            "boundary_dist_stable": float(np.median(dist[~flipped_mask])) if (~flipped_mask).any() else float("nan")}


def instrument_seat(fi: dict) -> dict | None:
    """Per-form-kind aggregation of the pair records. None if the forminv predates `pairs`."""
    pairs = fi.get("pairs")
    if not pairs:
        return None
    out = {}
    for kind in sorted({p["form"] for p in pairs}):
        ps = [p for p in pairs if p["form"] == kind]
        drift = np.array([p["drift"] for p in ps]); bias = np.array([abs(p["bias"]) for p in ps])
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(drift > 0, np.minimum(bias / drift, 1.0), np.nan)
        out[kind] = {"n": len(ps),
                     "median_drift": float(np.median(drift)),
                     "median_flip": float(np.median([p["flip"] for p in ps])),
                     "median_abs_bias": float(np.median(bias)),
                     "main_effect_share": float(np.nanmedian(share))}   # |bias|/drift in [0,1]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", required=True, help="checkpoint dir (*_sigs.npz [+ *_forminv.json])")
    p.add_argument("--json", default=None, help="output path (default <dir>/form_decompose.json)")
    a = p.parse_args()

    rows, n_tseat, n_iseat = {}, 0, 0
    for f in sorted(glob.glob(os.path.join(a.dir, "*_sigs.npz"))):
        key = os.path.basename(f).replace("_sigs.npz", "")
        z = np.load(f, allow_pickle=True)
        rec = {"name": str(z["name"]) if "name" in z.files else key}
        ts = target_seat(z)
        if ts: rec["target"] = ts; n_tseat += 1
        fip = f.replace("_sigs.npz", "_forminv.json")
        if os.path.exists(fip):
            iseat = instrument_seat(json.load(open(fip)))
            if iseat: rec["instrument"] = iseat; n_iseat += 1
        rows[key] = rec

    print(f"{len(rows)} ckpts | target-seat data (M_i_forms): {n_tseat} | "
          f"instrument-seat data (forminv pairs): {n_iseat}")
    if n_tseat:
        fr = [r["target"]["flip_raw"] for r in rows.values() if "target" in r]
        fc = [r["target"]["flip_cal"] for r in rows.values() if "target" in r]
        print(f"\nTARGET seat (median over {n_tseat} metrics): "
              f"flip_raw={np.median(fr):.3f} -> flip_cal={np.median(fc):.3f} "
              f"(calibration removes {100*(1 - np.median(fc)/max(np.median(fr),1e-9)):.0f}%)")
    if n_iseat:
        print("\nINSTRUMENT seat, by transformation (medians over metrics):")
        kinds = sorted({k for r in rows.values() for k in r.get("instrument", {})})
        print(f"{'form':<12} {'drift':>7} {'flip':>7} {'|bias|':>7} {'main-effect share':>18}")
        for k in kinds:
            rs = [r["instrument"][k] for r in rows.values() if k in r.get("instrument", {})]
            print(f"{k:<12} {np.median([x['median_drift'] for x in rs]):>7.3f} "
                  f"{np.median([x['median_flip'] for x in rs]):>7.3f} "
                  f"{np.median([x['median_abs_bias'] for x in rs]):>7.3f} "
                  f"{np.median([x['main_effect_share'] for x in rs]):>18.2f}")
    if not (n_tseat or n_iseat):
        print("no decomposable data: ckpts predate M_i_forms / forminv-pairs (2026-07-01 patches); "
              "re-run rescore_executor --orbit-target (retarget ok) or run_alpha_probe to backfill")

    out = a.json or os.path.join(a.dir, "form_decompose.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
