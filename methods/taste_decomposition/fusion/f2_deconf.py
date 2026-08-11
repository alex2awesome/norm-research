#!/usr/bin/env python3
"""F2 DECONFOUNDED-FUSION battery.

Frozen spec: notes/2026-08-09__full_sweep_queue.md §F2, run under the §13 certified
stack (notes/2026-08-05__taste-decomposition-design.md §13.0-§13.4).

Per cell, on the SAME E rows and the SAME frozen Layer-1 stack as the master ledger:

  (a) VA_enr_nl          stack on [bank_enriched]
  (b) NUIS               stack on [nuisance only]              (= "spurious-alone")
  (c) VA_enr+NUIS        stack on [bank_enriched + nuisance]
  (d) VAT_dec_trained    stack on [bank_enriched + nuisance + T]
  (e) VAT_dec_untrained  stack on [bank_enriched + nuisance + T0]

PRIMARY readout = the stacked increment (d)-(c), group bootstrap (2,000 draws) on the
cell's grouping unit.  SECONDARY = (e)-(c) (expected ~0 from the T0 arm; a positive
here flags nuisance-prior interaction).

§13 obligations discharged here:
  * Westfall & Yarkoni reliability band on every POSITIVE increment: the nuisance
    scores are noise-injected to simulated reliability r in {.5,.7,.9}, the nuisance
    model is refit, and (d)-(c) is recomputed grouped-OOF.  The band is quoted, not
    the point.
  * Leg-2 matched-sampling SIGN CHECK on the cell's top nuisance channel, full caliper
    sweep {.01,.02,.05} on the channel's percentile rank; |estimate| < .01 is reported
    as "indeterminate", never as sign evidence.
  * spurious-alone = (b); > .65 is flagged (the regime where LEACE becomes the second
    quantitative leg and matched magnitude is untrustworthy).
  * The Feng et al. 2019 limitation sentence travels in the results JSON.

The stack machinery is NOT reimplemented: direction1_mirror.fit_arm is called four
times on shared folds (L1.outer_folds is a pure function of (n, groups)), so (c)
appears in two calls and is asserted bit-identical, which is what makes the paired
bootstraps legitimate.

CPU only.  Usage: python3 f2_deconf.py --cell jokes_community --box sk3
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
TD = HERE.parent
RESULTS = TD / "results"
ROWS = HERE / "t0_rows"
SCORES = HERE / "t0_scores"


def _mod(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


D1 = _mod(HERE / "direction1_mirror.py", "d1_f2")
fit_arm = D1.fit_arm
gboot = D1.group_paired_boot
F2C = _mod(HERE / "f2_cells.py", "f2_cells_mod")

N_BOOT = 2000
WY_R = (0.5, 0.7, 0.9)
WY_SEEDS = (11, 12)
CALIPERS = (0.01, 0.02, 0.05)

FENG_LIMITATION = (
    "A clean stacked increment rules out only the channels we scored -- not their "
    "interactions, and not channels never named (Feng et al. 2019). Track-B maps are "
    "lower bounds on the channel set (B-side missing mass > A-side, registry 2026-08-07)."
)
WY_CAVEAT = (
    "Westfall & Yarkoni (2016) incremental-validity caveat: the nuisance scores are "
    "single noisy LLM-judged indicators, and unreliability plus large n biases TOWARD "
    "declaring signal beyond the nuisance. Positive increments are quoted as a "
    "reliability band, never as a point."
)


def env_block():
    import numpy as _np
    import scipy as _sp
    return {"platform": f"{platform.system()}-{platform.machine()}",
            "python": sys.version.split()[0], "sklearn": sklearn.__version__,
            "numpy": _np.__version__, "scipy": _sp.__version__}


# --------------------------------------------------------------------- data
def load_E(cell):
    z = np.load(ROWS / f"{cell}.npz", allow_pickle=True)
    meta = json.loads((ROWS / f"{cell}.meta.json").read_text())
    uids = [str(u) for u in z["uids"]]
    ids = [str(i) for i in z["ids"]]
    y = z["y"].astype(int)
    groups = np.array([str(g) for g in z["groups"]], dtype=object)
    dense = z["dense"].astype(float)
    pm = {}
    with gzip.open(SCORES / f"{cell}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            pm[r["uid"]] = float(r["p_yes"])
    t0 = np.array([pm[u] for u in uids], dtype=float)
    return meta, ids, y, groups, dense, t0


def align(cell, adapter_out, ids_E, y_E, groups_E):
    """Subset the adapter's population matrices to E, in the master ledger's order."""
    a = adapter_out
    pos = {str(i): k for k, i in enumerate(a["ids"])}
    assert len(pos) == len(a["ids"]), f"{cell}: duplicate ids in the closure population"
    missing = [i for i in ids_E if i not in pos]
    assert not missing, f"{cell}: {len(missing)} E ids absent from the closure population"
    idx = np.array([pos[i] for i in ids_E])
    assertions = {
        "n_E": int(len(idx)),
        "y_equal_elementwise": bool(np.array_equal(np.asarray(a["y"])[idx], y_E)),
        "groups_equal_elementwise": bool(np.array_equal(
            np.array([str(g) for g in np.asarray(a["groups"], dtype=object)[idx]]),
            np.asarray(groups_E, dtype=object).astype(str))),
        "E_mask_agrees_with_adapter": bool(int(np.asarray(a["E"])[idx].sum()) == len(idx)),
    }
    assert assertions["y_equal_elementwise"], f"{cell}: y mismatch on the E join"
    assert assertions["E_mask_agrees_with_adapter"], f"{cell}: E rows outside adapter E"
    return a["bank"][idx], a["nuis"][idx], assertions


# --------------------------------------------------------------------- arms
def alone_auc(y, col):
    v = np.asarray(col, dtype=float)
    if not np.isfinite(v).any():
        return 0.5
    v = np.where(np.isfinite(v), v, np.nanmedian(v))
    if np.nanstd(v) == 0:
        return 0.5
    return float(roc_auc_score(y, v))


def noise_inject(M, r, rng):
    """Attenuate to simulated reliability r: add noise with Var = Var(x)*(1-r)/r."""
    out = M.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        fin = np.isfinite(col)
        if fin.sum() < 5:
            continue
        s = np.std(col[fin])
        if s == 0:
            continue
        col[fin] = col[fin] + rng.normal(0.0, s * np.sqrt((1 - r) / r), fin.sum())
        out[:, j] = col
    return out


def matched_sign_check(y, groups, chan, oof_d, oof_c, dense, calipers=CALIPERS, seed=7):
    """Leg 2: greedy caliper matching of positives to negatives on the top nuisance
    channel's PERCENTILE RANK, then the primary increment re-read inside the matched
    subsample.  Sign/consistency only (§13.1 Leg 2)."""
    v = np.asarray(chan, dtype=float)
    v = np.where(np.isfinite(v), v, np.nanmedian(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0)
    order = np.argsort(v, kind="mergesort")
    pct = np.empty(len(v))
    pct[order] = np.linspace(0, 1, len(v))
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    out = {}
    for cal in calipers:
        avail = sorted(neg, key=lambda i: pct[i])
        av_p = np.array([pct[i] for i in avail])
        used = np.zeros(len(avail), dtype=bool)
        keep_p, keep_n = [], []
        for i in rng.permutation(pos):
            lo, hi = np.searchsorted(av_p, pct[i] - cal), np.searchsorted(av_p, pct[i] + cal)
            cand = [k for k in range(lo, hi) if not used[k]]
            if not cand:
                continue
            k = cand[int(rng.integers(len(cand)))]
            used[k] = True
            keep_p.append(i)
            keep_n.append(avail[k])
        idx = np.array(keep_p + keep_n)
        blk = {"caliper": cal, "n_pairs": len(keep_p), "n_matched": int(len(idx))}
        if len(keep_p) >= 30:
            ym = y[idx]
            blk["increment_d_minus_c"] = float(roc_auc_score(ym, oof_d[idx])
                                               - roc_auc_score(ym, oof_c[idx]))
            blk["T_auc_matched"] = float(roc_auc_score(ym, dense[idx]))
            blk["T_auc_all"] = float(roc_auc_score(y, dense))
            blk["verdict"] = ("indeterminate (|est| < .01, inside the demonstrated "
                              "protocol-sensitivity band)"
                              if abs(blk["increment_d_minus_c"]) < 0.01 else
                              ("same sign as Leg 1" if blk["increment_d_minus_c"] > 0
                               else "OPPOSITE sign to Leg 1"))
        else:
            blk["verdict"] = "too few matched pairs (<30) -- not read"
        out[f"caliper_{cal}"] = blk
    return out


# --------------------------------------------------------------------- main
def run_cell(cell, box, n_boot=N_BOOT):
    t0 = time.time()
    meta, ids_E, y, groups, dense, t0col = load_E(cell)
    family = meta["family"]
    a = F2C.ADAPTERS[cell]()
    bank, nuis, join = align(cell, a, ids_E, y, groups)
    bn = np.column_stack([bank, nuis])

    print(f"  [{cell}] n_E={len(y)} groups={len(set(groups))} bank={bank.shape} "
          f"nuis={nuis.shape} family={family}", flush=True)

    r_bank = fit_arm(family, bank, dense, y, groups)      # (a)=VA ; bonus bank+T
    r_nuis = fit_arm(family, nuis, dense, y, groups)      # (b)=VA
    r_bn_T = fit_arm(family, bn, dense, y, groups)        # (c)=VA ; (d)=VAT
    r_bn_0 = fit_arm(family, bn, t0col, y, groups)        # (c) again ; (e)=VAT
    assert np.allclose(r_bn_T["_oof_VA_nl0"], r_bn_0["_oof_VA_nl0"]), \
        f"{cell}: arm (c) differs between the two calls -- folds not shared"
    assert abs(r_bn_T["VA_nl_mean"] - r_bn_0["VA_nl_mean"]) < 1e-12

    oof_c, oof_d, oof_e = r_bn_T["_oof_VA_nl0"], r_bn_T["_oof_VAT_nl0"], r_bn_0["_oof_VAT_nl0"]
    A_nl, B_nl, C_nl = r_bank["VA_nl_mean"], r_nuis["VA_nl_mean"], r_bn_T["VA_nl_mean"]
    D_nl, E_nl = r_bn_T["VAT_nl_mean"], r_bn_0["VAT_nl_mean"]

    prim = gboot(y, oof_d, oof_c, groups, n_boot=n_boot)
    sec = gboot(y, oof_e, oof_c, groups, n_boot=n_boot)

    # ---- Westfall-Yarkoni reliability band on the primary --------------------
    # §13.1 Leg 1: the band is REQUIRED for positive increments (unreliability biases
    # toward declaring signal beyond the nuisance).  NULL/NEGATIVE increments are
    # robust to that critique and "may be quoted directly", so the band is skipped
    # there and the skip is recorded.
    wy = {}
    if prim["estimate"] <= 0:
        wy["skipped"] = ("increment is null/negative; §13.1 Leg 1 states such increments "
                         "are robust to the Westfall-Yarkoni critique (the asymmetry cuts "
                         "the other way) and may be quoted directly")
    for r in (WY_R if prim["estimate"] > 0 else ()):
        pts = []
        for s in WY_SEEDS:
            rng = np.random.default_rng(s)
            bn_r = np.column_stack([bank, noise_inject(nuis, r, rng)])
            rr = fit_arm(family, bn_r, dense, y, groups)
            pts.append(rr["VAT_nl_mean"] - rr["VA_nl_mean"])
        wy[f"r_{r}"] = {"increment_mean": float(np.mean(pts)),
                        "increment_per_seed": [float(x) for x in pts],
                        "n_noise_seeds": len(WY_SEEDS)}
    if prim["estimate"] > 0:
        band = [wy[f"r_{r}"]["increment_mean"] for r in WY_R] + [prim["estimate"]]
        wy["band"] = {"lo": float(min(band)), "hi": float(max(band)),
                      "note": "band spans simulated nuisance reliability r in {.5,.7,.9} "
                              "plus the observed-reliability point estimate (r=1 implied)"}
    else:
        wy["band"] = {"lo": prim["estimate"], "hi": prim["estimate"],
                      "note": "not applicable (null/negative increment); point quoted"}

    # ---- top nuisance channel + Leg 2 ---------------------------------------
    alone = [(a["nuis_names"][j], alone_auc(y, nuis[:, j])) for j in range(nuis.shape[1])]
    ranked = sorted(alone, key=lambda x: -abs(x[1] - 0.5))
    jtop = a["nuis_names"].index(ranked[0][0])
    leg2 = matched_sign_check(y, groups, nuis[:, jtop], oof_d, oof_c, dense)

    fused = {"d_VAT_dec_trained": D_nl, "e_VAT_dec_untrained": E_nl}
    best_fused = max(fused.values())
    out = {
        "cell": cell, "box": box, "env": env_block(),
        "spec": "notes/2026-08-09__full_sweep_queue.md §F2 (frozen before any F2 run)",
        "stack": ("direction1_mirror.fit_arm (frozen Layer-1: GroupKFold(5) on the "
                  "cell's grouping unit, HistGB seeds {0,1,2} mean, nested grid per "
                  "fold), called 4x on shared folds"),
        "family": family, "group_column": meta["group_column"],
        "n_E": int(len(y)), "n_groups_E": int(len(set(groups))),
        "pos_rate_E": float(y.mean()),
        "join_assertions": join,
        "ids_sha256": meta["ids_sha256"],
        "shapes": {"bank_enriched": list(bank.shape), "nuisance": list(nuis.shape),
                   "nuisance_gemma_B": a["n_nuis_gemma"], "nuisance_struct": a["n_struct"]},
        "bank_blocks": a["bank_names"], "nuisance_names": a["nuis_names"],
        "collapse_gate_dropped": a["collapse_gate_dropped"],
        "provenance": a["provenance"],

        "arms": {
            "a_VA_enr_nl": A_nl, "a_VA_enr_lin": r_bank["VA_lin"],
            "b_NUIS_nl": B_nl, "b_NUIS_lin": r_nuis["VA_lin"],
            "c_VA_enr_plus_NUIS_nl": C_nl, "c_VA_enr_plus_NUIS_lin": r_bn_T["VA_lin"],
            "d_VAT_dec_trained_nl": D_nl, "d_VAT_dec_trained_lin": r_bn_T["VAT_lin"],
            "e_VAT_dec_untrained_nl": E_nl, "e_VAT_dec_untrained_lin": r_bn_0["VAT_lin"],
            "bonus_bank_plus_T_nl": r_bank["VAT_nl_mean"],
            "T": float(roc_auc_score(y, dense)), "T0": float(roc_auc_score(y, t0col)),
            "seed_spread": {"a": r_bank["VA_nl_spread"], "c": r_bn_T["VA_nl_spread"],
                            "d": r_bn_T["VAT_nl_spread"], "e": r_bn_0["VAT_nl_spread"]},
        },

        "PRIMARY_stacked_increment_d_minus_c": prim,
        "SECONDARY_untrained_increment_e_minus_c": sec,
        "westfall_yarkoni_reliability_band": wy,
        "spurious_alone_b": B_nl,
        "spurious_alone_gt_065": bool(B_nl > 0.65),
        "top_nuisance_channels": [{"name": n, "alone_auc": v} for n, v in ranked[:8]],
        "leg2_matched_sign_check": {"channel": ranked[0][0],
                                    "channel_alone_auc": ranked[0][1],
                                    "matching_scale": "percentile rank of the channel on E",
                                    "calipers": leg2},

        "fused_must_beat_bank": {
            "rule": "design note §11: max(fused) must beat the bank, else AUTO-FABLE-AUDIT",
            "bank_enriched_a": A_nl, "best_fused": best_fused,
            "margin": best_fused - A_nl,
            "verdict": "PASS" if best_fused > A_nl else "AUTO-FABLE-AUDIT",
        },
        "caveats": {"feng_2019_limitation": FENG_LIMITATION,
                    "westfall_yarkoni": WY_CAVEAT,
                    "never_quote_against_old_delta_beyond":
                        "NEVER quote (d)-(c) against the old Delta_beyond without naming "
                        "both designs: this arm conditions on the ENRICHED bank plus named "
                        "nuisance and refits on E, the old one did neither."},
        "runtime_sec": None,
    }
    out["runtime_sec"] = time.time() - t0
    p = RESULTS / f"f2_deconf_{cell}.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"  [{cell}] (a) bank_enr {A_nl:.4f} | (b) NUIS {B_nl:.4f} | (c) enr+nuis {C_nl:.4f} "
          f"| (d) +T {D_nl:.4f} | (e) +T0 {E_nl:.4f}", flush=True)
    print(f"  [{cell}] PRIMARY (d)-(c) = {prim['estimate']:+.4f} "
          f"[{prim['ci95'][0]:+.4f},{prim['ci95'][1]:+.4f}] P={prim['p_gt_0']:.3f} "
          f"| WY band [{wy['band']['lo']:+.4f},{wy['band']['hi']:+.4f}]", flush=True)
    print(f"  [{cell}] SECONDARY (e)-(c) = {sec['estimate']:+.4f} P={sec['p_gt_0']:.3f} "
          f"| fused-vs-bank {out['fused_must_beat_bank']['verdict']} "
          f"({out['fused_must_beat_bank']['margin']:+.4f}) | {out['runtime_sec']:.0f}s", flush=True)
    print(f"  wrote {p}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--box", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    for c in args.cell:
        print(f"=== F2 {c} ===", flush=True)
        run_cell(c, args.box, args.n_boot)


if __name__ == "__main__":
    main()
