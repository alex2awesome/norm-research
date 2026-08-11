#!/usr/bin/env python3
"""F2 addendum: the E-VALUE ANALOG column.

FROZEN DEFINITION: notes/2026-08-11__f2_deconfounded_fusion.md, section
"E-VALUE ANALOG -- FROZEN DEFINITION (written before any cell's value was computed)",
committed as 89a154dd0 / sha256 3a2be4bf... BEFORE any value below was computed.
This file implements that definition and nothing else; it never edits an existing arm.

Emits, per cell, a schema-versioned `evalue_analog` block ("evalue_analog/v1") into
results/f2_deconf_<cell>.json:

    X   inf{ alone-AUC(U) : Delta(U) <= 0 } over the frozen adversarial family
        (U = rank(T) blended with Uniform noise, seed 2026, calibrated to alone-AUC s)
        -- a LOWER BOUND on what a real unfound channel would need
    Y   max alone-AUC over the channels the sealed fleets actually found
    RR  (X - .5) / (Y - .5)      robustness ratio on excess-over-chance
    Z   expected max of S_obs*Mhat/(1-Mhat) further draws from an exponential upper
        tail fitted to the found channels' odds -- the strongest single channel the
        strict Track-B missing mass can still be hiding
    verdict  ROBUST (X > Z) / ABSORBABLE-IN-PRINCIPLE (X <= Z) / n/a (Delta <= 0)

CPU only.  Usage: python3 f2_evalue.py --cell jokes_community [--cell ...]
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
TD = HERE.parent
CLOSURE = TD / "closure"
RESULTS = TD / "results"
ROWS = HERE / "t0_rows"
SCORES = HERE / "t0_scores"

SCHEMA = "evalue_analog/v1"
FREEZE = ("notes/2026-08-11__f2_deconfounded_fusion.md, section 'E-VALUE ANALOG -- "
          "FROZEN DEFINITION', commit 89a154dd0 (sha256 3a2be4bf...), written and "
          "committed BEFORE any value in this block was computed")
U_SEED = 2026
S_GRID = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
N_BISECT = 2
AUC_TOL = 0.002
MONO_TOL = 0.01
SWEEP_SEEDS = (0,)

LIMITATIONS = [
    "X is a LOWER BOUND: it is achieved only by a channel maximally aligned with what "
    "T contributes beyond (c). A real channel of the same alone-AUC but arbitrary "
    "orientation absorbs strictly less, so the true requirement is higher than X.",
    "Z inherits every assumption of Good-Turing on a sealed-fleet species count, plus "
    "the assumption that unfound channels are exchangeable in strength with found ones "
    "(discovery is strength-biased, so that assumption is generous to the skeptic). "
    "Z is an order-of-magnitude guide, not a confidence bound.",
    "Feng et al. 2019: this bounds a SINGLE unfound channel. It does not bound "
    "interactions among unfound channels, nor a coordinated set of several, nor "
    "channels outside the proposable space entirely.",
]


def _mod(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


D1 = _mod(HERE / "direction1_mirror.py", "d1_ev")
F2C = _mod(HERE / "f2_cells.py", "f2_cells_ev")
fit_arm = D1.fit_arm


# ------------------------------------------------------- adversarial channel
def build_U(T, y, s, seed=U_SEED, tol=AUC_TOL):
    """U(w) = (1-w)*rank(T)/n + w*Uniform, w bisected so alone-AUC(U) == s."""
    n = len(T)
    order = np.argsort(np.asarray(T, dtype=float), kind="mergesort")
    u = np.empty(n)
    u[order] = np.linspace(0.0, 1.0, n)
    z = np.random.default_rng(seed).random(n)
    auc_u = roc_auc_score(y, u)
    if auc_u < 0.5:                       # orient toward y so blending is monotone
        u = 1.0 - u
        auc_u = 1.0 - auc_u
    lo, hi = 0.0, 1.0                     # w=0 -> AUC(u); w=1 -> ~.5
    for _ in range(40):
        w = 0.5 * (lo + hi)
        a = roc_auc_score(y, (1 - w) * u + w * z)
        if abs(a - s) <= tol:
            return (1 - w) * u + w * z, float(a), float(w)
        if a > s:
            lo = w
        else:
            hi = w
    w = 0.5 * (lo + hi)
    U = (1 - w) * u + w * z
    return U, float(roc_auc_score(y, U)), float(w)


def delta_with(family, bank, nuis, U, T, y, groups):
    M = np.column_stack([bank, nuis, U.reshape(-1, 1)])
    r = fit_arm(family, M, T, y, groups, gbm_seeds=SWEEP_SEEDS)
    return float(r["VAT_nl_mean"] - r["VA_nl_mean"])


# ------------------------------------------------------------------ Z bound
MASS_DIR = {
    "jokes_community": "jokes_community", "press_verdict": "press_verdict",
    "peer_curation": "peer_curation_ext", "peer_revealed": "peer_revealed",
    "hashtagwars_verdict": "maps_hw_si", "mathse_accepted_verdict": "mathse_accepted",
    "mathse_vote_score": "mathse_vote", "cap_finalist": "cap_finalist",
    "nc_responded": "nc_responded", "cw_community": "cw_community",
    "peer_verdict": ".",
}


def _mass_from(obj):
    """Pull (M_hat, S_obs) out of any of the campaign's missing-mass shapes."""
    if not isinstance(obj, dict):
        return None
    tr = obj.get("tracks", {})
    for key in ("b", "B"):
        t = tr.get(key)
        if isinstance(t, dict) and "M_hat" in t:
            return float(t["M_hat"]), int(t.get("S_obs", 0)), f"tracks.{key}.M_hat"
        if isinstance(t, dict) and "good_turing_missing_mass" in t:
            return (float(t["good_turing_missing_mass"]), int(t.get("S_obs", 0)),
                    f"tracks.{key}.good_turing_missing_mass")
        if isinstance(t, dict):                      # nested per-round (cw shape)
            rk = sorted([k for k in t if k.startswith("round")],
                        key=lambda k: int("".join(c for c in k if c.isdigit()) or 0))
            for k in reversed(rk):
                v = t[k]
                if isinstance(v, dict) and "good_turing_missing_mass" in v:
                    return (float(v["good_turing_missing_mass"]), int(v.get("S_obs", 0)),
                            f"tracks.{key}.{k}.good_turing_missing_mass")
                if isinstance(v, dict) and "M_hat" in v:
                    return float(v["M_hat"]), int(v.get("S_obs", 0)), f"tracks.{key}.{k}.M_hat"
    return None


def _species_mass(path):
    """Track-B Good-Turing from a species.json: STRICT (post-b_merge) when the file
    declares b_merge.strict, else the tau-only PREMERGE figure, flagged."""
    try:
        d = json.loads(Path(path).read_text())
    except Exception:
        return None
    B = (d.get("tracks") or {}).get("B")
    if not isinstance(B, dict):
        return None
    strict = bool((d.get("b_merge") or {}).get("strict"))
    gt = B.get("good_turing") if strict else None
    field = "tracks.B.good_turing"
    if not isinstance(gt, dict):
        gt = B.get("good_turing") or B.get("good_turing_PREMERGE_tau_only")
        field = ("tracks.B.good_turing" if B.get("good_turing")
                 else "tracks.B.good_turing_PREMERGE_tau_only")
    if not isinstance(gt, dict):
        return None
    m = gt.get("good_turing_missing_mass")
    if m is None:
        return None
    return {"M_hat": float(m), "S_obs_reported": int(gt.get("S_obs", 0)),
            "field": field, "strict_marker_present": strict,
            "n_proposals": gt.get("N_proposals"), "f1": gt.get("f1"), "f2": gt.get("f2")}


def find_mass(cell):
    """Prefer the campaign's own species.json Track-B Good-Turing (strict where the
    b_merge certificate exists); fall back to a *_missing_mass.json."""
    d_ = CLOSURE / MASS_DIR.get(cell, cell)
    if not d_.exists():
        return {"available": False, "note": f"no closure dir {d_}"}

    sp = [q for q in d_.glob("*species*.json") if "PREMERGE" not in q.name]
    def _rnum(q):
        s = "".join(c for c in q.stem.split("species")[0] if c.isdigit())
        return int(s or 0)
    best = None
    for q in sorted(sp, key=_rnum):
        got = _species_mass(q)
        if got:
            got["source"] = str(q.relative_to(TD.parents[1]))
            got["round"] = _rnum(q)
            best = got                      # keep the highest-numbered round that parses
    if best:
        best.update({"available": True, "TAU_ERA_MASS": not best["strict_marker_present"],
                     "resolver": "species.json Track-B Good-Turing"})
        return best

    cands = [q for q in sorted(d_.glob("*missing_mass*.json")) if "P6" not in q.name]
    picked, best_r = None, -1
    for q in cands:
        try:
            obj = json.loads(q.read_text())
        except Exception:
            continue
        got = _mass_from(obj)
        if not got:
            continue
        r = int("".join(c for c in q.stem if c.isdigit()) or obj.get("round", 0) or 0)
        if r >= best_r:
            best_r, picked = r, (got, q, r)
    if picked is None:
        return {"available": False,
                "searched": [str(q) for q in list(sp) + cands] or [str(d_)],
                "note": "no Track-B Good-Turing missing mass on disk for this cell"}
    (m, s_obs, field), path, rnd = picked
    txt = path.read_text().lower()
    strict = ("strict" in txt) or ("b_merge" in txt)
    return {"available": True, "M_hat": m, "S_obs_reported": s_obs,
            "source": str(path.relative_to(TD.parents[1])), "field": field,
            "round": rnd, "strict_marker_present": bool(strict),
            "TAU_ERA_MASS": (not strict), "resolver": "*_missing_mass.json"}


def z_bound(found_aucs, M_hat, S_obs):
    """Expected max of S_unf further draws from an exponential upper tail fitted to
    the found channels' log-odds (frozen definition)."""
    a = np.array([max(x, 1 - x) for x in found_aucs], dtype=float)
    a = np.clip(a, 0.5 + 1e-6, 1 - 1e-6)
    o = np.log(a / (1 - a))                      # log-odds, all >= 0
    if S_obs <= 0:
        S_obs = len(a)
    S_unf = S_obs * M_hat / max(1 - M_hat, 1e-6)
    if S_unf < 1:
        S_unf = 1.0
    med = float(np.median(o))
    tail = o[o >= med] - med
    lam = 1.0 / max(float(np.mean(tail)), 1e-9)  # exponential rate above the median
    q = med + (-np.log(1.0 / S_unf) / lam)       # (1 - 1/S_unf) quantile
    z_auc = float(1.0 / (1.0 + np.exp(-q)))
    return {"Z": z_auc, "S_unf_effective": float(S_unf), "S_obs_used": int(S_obs),
            "tail_rate_lambda": float(lam), "median_log_odds": med,
            "n_found_channels": int(len(a))}


# ------------------------------------------------------------------- driver
def run(cell, dry=False, matched=False):
    p = RESULTS / f"f2_deconf_{cell}.json"
    res = json.loads(p.read_text())
    if matched:
        ms = res.get("matched_strength_companion") or {}
        if not ms.get("applicable"):
            print(f"  [{cell}] no applicable matched-strength companion -- skipping", flush=True)
            return None
        delta3 = ms["COMPANION_increment_dstar_minus_cstar"]["estimate"]
    else:
        delta3 = res["PRIMARY_stacked_increment_d_minus_c"]["estimate"]
    Y = res["top_nuisance_channels"][0]["alone_auc"]
    Y = max(Y, 1 - Y)
    found = [c["alone_auc"] for c in res["top_nuisance_channels"]]

    blk = {"schema": (SCHEMA if not matched else "evalue_analog_matched/v1"),
           "conditioning_block": ("[bank_enriched + nuisance] (E-refit primary)" if not matched
                                  else "[bank_full_oof + nuisance] (matched-strength companion)"),
           "frozen_definition": FREEZE,
           "Y_strongest_found_channel": Y,
           "Y_channel": res["top_nuisance_channels"][0]["name"],
           "primary_delta_3seed": delta3,
           "adversarial_family": ("U = rank(T) blended with Uniform(0,1) (seed 2026), "
                                  "blend bisected to hit the target alone-AUC within .002"),
           "sweep": {"s_grid": list(S_GRID), "n_bisections": N_BISECT,
                     "gbm_seeds": list(SWEEP_SEEDS),
                     "why_one_seed": "the crossing is a threshold, not a quoted AUC"},
           "limitations": LIMITATIONS}

    mass = find_mass(cell)
    blk["missing_mass"] = mass

    if delta3 <= 0:
        blk.update({"X": None, "robustness_ratio_excess": None, "X_over_Y": None,
                    "Z": None, "verdict": "n/a",
                    "verdict_reason": "PRIMARY increment is null/negative -- there is "
                                      "nothing to absorb, so the E-value analog is "
                                      "undefined for this cell"})
        res["evalue_analog_matched" if matched else "evalue_analog"] = blk
        if not dry:
            p.write_text(json.dumps(res, indent=2, default=str))
        print(f"  [{cell}] Delta={delta3:+.4f} <= 0 -> E-value n/a", flush=True)
        return blk

    # ---- rebuild the matrices (arms are NOT recomputed) ---------------------
    meta = json.loads((ROWS / f"{cell}.meta.json").read_text())
    z = np.load(ROWS / f"{cell}.npz", allow_pickle=True)
    ids_E = [str(i) for i in z["ids"]]
    y = z["y"].astype(int)
    groups = np.array([str(g) for g in z["groups"]], dtype=object)
    T = z["dense"].astype(float)
    a = F2C.ADAPTERS[cell]()
    pos = {str(i): k for k, i in enumerate(a["ids"])}
    idx = np.array([pos[i] for i in ids_E])
    bank, nuis = a["bank"][idx], a["nuis"][idx]
    assert np.array_equal(np.asarray(a["y"])[idx], y), f"{cell}: y mismatch"
    if matched:
        f = ROWS / f"{cell}.bank_full_oof_E.npy"
        assert f.exists(), f"{cell}: run f2_matched.py first ({f} missing)"
        bank = np.load(f).reshape(-1, 1)          # stage-1 column replaces the bank block
    family = meta["family"]
    auc_T = float(roc_auc_score(y, T))

    grid = [s for s in S_GRID if s <= auc_T + 1e-9]
    blk["auc_T"] = auc_T
    blk["s_grid_used"] = grid

    d0 = fit_arm(family, np.column_stack([bank, nuis]), T, y, groups, gbm_seeds=SWEEP_SEEDS)
    blk["primary_delta_seed0_unswept"] = float(d0["VAT_nl_mean"] - d0["VA_nl_mean"])

    pts = []
    for s in grid:
        U, a_real, w = build_U(T, y, s)
        dv = delta_with(family, bank, nuis, U, T, y, groups)
        pts.append({"s_target": s, "s_realised": a_real, "blend_w": w, "delta": dv})
        print(f"    [{cell}] s={s:.3f} (real {a_real:.4f}) -> Delta={dv:+.4f}", flush=True)
        if dv <= 0:
            break
    blk["sweep_points"] = pts

    ds = [q["delta"] for q in pts]
    blk["monotone_ok"] = bool(all(ds[i + 1] <= ds[i] + MONO_TOL for i in range(len(ds) - 1)))
    if not blk["monotone_ok"]:
        blk["NON_MONOTONE"] = True

    if ds[-1] > 0:
        blk.update({"X": None, "X_lower_bound_exceeds": auc_T,
                    "X_statement": f"> AUC(T) = {auc_T:.4f}",
                    "robustness_ratio_excess": float((auc_T - .5) / max(Y - .5, 1e-9)),
                    "X_over_Y": float(auc_T / Y),
                    "not_absorbable_by_single_channel_weaker_than_T": True})
        X_for_verdict = auc_T
    else:
        lo = pts[-2]["s_realised"] if len(pts) > 1 else 0.5
        hi = pts[-1]["s_realised"]
        for _ in range(N_BISECT):
            mid = 0.5 * (lo + hi)
            U, a_real, w = build_U(T, y, mid)
            dv = delta_with(family, bank, nuis, U, T, y, groups)
            blk["sweep_points"].append({"s_target": mid, "s_realised": a_real,
                                        "blend_w": w, "delta": dv, "bisection": True})
            print(f"    [{cell}] bisect s={mid:.4f} -> Delta={dv:+.4f}", flush=True)
            if dv > 0:
                lo = a_real
            else:
                hi = a_real
        X = 0.5 * (lo + hi)
        blk.update({"X": float(X), "X_resolution": float(abs(hi - lo) / 2),
                    "robustness_ratio_excess": float((X - .5) / max(Y - .5, 1e-9)),
                    "X_over_Y": float(X / Y)})
        X_for_verdict = X

    if mass.get("available"):
        zb = z_bound(found, mass["M_hat"], mass.get("S_obs_reported") or len(found))
        blk["Z_detail"] = zb
        blk["Z"] = zb["Z"]
        blk["verdict"] = "ROBUST" if X_for_verdict > zb["Z"] else "ABSORBABLE-IN-PRINCIPLE"
        blk["verdict_reason"] = (f"X {'>' if X_for_verdict > zb['Z'] else '<='} Z "
                                 f"({X_for_verdict:.4f} vs {zb['Z']:.4f})")
        if mass.get("TAU_ERA_MASS"):
            blk["Z_FLAG"] = ("TAU_ERA_MASS -- no strict-merge marker found in this "
                             "cell's missing-mass artifact; Z is provisional pending "
                             "the strict certificate backfill (certA_strict.json)")
    else:
        blk["Z"] = None
        blk["verdict"] = "Z_UNAVAILABLE"
        blk["verdict_reason"] = ("no Track-B Good-Turing missing mass on disk for this "
                                 "cell; X and RR stand, the M-hat-coupled bound does not")

    res["evalue_analog_matched" if matched else "evalue_analog"] = blk
    if not dry:
        p.write_text(json.dumps(res, indent=2, default=str))
    print(f"  [{cell}] X={blk.get('X')} Y={Y:.4f} RR={blk.get('robustness_ratio_excess')} "
          f"Z={blk.get('Z')} -> {blk['verdict']}", flush=True)
    return blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--matched", action="store_true",
                    help="condition X on the matched-strength block [bank_full_oof + nuisance]")
    args = ap.parse_args()
    for c in args.cell:
        t = time.time()
        print(f"=== E-value {c} ===", flush=True)
        try:
            run(c, dry=args.dry, matched=args.matched)
        except Exception as e:
            print(f"  [{c}] FAILED: {type(e).__name__}: {e}", flush=True)
        print(f"  [{c}] {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
