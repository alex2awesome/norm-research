#!/usr/bin/env python3
"""ROUND 0 for the code_v3 Layer-3 closure cell.

Produces, all under the WITHIN-REPO protocol (pooled AUC is never quoted as a residual
on this cell):

  1. GATE TABLE      -- within-repo T per dense seed + across-seed spread, and the
                        within-repo residual Δ = T − VA_nl in the LAYER-1 protocol
                        (VA_nl fit grouped-OOF within each split), i.e. exactly the
                        instrument that produced the gated +.0576 / +.0390.
  2. CLOSURE ROUND-0 -- VA_lin / VA_nl refit under the CLOSURE protocol
                        (fit on FIT+MINE only, predict MONITOR) and the within-repo Δ
                        on MONITOR / honest-full / eval / test.  These levels are
                        protocol-specific and are NOT comparable to the gate table.
  3. POSITION LINE   -- within-repo PR-number percentile as an observed covariate.
  4. SWAP BASELINE   -- within-repo (C+, C-) pair algebra.

CPU only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "maps_hw_si"))

import cells_code as C                                       # noqa: E402
import closure_core as CC                                    # noqa: E402

AB = HERE / "abank_rescore"


# ------------------------------------------------------------- gate table ---
def gate_table(d):
    """Layer-1-protocol within-repo readout: VA_nl = the stored grouped-OOF vectors
    fit WITHIN each split (the instrument behind +.0576 eval / +.0390 test)."""
    out = {"protocol": "LAYER-1 (VA_nl grouped-OOF fit WITHIN each split), "
                       "within-repo n-weighted, repos with n>=20 and both classes",
           "dense_seeds_available": d["dense_seeds_have"], "splits": {}}
    for sp in ("eval", "test"):
        m = d["split"] == sp
        va = np.mean([np.load(AB / f"code_v3_{sp}_va_nl_oof_seed{s}.npy") for s in (0, 1, 2)],
                     axis=0)
        # the stored OOF vectors are in split-file row order == our row order for that split
        vfull = np.full(len(d["y"]), np.nan)
        vfull[m] = va
        rec = {"n_rows": int(m.sum()), "per_seed": {}}
        for k, s in enumerate(d["dense_seeds_have"]):
            p = d["dense_seed_probs"][:, k]
            w = C.within_repo_delta(d["y"], p, vfull, d["groups"], m)
            rec["per_seed"][f"seed{s}"] = {
                "T_within": w["a_nwtd"], "VA_nl_within": w["b_nwtd"],
                "delta": w["delta_nwtd"], "T_pooled": float(roc_auc_score(d["y"][m], p[m])),
                "dense_wins_repos": w["a_wins_repos"], "n_repos": w["n_repos"],
                "wilcoxon_p": w.get("wilcoxon_p"), "jackknife_se": w.get("jackknife_se"),
                "jackknife_ci95": w.get("jackknife_ci95")}
        # seed-ensemble row-level readout
        we = C.within_repo_delta(d["y"], d["dense"], vfull, d["groups"], m)
        Ts = [rec["per_seed"][f"seed{s}"]["T_within"] for s in d["dense_seeds_have"]]
        Ds = [rec["per_seed"][f"seed{s}"]["delta"] for s in d["dense_seeds_have"]]
        rec["ensemble"] = {"T_within": we["a_nwtd"], "VA_nl_within": we["b_nwtd"],
                           "delta": we["delta_nwtd"], "n_repos": we["n_repos"],
                           "wilcoxon_p": we.get("wilcoxon_p"),
                           "jackknife_ci95": we.get("jackknife_ci95")}
        rec["across_seed"] = {
            "T_within_mean": float(np.mean(Ts)), "T_within_sd": float(np.std(Ts, ddof=1)) if len(Ts) > 1 else None,
            "T_within_range": [float(min(Ts)), float(max(Ts))],
            "delta_mean": float(np.mean(Ds)), "delta_sd": float(np.std(Ds, ddof=1)) if len(Ds) > 1 else None,
            "delta_range": [float(min(Ds)), float(max(Ds))]}
        out["splits"][sp] = rec
    return out


# --------------------------------------------------------- closure round 0 --
def _expand(vec, mask, n):
    """MONITOR-length vector -> full-length vector with NaN elsewhere."""
    v = np.full(n, np.nan)
    v[mask] = vec
    return v


def closure_round0(d, fitmask, monmask):
    """VA refit under the CLOSURE protocol.

    Cost note (recorded): only the VA block needs the honest-full vector, because Δ is
    read off VA.  V and A are fit MONITOR-only (`want_oof=False`), which removes two
    thirds of the HistGB work and changes no quoted number.
    """
    A = d["A"]
    Aaug = np.column_stack([A, (~np.isnan(A)).astype(float)])   # score + applied, as Layer 1
    blocks = {"V": ([d["V"]], False), "A": ([Aaug], False), "VA": ([d["V"], Aaug], True)}
    y, g, n = d["y"], d["groups"], len(d["y"])
    res = {"protocol": "CLOSURE (fit on FIT+MINE only, predict MONITOR); "
                       "levels are protocol-specific and NOT comparable to the gate table",
           "blocks": {}}
    store = {}
    tiers = (("monitor", monmask), ("honest_full", None),
             ("eval", d["split"] == "eval"), ("test", d["split"] == "test"))
    for name, (raw, want_oof) in blocks.items():
        print(f"  [closure] fitting block {name} (want_oof={want_oof}) ...", flush=True)
        fb = CC.fit_block(raw, fitmask, monmask, y, g, want_oof=want_oof)
        if want_oof:
            lin, nl = np.full(n, np.nan), np.full(n, np.nan)
            lin[fitmask] = fb["oof_lin_fitmine"]; lin[monmask] = fb["lin_mon"]
            nl[fitmask] = fb["oof_nl_fitmine"];   nl[monmask] = fb["nl_mon"]
        else:
            lin = _expand(fb["lin_mon"], monmask, n)
            nl = _expand(fb["nl_mon"], monmask, n)
        store[name] = {"lin": lin, "nl": nl, "n_features": fb["n_features"]}
        r = {"n_features": fb["n_features"], "grid_picks": fb["picks"],
             "monitor_only": not want_oof}
        for tier, m in (tiers if want_oof else (("monitor", monmask),)):
            wr = C.within_repo_auc(y, nl, g, m)
            r[tier] = {"VA_lin_within": C.within_repo_auc(y, lin, g, m)["nwtd"],
                       "VA_nl_within": wr["nwtd"],
                       "T_within": C.within_repo_auc(y, d["dense"], g, m)["nwtd"],
                       "n_repos": wr["n_repos"]}
        per_seed = [C.within_repo_auc(y, _expand(v, monmask, n), g, monmask)["nwtd"]
                    for v in fb["nl_mon_seeds"]]
        r["monitor"]["VA_nl_within_per_seed"] = per_seed
        r["monitor"]["VA_nl_within_seed_spread"] = float(max(per_seed) - min(per_seed))
        res["blocks"][name] = r
        print(f"  [closure] {name}: MONITOR within-repo VA_nl "
              f"{r['monitor']['VA_nl_within']:.4f} (seed spread "
              f"{r['monitor']['VA_nl_within_seed_spread']:.4f})", flush=True)

    res["delta"] = {}
    for tier, m in tiers:
        best = "nl" if (res["blocks"]["VA"][tier]["VA_nl_within"]
                        >= res["blocks"]["VA"][tier]["VA_lin_within"]) else "lin"
        w = C.within_repo_delta(y, d["dense"], store["VA"][best], g, m)
        res["delta"][tier] = {"best_VA": best, "T_within": w["a_nwtd"],
                              "VA_within": w["b_nwtd"], "delta": w["delta_nwtd"],
                              "n_repos": w["n_repos"], "n_rows": w["n_rows"],
                              "dense_wins_repos": w["a_wins_repos"],
                              "wilcoxon_p": w.get("wilcoxon_p"),
                              "jackknife_se": w.get("jackknife_se"),
                              "jackknife_ci95": w.get("jackknife_ci95")}
        print(f"  [closure] delta {tier}: {w['delta_nwtd']:+.4f} "
              f"(T {w['a_nwtd']:.4f} VA {w['b_nwtd']:.4f}, {w['n_repos']} repos)", flush=True)
    np.savez_compressed(HERE / "round0_state.npz", ids=np.array(d["ids"]),
                        **{f"{k}_{s}": v[s] for k, v in store.items() for s in ("lin", "nl")})
    return res, store


# ------------------------------------------------------------ position ------
def position_line(d):
    """FREEZE ADDENDUM 4 -- position-in-container, as an OBSERVED COVARIATE only.
    Container = repository; position = PR number (monotone in submission time)."""
    y, g = d["y"], d["groups"]
    pct = d["position"]["within_repo_pr_pct"]
    out = {"channel": "within-repo PR-number percentile (repo-local recency)",
           "status": "OBSERVED COVARIATE -- never enters V, A, the bank or any closure fit"}
    for tier, m in (("all", None), ("eval", d["split"] == "eval"),
                    ("test", d["split"] == "test")):
        mm = np.ones(len(y), bool) if m is None else m
        w = C.within_repo_auc(y, pct, g, mm)
        per = np.array([x["auc"] for x in w["per_repo"]])
        out[tier] = {"pooled_auc": float(roc_auc_score(y[mm], pct[mm])),
                     "within_repo_nwtd": w["nwtd"], "within_repo_median": w["median"],
                     "n_repos": w["n_repos"], "n_rows": w["n_rows"],
                     "frac_repos_absdev_gt_.15": float((np.abs(per - .5) > .15).mean()) if len(per) else None,
                     "repo_auc_sd": float(per.std(ddof=1)) if len(per) > 1 else None}
    # does the dense model track it?  (correlation of dense score with position, within repo)
    import pandas as pd
    s = pd.DataFrame({"g": g, "p": d["dense"], "x": pct})
    rho = s.groupby("g").apply(
        lambda t: t["p"].corr(t["x"], method="spearman") if len(t) >= C.MIN_REPO_N else np.nan,
        include_groups=False)
    rho = rho.dropna()
    out["dense_vs_position_within_repo_spearman"] = {
        "n_repos": int(len(rho)), "mean": float(rho.mean()), "median": float(rho.median()),
        "frac_abs_gt_.2": float((rho.abs() > .2).mean())}
    return out


# ---------------------------------------------------------------- swap ------
def swap_pairs(d, va, mask=None, min_n=C.MIN_REPO_N):
    """(C+, C-) pair algebra restricted to WITHIN-REPO (merged, unmerged) pairs."""
    y, g = d["y"], d["groups"]
    dense = d["dense"]
    m = np.ones(len(y), bool) if mask is None else mask
    cp_n = cp_d = cm_n = cm_d = 0
    npairs = 0
    for r in np.unique(g[m]):
        s = m & (g == r)
        if s.sum() < min_n:
            continue
        yi = y[s]
        if len(set(yi)) < 2:
            continue
        pi, vi = dense[s], va[s]
        P, N = np.where(yi == 1)[0], np.where(yi == 0)[0]
        dp = pi[P][:, None] - pi[N][None, :]
        dv = vi[P][:, None] - vi[N][None, :]
        dense_ok = dp > 0
        bank_ok = dv > 0
        npairs += dp.size
        cp_d += int(dense_ok.sum());  cp_n += int((bank_ok & dense_ok).sum())
        cm_d += int((~dense_ok).sum()); cm_n += int((bank_ok & ~dense_ok).sum())
    return {"n_within_repo_pairs": npairs,
            "C_plus": cp_n / cp_d if cp_d else float("nan"),
            "C_minus": cm_n / cm_d if cm_d else float("nan"),
            "w_plus": cp_d / npairs if npairs else float("nan")}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-only", action="store_true",
                    help="recompute ONLY the seed gate table (cheap; use when dense "
                         "seeds 1/2 land) and merge it into round0_results.json")
    args = ap.parse_args()

    if args.gate_only:
        d = C.load()
        gt = gate_table(d)
        p = HERE / "round0_results.json"
        cur = json.loads(p.read_text()) if p.exists() else {}
        cur["gate_table"] = gt
        p.write_text(json.dumps(cur, indent=1, default=float))
        print(json.dumps(gt, indent=1, default=float))
        return

    import build_splits_code as BS
    d, fitmask, monmask = BS.build()
    out = {"cell": "code_v3 (GitHub PR merge, enriched text)",
           "gate_table": gate_table(d)}
    print("\n--- gate table done ---", flush=True)
    r0, store = closure_round0(d, fitmask, monmask)
    out["closure_round0"] = r0
    print("--- closure round 0 done ---", flush=True)
    out["position"] = position_line(d)
    out["swap_baseline"] = {
        tier: swap_pairs(d, store["VA"]["nl"], m)
        for tier, m in (("monitor", monmask), ("honest_full", None),
                        ("eval", d["split"] == "eval"), ("test", d["split"] == "test"))}
    (HERE / "round0_results.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "gate_table"}, indent=1,
                     default=float)[:4000])
    print("\nGATE:", json.dumps(out["gate_table"], indent=1, default=float)[:2500])


if __name__ == "__main__":
    main()
