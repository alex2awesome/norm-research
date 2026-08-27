#!/usr/bin/env python3
"""The closure curve for the press cell, on the DECLARED T convention.

`readout.py` computes its internal `Delta_beyond_*` fields against `d["dense"]`, which is
the per-row mean of the three dense seeds' probabilities.  Averaging denoises, so that
vector's AUC (.7744 on HONEST) sits well above the quantity this programme calls T --
the MEAN OVER SEEDS OF THE AUC (.7508 on HONEST), which is the convention VA_nl itself
uses.  This script re-reads every round's VA_nl from its results file and recomputes
Delta against the declared T, so the curve and the round table are internally consistent
and comparable to the round-0 anchor.

It also assembles: the per-round gains and their company-cluster bootstrap CIs, the
signed saturation flags (`gain < eps`, the frozen prereg reading), the swap pair, the
spurious-alone AUCs, the discount, the stacked increment and both tracks' missing mass.

CPU only, reads saved artifacts only.  Usage: python curve.py [--upto 5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EPS = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, default=5)
    a = ap.parse_args()

    r0 = json.loads((HERE / "press_verdict_r0_context.json").read_text())
    T_H = r0["T"]["HONEST_eval_plus_test"]["T"]
    T_M = r0["T"]["MONITOR"]["T"]
    T_ens_H = r0["T"]["HONEST_eval_plus_test"]["T_seed_ensemble_NOT_QUOTED"]

    rows = [{
        "round": 0,
        "feats": r0["state0_bank"]["PRIMARY_layer1_const05"]["n_features"],
        "VA_H": r0["state0_bank"]["PRIMARY_layer1_const05"]["VA_nl_HONEST"],
        "VA_M": r0["state0_bank"]["PRIMARY_layer1_const05"]["VA_nl_MONITOR"],
    }]
    detail = {}
    for r in range(1, a.upto + 1):
        f = HERE / f"press_verdict_r{r}_results.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        st = d["track_A"][f"state_after_round_{r}"]
        rows.append({"round": r, "feats": st["n_features"],
                     "VA_H": st["VA_nl_HONEST"], "VA_M": st["VA_nl_MONITOR"]})
        detail[r] = d

    for i, row in enumerate(rows):
        row["D_H"] = T_H - row["VA_H"]
        row["D_M"] = T_M - row["VA_M"]
        if i:
            row["gain_H"] = row["VA_H"] - rows[i - 1]["VA_H"]
            row["gain_M"] = row["VA_M"] - rows[i - 1]["VA_M"]
            row["sub_eps_signed_H"] = row["gain_H"] < EPS
            row["sub_eps_signed_M"] = row["gain_M"] < EPS
            row["sub_eps_abs_H"] = abs(row["gain_H"]) < EPS

    # trailing run of signed sub-eps rounds on the primary (HONEST) statistic
    flags = [r.get("sub_eps_signed_H") for r in rows[1:]]
    trailing = 0
    for f in reversed(flags):
        if f:
            trailing += 1
        else:
            break

    out = {
        "cell": "press_verdict",
        "T_convention": "mean over dense seeds {42,1,2} of AUC",
        "T_HONEST": T_H, "T_MONITOR": T_M,
        "T_HONEST_seed_ensemble_NOT_QUOTED": T_ens_H,
        "epsilon": EPS,
        "primary_saturation_statistic": "VA_nl gain on HONEST (n=605, 45 companies) -- "
                                        "DECISION 2, recorded before round 1",
        "curve": rows,
        "signed_sub_eps_flags_HONEST": flags,
        "trailing_sub_eps_run": trailing,
        "saturation_declared": trailing >= 2,
        "best_bank_so_far": None,
        "per_round": {},
    }
    best = min(rows, key=lambda r: r["D_H"])
    out["best_bank_so_far"] = {"round": best["round"], "feats": best["feats"],
                               "VA_nl_HONEST": best["VA_H"], "Delta_beyond_HONEST": best["D_H"]}

    for r, d in detail.items():
        ta, sp = d["track_A"], d.get("swap_pair_HONEST", {}).get("delta", {})
        blk = {
            "routing": d["routing"],
            "gain_ci_HONEST": ta["gain_ci_HONEST"], "gain_ci_MONITOR": ta["gain_ci_MONITOR"],
            "swap": sp,
            "missing_mass": d.get("missing_mass"), "n_species": d.get("n_species"),
            "anchors": (d.get("score_report") or {}).get("anchors"),
            "n_collapsed": (d.get("score_report") or {}).get("n_collapsed"),
            "spurious_alone_HONEST": d["discount_ALL_B"]["spurious_alone_AUC_histgb_HONEST"],
            "spurious_alone_linear_MONITOR": d["discount_ALL_B"]["spurious_alone_AUC_linear_MONITOR"],
            "decile_Delta_adj_ALL_B": d["discount_ALL_B"]["stratified_HONEST_q10"]["joint_B_score"]["Delta_adj"],
            "stacked": {k: v for k, v in d["stacked_increment_HONEST"].items()
                        if not isinstance(v, dict)},
            "stacked_ci_dense": d["stacked_increment_HONEST"]["ci_dense_increment"],
            "top_channels": [{"auc": c["alone_AUC_HONEST"], "mixed": c["mixed"],
                              "name": c["name"], "parent": c["upstream_parent"]}
                             for c in d["spurious_map"]["channels"][:6]],
        }
        if "discount_STRICT_no_mixed" in d:
            blk["strict"] = {
                "spurious_alone_HONEST": d["discount_STRICT_no_mixed"]["spurious_alone_AUC_histgb_HONEST"],
                "decile_Delta_adj": d["discount_STRICT_no_mixed"]["stratified_HONEST_q10"]["joint_B_score"]["Delta_adj"]}
        out["per_round"][str(r)] = blk

    (HERE / "press_verdict_curve.json").write_text(json.dumps(out, indent=1, default=float))

    print(f"T (declared convention): HONEST {T_H:.4f}  MONITOR {T_M:.4f}   "
          f"[seed-ensemble {T_ens_H:.4f}, never quoted]\n")
    print(f"{'r':>2} {'feats':>6} {'VA_H':>7} {'gain_H':>8} {'D_H':>8} | "
          f"{'VA_M':>7} {'gain_M':>8} {'D_M':>8}  sub-eps(signed,H)")
    for row in rows:
        g = f"{row['gain_H']:+.4f}" if "gain_H" in row else "   --  "
        gm = f"{row['gain_M']:+.4f}" if "gain_M" in row else "   --  "
        se = ("YES" if row.get("sub_eps_signed_H") else "no") if "gain_H" in row else ""
        print(f"{row['round']:>2} {row['feats']:>6} {row['VA_H']:.4f} {g:>8} "
              f"{row['D_H']:+.4f} | {row['VA_M']:.4f} {gm:>8} {row['D_M']:+.4f}   {se}")
    print(f"\ntrailing signed sub-eps run = {trailing}; saturation declared = "
          f"{out['saturation_declared']}")
    print(f"best bank so far: round {best['round']} ({best['feats']} feats), "
          f"VA_nl {best['VA_H']:.4f}, Delta_beyond {best['D_H']:+.4f}")


if __name__ == "__main__":
    main()
