#!/usr/bin/env python3
"""Round-3 (TIER R) readout: reconstruct the round-1 and round-2 closure curve from the
REPAIRED scores, side by side with the pre-repair numbers.

This is the deliverable the coordinator asked for -- "reconstruct their bank state and
Delta curve under the closure protocol" -- computed once the item-view defect is fixed.
It runs the frozen Layer-1 fitting protocol (closure_core.fit_block, collapse gate
enforced) on exactly the bank states rounds 1 and 2 actually produced:

    bank 0 = [V, A]
    bank 1 = bank 0 + the round-1 A-routed columns
    bank 2 = bank 1 + the round-2 A-routed columns

and reports, for each of MONITOR (TIER 1), HONEST and within-contest (TIER 2):
VA_lin / VA_nl, the gain, its group and item bootstrap bands, Delta_beyond, the swap pair,
and the Track-A / Track-B alone-AUCs under both views.

No proposals, no routing, no Good-Turing.  CPU only.  Usage: python view_repair_readout.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import cells as C
import closure_core as L
from contest_line import within_group_auc
from readout import item_boot_ci, stack_oof, swap_pair

HERE = Path(__file__).resolve().parent


def a_ids_of(r):
    rt = json.loads((HERE / f"cap_crowd_r{r}_routing_final.json").read_text())
    return ([x["blind_id"] for x in rt["final"] if x["final_route"] == "A"],
            [x for x in rt["final"] if x["final_route"] == "B"])


def cols(path, ids):
    z = np.load(path, allow_pickle=True)
    cid = [str(s) for s in z["crit_ids"]]
    idx = [cid.index(i) for i in ids if i in cid]
    return z["X"][:, idx], [cid[j] for j in idx]


def main():
    d = C.load("cap_crowd")
    sp = json.loads((HERE / "cap_crowd_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    y, groups, dense = d["y"], d["groups"], d["dense"]
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])
    ymon, gmon = y[monm], groups[monm]
    T_mon, T_hon = L.auc(ymon, dense[monm]), L.auc(y[held], dense[held])
    da = d["dense_archived"]
    T_mon_arch, T_hon_arch = L.auc(ymon, da[monm]), L.auc(y[held], da[held])

    out = {"cell": "cap_crowd", "round": "3", "tier": "R",
           "kind": "VIEW-REPAIR reconstruction of the round-1/round-2 closure curve",
           "n": {"MONITOR": int(monm.sum()), "n_groups_MONITOR": int(len(set(gmon))),
                 "n_pos_MONITOR": int(ymon.sum()), "HONEST": int(held.sum()),
                 "n_groups_HONEST": int(len(set(groups[held])))},
           "T_convention": ("PRIMARY = matched fresh vanilla (D20_cap_vanilla, lambda_adv=0); "
                            "ARCHIVED reported beside it, never differenced against it"),
           "T_archived": {"MONITOR": T_mon_arch, "HONEST": T_hon_arch},
           "T": {"MONITOR": T_mon, "HONEST": T_hon,
                 "MONITOR_within_contest": within_group_auc(ymon, dense[monm], gmon)[0],
                 "HONEST_within_contest": within_group_auc(y[held], dense[held],
                                                           groups[held])[0]},
           "score_report_r3": json.loads(
               (HERE / "cap_crowd_r3_score_report.json").read_text()),
           "views": {}}
    # keep the report small: the anchor text dump lives in the score report file itself
    out["score_report_r3"]["anchors"].pop("anchor_scram_texts", None)
    out["score_report_r3"].pop("per_criterion", None)

    for view, suffix in (("REPAIRED (cartoon+caption)", "_scores.npz"),
                         ("PRE-REPAIR (caption only)", "_scores.MISMATCHEDVIEW.npz")):
        blocks, tags = [d["V"], d["A"]], ["V", "A_base"]
        states, prev = {}, None
        vprev = None
        rec = {}
        for r in ("1", "2"):
            aids, brows = a_ids_of(r)
            XA, _ = cols(HERE / f"cap_crowd_r{r}{suffix}", aids)
            fit_prev = L.fit_block(list(blocks), fitm, monm, y, groups)
            blocks.append(XA)
            tags.append(f"A_round{r}")
            fit_new = L.fit_block(list(blocks), fitm, monm, y, groups)

            def full(f):
                v = np.full(len(y), np.nan)
                v[fitm] = f["oof_nl_fitmine"]
                v[monm] = f["nl_mon"]
                return v

            vp, vn = full(fit_prev), full(fit_new)
            rec[f"round{r}"] = {
                "n_features_before": fit_prev["n_features"],
                "n_features_after": fit_new["n_features"],
                "n_A_routed": len(aids),
                "VA_lin_MONITOR": L.auc(ymon, fit_new["lin_mon"]),
                "VA_nl_MONITOR_before": L.auc(ymon, fit_prev["nl_mon"]),
                "VA_nl_MONITOR_after": L.auc(ymon, fit_new["nl_mon"]),
                "VA_nl_MONITOR_seed_spread": float(
                    max(L.auc(ymon, p) for p in fit_new["nl_mon_seeds"])
                    - min(L.auc(ymon, p) for p in fit_new["nl_mon_seeds"])),
                "VA_nl_HONEST_before": L.auc(y[held], vp[held]),
                "VA_nl_HONEST_after": L.auc(y[held], vn[held]),
                "gain_MONITOR": L.auc(ymon, fit_new["nl_mon"]) - L.auc(ymon, fit_prev["nl_mon"]),
                "gain_HONEST": L.auc(y[held], vn[held]) - L.auc(y[held], vp[held]),
                "gain_ci_MONITOR_group": L.group_boot_ci(ymon, fit_new["nl_mon"],
                                                         fit_prev["nl_mon"], gmon),
                "gain_ci_MONITOR_item": item_boot_ci(ymon, fit_new["nl_mon"],
                                                     fit_prev["nl_mon"]),
                "gain_ci_HONEST_group": L.group_boot_ci(y[held], vn[held], vp[held],
                                                        groups[held]),
                "gain_MONITOR_within_contest": (
                    within_group_auc(ymon, fit_new["nl_mon"], gmon)[0]
                    - within_group_auc(ymon, fit_prev["nl_mon"], gmon)[0]),
                "Delta_beyond_MONITOR_after": T_mon - L.auc(ymon, fit_new["nl_mon"]),
                "Delta_beyond_HONEST_after": T_hon - L.auc(y[held], vn[held]),
                "swap": {"before": swap_pair(y[held], dense[held], vp[held]),
                         "after": swap_pair(y[held], dense[held], vn[held])},
            }
            s0, s1 = rec[f"round{r}"]["swap"]["before"], rec[f"round{r}"]["swap"]["after"]
            rec[f"round{r}"]["swap"]["delta"] = {
                "dC_plus": s1["C_plus"] - s0["C_plus"],
                "dC_minus": s1["C_minus"] - s0["C_minus"],
                "d_rho": s1["spearman_bank_vs_dense"] - s0["spearman_bank_vs_dense"],
                "adverse_by_registry_rule": bool(s1["C_plus"] > s0["C_plus"]
                                                 and s1["C_minus"] <= s0["C_minus"])}
            vprev = vn

            # alone-AUCs, both tracks
            XAall = L.clean_apply(XA, *L.clean_fit(XA[fitm]))
            keepA, _ = L.clean_fit(XA[fitm])
            rec[f"round{r}"]["track_A_alone_AUC_HONEST"] = sorted(
                [{"id": aids[j], "auc": L.auc(y[held], XAall[held, k])}
                 for k, j in enumerate(keepA)],
                key=lambda z: -abs(z["auc"] - .5))
            bids = [b["blind_id"] for b in brows]
            XB, _ = cols(HERE / f"cap_crowd_r{r}{suffix}", bids)
            keepB, medB = L.clean_fit(XB[fitm])
            XBall = L.clean_apply(XB, keepB, medB)
            nameB = {b["blind_id"]: b["name"] for b in brows}
            mixB = {b["blind_id"]: bool(b.get("mixed")) for b in brows}
            rec[f"round{r}"]["track_B_alone_AUC_HONEST"] = sorted(
                [{"id": bids[j], "name": nameB[bids[j]], "mixed": mixB[bids[j]],
                  "auc": L.auc(y[held], XBall[held, k])}
                 for k, j in enumerate(keepB)], key=lambda z: -abs(z["auc"] - .5))
            rec[f"round{r}"]["n_B_dropped_by_gate"] = int(len(bids) - len(keepB))

        # cumulative state after round 2, and the stacked increment of record
        rec["cumulative_after_r2"] = {
            "bank_blocks": tags,
            "VA_nl_MONITOR": rec["round2"]["VA_nl_MONITOR_after"],
            "VA_nl_HONEST": rec["round2"]["VA_nl_HONEST_after"],
            "total_gain_MONITOR": (rec["round2"]["VA_nl_MONITOR_after"]
                                   - rec["round1"]["VA_nl_MONITOR_before"]),
            "total_gain_HONEST": (rec["round2"]["VA_nl_HONEST_after"]
                                  - rec["round1"]["VA_nl_HONEST_before"]),
        }
        # joint nuisance model over BOTH rounds' B channels
        allB = []
        for r in ("1", "2"):
            _, brows = a_ids_of(r)
            XB, _ = cols(HERE / f"cap_crowd_r{r}{suffix}", [b["blind_id"] for b in brows])
            allB.append(XB)
        XBc = np.column_stack(allB)
        rb = L.fit_block([XBc], fitm, monm, y, groups)
        jb = np.full(len(y), np.nan)
        jb[fitm] = rb["oof_nl_fitmine"]
        jb[monm] = rb["nl_mon"]
        for pop, mask in (("HONEST", held), ("MONITOR", monm)):
            yy, gg = y[mask], groups[mask]
            b, dn, va = jb[mask], dense[mask], vprev[mask]
            s_bd, s_bv = stack_oof([b, dn], yy, gg), stack_oof([b, va], yy, gg)
            s_all = stack_oof([b, dn, va], yy, gg)
            rec["cumulative_after_r2"][f"stacked_increment_{pop}"] = {
                "n": int(mask.sum()), "n_B_channels": int(rb["n_features"]),
                "AUC_jointB": L.auc(yy, b), "AUC_dense": L.auc(yy, dn),
                "AUC_bank": L.auc(yy, va),
                "dense_increment_over_B_plus_bank": L.auc(yy, s_all) - L.auc(yy, s_bv),
                "bank_increment_over_B_plus_dense": L.auc(yy, s_all) - L.auc(yy, s_bd),
                "ci_dense_increment_over_B_plus_bank": L.group_boot_ci(yy, s_all, s_bv, gg),
            }
        out["views"][view] = rec

    (HERE / "cap_crowd_r3_viewrepair_results.json").write_text(
        json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "views"}, indent=1, default=float))
    for view, rec in out["views"].items():
        print(f"\n===== {view}")
        for r in ("1", "2"):
            g = rec[f"round{r}"]
            ci = g["gain_ci_MONITOR_group"]
            print(f"  r{r}: VA_nl MONITOR {g['VA_nl_MONITOR_before']:.4f} -> "
                  f"{g['VA_nl_MONITOR_after']:.4f}  gain {g['gain_MONITOR']:+.4f} "
                  f"[{ci['lo']:+.4f},{ci['hi']:+.4f}] p={ci['p_gt0']:.3f} | "
                  f"HONEST gain {g['gain_HONEST']:+.4f} | "
                  f"best A alone {max((a['auc'] for a in g['track_A_alone_AUC_HONEST']), default=0):.3f} "
                  f"| best |B-.5| {max((abs(b['auc']-.5) for b in g['track_B_alone_AUC_HONEST']), default=0):.3f}")
        c = rec["cumulative_after_r2"]
        print(f"  cumulative: MONITOR {c['total_gain_MONITOR']:+.4f}  "
              f"HONEST {c['total_gain_HONEST']:+.4f}  "
              f"dense increment over B+bank HONEST "
              f"{c['stacked_increment_HONEST']['dense_increment_over_B_plus_bank']:+.4f}")
    print("\nwrote cap_crowd_r3_viewrepair_results.json")


if __name__ == "__main__":
    main()
