#!/usr/bin/env python3
"""Assemble the campaign's markdown tables from the per-round JSON artifacts.

Emits (to stdout) the round table, the closure curve, the spurious map with
per-channel alone-AUCs / upstream parents / MIXED flags, the discount band, the
stacked-increment table and the missing-mass table -- ready to paste into
notes/2026-08-06__closure_cw_community.md.

Usage: python build_report.py --rounds 1,2,3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def f(x, n=4, signed=False):
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "n/a"
    return f"{v:+.{n}f}" if signed else f"{v:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", required=True)
    a = ap.parse_args()
    rounds = [int(x) for x in a.rounds.split(",") if x]
    r0 = json.loads((HERE / "round0_results.json").read_text())
    res = {r: json.loads((HERE / f"round{r}_results.json").read_text())
           for r in rounds if (HERE / f"round{r}_results.json").exists()}

    print("### Closure curve\n")
    print("| round | criteria added (post-gate) | bank cols | VA_nl MONITOR | Δ_r (MONITOR) "
          "| Δ_beyond MONITOR | Δ_beyond population |")
    print("|---|---|---|---|---|---|---|")
    print(f"| 0 | — (45 A + 15 V) | {r0['n_features']['VA']} | "
          f"{f(r0['VA']['va_nl_monitor'])} | — | "
          f"{f(r0['Delta_beyond_monitor'], signed=True)} | "
          f"{f(r0['Delta_beyond_population'], signed=True)} |")
    for r in sorted(res):
        d = res[r]
        print(f"| {r} | +{d['n_A_kept']} A / +{d['n_B_kept']} B | "
              f"{d['bank_size_after']} | {f(d['curr']['va_nl_monitor'])} | "
              f"{f(d['Delta_VA_nl_monitor'], signed=True)} | "
              f"{f(d['Delta_beyond_monitor'], signed=True)} | "
              f"{f(d['Delta_beyond_population'], signed=True)} |")

    print("\n### Per-round instrument bookkeeping\n")
    print("| round | scored | collapsed | misrouting | probe gate | anchor pos/neg AUC "
          "| coherent-vs-scrambled | sign-contradicting A | ΔC₊ | ΔC₋ |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(res):
        d = res[r]
        rt = json.loads((HERE / f"round{r}_routing.json").read_text())
        ab = d["anchor_battery"]
        print(f"| {r} | {d['n_scored']} | {d['n_collapsed']} | "
              f"{rt['misrouting_rate']*100:.0f}% | "
              f"{'PASS' if rt['PROBE_GATE_PASS'] else 'FAIL'} | "
              f"{f(ab['pos_vs_neg_auc'],3)} | {f(ab['coherent_vs_scrambled_auc'],3)} | "
              f"{len(d['sign_contradicting_A'])} | "
              f"{f(d['swap']['dC_plus'],4,True)} | {f(d['swap']['dC_minus'],4,True)} |")

    print("\n### Track-B discount band (FREEZE ADDENDUM 2: MIXED channels in both)\n")
    print("| round | nuisance cols | mixed | upstream-traced | spurious-alone (lin/gb) "
          "| estimator | T_adj | VA_adj | Δ_adj (full set) | Δ_adj (strict, no MIXED) "
          "| Δ undiscounted |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(res):
        d = res[r]["discount"]
        if not d.get("n_nuisance_cumulative"):
            continue
        strict = d.get("discount_band", {}).get("strict_set_no_mixed", {})
        print(f"| {r} | {d['n_nuisance_cumulative']} | {d.get('n_mixed','—')} | "
              f"{d.get('n_upstream_traced','—')} | "
              f"{f(d.get('spurious_alone_linear'),3)}/{f(d.get('spurious_alone_histgb'),3)} | "
              f"{d.get('estimator','—')} | {f(d.get('T_adj'))} | {f(d.get('VA_adj'))} | "
              f"{f(d.get('Delta_adj'), signed=True)} | "
              f"{f(strict.get('Delta_adj'), signed=True)} | "
              f"{f(d.get('undiscounted',{}).get('Delta'), signed=True)} |")

    print("\n### Stacked increment (stratification-free control)\n")
    print("| round | AUC(joint B) | AUC(B + dense) | dense increment over all named "
          "channels | AUC(bank) | AUC(bank + dense) | dense increment over bank |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(res):
        s = res[r]["discount"].get("stacked_increment")
        if not s:
            continue
        print(f"| {r} | {f(s['AUC_joint_B'])} | {f(s['AUC_B_plus_dense'])} | "
              f"{f(s['dense_increment_over_all_named_channels'], signed=True)} | "
              f"{f(s['AUC_bank'])} | {f(s['AUC_bank_plus_dense'])} | "
              f"{f(s['dense_increment_over_bank'], signed=True)} |")

    print("\n### Spurious map — per-channel alone-AUC (population), upstream parent, MIXED\n")
    print("| channel | alone-AUC | upstream parent | MIXED | first seen |")
    print("|---|---|---|---|---|")
    seen = set()
    for r in sorted(res):
        d = res[r]["discount"]
        pc = d.get("per_channel_alone_auc", {})
        ups = d.get("upstream_parents", {})
        mixed = set(d.get("mixed_channels", []))
        for name, auc in sorted(pc.items(), key=lambda kv: -(kv[1] or 0.5)):
            if name in seen:
                continue
            seen.add(name)
            print(f"| {name} | {f(auc,3)} | {ups.get(name,'—')} | "
                  f"{'yes' if name in mixed else 'no'} | r{r} |")

    mm = HERE / "missing_mass.json"
    if mm.exists():
        m = json.loads(mm.read_text())
        print("\n### Fleet missing mass (Good-Turing, blind full-recall species)\n")
        print("| track | round | P | families | N | S_obs | f1 | f2 | M̂ | jackknife M̂ "
              "[min,max] | cross-proposer recapture | remaining AUC (odds form) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for track in ("A", "B"):
            for k, s in sorted(m["tracks"].get(track, {}).items()):
                jk = s.get("jackknife_leave1out", {}).get("missing_mass", {})
                print(f"| {track} | {k} | {s['P']} | {s['n_families']} | "
                      f"{s['N_proposals']} | {s['S_obs']} | {s['f1']} | {s['f2']} | "
                      f"{f(s['good_turing_missing_mass'],3)} | "
                      f"[{f(jk.get('min'),3)}, {f(jk.get('max'),3)}] | "
                      f"{f(s['cross_proposer_recapture_rate'],2)} | "
                      f"{f(s.get('remaining_auc_odds_form'),4,True)} |")


if __name__ == "__main__":
    main()
