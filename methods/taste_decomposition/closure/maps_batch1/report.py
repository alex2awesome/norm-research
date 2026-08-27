#!/usr/bin/env python3
"""Render the per-cell markdown sections of notes/2026-08-06__spurious_maps_batch1.md
from the round result JSONs.  Keeps the note mechanically consistent with the
artifacts instead of hand-transcribed.

Usage: python report.py > sections.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = ["peer_curation", "peer_revealed", "cap_crowd", "cap_finalist",
         "nc_outcome", "nc_agree"]
TITLE = {
    "peer_curation": "peer curation (ICLR oral/spotlight selection)",
    "peer_revealed": "peer revealed (citation percentile)",
    "cap_crowd": "humor caption crowd-C (median split, votes>=100)",
    "cap_finalist": "humor caption finalist-B (finalist vs hard negative)",
    "nc_outcome": "N&C outcome (agency response outcome)",
    "nc_agree": "N&C agree (agency agrees vs disagrees)",
}


def f(x, n=3, sign=False):
    if x is None:
        return "n/a"
    try:
        return (f"{x:+.{n}f}" if sign else f"{x:.{n}f}")
    except (TypeError, ValueError):
        return str(x)


def load(cell, rnd):
    p = HERE / f"{cell}_r{rnd}_results.json"
    return json.loads(p.read_text()) if p.exists() else None


def spurious_map_table(res, top=None):
    rows = res["spurious_map"]["channels"]
    if top:
        rows = rows[:top]
    out = ["| channel | alone AUC (HONEST) | alone AUC (MONITOR) | upstream parent | mixed |",
           "|---|---:|---:|---|:--:|"]
    for c in rows:
        out.append(f"| {c['name']} | {f(c['alone_AUC_HONEST'])} | "
                   f"{f(c['alone_AUC_MONITOR'])} | {c['upstream_parent']} | "
                   f"{'YES' if c['mixed'] else ''} |")
    return "\n".join(out)


def discount_table(res):
    out = ["| readout | spurious-alone AUC (joint) | T_adj | VA_adj | Δ_adj | pooled T | pooled VA | pooled Δ |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, label in (("discount_ALL_B", "ALL B channels"),
                       ("discount_STRICT_no_mixed", "STRICT (mixed dropped)")):
        d = res.get(key)
        if not d:
            continue
        for pop, tag in (("stratified_HONEST_q10", "HONEST q10"),
                         ("stratified_MONITOR_q5", "MONITOR q5")):
            s = d[pop]
            j = s["joint_B_score"]
            out.append(f"| {label}, {tag} | {f(d['spurious_alone_AUC_histgb_HONEST'])} | "
                       f"{f(j['T_adj'])} | {f(j['VA_adj'])} | {f(j['Delta_adj'], sign=True)} | "
                       f"{f(s['pooled_T'])} | {f(s['pooled_VA'])} | {f(s['pooled_Delta'], sign=True)} |")
    return "\n".join(out)


def stack_table(res):
    out = ["| population | AUC(B) | AUC(dense) | AUC(bank) | AUC(B+dense) | dense increment | AUC(B+bank) | bank increment |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, lab in (("stacked_increment_HONEST", "HONEST"),
                   ("stacked_increment_MONITOR", "MONITOR")):
        s = res[k]
        out.append(f"| {lab} (n={s['n']}) | {f(s['AUC_jointB'])} | {f(s['AUC_dense'])} | "
                   f"{f(s['AUC_bank_VA_nl'])} | {f(s['AUC_stack_B_plus_dense'])} | "
                   f"{f(s['dense_increment_over_B'], sign=True)} | "
                   f"{f(s['AUC_stack_B_plus_bank'])} | "
                   f"{f(s['bank_increment_over_B'], sign=True)} |")
    return "\n".join(out)


def track_a_table(rounds):
    out = ["| round | bank features | VA_nl MONITOR | VA_nl HONEST | gain MONITOR | gain HONEST | gain CI (HONEST) | Δ_beyond HONEST |",
           "|---|---:|---:|---:|---:|---:|---|---:|"]
    for r, res in rounds:
        ta = res["track_A"]
        cur = ta[f"round{r}"]
        out.append(f"| r{r} | {cur['n_features']} | {f(cur['VA_nl_MONITOR'])} | "
                   f"{f(cur['VA_nl_HONEST'])} | {f(ta['gain_MONITOR'], sign=True)} | "
                   f"{f(ta['gain_HONEST'], sign=True)} | "
                   f"[{f(ta['gain_ci_HONEST']['lo'], sign=True)}, "
                   f"{f(ta['gain_ci_HONEST']['hi'], sign=True)}] "
                   f"p={f(ta['gain_ci_HONEST']['p_gt0'], 2)} | "
                   f"{f(ta['Delta_beyond_HONEST_new'], sign=True)} |")
    return "\n".join(out)


def mm_table(rounds):
    out = ["| round | track | N | P | families | S_obs | f1 | f2 | Good-Turing M̂ | LOPO jackknife | cross-proposer recapture |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for r, res in rounds:
        for t in ("A", "B"):
            g = res.get("missing_mass", {}).get(t)
            if not g:
                continue
            jk = g["jackknife_LOPO_missing_mass"]
            out.append(f"| r{r} | {t} | {g['N_proposals']} | {g['P']} | {g['n_families']} | "
                       f"{g['S_obs']} | {g['f1']} | {g['f2']} | "
                       f"{f(g['good_turing_missing_mass'])} | "
                       f"[{f(jk['min'])}, {f(jk['max'])}] | "
                       f"{f(g['cross_proposer_recapture'], 2)} |")
    return "\n".join(out)


def main():
    r0all = json.loads((HERE / "round0_context_all.json").read_text())
    print("<!-- generated by maps_batch1/report.py -->")
    for cell in CELLS:
        rounds = [(r, load(cell, r)) for r in (1, 2)]
        rounds = [(r, x) for r, x in rounds if x]
        if not rounds:
            continue
        r0 = r0all.get(cell, {})
        last = rounds[-1][1]
        print(f"\n## {cell} — {TITLE[cell]}\n")
        print(f"Population n={r0.get('n')}; HONEST (dense-held-out) n={r0.get('n_HONEST')}; "
              f"MONITOR n={r0.get('n_MONITOR')} ({last['splits']['n_groups_monitor']} groups). "
              f"Round-0: T {f(r0.get('T_HONEST'))} / VA_nl {f(r0.get('VA_nl_HONEST'))} / "
              f"**Δ_beyond {f(r0.get('Delta_beyond_HONEST'), sign=True)}** on HONEST; "
              f"T {f(r0.get('T_MONITOR'))} / VA_nl {f(r0.get('VA_nl_MONITOR'))} / "
              f"Δ {f(r0.get('Delta_beyond_MONITOR'), sign=True)} on MONITOR.\n")
        print("### Spurious map (headline)\n")
        print(spurious_map_table(last))
        sm = last["spurious_map"]
        if sm["n_dropped_by_degeneracy_screen"]:
            print(f"\n({sm['n_dropped_by_degeneracy_screen']} channel(s) dropped by the "
                  f"FIT+MINE degeneracy screen: {', '.join(sm['dropped_ids'])})")
        print("\n### Discount\n")
        print(discount_table(last))
        print("\n### Stacked increment\n")
        print(stack_table(last))
        print("\n### Track A (secondary)\n")
        print(track_a_table(rounds))
        print("\n### Missing mass (both tracks)\n")
        print(mm_table(rounds))
        print("\n### Round bookkeeping\n")
        for r, res in rounds:
            sr = res.get("score_report") or {}
            anc = sr.get("anchors", {})
            print(f"- r{r}: routing A={res['routing']['n_A']} / B={res['routing']['n_B']} "
                  f"(mixed {res['routing']['n_mixed_B']}), misrouting "
                  f"{f(res['routing']['misrouting_rate'], 2)}, planted probes "
                  f"{res['routing']['probe_pass']}; anchors pos-vs-neg AUC "
                  f"{f(anc.get('pos_vs_neg_auc'))}, coherent-vs-scrambled "
                  f"{f(anc.get('coherent_vs_scrambled_auc'))} "
                  f"({'PASS' if anc.get('pass_scrambled') else 'FAIL'}), "
                  f"collapsed criteria {sr.get('n_collapsed')}, NA rate "
                  f"{f(sr.get('overall_na_rate'))}")


if __name__ == "__main__":
    sys.exit(main())
