#!/usr/bin/env python3
"""Render the per-cell markdown sections for the humor map batch.

Usage: python report.py > sections.md
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = ["hashtagwars_verdict", "style_inv_toptier"]
TITLE = {"hashtagwars_verdict": "HashtagWars humor verdict (top-10-or-winner)",
         "style_inv_toptier": "Style Invitational top-tier (winner/runner-up)"}


def f(x, n=3):
    return "n/a" if x is None else f"{x:.{n}f}"


def sec(cell):
    out = []
    r0 = json.loads((HERE / f"{cell}_r0_context.json").read_text())
    out.append(f"## {cell} — {TITLE[cell]}\n")
    out.append(f"Population n={r0['n']}; HONEST (dense-held-out) n={r0['n_HONEST']}; "
               f"MONITOR n={r0['n_MONITOR']}; MONITOR_FULL n={r0['n_MONITOR_FULL']}.\n")
    T = r0["T"]
    out.append("### Round-0 baseline\n")
    out.append("| population | n | T (mean of seed AUCs) | T (seed ensemble) | VA_nl | VA_lin | Δ_beyond (ensemble T) |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    out.append(f"| HONEST | {T['HONEST']['n']} | {f(T['HONEST']['mean_of_seed_AUC'],4)} | "
               f"{f(T['HONEST']['ensemble'],4)} | {f(r0['VA_nl_HONEST'],4)} | "
               f"{f(r0['VA_lin_HONEST'],4)} | **{r0['Delta_beyond_HONEST_ensembleT']:+.4f}** |")
    out.append(f"| MONITOR | {T['MONITOR']['n']} | {f(T['MONITOR']['mean_of_seed_AUC'],4)} | "
               f"{f(T['MONITOR']['ensemble'],4)} | {f(r0['VA_nl_MONITOR'],4)} | "
               f"{f(r0['VA_lin_MONITOR'],4)} | {r0['Delta_beyond_MONITOR_ensembleT']:+.4f} |")
    out.append(f"| eval only (SELECTED-ON) | {T['eval_selected_on']['n']} | "
               f"{f(T['eval_selected_on']['mean_of_seed_AUC'],4)} | "
               f"{f(T['eval_selected_on']['ensemble'],4)} | — | — | — |")
    out.append(f"| test only (selection-free) | {T['test_selection_free']['n']} | "
               f"{f(T['test_selection_free']['mean_of_seed_AUC'],4)} | "
               f"{f(T['test_selection_free']['ensemble'],4)} | — | — | — |")
    ci = r0["ci_Delta_HONEST"]
    out.append(f"\nGroup-clustered bootstrap on Δ (HONEST): [{ci['lo']:+.4f}, {ci['hi']:+.4f}], "
               f"P(>0) = {ci['p_gt0']:.3f}. Layer-1 ledger VA_nl (pooled, all rows) = "
               f"{r0['layer1_ledger_VA_nl']:.4f}.\n")

    # decomposition
    p = HERE / f"{cell}_rd_results.json"
    if p.exists():
        d = json.loads(p.read_text())
        out.append("### Decomposition pass (FREEZE ADDENDUM 3), run before round 1\n")
        out.append("| parent bank criterion | P alone | R (real) alone | F (surface) alone | "
                   "P \\| F strata | R \\| F strata | ρ(R,F) | R route | F route |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
        for r in d["parents"]:
            a = r["alone_AUC"]
            s = r["stratified_on_surface_component_HONEST"]
            out.append(f"| {r['parent']} | {f(a['parent_HONEST'])} | {f(a['real_HONEST'])} | "
                       f"{f(a['surface_HONEST'])} | {f(s['parent']['AUC_adj'])} | "
                       f"{f(s['real']['AUC_adj'])} | "
                       f"{r['separability']['spearman_real_vs_surface']:+.2f} | "
                       f"{r['component_real']['final_route']} | "
                       f"{r['component_surface']['final_route']} |")
        bs = d["bank_sensitivity"]
        out.append("\n| bank state | features | VA_nl HONEST | VA_nl MONITOR | Δ_beyond HONEST |")
        out.append("|---|---:|---:|---:|---:|")
        for k, lab in (("frozen_layer1_bank", "frozen Layer-1 bank"),
                       ("bank_plus_A_routed_components", "bank + A-routed components"),
                       ("bank_minus_parents_plus_components", "bank − parents + components"),
                       ("bank_minus_parents_only", "bank − parents")):
            b = bs[k]
            dk = {"frozen_layer1_bank": "frozen",
                  "bank_plus_A_routed_components": "plus_components",
                  "bank_minus_parents_plus_components": "replaced",
                  "bank_minus_parents_only": "parents_dropped"}[k]
            out.append(f"| {lab} | {b['n_features']} | {f(b['VA_nl_HONEST'],4)} | "
                       f"{f(b['VA_nl_MONITOR'],4)} | {bs['Delta_beyond_HONEST'][dk]:+.4f} |")
        rp = d["score_report"]
        out.append(f"\nJudging: anchors coherent-vs-scrambled "
                   f"{rp['anchors']['coherent_vs_scrambled_auc']:.3f} "
                   f"({'PASS' if rp['anchors']['pass_scrambled'] else 'FAIL'}), pos-vs-neg "
                   f"{rp['anchors']['pos_vs_neg_auc']:.3f}, collapsed {rp['n_collapsed']}, "
                   f"NA {rp['overall_na_rate']:.4f}; routing misrouting "
                   f"{d['routing']['misrouting_rate']:.2f}, probes {d['routing']['probe_pass']}.\n")

    # position audit
    pa = HERE / f"{cell}_position_audit.json"
    if pa.exists():
        d = json.loads(pa.read_text())
        out.append("### Position-in-container audit (FREEZE ADDENDUM 4) — observed covariate\n")
        out.append("| variable | alone AUC (full) | alone AUC (HONEST) | ρ with T | ρ with VA_nl | direction |")
        out.append("|---|---:|---:|---:|---:|---|")
        for n, v in sorted(d["variables"].items(),
                           key=lambda kv: -abs(kv[1]["alone_AUC_full"] - .5)):
            out.append(f"| {n} | {v['alone_AUC_full']:.4f} | {v['alone_AUC_HONEST']:.4f} | "
                       f"{v['spearman_with_T_HONEST']:+.3f} | "
                       f"{v['spearman_with_VA_nl_HONEST']:+.3f} | {v['direction'][:90]} |")
        j = d["joint_position_model"]
        out.append(f"| **JOINT (grouped-OOF)** | **{j['grouped_OOF_AUC_full']:.4f}** | "
                   f"{j['AUC_HONEST']:.4f} | {j['spearman_with_T_HONEST']:+.3f} | "
                   f"{j['spearman_with_VA_nl_HONEST']:+.3f} | — |")
        u = d.get("joint_position_model_UPSTREAM_ONLY")
        if u:
            out.append(f"| **JOINT, upstream-only** | **{u['grouped_OOF_AUC_full']:.4f}** | "
                       f"{u['AUC_HONEST']:.4f} | — | — | {', '.join(u['variables'])} |")
        out.append("\n| container-size quintile | n | mean size | label rate |")
        out.append("|---:|---:|---:|---:|")
        for r in d["label_rate_by_container_size_quintile"]:
            out.append(f"| {r['quintile']+1} | {r['n']} | {r['mean_container_size']:.0f} | "
                       f"{r['label_rate']:.3f} |")
        for k in d:
            if k.startswith("Delta_stratified"):
                v = d[k]
                out.append(f"\n{k}: pooled Δ {v['pooled_Delta']:+.4f} → Δ_adj {v['Delta_adj']:+.4f} "
                           f"(T {v['pooled_T']:.4f}→{v['T_adj']:.4f}, "
                           f"VA {v['pooled_VA']:.4f}→{v['VA_adj']:.4f})")
        out.append("")

    # rounds
    for r in (1, 2):
        p = HERE / f"{cell}_r{r}_results.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        out.append(f"### Round {r} — spurious map (headline)\n")
        out.append("| channel | alone AUC (HONEST) | alone AUC (MONITOR) | upstream parent | mixed |")
        out.append("|---|---:|---:|---|:--:|")
        for c in d["spurious_map"]["channels"]:
            out.append(f"| {c['name']} | {c['alone_AUC_HONEST']:.3f} | "
                       f"{c['alone_AUC_MONITOR']:.3f} | {c['upstream_parent'][:60]} | "
                       f"{'YES' if c['mixed'] else ''} |")
        out.append("\n#### Discount\n")
        out.append("| readout | joint spurious-alone | T_adj | VA_adj | Δ_adj | pooled T | pooled VA | pooled Δ |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for key, lab in (("discount_ALL_B", "ALL B channels"),
                         ("discount_STRICT_no_mixed", "STRICT (mixed dropped)")):
            if key not in d:
                continue
            dd = d[key]
            for pop, q in (("stratified_HONEST_q10", "HONEST q10"),
                           ("stratified_MONITOR_q5", "MONITOR q5")):
                s = dd[pop]["joint_B_score"]
                out.append(f"| {lab}, {q} | {dd['spurious_alone_AUC_histgb_HONEST']:.3f} | "
                           f"{s['T_adj']:.3f} | {s['VA_adj']:.3f} | {s['Delta_adj']:+.3f} | "
                           f"{dd[pop]['pooled_T']:.3f} | {dd[pop]['pooled_VA']:.3f} | "
                           f"{dd[pop]['pooled_Delta']:+.3f} |")
        out.append("\n#### Stacked increment\n")
        out.append("| population | AUC(B) | AUC(dense) | AUC(bank) | dense increment | p | bank increment | p |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for k, lab in (("stacked_increment_HONEST", "HONEST"),
                       ("stacked_increment_MONITOR", "MONITOR")):
            s = d[k]
            out.append(f"| {lab} (n={s['n']}) | {s['AUC_jointB']:.3f} | {s['AUC_dense']:.3f} | "
                       f"{s['AUC_bank_VA_nl']:.3f} | **{s['dense_increment_over_B']:+.4f}** | "
                       f"{f(s['ci_dense_increment']['p_gt0'],2)} | "
                       f"**{s['bank_increment_over_B']:+.4f}** | "
                       f"{f(s['ci_bank_increment']['p_gt0'],2)} |")
        ta = d["track_A"]
        out.append("\n#### Track A (secondary)\n")
        out.append("| state | features | VA_nl MONITOR | VA_nl HONEST | gain MONITOR | gain HONEST | gain CI (HONEST) | Δ_beyond HONEST |")
        out.append("|---|---:|---:|---:|---:|---:|---|---:|")
        prev = ta["state_entering_round"]
        new = ta[f"state_after_round_{r}"]
        out.append(f"| entering r{r} | {prev['n_features']} | {prev['VA_nl_MONITOR']:.4f} | "
                   f"{prev['VA_nl_HONEST']:.4f} | — | — | — | "
                   f"{ta['Delta_beyond_HONEST_prev']:+.4f} |")
        g = ta["gain_ci_HONEST"]
        out.append(f"| after r{r} | {new['n_features']} | {new['VA_nl_MONITOR']:.4f} | "
                   f"{new['VA_nl_HONEST']:.4f} | {ta['gain_MONITOR']:+.4f} | "
                   f"{ta['gain_HONEST']:+.4f} | [{g['lo']:+.4f}, {g['hi']:+.4f}] p={g['p_gt0']:.2f} | "
                   f"**{ta['Delta_beyond_HONEST_new']:+.4f}** |")
        if "missing_mass" in d:
            out.append("\n#### Missing mass\n")
            out.append("| track | N | P | families | S_obs | f1 | f2 | Good-Turing M̂ | LOPO | recapture |")
            out.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
            for t, m in d["missing_mass"].items():
                jk = m["jackknife_LOPO_missing_mass"]
                out.append(f"| {t} | {m['N_proposals']} | {m['P']} | {m['n_families']} | "
                           f"{m['S_obs']} | {m['f1']} | {m['f2']} | "
                           f"**{m['good_turing_missing_mass']:.3f}** | "
                           f"[{f(jk['min'])}, {f(jk['max'])}] | "
                           f"{m['cross_proposer_recapture']:.2f} |")
        sr = d.get("score_report")
        if sr:
            out.append(f"\nRound {r} bookkeeping: routing A={d['routing']['n_A']} / "
                       f"B={d['routing']['n_B']} (mixed {d['routing']['n_mixed_B']}), "
                       f"misrouting {d['routing']['misrouting_rate']:.2f}, probes "
                       f"{d['routing']['probe_pass']}; anchors coherent-vs-scrambled "
                       f"{sr['anchors']['coherent_vs_scrambled_auc']:.3f} "
                       f"({'PASS' if sr['anchors']['pass_scrambled'] else 'FAIL'}), pos-vs-neg "
                       f"{sr['anchors']['pos_vs_neg_auc']:.3f}, collapsed {sr['n_collapsed']}, "
                       f"NA {sr['overall_na_rate']:.4f}.\n")
    return "\n".join(out)


if __name__ == "__main__":
    print("<!-- generated by maps_hw_si/report.py -->\n")
    for c in CELLS:
        print(sec(c))
        print()
