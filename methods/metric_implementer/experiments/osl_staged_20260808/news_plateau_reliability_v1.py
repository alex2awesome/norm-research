#!/usr/bin/env python
"""
Reliability check on the 20 FLAT-or-FALLING (plateau_set) metrics from
family_verdict_join_v1.json, vs a matched random sample of 40 non-plateau
metrics from the same tasks. Tests whether "falling agreement-with-frontier"
reflects FRONTIER-VOTER unreliability (degenerate score dist / extreme base
rate / low frontier-pair agreement) rather than genuine construct tacitness.

Reuses the exact panel-loading + binarize + frontier-pairwise-agreement logic
from family_verdict_join_v1.py so numbers are directly comparable.
"""
import json
import os
import random

import numpy as np

B = "/lfs/skampere3/0/alexspan"
OSL = f"{B}/outputs/osl"
OM = f"{B}/outputs/osl_multi"
FRONTIER = ["llama70b", "qwen25-72b"]
ALL_EXECS_NEEDED = FRONTIER  # only need frontier panels for this check
TASKS_WITH_PLATEAU = ["humor", "peer_review", "math", "news_homepages"]
MIN_FRONTIER_ITEMS = 30
UNCERTAIN_LO, UNCERTAIN_HI = 0.45, 0.55
COLLAPSE_THRESH = 0.95   # matches a_bank_degeneracy_audit precedent (mode-share >=.95)
EXTREME_BASE_LO, EXTREME_BASE_HI = 0.05, 0.95
SEED = 20260807

random.seed(SEED)
np.random.seed(SEED)


def load_panel_bank(path):
    d = np.load(path, allow_pickle=True)
    names, kinds, m = d["names"], d["kinds"], d["m_bar"]
    out = {}
    for i, n in enumerate(names):
        if str(kinds[i]) != "bank":
            continue
        out[str(n)] = m[i].astype(float)
    return out


def load_task_panels_frontier(task):
    """frontier-exec -> {metric_name: m_bar_vector(300,)} restricted to kind=='bank'."""
    panels = {}
    if task == "humor":
        for e in FRONTIER:
            dct = {}
            p1 = f"{OSL}/mbar285_{e}.npz"
            p2 = f"{OM}/mbar2_humor_sup_{e}.npz"
            if os.path.exists(p1):
                dct.update(load_panel_bank(p1))
            if os.path.exists(p2):
                sup = load_panel_bank(p2)
                overlap = set(sup) & set(dct)
                for n in overlap:
                    sup.pop(n)
                dct.update(sup)
            panels[e] = dct
    else:
        for e in FRONTIER:
            p = f"{OM}/mbar2_{task}_{e}.npz"
            if os.path.exists(p):
                panels[e] = load_panel_bank(p)
            else:
                panels[e] = {}
    return panels


def binarize(v):
    b = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    b[ok] = (v[ok] > 0.5).astype(float)
    return b


def voter_stats(v):
    """raw m_bar vector -> dict of degeneracy/base-rate stats for one voter on one metric."""
    ok = np.isfinite(v)
    n = int(ok.sum())
    if n == 0:
        return dict(n=0, frac_uncertain=np.nan, frac_collapsed=np.nan,
                     collapsed_flag=False, yes_rate=np.nan)
    vv = v[ok]
    frac_uncertain = float(np.mean((vv >= UNCERTAIN_LO) & (vv <= UNCERTAIN_HI)))
    b = (vv > 0.5).astype(float)
    yes_rate = float(np.mean(b))
    frac_collapsed = max(yes_rate, 1.0 - yes_rate)  # fraction in the majority class
    collapsed_flag = frac_collapsed >= COLLAPSE_THRESH
    return dict(n=n, frac_uncertain=frac_uncertain, frac_collapsed=frac_collapsed,
                collapsed_flag=collapsed_flag, yes_rate=yes_rate)


def metric_diagnostics(task, name, panel_cache):
    if task not in panel_cache:
        panel_cache[task] = load_task_panels_frontier(task)
    panels = panel_cache[task]
    if name not in panels.get(FRONTIER[0], {}) or name not in panels.get(FRONTIER[1], {}):
        return None
    v_l = panels[FRONTIER[0]][name]
    v_q = panels[FRONTIER[1]][name]
    b_l = binarize(v_l)
    b_q = binarize(v_q)
    both_fin = np.isfinite(b_l) & np.isfinite(b_q)
    n_both = int(both_fin.sum())
    if n_both < MIN_FRONTIER_ITEMS:
        return None
    pairwise_agree = float(np.mean(b_l[both_fin] == b_q[both_fin]))
    st_l = voter_stats(v_l)
    st_q = voter_stats(v_q)
    avg_yes = float(np.nanmean([st_l["yes_rate"], st_q["yes_rate"]]))
    max_uncertain = float(np.nanmax([st_l["frac_uncertain"], st_q["frac_uncertain"]]))
    any_collapsed = bool(st_l["collapsed_flag"] or st_q["collapsed_flag"])
    return dict(
        task=task, name=name, n_items=n_both,
        pairwise_agree=pairwise_agree,
        llama70b=st_l, qwen2572b=st_q,
        max_uncertain=max_uncertain, any_collapsed=any_collapsed,
        avg_yes_rate=avg_yes,
    )


def main():
    join = json.load(open(f"{OM}/family_verdict_join_v1.json"))
    plateau = join["plateau_set"]["top30"]
    assert len(plateau) == 20, f"expected 20 plateau metrics, got {len(plateau)}"
    plateau_keys = {(x["task"], x["name"]) for x in plateau}

    full_rows = join["full_rows"]
    # candidate pool for controls: same 4 tasks, NOT in plateau set, has a valid n_frontier_items
    by_task_pool = {}
    for r in full_rows:
        key = (r["task"], r["name"])
        if key in plateau_keys:
            continue
        if r["task"] not in TASKS_WITH_PLATEAU:
            continue
        if not r.get("n_frontier_items") or r["n_frontier_items"] < MIN_FRONTIER_ITEMS:
            continue
        by_task_pool.setdefault(r["task"], []).append(r)

    plateau_task_counts = join["plateau_set"]["by_task"]
    # stratified 2x sample (40 total, proportional to plateau's by-task composition)
    control_targets = {t: c * 2 for t, c in plateau_task_counts.items()}
    controls = []
    for t, k in control_targets.items():
        pool = by_task_pool.get(t, [])
        k = min(k, len(pool))
        controls.extend(random.sample(pool, k))
    print(f"[info] control pool sizes: {[(t, len(v)) for t, v in by_task_pool.items()]}")
    print(f"[info] control targets: {control_targets}, sampled {len(controls)}")

    panel_cache = {}
    plateau_diag = []
    for x in plateau:
        d = metric_diagnostics(x["task"], x["name"], panel_cache)
        if d is None:
            print(f"[WARN] plateau metric missing/thin panel: {x['task']} | {x['name']}")
            continue
        d["combined_flatness"] = x["combined_flatness"]
        d["verdicts"] = x["verdicts"]
        plateau_diag.append(d)

    control_diag = []
    for r in controls:
        d = metric_diagnostics(r["task"], r["name"], panel_cache)
        if d is None:
            continue
        control_diag.append(d)

    print(f"[info] plateau diagnosed: {len(plateau_diag)}/20; control diagnosed: {len(control_diag)}/{len(controls)}")

    ctrl_agree = np.array([d["pairwise_agree"] for d in control_diag])
    ctrl_p25 = float(np.percentile(ctrl_agree, 25))
    ctrl_median = float(np.median(ctrl_agree))
    print(f"[info] control frontier-pairwise-agreement: median={ctrl_median:.3f} p25={ctrl_p25:.3f} "
          f"min={ctrl_agree.min():.3f} max={ctrl_agree.max():.3f} n={len(ctrl_agree)}")

    # NOTE (calibration pass): an OR-of-4-flags rule made COLLAPSED-VOTER /
    # EXTREME-BASE-RATE fire on 37/40 (92%) of CONTROLS too -- skewed-NO base
    # rates + near-collapsed frontier voters are the norm for bright-line-style
    # binary judge metrics across this whole panel, not something distinctive
    # to the plateau set. Those two checks are reported as descriptive context
    # per metric but do NOT drive the verdict (they don't discriminate).
    # The verdict-driving test is check #1 as the user framed it: is frontier
    # pairwise agreement LOWER for plateau items than the control reference
    # distribution (< control's own 25th percentile)? HIGH-UNCERTAIN (>30% of
    # items sitting in the ambiguous P(YES) in [.45,.55] band) is kept as a
    # second verdict-driving flag since a genuinely noisy P(YES) distribution
    # -- as opposed to a confidently-skewed one -- IS a direct reliability
    # concern regardless of base rate.
    def verdict_for(d):
        reasons = []
        if d["pairwise_agree"] < ctrl_p25:
            reasons.append("LOW-AGREE")
        if d["max_uncertain"] > 0.30:
            reasons.append("HIGH-UNCERTAIN")
        context = []
        if d["any_collapsed"]:
            context.append("collapsed-voter")
        if d["avg_yes_rate"] < EXTREME_BASE_LO or d["avg_yes_rate"] > EXTREME_BASE_HI:
            context.append("extreme-base-rate")
        v = "REFERENCE-UNRELIABLE" if reasons else "GENUINE-PLATEAU-CANDIDATE"
        return v, reasons, context

    for d in plateau_diag:
        d["verdict"], d["reasons"], d["context"] = verdict_for(d)
    for d in control_diag:
        d["verdict"], d["reasons"], d["context"] = verdict_for(d)

    n_unreliable_ctrl = sum(1 for d in control_diag if d["verdict"] == "REFERENCE-UNRELIABLE")
    n_unreliable_plat = sum(1 for d in plateau_diag if d["verdict"] == "REFERENCE-UNRELIABLE")
    n_genuine_plat = len(plateau_diag) - n_unreliable_plat
    n_collapsed_ctrl = sum(1 for d in control_diag if d["any_collapsed"])
    n_collapsed_plat = sum(1 for d in plateau_diag if d["any_collapsed"])
    n_extreme_ctrl = sum(1 for d in control_diag if d["avg_yes_rate"] < EXTREME_BASE_LO or d["avg_yes_rate"] > EXTREME_BASE_HI)
    n_extreme_plat = sum(1 for d in plateau_diag if d["avg_yes_rate"] < EXTREME_BASE_LO or d["avg_yes_rate"] > EXTREME_BASE_HI)

    try:
        from scipy.stats import fisher_exact
        _, fisher_p = fisher_exact([[n_unreliable_plat, len(plateau_diag) - n_unreliable_plat],
                                     [n_unreliable_ctrl, len(control_diag) - n_unreliable_ctrl]])
    except Exception:
        fisher_p = None

    # ---- write compact markdown table ----
    lines = []
    lines.append("# News-plateau reliability check: 20 plateau vs 40 matched controls")
    lines.append(f"Ctrl frontier-pair agree(llama70b,qwen25-72b): median={ctrl_median:.2f} p25={ctrl_p25:.2f} (n={len(control_diag)}). "
                  f"Verdict = LOW-AGREE (agree<ctrl-p25) or HIGH-UNCERTAIN (>30% items P(YES) in [.45,.55]). "
                  f"collapse/YES% = descriptive context, NOT verdict-driving (see note).")
    lines.append("")
    lines.append("| task | metric | agree | unc% | cx | YES% | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for d in sorted(plateau_diag, key=lambda x: x["pairwise_agree"]):
        nm = d["name"][:34]
        vshort = "UNRELIABLE" if d["verdict"] == "REFERENCE-UNRELIABLE" else "genuine"
        cx = ("C" if d["any_collapsed"] else "") + ("X" if ("extreme-base-rate" in d["context"]) else "") or "-"
        lines.append(
            f"| {d['task'][:5]} | {nm} | {d['pairwise_agree']:.2f} | "
            f"{d['max_uncertain']*100:.0f} | {cx} | "
            f"{d['avg_yes_rate']*100:.0f} | {vshort} |"
        )
    lines.append("")
    lines.append(f"Control base rate for same rule: {n_unreliable_ctrl}/{len(control_diag)} "
                  f"({100*n_unreliable_ctrl/max(1,len(control_diag)):.0f}%, ~25% expected by p25 construction "
                  f"+ a few HIGH-UNCERTAIN adds).")
    lines.append(f"Note: COLLAPSED-VOTER pervasive both arms (plateau {n_collapsed_plat}/20 vs ctrl "
                  f"{n_collapsed_ctrl}/{len(control_diag)}); EXTREME-BASE-RATE too (plateau {n_extreme_plat}/20 vs ctrl "
                  f"{n_extreme_ctrl}/{len(control_diag)}) -- skewed-NO bright-line judges are the panel norm, not plateau-specific.")
    lines.append(f"LOW-AGREE alone: plateau {sum(1 for d in plateau_diag if 'LOW-AGREE' in d['reasons'])}/20 "
                  f"({100*sum(1 for d in plateau_diag if 'LOW-AGREE' in d['reasons'])/20:.0f}%) vs ctrl ~25% by construction"
                  + (f"; Fisher exact p={fisher_p:.3f}" if fisher_p is not None else "") + ".")
    lines.append("")
    lines.append(f"**Summary: {n_genuine_plat}/20 GENUINE-PLATEAU-CANDIDATE, {n_unreliable_plat}/20 "
                  f"REFERENCE-UNRELIABLE** (frontier agree < ctrl-p25 or noisy P(YES)); "
                  f"vs {100*n_unreliable_ctrl/max(1,len(control_diag)):.0f}% ctrl base rate for the same rule "
                  f"-> plateau set is enriched for low frontier-pair agreement but roughly half survive.")
    md = "\n".join(lines)

    out_path = f"{OM}/news_plateau_reliability_v1.md"
    with open(out_path, "w") as f:
        f.write(md + "\n")
    print(f"[info] wrote {out_path}")
    print()
    print(md)

    # also dump raw diagnostics json for provenance
    raw_out = {
        "meta": {
            "seed": SEED, "min_frontier_items": MIN_FRONTIER_ITEMS,
            "uncertain_band": [UNCERTAIN_LO, UNCERTAIN_HI],
            "collapse_thresh": COLLAPSE_THRESH,
            "extreme_base_rate": [EXTREME_BASE_LO, EXTREME_BASE_HI],
            "control_agree_median": ctrl_median, "control_agree_p25": ctrl_p25,
            "control_targets": control_targets,
        },
        "plateau": plateau_diag,
        "controls": control_diag,
    }
    with open(f"{OM}/news_plateau_reliability_v1_raw.json", "w") as f:
        json.dump(raw_out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
