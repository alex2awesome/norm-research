"""D4 — optimality MARGIN: does the best achieved prompt sit strictly below a projected ceiling?

Run with the REPO python (needs scipy), not the folder venv:  python3 analyze_bounds.py
Estimators are imported from the main repo (code import only; DATA never crosses the boundary —
every input and output stays inside datasets/prompt-optimality-test/).

WHAT THIS COMPUTES, per dataset, on the POOLED candidate pool (all three arms' rescored prompts,
deduped by the "hash" field — two resumable rescorers may have double-appended, so the same dedupe
rule as analyze.py is applied):
  (1) best ACHIEVED      = max over candidates of val accuracy (max row mean of the candidate x item
                           binary matrix).
  (2) projected POOL CEILING = y_inf from fit_saturating on the exchangeable best-of-m curve
                           E[max val-mean among m candidates drawn WITHOUT replacement], with a
                           CANDIDATE-level bootstrap CI (resample candidates with replacement,
                           rebuild the curve, refit).
  (3) ORACLE within-pool cap = union-of-all coverage, the fraction of val items solved by >=1
                           candidate in the pool. No single prompt can exceed this.
  (4) the ordering check  best_achieved < y_inf < union_all, and specifically whether the LOWER edge
                           of the y_inf bootstrap CI exceeds best_achieved (the "strictly larger"
                           claim).

=========================== HONESTY / SCOPE — READ BEFORE QUOTING ===========================
(i)   ALL bounds here are CONDITIONAL ON THE CANDIDATE POOL. They bound what THIS pool of prompts,
      searched by THIS process, can reach. They are NOT bounds over prompt space. A different
      seed, proposer, or budget produces a different pool and different numbers.
(ii)  best-of-m is a SELECTION statistic and is monotonically increasing in m by construction, so
      the saturating fit's y_inf is an extrapolated ceiling OF THE SAME SELECTION PROCESS run
      longer on the same pool — not a supremum over prompts.
(iii) NOTHING here is "certified". No quantity in this file is a certificate. The only certified
      all-prompt bound style in this project is the DPI fixed-target cap (see memory
      project_momega_audit_bracket); this is a pool-conditional projection, a strictly weaker object.

STRUCTURAL RESULT FOUND WHEN THIS WAS RUN (see runs/bounds_summary.md): for an exchangeable
best-of-m statistic over a FIXED FINITE pool, sup_m E[max among m] = max over the pool = the best
ACHIEVED value, attained exactly at m = n. The best-of-m curve therefore terminates at
best_achieved, and no monotone fit to it can project strictly above best_achieved. The requested
middle term of the ordering is unattainable by construction, not by empirical accident. Two things
are reported so a reader can see this rather than take it on faith:
  * the literal computation (fit_saturating on the raw curve), which lands BELOW best_achieved and
    is additionally MISSPECIFIED: fit_saturating imposes y = y_inf*(1 - e^{-m/tau}), forced through
    y(0) = 0, while a best-of-m curve has a large floor at m = 1 (= the mean candidate value). The
    fit absorbs the floor by collapsing tau toward 0; R^2 falls to 0.30-0.59.
  * the same estimator applied to the EXCESS curve y(m) - y(1) over m - 1, which does start at 0 and
    so satisfies the estimator's structural assumption (R^2 rises to 0.87-0.94). Its ceiling still
    lands at/below best_achieved — confirming the finite-pool cap, not the misspecification, binds.
The only term that legitimately sits above best_achieved is the union/oracle cap, which is a
DIFFERENT object (the ceiling for a per-item selector, not for any single prompt) and is itself
inflated by multiple comparisons over noisy binary scoring — quantified in the md.
============================================================================================
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))                      # repo root, code-only import
from methods.metric_implementer.experiments import unseen_value_scaling as uvs  # noqa: E402

DATASETS = ("hover", "hotpotqa", "aime2025")
ARMS = ("official", "inhouse", "unitrecomb")

N_SUBSETS = 400          # subset draws per m grid point on the observed pool (>= 300 required)
N_SUBSETS_BOOT = 150     # subset draws per m grid point inside each bootstrap replicate
N_BOOT = 250             # candidate-level bootstrap replicates (>= 200 required)
SEED = 0


# ------------------------------------------------------------------------------------------
# pooling
# ------------------------------------------------------------------------------------------
def pool_candidates(ds: str) -> dict:
    """Pool all arms' rescored candidates, dedupe by hash (first arm in ARMS order wins).

    Hashes appearing in more than one arm are INDEPENDENT RESCORES of the same prompt text; their
    per-item disagreement is a direct measurement of scoring noise, kept for the noise floor.
    """
    rows, seen, repeats = [], {}, []
    for arm in ARMS:
        p = HERE / "runs" / ds / arm / "rescore.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            h = r["hash"]
            if h in seen:
                prev = seen[h]
                if prev["arm"] != arm:                       # independent rescore of same prompt
                    a, b = np.asarray(prev["scores"]), np.asarray(r["item_scores"], float)
                    repeats.append({"hash": h, "arms": [prev["arm"], arm],
                                    "val_means": [float(a.mean()), float(b.mean())],
                                    "per_item_disagreement": float((a != b).mean())})
                continue
            seen[h] = {"arm": arm, "scores": r["item_scores"]}
            rows.append({**r, "arm": arm})
    M = np.array([r["item_scores"] for r in rows], float)
    return {"rows": rows, "M": M, "repeats": repeats}


# ------------------------------------------------------------------------------------------
# curves
# ------------------------------------------------------------------------------------------
def _m_grid(n: int) -> np.ndarray:
    """Hybrid geometric+linear grid: resolution at small m (where best-of-m bends) and coverage of
    the tail (which is what the saturating extrapolation is actually driven by)."""
    g = np.concatenate([np.geomspace(1, n, 10), np.linspace(1, n, 8)])
    return np.unique(np.round(g).astype(int))


def curves(M: np.ndarray, means: np.ndarray, m_grid: np.ndarray, rng, n_subsets: int):
    """E[max val-mean among m candidates] and E[fraction of items solved by >=1 of m], both drawn
    uniformly WITHOUT replacement. At m = n only one subset exists, so both are deterministic."""
    n = len(means)
    best = np.empty(len(m_grid), float)
    union = np.empty(len(m_grid), float)
    for i, m in enumerate(m_grid):
        m = int(m)
        if m >= n:
            best[i], union[i] = means.max(), M.max(0).mean()
            continue
        keys = rng.random((n_subsets, n))
        idx = np.argpartition(keys, m - 1, axis=1)[:, :m]     # uniform m-subsets, no replacement
        best[i] = float(means[idx].max(axis=1).mean())
        union[i] = float(np.mean([M[j].max(0).mean() for j in idx]))
    return best, union


def _excess_ceiling(m_grid: np.ndarray, y: np.ndarray) -> dict:
    """fit_saturating on the EXCESS curve y(m) - y(1) vs (m - 1), which genuinely starts at 0 and so
    satisfies the estimator's y(0)=0 assumption. Ceiling = y(1) + y_inf_excess."""
    m2, y2 = m_grid.astype(float) - 1.0, y - y[0]
    keep = m2 > 0
    if keep.sum() < 3:
        return {"ok": False}
    f = uvs.fit_saturating(m2[keep], y2[keep], n_boot=0)
    if not f.get("ok"):
        return {"ok": False}
    return {"ok": True, "ceiling": float(y[0] + f["y_inf"]), "y_inf_excess": float(f["y_inf"]),
            "tau": float(f["tau"]), "r2": float(f["r2_linear"]), "floor_at_m1": float(y[0])}


def analyze_dataset(ds: str) -> dict:
    pool = pool_candidates(ds)
    M, rows = pool["M"], pool["rows"]
    n_cand, n_items = M.shape
    means = M.mean(1)
    rng = np.random.default_rng(SEED)
    m_grid = _m_grid(n_cand)

    best_curve, union_curve = curves(M, means, m_grid, rng, N_SUBSETS)
    fit = uvs.fit_saturating(m_grid.astype(float), best_curve, n_boot=0)
    forms = uvs.compare_scaling_forms(m_grid.astype(float), best_curve, n_boot=0)
    front = uvs.value_frontloading_stat(m_grid.astype(float), best_curve, m_grid.astype(float))
    excess = _excess_ceiling(m_grid, best_curve)

    # --- candidate-level bootstrap: resample the POOL with replacement, rebuild curve, refit ---
    boot_rng = np.random.default_rng(SEED + 1)
    y_inf_boot, tau_boot, exc_boot = [], [], []
    for _ in range(N_BOOT):
        bidx = boot_rng.integers(0, n_cand, n_cand)
        cb, _ = curves(M[bidx], means[bidx], m_grid, boot_rng, N_SUBSETS_BOOT)
        fb = uvs.fit_saturating(m_grid.astype(float), cb, n_boot=0)
        if fb.get("ok"):
            y_inf_boot.append(fb["y_inf"])
            tau_boot.append(fb["tau"])
        eb = _excess_ceiling(m_grid, cb)
        if eb.get("ok"):
            exc_boot.append(eb["ceiling"])
    pct = lambda v: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                     if v else [float("nan")] * 2)
    y_inf_ci, exc_ci = pct(y_inf_boot), pct(exc_boot)

    best_achieved = float(means.max())
    best_i = int(np.argmax(means))
    union_all = float(M.max(0).mean())
    y_inf = float(fit["y_inf"]) if fit.get("ok") else float("nan")
    tau = float(fit["tau"]) if fit.get("ok") else float("nan")

    # --- noise floor ---
    dis = [r["per_item_disagreement"] for r in pool["repeats"]]
    q_dis = float(np.mean(dis)) if dis else 0.0
    binom_se = float(np.sqrt(best_achieved * (1 - best_achieved) / n_items))
    rep_spread = [abs(r["val_means"][0] - r["val_means"][1]) for r in pool["repeats"]]
    noise_floor = max([binom_se] + rep_spread)

    # --- is the oracle cap's gain over best-achieved explainable by scoring noise alone? ---
    # Two independent rescores of one prompt disagree on an item with prob 2p(1-p), so a genuinely
    # hard item (small solve-probability p) has p ~ q_dis/2. The chance it LOOKS solved by at least
    # one of the other k candidates is then 1-(1-p)^k. FIRST-ORDER only: assumes independence across
    # candidates and a single flip rate, and q_dis is estimated from very few repeat pairs.
    n_miss = int((M[best_i] == 0).sum())
    complement = int(((M[best_i] == 0) & (M.max(0) == 1)).sum())
    p_eff = q_dis / 2.0
    exp_false = float(n_miss * (1.0 - (1.0 - p_eff) ** max(0, n_cand - 1)))

    ordering_holds = bool(best_achieved < y_inf < union_all)
    ci_lo_exceeds = bool(y_inf_ci[0] > best_achieved)
    margin = y_inf - best_achieved
    if ci_lo_exceeds and margin > noise_floor:
        verdict = "HOLDS"
    elif ci_lo_exceeds:
        verdict = "MARGINAL — CI lower edge clears best-achieved but the margin is inside the noise floor"
    else:
        verdict = "DOES NOT HOLD — CI lower edge does not exceed best-achieved"

    return {
        "dataset": ds, "n_candidates": n_cand, "n_items": n_items,
        "per_arm_counts": {a: sum(1 for r in rows if r["arm"] == a) for a in ARMS},
        "best_achieved": best_achieved,
        "best_candidate_arm": rows[best_i]["arm"], "best_candidate_hash": rows[best_i]["hash"],
        "m_grid": m_grid.tolist(), "best_of_m": best_curve.tolist(),
        "union_of_m": union_curve.tolist(),
        "y_inf": y_inf, "tau": tau, "sat_r2": fit.get("r2_linear"),
        "y_inf_boot_ci_2p5_97p5": y_inf_ci, "n_boot_ok": len(y_inf_boot),
        "tau_boot_median": float(np.median(tau_boot)) if tau_boot else None,
        "excess_form_ceiling": excess, "excess_form_ceiling_ci": exc_ci,
        "scaling_form_verdict": forms.get("verdict"),
        "scaling_form_note": forms.get("verdict_note"),
        "frontloading_D": front.get("D") if front.get("ok") else None,
        "frontloading_note": ("NOT INTERPRETABLE here: value_frontloading_stat normalizes by the "
                              "terminal value and assumes both curves start near 0; best-of-m starts "
                              "at the mean candidate value (a high floor), which forces D large and "
                              "positive regardless of shape."),
        "union_all": union_all, "sum_individual_cov": float(means.sum()),
        "ordering_best_lt_yinf_lt_union": ordering_holds,
        "ci_lower_exceeds_best_achieved": ci_lo_exceeds,
        "margin_yinf_minus_best": margin,
        "best_lt_union": bool(best_achieved < union_all),
        "oracle_gap": {"items_best_misses": n_miss, "items_recovered_by_pool": complement,
                       "gap_fraction": union_all - best_achieved,
                       "mean_repeat_disagreement_q": q_dis,
                       "expected_false_recoveries_from_noise": exp_false,
                       "read": ("oracle gap is within what independent-rescore noise alone would "
                                "produce — not evidence of complementary capability"
                                if exp_false >= complement else
                                "most of the oracle gap is consistent with rescore noise "
                                "(noise expectation covers over half the recovered items)"
                                if exp_false >= 0.5 * complement else
                                "oracle gap exceeds the first-order noise expectation")},
        "noise": {"per_item_disagreement": dis, "repeat_rescore_abs_val_mean_spread": rep_spread,
                  "binomial_se_at_best": binom_se, "noise_floor_used": noise_floor},
        "verdict": verdict,
    }


# ------------------------------------------------------------------------------------------
# report
# ------------------------------------------------------------------------------------------
HEADER = """# prompt-optimality-test — D4 optimality margin (pool-conditional bounds)

**Scope, stated up front.** Every number below is **conditional on the candidate pool** — the union
of all three arms' rescored prompts for that dataset. These bound what *this pool of prompts*, under
*this* selection process, can reach. They are **not** bounds over prompt space. A different seed,
proposer, or budget gives a different pool and different numbers.

**best-of-m is a selection statistic**, monotonically increasing in m by construction. The fitted
`y_inf` is an extrapolated ceiling *of the same selection process continued on the same pool*, not a
supremum over prompts.

**Nothing here is certified.** No quantity in this file is a certificate. The only certified
all-prompt bound style in this project is the DPI fixed-target cap (see memory
`project_momega_audit_bracket`); what follows is a pool-conditional projection, a strictly weaker
object.

## Headline: the requested ordering does NOT hold, and cannot

`best_achieved < y_inf < union_all` fails on all three datasets, with `y_inf` landing *below*
`best_achieved`. This is **structural, not empirical**: for an exchangeable best-of-m statistic over
a **fixed finite pool**, `sup_m E[max among m] = max over the pool = best_achieved`, attained exactly
at `m = n`. The curve terminates at the achieved value, so no monotone fit to it can project strictly
above that value. The middle term of the requested ordering is unattainable by construction.
"""


def main():
    results = [analyze_dataset(ds) for ds in DATASETS]

    lines = [HEADER, "## Bounds", "",
             "| dataset | n cand | n items | best achieved | y_inf (projected pool ceiling) | "
             "y_inf CI (candidate boot) | union-of-all (oracle cap) | ordering holds | "
             "CI lo > best achieved |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lo, hi = r["y_inf_boot_ci_2p5_97p5"]
        lines.append(
            f"| {r['dataset']} | {r['n_candidates']} | {r['n_items']} | {r['best_achieved']:.4f} | "
            f"{r['y_inf']:.4f} | [{lo:.4f}, {hi:.4f}] | {r['union_all']:.4f} | "
            f"{'yes' if r['ordering_best_lt_yinf_lt_union'] else 'NO'} | "
            f"{'yes' if r['ci_lower_exceeds_best_achieved'] else 'no'} |")

    lines += ["", "## Per-dataset verdict on the strictly-larger claim", ""]
    for r in results:
        lo = r["y_inf_boot_ci_2p5_97p5"][0]
        lines.append(
            f"- **{r['dataset']}** — {r['verdict']}. Margin `y_inf − best` = "
            f"{r['margin_yinf_minus_best']:+.4f}; CI lower edge {lo:.4f} vs best achieved "
            f"{r['best_achieved']:.4f}; noise floor {r['noise']['noise_floor_used']:.4f} "
            f"(binomial SE {r['noise']['binomial_se_at_best']:.4f} on {r['n_items']} items"
            + (f", repeat-rescore val spread {[round(x, 4) for x in r['noise']['repeat_rescore_abs_val_mean_spread']]}"
               if r["noise"]["repeat_rescore_abs_val_mean_spread"] else "") + ").")

    lines += ["", "## Why y_inf lands below best-achieved (two compounding reasons)", "",
              "**1. `fit_saturating` is misspecified for a best-of-m curve.** It imposes "
              "`y = y_inf*(1 − e^{−m/τ})`, forced through `y(0)=0`. A best-of-m curve has a large "
              "floor at m=1 (the mean candidate value). The fit absorbs the floor by collapsing τ "
              "toward 0 — τ below 1 means it claims the curve is already saturated at the first "
              "point — and `y_inf` settles near the middle of the observed curve rather than its "
              "asymptote. `compare_scaling_forms` prefers the power law on all three, which is the "
              "same signal: the finite ceiling is an artifact of the imposed form.", "",
              "**2. The finite-pool cap, which is the binding reason.** Refitting the same estimator "
              "to the EXCESS curve `y(m) − y(1)` over `m − 1` (which does start at 0, so the "
              "estimator's assumption holds) raises R² from 0.30–0.59 to 0.87–0.94 — and the implied "
              "ceiling still lands at or below best-achieved:", "",
              "| dataset | raw y_inf (R²) | excess-form ceiling (R²) | excess-form CI | best achieved |",
              "|---|---|---|---|---|"]
    for r in results:
        e = r["excess_form_ceiling"]
        lo, hi = r["excess_form_ceiling_ci"]
        lines.append(f"| {r['dataset']} | {r['y_inf']:.4f} ({r['sat_r2']:.2f}) | "
                     f"{e['ceiling']:.4f} ({e['r2']:.2f}) | [{lo:.4f}, {hi:.4f}] | "
                     f"{r['best_achieved']:.4f} |")

    lines += ["", "Correcting the misspecification does not rescue the ordering, which is the point: "
              "the cap is structural.", "",
              "## The one term that does sit above best-achieved — and why it is not a usable ceiling",
              "",
              "`union_all` (fraction of items solved by ≥1 candidate) strictly exceeds best-achieved "
              "everywhere. But it is the ceiling for a **per-item oracle selector**, not for any "
              "single prompt, and it is inflated by multiple comparisons over noisy binary scoring. "
              "Independent rescores of the *same* prompt (the seed appears in more than one arm) "
              "measure that noise directly:", "",
              "| dataset | best achieved | union-of-all | items best misses | items pool recovers | "
              "mean repeat disagreement q | expected false recoveries from noise alone |",
              "|---|---|---|---|---|---|---|"]
    for r in results:
        g = r["oracle_gap"]
        lines.append(f"| {r['dataset']} | {r['best_achieved']:.4f} | {r['union_all']:.4f} | "
                     f"{g['items_best_misses']} | {g['items_recovered_by_pool']} | "
                     f"{g['mean_repeat_disagreement_q']:.4f} | "
                     f"{g['expected_false_recoveries_from_noise']:.1f} |")

    lines += ["", "Two independent rescores of one prompt disagree on an item with probability "
              "`2p(1−p)`, so a genuinely hard item has solve-probability `p ≈ q/2`, and the chance it "
              "*looks* solved by at least one of the other k candidates is `1 − (1−p)^k`. This is a "
              "**first-order** check only (independence across candidates, a single flip rate, and q "
              "estimated from very few repeat pairs). Read per dataset:", ""]
    for r in results:
        lines.append(f"- **{r['dataset']}** — {r['oracle_gap']['read']}.")

    lines += ["", "## Fit diagnostics", "",
              "| dataset | τ | τ inside observed m range | saturating R² | form comparison verdict |",
              "|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['dataset']} | {r['tau']:.2f} | "
            f"{'yes' if np.isfinite(r['tau']) and r['tau'] <= r['n_candidates'] else 'NO'} | "
            f"{r['sat_r2']:.4f} | {r['scaling_form_verdict']} |")
    lines += ["", "`value_frontloading_stat` was computed (kept in the JSON) but is **not "
              "interpretable** on these curves: it normalizes by the terminal value and assumes both "
              "curves start near 0, whereas best-of-m starts at the mean candidate value, which "
              "forces D large and positive regardless of shape.", ""]

    (HERE / "runs" / "bounds_summary.md").write_text("\n".join(lines) + "\n")
    (HERE / "runs" / "bounds_summary.json").write_text(
        json.dumps({"scope": "pool-conditional; not all-prompt; nothing certified; "
                             "only certified all-prompt bound style = DPI fixed-target cap",
                    "headline": "requested ordering best<y_inf<union fails on all 3; y_inf lands "
                                "below best_achieved because sup_m E[max among m] over a fixed "
                                "finite pool IS the pool max",
                    "config": {"n_subsets": N_SUBSETS, "n_subsets_boot": N_SUBSETS_BOOT,
                               "n_boot": N_BOOT, "seed": SEED, "arms": list(ARMS)},
                    "datasets": results}, indent=2, default=float))
    print("\n".join(lines))
    print(f"\nwrote {HERE/'runs'/'bounds_summary.md'} and runs/bounds_summary.json")


if __name__ == "__main__":
    main()
