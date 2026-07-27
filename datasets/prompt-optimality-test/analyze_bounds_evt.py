"""D4 follow-up — EXTREME-VALUE endpoint estimate of the prompt-score distribution.

Run with the REPO python (needs scipy):  python3 analyze_bounds_evt.py

WHY THIS EXISTS. analyze_bounds.py proved the requested bound cannot come from best-of-m over a
fixed finite pool: sup_m E[max among m] = pool max, so every monotone asymptote lands at/below
best_achieved (structural, verified). The correctly-specified object for "a ceiling strictly
above best-achieved" is the UPPER ENDPOINT x* of the distribution F that the search process
draws candidate scores from. If F has a short upper tail (extreme-value index gamma < 0), x* is
finite and estimable from the top order statistics — and x* >= best_achieved by construction,
with the interesting question being whether the margin clears the noise.

WHAT IS COMPUTED, per dataset, on the pooled deduped candidate scores (same pool as
analyze_bounds.py):
  (1) two endpoint estimators, cross-checked over a sweep of tail sizes k:
      - GPD-MLE:  fit a Generalized Pareto to exceedances over u = X_(n-k); if the shape
        xi < 0 the implied endpoint is u + sigma/(-xi).
      - Pickands (closed form): gamma_P = ln[(Q1-Q2)/(Q2-Q4)]/ln 2 with Q_j = X_(n-jk+1); if
        gamma_P < 0, endpoint = Q1 + (Q1-Q2)/(2^{-gamma_P} - 1).
  (2) a candidate-level bootstrap CI for each (resample candidates with replacement, re-estimate
      at the same k, percentile 2.5/97.5 over the k-median).
  (3) tie/discreteness diagnostics (scores live on a 1/n_items grid; EVT assumes continuity), and
      a dequantization sensitivity: uniform(-0.5,0.5)/n_items smoothing, many replicates,
      reported SEPARATELY and never as the headline.
  (4) binomial SE at best_achieved beside every margin.

=========================== HONESTY / SCOPE — READ BEFORE QUOTING ===========================
(i)   The estimand is the endpoint of the distribution of THIS search process's candidate draws
      (this proposer, this budget, this seed prompt) — a PROCESS-conditional ceiling. It is NOT
      an all-prompt bound. The only certified all-prompt bound style in this project remains the
      DPI fixed-target cap (memory project_momega_audit_bracket).
(ii)  I.I.D. IS VIOLATED BY CONSTRUCTION: candidates come from an ADAPTIVE search (GEPA's later
      proposals condition on earlier scores), so the draws are neither independent nor
      identically distributed. EVT asymptotics are being applied to a small (n=59-85), dependent,
      discretized sample. Numbers here are exploratory diagnostics, not certificates.
(iii) Scores are means of n_items BINARY judgments: the OBSERVED-score distribution is the true-
      skill distribution convolved with binomial noise (SE ~ sqrt(p(1-p)/n_items) ~ 0.04-0.12).
      An endpoint of the observed distribution therefore overestimates the true-skill endpoint by
      up to about one SE; margins smaller than the binomial SE mean NOTHING. On the current
      100/17-item splits that SE swamps most margins — the real version of this analysis belongs
      on the 300-item paper-exact splits (T2b), and a clean negative here is an acceptable result.
============================================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from analyze_bounds import DATASETS, pool_candidates  # noqa: E402  (same pool, same dedupe)


def pool_candidates_paperexact(bench: str, lm_tag: str) -> dict:
    """Pool the paperexact TEST-split rescore matrices (paperexact_rescore.py output).

    Same shape as analyze_bounds.pool_candidates: {"rows", "M", "repeats"}. Dedupe by hash
    (arm order official > inhouse > unitrecomb wins); repeats list stays empty because the
    rescorer skips already-rescored hashes (noise is instead carried by the binomial SE and the
    seed's per-arm test passes recorded in result.json).
    """
    rows, seen = [], set()
    for arm in ("official", "inhouse", "unitrecomb"):
        p = HERE / "runs_paperexact" / bench / lm_tag / arm / "rescore.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            if r["hash"] in seen:
                continue
            seen.add(r["hash"])
            rows.append({"hash": r["hash"], "item_scores": r["item_scores"], "arm": arm})
    M = np.array([r["item_scores"] for r in rows], float)
    return {"rows": rows, "M": M, "repeats": []}

try:
    from scipy.stats import genpareto
except ImportError as exc:  # pragma: no cover
    raise SystemExit("needs scipy — run with the repo python, not the folder venv") from exc

N_BOOT = 400
SEED = 0
K_MIN = 5


def _k_grid(n: int) -> list[int]:
    """Tail sizes to sweep: from K_MIN up to ~n/2 (Pickands needs 4k <= n; GPD capped at n-3)."""
    ks = sorted(set(int(round(k)) for k in np.geomspace(K_MIN, max(K_MIN + 1, n // 2), 8)))
    return [k for k in ks if K_MIN <= k <= n - 3]


def gpd_endpoint(x: np.ndarray, k: int) -> dict:
    """GPD fit to the k exceedances over the (k+1)-th largest value. Finite endpoint iff xi<0."""
    xs = np.sort(x)
    u = xs[-(k + 1)]
    exc = xs[-k:] - u
    if np.all(exc <= 0):
        return {"ok": False, "reason": "all-exceedances-zero (ties at threshold)"}
    try:
        xi, loc, sigma = genpareto.fit(exc, floc=0.0)
    except Exception as e:  # noqa: BLE001 — fit can fail on degenerate tails
        return {"ok": False, "reason": f"fit-failed: {e}"}
    if xi >= 0:
        return {"ok": False, "reason": f"xi={xi:.3f} >= 0 (no finite endpoint at this k)",
                "xi": float(xi)}
    return {"ok": True, "endpoint": float(u + sigma / (-xi)), "xi": float(xi),
            "sigma": float(sigma), "threshold": float(u)}


def pickands_endpoint(x: np.ndarray, k: int) -> dict:
    """Pickands estimator from X_(n-k+1), X_(n-2k+1), X_(n-4k+1). Finite endpoint iff gamma<0."""
    xs = np.sort(x)
    n = len(xs)
    if 4 * k > n:
        return {"ok": False, "reason": "4k > n"}
    q1, q2, q4 = xs[n - k], xs[n - 2 * k], xs[n - 4 * k]
    if q1 - q2 <= 0 or q2 - q4 <= 0:
        return {"ok": False, "reason": "tied quartile order statistics (discreteness)"}
    gamma = np.log((q1 - q2) / (q2 - q4)) / np.log(2.0)
    if gamma >= 0:
        return {"ok": False, "reason": f"gamma={gamma:.3f} >= 0 (no finite endpoint at this k)",
                "gamma": float(gamma)}
    return {"ok": True, "endpoint": float(q1 + (q1 - q2) / (2.0 ** (-gamma) - 1.0)),
            "gamma": float(gamma)}


def _median_endpoint(x: np.ndarray, ks: list[int], fn) -> float:
    """Median finite-endpoint estimate over the k sweep (nan if none finite). Capped at 1.0:
    scores are frequencies, so an implied endpoint above 1 is an artifact of the tail fit."""
    vals = [min(r["endpoint"], 1.0) for r in (fn(x, k) for k in ks) if r.get("ok")]
    return float(np.median(vals)) if vals else float("nan")


def analyze_dataset(ds: str, pool: dict | None = None) -> dict:
    pool = pool if pool is not None else pool_candidates(ds)
    M = pool["M"]
    means = M.mean(1)
    n_cand, n_items = M.shape
    best = float(means.max())
    binom_se = float(np.sqrt(best * (1 - best) / n_items))
    ks = _k_grid(n_cand)

    sweep = {"gpd": {k: gpd_endpoint(means, k) for k in ks},
             "pickands": {k: pickands_endpoint(means, k) for k in ks}}
    point = {name: _median_endpoint(means, ks, fn)
             for name, fn in (("gpd", gpd_endpoint), ("pickands", pickands_endpoint))}

    # ---- candidate bootstrap of the k-median endpoint --------------------------------------
    rng = np.random.default_rng(SEED)
    boots = {"gpd": [], "pickands": []}
    n_finite = {"gpd": 0, "pickands": 0}
    for _ in range(N_BOOT):
        bx = means[rng.integers(0, n_cand, n_cand)]
        for name, fn in (("gpd", gpd_endpoint), ("pickands", pickands_endpoint)):
            v = _median_endpoint(bx, ks, fn)
            if np.isfinite(v):
                boots[name].append(v)
                n_finite[name] += 1
    ci = {name: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v
                 else [float("nan")] * 2) for name, v in boots.items()}

    # ---- dequantization sensitivity (NEVER the headline) -----------------------------------
    deq_rng = np.random.default_rng(SEED + 1)
    deq = {"gpd": [], "pickands": []}
    for _ in range(200):
        xj = means + deq_rng.uniform(-0.5, 0.5, n_cand) / n_items
        for name, fn in (("gpd", gpd_endpoint), ("pickands", pickands_endpoint)):
            v = _median_endpoint(xj, ks, fn)
            if np.isfinite(v):
                deq[name].append(v)
    deq_med = {name: (float(np.median(v)) if v else float("nan")) for name, v in deq.items()}

    # ---- diagnostics ----------------------------------------------------------------------
    top10 = np.sort(means)[-10:]
    n_distinct_top10 = int(len(np.unique(np.round(top10 * n_items))))
    finite_frac = {name: sum(1 for k in ks if sweep[name][k].get("ok")) / len(ks)
                   for name in sweep}

    margins = {name: (point[name] - best if np.isfinite(point[name]) else float("nan"))
               for name in point}
    agree = (np.isfinite(point["gpd"]) and np.isfinite(point["pickands"])
             and abs(point["gpd"] - point["pickands"]) <= max(binom_se, 0.02))

    both_finite = np.isfinite(point["gpd"]) and np.isfinite(point["pickands"])
    any_margin_clears = any(np.isfinite(m) and m > binom_se for m in margins.values())
    ci_lo_clears = {name: bool(np.isfinite(ci[name][0]) and ci[name][0] > best + binom_se)
                    for name in ci}
    if not both_finite:
        verdict = ("NO USABLE ENDPOINT — one or both estimators find no finite endpoint "
                   "(gamma/xi >= 0 or tail ties) at most k")
    elif not agree:
        verdict = "ESTIMATORS DISAGREE beyond the noise floor — endpoint not stable, do not quote"
    elif not any_margin_clears:
        verdict = ("ENDPOINT ~= BEST-ACHIEVED — margin inside binomial SE; consistent with the "
                   "search having exhausted its own draw distribution")
    elif not any(ci_lo_clears.values()):
        verdict = ("SUGGESTIVE ONLY — point margin clears the SE but the bootstrap CI lower edge "
                   "does not; needs the 300-item splits")
    else:
        verdict = ("ENDPOINT STRICTLY ABOVE BEST-ACHIEVED (CI lower edge clears best + SE) — "
                   "process-conditional headroom exists")

    return {
        "dataset": ds, "n_candidates": n_cand, "n_items": n_items,
        "best_achieved": best, "binomial_se_at_best": binom_se,
        "k_grid": ks,
        "endpoint_point_kmedian": point, "endpoint_margin_vs_best": margins,
        "endpoint_boot_ci_2p5_97p5": ci,
        "boot_finite_fraction": {n: n_finite[n] / N_BOOT for n in n_finite},
        "estimators_agree_within_noise": bool(agree),
        "ci_lower_clears_best_plus_se": ci_lo_clears,
        "dequantized_median_sensitivity": deq_med,
        "sweep": {name: {int(k): sweep[name][k] for k in ks} for name in sweep},
        "diagnostics": {"score_grid": 1.0 / n_items,
                        "n_distinct_scores_in_top10": n_distinct_top10,
                        "finite_endpoint_fraction_of_k": finite_frac},
        "verdict": verdict,
    }


HEADER = """# prompt-optimality-test — EVT endpoint estimate (process-conditional ceiling)

**Scope.** The estimand is the upper endpoint of the score distribution *this search process*
draws candidates from — a PROCESS-conditional ceiling, not an all-prompt bound (that remains the
DPI fixed-target cap only). I.i.d. is violated by construction (adaptive search); n is small
(59-85); scores are binomial means on a 1/n_items grid. Treat everything below as exploratory
diagnostics. Margins smaller than the binomial SE mean nothing; on the current 100/17-item
splits that SE swamps most margins, which is why the real version of this analysis is scheduled
for the 300-item paper-exact splits.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paperexact", default=None,
                    choices=["aime", "hover", "hotpot", "ifbench", "livebench", "pupa"],
                    help="analyze a paperexact TEST-split rescore pool instead of the runs/ pools")
    ap.add_argument("--lm-tag", default=None, help="[--paperexact] run-dir LM tag, e.g. Qwen3-8B")
    a = ap.parse_args()

    if a.paperexact:
        if not a.lm_tag:
            raise SystemExit("--paperexact needs --lm-tag")
        tag = f"{a.paperexact}:{a.lm_tag} (paper test split)"
        results = [analyze_dataset(tag, pool_candidates_paperexact(a.paperexact, a.lm_tag))]
        suffix = f"_paperexact_{a.paperexact}_{a.lm_tag}"
    else:
        results = [analyze_dataset(ds) for ds in DATASETS]
        suffix = ""

    lines = [HEADER, "## Endpoint estimates (k-median over the sweep, capped at 1.0)", "",
             "| dataset | n cand | best achieved | binom SE | GPD endpoint | Pickands endpoint | "
             "GPD CI | Pickands CI | agree | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        p, c = r["endpoint_point_kmedian"], r["endpoint_boot_ci_2p5_97p5"]
        lines.append(
            f"| {r['dataset']} | {r['n_candidates']} | {r['best_achieved']:.4f} | "
            f"{r['binomial_se_at_best']:.4f} | {p['gpd']:.4f} | {p['pickands']:.4f} | "
            f"[{c['gpd'][0]:.4f}, {c['gpd'][1]:.4f}] | "
            f"[{c['pickands'][0]:.4f}, {c['pickands'][1]:.4f}] | "
            f"{'yes' if r['estimators_agree_within_noise'] else 'NO'} | {r['verdict']} |")

    lines += ["", "## Margins and stability", "",
              "| dataset | GPD margin | Pickands margin | finite-endpoint k-fraction (GPD/Pick) | "
              "boot finite fraction (GPD/Pick) | distinct scores in top-10 | dequantized medians |",
              "|---|---|---|---|---|---|---|"]
    for r in results:
        m, d = r["endpoint_margin_vs_best"], r["diagnostics"]
        ff, bf = d["finite_endpoint_fraction_of_k"], r["boot_finite_fraction"]
        dq = r["dequantized_median_sensitivity"]
        lines.append(
            f"| {r['dataset']} | {m['gpd']:+.4f} | {m['pickands']:+.4f} | "
            f"{ff['gpd']:.2f}/{ff['pickands']:.2f} | {bf['gpd']:.2f}/{bf['pickands']:.2f} | "
            f"{d['n_distinct_scores_in_top10']} | {dq['gpd']:.4f}/{dq['pickands']:.4f} |")

    lines += ["", "## Read", ""]
    for r in results:
        lines.append(f"- **{r['dataset']}** — {r['verdict']}.")
    lines += ["", "A finite-endpoint failure (xi/gamma >= 0) at small n is NOT evidence of an "
              "infinite ceiling — scores are bounded by 1 — it means the top order statistics "
              "are too few/too tied to pin the tail. Ties (`tied quartile order statistics`) are "
              "the discreteness of the 1/n_items grid, not a property of prompts.", ""]

    md = HERE / "runs" / f"bounds_evt_summary{suffix}.md"
    js = HERE / "runs" / f"bounds_evt_summary{suffix}.json"
    md.write_text("\n".join(lines) + "\n")
    js.write_text(
        json.dumps({"scope": "process-conditional endpoint; NOT all-prompt; i.i.d. violated "
                             "(adaptive search); nothing certified",
                    "config": {"n_boot": N_BOOT, "seed": SEED, "k_min": K_MIN},
                    "datasets": results}, indent=2, default=float))
    print("\n".join(lines))
    print(f"\nwrote {md} and {js}")


if __name__ == "__main__":
    main()
