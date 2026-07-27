"""Rung 1.5 — unit-generator-conditional projected ceiling via missing-mass + value replay.

Run with the REPO python (needs scipy):  python3 analyze_unit_missing_mass.py

THE QUESTION (user, 2026-07-20): the certified bound is vacuous (deterministic labels => 1.0);
can we still ESTIMATE the value left in undiscovered units from the missing mass — an OSL-style
readout from (# units seen, p(new unit), value-saturation curve)?

THE OBJECT. Three coupled estimates per benchmark, all conditional on the UNIT GENERATOR
(clause-mining over this run's GEPA/inhouse trajectories; the LLM-suggester axis is a separate
source and is reported separately when sampled):
  (1) MASS  — clause-species frequencies across trajectory candidates -> Good-Turing missing
      mass U0 = F1/n and Chao1 richness S_hat = S_obs + F1^2/(2 F2) (bias-corrected when F2=0).
  (2) VALUE — the logged per-unit paired marginal deltas; discovery-order REPLAY (uniformly
      resampled orders — valid under exchangeability, per the capture-recapture doctrine) of
      E[best single-unit delta | n units discovered]; saturating fit on the excess curve;
      evaluated at n = S_obs (observed), n = S_hat (Chao-projected), n -> inf (generator limit).
  (3) RECOMBINATION SLACK — measured, never extrapolated: (best compiled select score) minus
      (init + best single-unit delta) on the same panel. Projected ceiling = init + projected
      best-single delta + measured slack.

=========================== HONESTY / SCOPE — READ BEFORE QUOTING ===========================
(i)   POINT ESTIMATE, NOT A CERTIFICATE. Extrapolation is banned from certificates in this
      project (v13 doctrine: certification needs exact factorization of value through the
      species quotient; the approximation error is exactly the unmeasurable part).
(ii)  ANTI-CONSERVATIVE UNDER DEPENDENCE. All trajectory clauses come from one correlated LM
      process; positive dependence UNDERESTIMATES the missing mass => "saturated" readings can
      be false. Independence of sources buys a point estimate, never a bound.
(iii) VALUE COUNTERWEIGHT (standing lesson): behaviorally-new units are usually near-worthless;
      never read U0 alone as "much value missing" — that is why the value replay, not the mass,
      carries this estimate.
(iv)  Panel-scale caveat: deltas are measured on the selection panel (27-100 items); panel noise
      inflates the top order statistics, so projections inherit selection-noise bias upward.
============================================================================================
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from methods.metric_implementer.experiments import unseen_value_scaling as uvs  # noqa: E402

CLAUSE_RE = r"(?<=[.!?])\s+|\n+|^[-*•]\s*"
N_REPLAY = 400
N_BOOT = 300
SEED = 0


def _norm(c: str) -> str:
    return " ".join(c.lower().split())


def clause_frequencies(candidate_texts: list[str]) -> dict[str, int]:
    """Clause-species occurrence counts across candidate draws (the mining process)."""
    freq: dict[str, int] = {}
    for text in candidate_texts:
        for clause in re.split(CLAUSE_RE, str(text)):
            c = clause.strip().strip("-*• ").rstrip()
            if 25 <= len(c) <= 400:
                freq[_norm(c)] = freq.get(_norm(c), 0) + 1
    return freq


def mass_estimates(freq: dict[str, int]) -> dict:
    counts = np.array(list(freq.values()))
    n = int(counts.sum())
    s_obs = len(counts)
    f1 = int((counts == 1).sum())
    f2 = int((counts == 2).sum())
    u0 = f1 / n if n else float("nan")
    if f2 > 0:
        s_hat = s_obs + f1 * f1 / (2.0 * f2)
    else:  # bias-corrected Chao1
        s_hat = s_obs + f1 * (f1 - 1) / 2.0
    return {"n_draws": n, "s_obs": s_obs, "f1": f1, "f2": f2,
            "gt_missing_mass_u0": float(u0), "chao1_s_hat": float(s_hat)}


def replay_best_single_curve(deltas: np.ndarray, rng) -> np.ndarray:
    """E[max delta among first n discovered units], uniform discovery orders (exchangeable)."""
    s = len(deltas)
    curve = np.zeros(s)
    for _ in range(N_REPLAY):
        perm = rng.permutation(s)
        curve += np.maximum.accumulate(deltas[perm])
    return curve / N_REPLAY


def project(deltas: np.ndarray, s_hat: float, rng) -> dict:
    """Fit the excess best-single curve, evaluate at s_obs / s_hat / inf."""
    s_obs = len(deltas)
    curve = replay_best_single_curve(deltas, rng)
    n_grid = np.arange(1, s_obs + 1, dtype=float)
    y_at = {"observed_best": float(deltas.max()), "curve_at_s_obs": float(curve[-1])}
    if s_obs < 4:
        return {**y_at, "ok": False, "reason": "too few units to fit"}
    fit = uvs.fit_saturating(n_grid - 1.0, curve - curve[0], n_boot=0)
    if not fit.get("ok"):
        return {**y_at, "ok": False, "reason": "fit failed"}
    y_inf, tau = float(fit["y_inf"]), float(fit["tau"])
    def y(n):
        return float(curve[0] + y_inf * (1.0 - np.exp(-max(n - 1.0, 0.0) / tau)))
    return {**y_at, "ok": True, "tau": tau, "r2": float(fit["r2_linear"]),
            "projected_delta_at_chao": y(s_hat),
            "projected_delta_at_inf": float(curve[0] + y_inf)}


def analyze(tag: str, traj_texts: list[str], marginals: list[float],
            init_score: float, compiled_score: float | None) -> dict:
    rng = np.random.default_rng(SEED)
    mass = mass_estimates(clause_frequencies(traj_texts))
    deltas = np.array(sorted(marginals, reverse=True))
    proj = project(deltas, mass["chao1_s_hat"], rng)

    # measured recombination slack (never extrapolated)
    best_single = init_score + max(0.0, float(deltas.max())) if len(deltas) else init_score
    slack = (max(0.0, compiled_score - best_single) if compiled_score is not None else 0.0)

    boots = []
    brng = np.random.default_rng(SEED + 1)
    for _ in range(N_BOOT):
        bd = deltas[brng.integers(0, len(deltas), len(deltas))]
        bp = project(np.sort(bd)[::-1], mass["chao1_s_hat"], brng)
        if bp.get("ok"):
            boots.append(init_score + max(0.0, bp["projected_delta_at_chao"]) + slack)
    ci = ([float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
          if boots else [float("nan")] * 2)

    ceiling = (init_score + max(0.0, proj.get("projected_delta_at_chao", 0.0)) + slack
               if proj.get("ok") else float("nan"))
    return {"pool": tag, "mass": mass, "n_units_valued": len(deltas),
            "value_projection": proj, "init_score": init_score,
            "compiled_score": compiled_score,
            "measured_recomb_slack": float(slack),
            "projected_unit_ceiling_at_chao": float(ceiling),
            "projected_ceiling_boot_ci": ci}


# ------------------------------------------------------------------ data plumbing per pool
def paperexact_pool(bench: str, lm: str) -> dict | None:
    base = HERE / "runs_paperexact" / bench / lm
    traj = []
    for arm in ("official", "inhouse"):
        p = base / arm / "proposals.jsonl"
        if p.exists():
            for line in open(p):
                traj += list(json.loads(line)["candidate"].values())
    up = base / "unitrecomb" / "proposals.jsonl"
    if not up.exists() or not traj:
        return None
    rows = [json.loads(l) for l in open(up)]
    init_rows = [r for r in rows if r["phase"] == "select_init"]
    marg_rows = [r for r in rows if r["phase"] == "select_marginal"]
    if not init_rows or not marg_rows:
        return None
    init = init_rows[-1]["mean_score"]
    prefix = [r["mean_score"] for r in rows if r["phase"].startswith(("prefix", "drop_one"))]
    compiled = max(prefix) if prefix else None
    return analyze(f"paperexact/{bench}/{lm}", traj,
                   [r["mean_score"] - init for r in marg_rows], init, compiled)


def t1_pool(ds: str) -> dict | None:
    rj = HERE / "runs" / ds / "momega_v2" / "result.json"
    if not rj.exists():
        return None
    r = json.loads(rj.read_text())
    traj = []
    for arm in ("official", "inhouse"):
        p = HERE / "runs" / ds / arm / "proposals.jsonl"
        if p.exists():
            for line in open(p):
                traj += list(json.loads(line)["candidate"].values())
    rows = [json.loads(l) for l in open(HERE / "runs" / ds / "momega_v2" / "proposals.jsonl")]
    init = [x for x in rows if x["phase"] == "select_init"][-1]["mean_score"]
    margs = [x["mean_score"] - init for x in rows if x["phase"] == "select_marginal"]
    return analyze(f"t1/{ds}", traj, margs, init, r["select"]["compiled"])


def main():
    results = []
    for bench in ("aime", "hover", "hotpot"):
        for lm in ("Qwen3-8B", "glm-5.2"):
            r = paperexact_pool(bench, lm)
            if r:
                results.append(r)
    for ds in ("hover", "hotpotqa"):
        r = t1_pool(ds)
        if r:
            results.append(r)

    lines = ["# Rung 1.5 — unit-generator missing-mass + value projection (POINT ESTIMATES, "
             "NOT certificates)", "",
             "| pool | draws | S_obs | F1 | U0 (GT) | Chao1 S_hat | init | best single | "
             "compiled | recomb slack | projected ceiling @Chao [boot CI] | fit R2 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        m, p = r["mass"], r["value_projection"]
        lo, hi = r["projected_ceiling_boot_ci"]
        best_single = r["init_score"] + max(0.0, p.get("observed_best", 0.0))
        comp = f"{r['compiled_score']:.3f}" if r["compiled_score"] is not None else "-"
        lines.append(
            f"| {r['pool']} | {m['n_draws']} | {m['s_obs']} | {m['f1']} | "
            f"{m['gt_missing_mass_u0']:.3f} | {m['chao1_s_hat']:.0f} | {r['init_score']:.3f} | "
            f"{best_single:.3f} | {comp} | {r['measured_recomb_slack']:+.3f} | "
            f"{r['projected_unit_ceiling_at_chao']:.3f} [{lo:.3f},{hi:.3f}] | "
            f"{p.get('r2', float('nan')):.2f} |")
    lines += ["", "Scope: conditional on THIS unit generator (trajectory clause-mining); "
              "anti-conservative under source dependence (false saturation possible); panel-"
              "noise inflates top order statistics; recombination slack is MEASURED, never "
              "extrapolated; nothing here is a certificate.", ""]
    (HERE / "runs" / "unit_missing_mass_summary.md").write_text("\n".join(lines) + "\n")
    (HERE / "runs" / "unit_missing_mass_summary.json").write_text(
        json.dumps({"scope": "generator-conditional point estimates; extrapolation banned from "
                             "certificates; anti-conservative under dependence",
                    "results": results}, indent=2, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
