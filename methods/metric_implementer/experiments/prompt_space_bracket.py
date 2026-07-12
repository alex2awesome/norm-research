"""Prompt-space bracket instruments — the (a)-(d) companions to the fixed-target DPI cap.

Post-audit (``notes/2026-07-10__momega-certificate-audit.md``) there is exactly ONE certified
all-prompt object: the fixed-target DPI cap ``T_f(M_omega) = I_f(M_omega; X)``
(``vinfo.target_channel_ceiling`` / ``vinfo.fixed_target_channel_certificate``). Everything here
tightens the *bracket around* the prompt-space optimum from both sides WITHOUT minting any new
certificate:

    lower (achieved) arm                          upper arm
    --------------------                          ---------
    R_single / GEPA* / frozen_head (b)   <=   I(M;U) estimate (d)   <=   T_soft (a)   <=   H(M)
    evt_endpoint (c): tail ESTIMATE of the reachable supremum — assumption-carrying, never certified

(a) ``reliability_ceiling`` — T from SOFT per-item probabilities (orbit/pass means): verdict noise
    is charged in ``mean H(q_i)`` (audit requirement #1: "preserve soft target probabilities").
    With hard 0/1 targets T degenerates to the loose H(M) cap. Split-half self-transmission is the
    attained self-consistency companion. Population scope inherits vinfo semantics (2-D
    pass-estimated input -> population bound stays blocked).
(b) ``frozen_head`` — retroactive freeze discipline on saved signature checkpoints: SELECT the
    checklist head on one probe half, EVALUATE the frozen rule on the other, both directions.
    Removes the search optimism the audit measured on CW#24 (selection .561/.625 -> frozen
    .444/.431). Same-f, same-heldout readout via ``vinfo.binary_soft_channel_mi`` (Shannon bits).
    Applies to EVERY attack family in the (b) portfolio: report frozen numbers, never selection.
(c) ``evt_endpoint`` — GPD fit to the upper tail of achieved recoveries over behavioral species:
    the type-(c) statistical endpoint. ``certified=False`` by construction (tail regularity +
    declared search measure are assumptions; a disconnected higher mode is invisible — the
    tail-XOR lesson).
(d) ``joint_combiner_ceiling`` — the combiner-class ladder over unit signals under the same freeze
    discipline: best-single -> linear-logistic -> +pairwise products -> pattern lookup. For any
    combiner g, ``I(M; g(U)) <= I(M; U) <= I(M; X) = T`` by DPI (units are functions of X up to
    conditionally independent execution noise), so rung values live INSIDE the certified bracket.
    Rung gaps decompose the gestalt: (lookup - linear) = interaction content within the unit span;
    ``span_residual`` locates a candidate prompt's behavior inside/outside the span (span vs
    channel content — re-mine loop separates the two operationally).

Scope discipline: every dict returned here carries ``certified=False`` (estimates / achieved
values). The only certified lines remain vinfo's, consumed unchanged. CPU-only; feeds on saved
artifacts (sigs matrices, per-form/pass panels, achieved-value pools). No edits to
``value_certificate.py`` — this module sits beside it so historical fields stay untouched.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from methods.metric_implementer import vinfo

_EPS = 1e-12


# --------------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------------

def _halves(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Even/odd index masks over the FROZEN probe order (stable under growth — house split rule)."""
    idx = np.arange(n)
    return (idx % 2 == 0), (idx % 2 == 1)


def _trans_bits(q: np.ndarray, p: np.ndarray) -> float:
    """Same-f readout: Shannon bits of the conditionally-independent soft channel (q, p)."""
    q = np.clip(np.asarray(q, float), 0.0, 1.0)
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    if q.size < 4 or np.std(q) < _EPS or np.std(p) < _EPS:
        return 0.0
    return float(vinfo.binary_soft_channel_mi(q, p)["shannon"])


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> np.ndarray:
    """L2-regularized logistic with SOFT labels (cross-entropy against y in [0,1])."""
    from scipy.optimize import minimize
    Xb = np.hstack([np.ones((len(X), 1)), X])
    y = np.clip(np.asarray(y, float), 0.0, 1.0)

    def nll(w):
        z = np.clip(Xb @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + l2 * np.sum(w[1:] ** 2))

    res = minimize(nll, np.zeros(Xb.shape[1]), method="L-BFGS-B")
    return res.x


def _pred_logistic(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.hstack([np.ones((len(X), 1)), X])
    return 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))


# --------------------------------------------------------------------------------------------
# (a) reliability ceiling — soft-target DPI cap + split-half self-consistency
# --------------------------------------------------------------------------------------------

def reliability_ceiling(per_item_passes: np.ndarray, *, delta: Optional[float] = None) -> dict:
    """Arm (a): the unconditional top line of the bracket, from SOFT verdict probabilities.

    ``per_item_passes``: (N,) already-soft per-item probabilities, or (N, F) per-item x per-form/
    pass P(YES) panel (the stored ``per_form`` layout, item-major). The 2-D form is the honest one:
    vinfo marks it pass-estimated and keeps the population bound blocked.

    Returns the soft-target ceiling ``T_soft`` (verdict noise charged in ``mean H(q_i)``), the
    hard-threshold degeneration ``T_hard = H(M)`` for comparison, and — when >=2 passes exist —
    the split-half self-transmission ``R_self`` (even vs odd passes), the attained
    self-consistency that any candidate is competing against.
    """
    A = np.asarray(per_item_passes, float)
    cap = vinfo.target_channel_ceiling(A, delta=delta)
    q = np.nanmean(A, axis=1) if A.ndim == 2 else A
    q = q[np.isfinite(q)]
    pi = float(np.mean(q > 0.5)) if q.size else float("nan")
    t_hard = float(vinfo._h_bits(pi)) if 0.0 < pi < 1.0 else 0.0
    out = {
        "T_soft": cap["empirical"]["shannon"], "T_soft_tvd": cap["empirical"]["tvd"],
        "T_hard_HM": t_hard, "noise_charged_bits": float(max(0.0, t_hard - cap["empirical"]["shannon"])),
        "n_items": cap["n_items"], "population": cap.get("population"),
        "population_error": cap.get("population_error"),
        "scope": cap["scope"], "certified": False,
        "note": "T_soft is the vinfo DPI cap fed soft targets; this wrapper adds no new bound.",
    }
    if A.ndim == 2 and A.shape[1] >= 2:
        ev, od = _halves(A.shape[1])
        q_ev, q_od = np.nanmean(A[:, ev], axis=1), np.nanmean(A[:, od], axis=1)
        keep = np.isfinite(q_ev) & np.isfinite(q_od)
        if keep.sum() >= 4:
            self_cert = vinfo.fixed_target_channel_certificate(q_ev[keep], q_od[keep])
            out["R_self_split_half"] = self_cert["shannon"]["R"] if self_cert.get("valid") else None
            out["self_cert"] = self_cert
    return out


# --------------------------------------------------------------------------------------------
# (b) frozen-head — freeze-before-eval discipline, retroactive on saved checkpoints
# --------------------------------------------------------------------------------------------

def frozen_head(sigs: np.ndarray, target: np.ndarray, *, n_select: int = 6, eps: float = 1e-3,
                combiner: str = "mean") -> dict:
    """Arm (b) core discipline: greedy-select on one probe half, evaluate FROZEN on the other.

    ``sigs``: (K, N) unit signatures, soft P(YES) in [0,1] (binary also fine). ``target``: (N,)
    soft or hard M_omega. Combiner 'mean' = unweighted soft vote over the selected units (fully
    frozen by the subset); 'logistic' = weights fitted on the selection half only, then frozen.

    Both split directions are run; ``optimism`` = selection-half value minus frozen value is the
    quantity the audit showed OOF-inside-candidates does not remove. Report ``R_frozen_mean`` as
    the achieved checklist value; never the selection value.
    """
    S = np.asarray(sigs, float)
    q = np.asarray(target, float)
    if S.ndim != 2 or S.shape[1] != q.size:
        raise ValueError("sigs must be (K, N) aligned with target (N,)")
    ev, od = _halves(q.size)
    runs = []
    for sel_mask, eval_mask, tag in [(ev, od, "even->odd"), (od, ev, "odd->even")]:
        qs, qe = q[sel_mask], q[eval_mask]
        chosen: List[int] = []
        best_val = 0.0
        while len(chosen) < n_select:
            gains = []
            for k in range(S.shape[0]):
                if k in chosen:
                    continue
                cols = chosen + [k]
                comb = S[cols][:, sel_mask].mean(axis=0)
                gains.append((_trans_bits(qs, comb), k))
            if not gains:
                break
            g, k = max(gains)
            if g < best_val + eps:
                break
            best_val = g
            chosen.append(k)
        if not chosen:
            runs.append({"direction": tag, "selected": [], "R_selection": 0.0, "R_frozen": 0.0,
                         "optimism": 0.0})
            continue
        if combiner == "logistic":
            w = _fit_logistic(S[chosen][:, sel_mask].T, qs)
            p_sel = _pred_logistic(S[chosen][:, sel_mask].T, w)
            p_eval = _pred_logistic(S[chosen][:, eval_mask].T, w)
        elif combiner == "mean":
            p_sel = S[chosen][:, sel_mask].mean(axis=0)
            p_eval = S[chosen][:, eval_mask].mean(axis=0)
        else:
            raise ValueError(f"unknown combiner {combiner!r}")
        r_sel, r_fro = _trans_bits(qs, p_sel), _trans_bits(qe, p_eval)
        runs.append({"direction": tag, "selected": [int(c) for c in chosen],
                     "R_selection": r_sel, "R_frozen": r_fro, "optimism": float(r_sel - r_fro)})
    return {
        "runs": runs, "combiner": combiner,
        "R_frozen_mean": float(np.mean([r["R_frozen"] for r in runs])),
        "R_selection_mean": float(np.mean([r["R_selection"] for r in runs])),
        "optimism_mean": float(np.mean([r["optimism"] for r in runs])),
        "certified": False,
        "note": "achieved in-class value under freeze discipline; a lower bound on the class "
                "optimum, never a prompt-space ceiling",
    }


# --------------------------------------------------------------------------------------------
# (c) EVT endpoint — statistical estimate of the reachable supremum (never certified)
# --------------------------------------------------------------------------------------------

def evt_endpoint(values: Sequence[float], *, tail_frac: float = 0.25, n_boot: int = 400,
                 min_tail: int = 20, seed: int = 0) -> dict:
    """Arm (c): generalized-Pareto fit to the upper tail of achieved recoveries.

    ``values`` should be SPECIES-level achieved recoveries (dedupe prompts to behavioral species
    first — prompts inducing the same behavior are one atom), pooled across attack families.
    xi < 0 => finite right endpoint ``u + sigma/(-xi)`` = the estimated reachable supremum.
    xi >= 0 => no finite endpoint under the fit (report None).

    Assumption-carrying by construction: GPD tail regularity, the declared search measure, and no
    disconnected higher mode (the tail-XOR failure shows exactly such a mode evading extrapolation).
    Optimization bias of the search families explores the right tail hardest, which HELPS endpoint
    estimation; it never certifies it. ``certified=False`` always.
    """
    from scipy.stats import genpareto
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if v.size < max(min_tail * 2, 40):
        return {"valid": False, "error": "too_few_values", "n": int(v.size), "certified": False}
    u = float(np.quantile(v, 1.0 - tail_frac))
    exc = v[v > u] - u
    if exc.size < min_tail:
        return {"valid": False, "error": "too_few_exceedances", "n_tail": int(exc.size),
                "certified": False}
    xi, _, sig = genpareto.fit(exc, floc=0.0)
    endpoint = float(u + sig / (-xi)) if xi < 0 else None
    rng = np.random.default_rng(seed)
    b_end, b_xi = [], []
    for _ in range(n_boot):
        vb = rng.choice(v, size=v.size, replace=True)
        ub = float(np.quantile(vb, 1.0 - tail_frac))
        eb = vb[vb > ub] - ub
        if eb.size < min_tail // 2:
            continue
        try:
            xib, _, sb = genpareto.fit(eb, floc=0.0)
        except Exception:
            continue
        b_xi.append(xib)
        if xib < 0:
            b_end.append(ub + sb / (-xib))
    b_end = np.asarray(b_end, float)
    return {
        "valid": True, "n": int(v.size), "threshold_u": u, "n_tail": int(exc.size),
        "xi": float(xi), "sigma": float(sig), "endpoint": endpoint,
        "endpoint_ci": ([float(np.percentile(b_end, 2.5)), float(np.percentile(b_end, 97.5))]
                        if b_end.size >= 50 else None),
        "boot_frac_finite_endpoint": float(np.mean(np.asarray(b_xi) < 0)) if b_xi else None,
        "max_observed": float(v.max()),
        "certified": False,
        "assumptions": ["GPD tail regularity", "declared search measure (attack portfolio)",
                        "no disconnected higher mode", "species-level dedupe upstream"],
    }


# --------------------------------------------------------------------------------------------
# (d) joint-combiner ladder + span residual — locating the gestalt inside the bracket
# --------------------------------------------------------------------------------------------

def joint_combiner_ceiling(sigs: np.ndarray, target: np.ndarray, *, max_pair_units: int = 10,
                           lookup_units: int = 6, l2: float = 1e-2) -> dict:
    """Arm (d): the combiner-expressivity ladder over unit signals, freeze-disciplined.

    Rungs (each selected/fitted on one probe half, evaluated frozen on the other, both directions):
      ``best_single``  argmax single unit
      ``linear``       logistic over all units (F1, parity-blind)
      ``pairwise``     logistic over units + pairwise products of the top ``max_pair_units`` (F2)
      ``lookup``       pattern table over the top ``lookup_units`` binarized units (free g; the
                       empirical stand-in for I(M;U) at small effective K)
    Every rung is read as a HARD decision channel (predictions thresholded at 0.5) so expressivity
    classes are compared at equal output sharpness — otherwise calibration differences (a shrunk
    logistic vs extreme small-cell lookup means) masquerade as interaction content (caught by the
    planted linear-rule null in tests). DPI chain: every rung <= T(q_eval) algebraically (checked,
    ``dpi_ok``). ``interaction_bits`` = lookup - linear (frozen) = configural content the linear
    class cannot express — the metric-level gestalt readout. All values are ESTIMATES inside the
    certified bracket.
    """
    S = np.asarray(sigs, float)
    q = np.asarray(target, float)
    ev, od = _halves(q.size)
    directions = []
    for sel_mask, eval_mask, tag in [(ev, od, "even->odd"), (od, ev, "odd->even")]:
        qs, qe = q[sel_mask], q[eval_mask]
        Xs, Xe = S[:, sel_mask].T, S[:, eval_mask].T
        add = np.array([_trans_bits(qs, S[k][sel_mask]) for k in range(S.shape[0])])
        hard = lambda p: (np.asarray(p, float) > 0.5).astype(float)   # equal-sharpness readout
        rung: Dict[str, float] = {}
        k_best = int(np.argmax(add))
        rung["best_single"] = _trans_bits(qe, hard(S[k_best][eval_mask]))
        w = _fit_logistic(Xs, qs, l2=l2)
        rung["linear"] = _trans_bits(qe, hard(_pred_logistic(Xe, w)))
        top = list(np.argsort(-add)[: min(max_pair_units, S.shape[0])])
        pair_s = [Xs[:, i] * Xs[:, j] for a, i in enumerate(top) for j in top[a + 1:]]
        pair_e = [Xe[:, i] * Xe[:, j] for a, i in enumerate(top) for j in top[a + 1:]]
        if pair_s:
            Xs2 = np.hstack([Xs, np.stack(pair_s, axis=1)])
            Xe2 = np.hstack([Xe, np.stack(pair_e, axis=1)])
            w2 = _fit_logistic(Xs2, qs, l2=l2)
            rung["pairwise"] = _trans_bits(qe, hard(_pred_logistic(Xe2, w2)))
        lk = list(np.argsort(-add)[: min(lookup_units, S.shape[0])])
        Bs, Be = (S[lk][:, sel_mask] > 0.5), (S[lk][:, eval_mask] > 0.5)
        table: Dict[bytes, list] = {}
        for col in range(Bs.shape[1]):
            table.setdefault(Bs[:, col].tobytes(), []).append(qs[col])
        means = {k: float(np.mean(v)) for k, v in table.items()}
        marg = float(np.mean(qs))
        rung["lookup"] = _trans_bits(qe, hard(np.array([means.get(Be[:, c].tobytes(), marg)
                                                        for c in range(Be.shape[1])])))
        t_eval = vinfo.target_channel_ceiling(qe)["empirical"]["shannon"]
        directions.append({"direction": tag, "rungs": rung, "T_eval": t_eval,
                           "dpi_ok": bool(all(vv <= t_eval + 1e-9 for vv in rung.values()))})
    names = sorted({k for d in directions for k in d["rungs"]})
    mean_rungs = {k: float(np.mean([d["rungs"][k] for d in directions if k in d["rungs"]]))
                  for k in names}
    return {
        "directions": directions, "rungs_mean": mean_rungs,
        "joint_estimate": mean_rungs.get("lookup"),
        "interaction_bits": float(max(0.0, mean_rungs.get("lookup", 0.0) - mean_rungs.get("linear", 0.0))),
        "dpi_ok": bool(all(d["dpi_ok"] for d in directions)),
        "certified": False,
        "note": "estimates of I(M; g(U)) inside the certified DPI bracket; not a prompt-space bound",
    }


def span_residual(sigs: np.ndarray, candidate: np.ndarray, *, target: Optional[np.ndarray] = None,
                  ridge: float = 1e-2) -> dict:
    """Arm (d) locator: is a candidate prompt's BEHAVIOR inside the unit span?

    Ridge-regress the candidate's per-item soft behavior on the unit signals (fit on even probes,
    R^2 on odd, both directions). With ``target`` supplied, also reports the transmission gap
    ``I(M;cand) - I(M;proj(cand))`` (projection clipped to [0,1]): the candidate's alignment that
    the unit span cannot reproduce. Interpretation protocol (see module docstring): a positive gap
    is SPAN content if re-mining units from the winning candidate closes it, CHANNEL content if it
    persists at the re-mine fixed point.
    """
    S = np.asarray(sigs, float)
    c = np.asarray(candidate, float)
    ev, od = _halves(c.size)
    out_dirs = []
    for tr, te, tag in [(ev, od, "even->odd"), (od, ev, "odd->even")]:
        X_tr, X_te = S[:, tr].T, S[:, te].T
        Xb_tr = np.hstack([np.ones((X_tr.shape[0], 1)), X_tr])
        Xb_te = np.hstack([np.ones((X_te.shape[0], 1)), X_te])
        w = np.linalg.solve(Xb_tr.T @ Xb_tr + ridge * np.eye(Xb_tr.shape[1]), Xb_tr.T @ c[tr])
        proj = Xb_te @ w
        ss_res = float(np.sum((c[te] - proj) ** 2))
        ss_tot = float(np.sum((c[te] - c[te].mean()) ** 2))
        d = {"direction": tag, "r2_heldout": float(1.0 - ss_res / ss_tot) if ss_tot > _EPS else 0.0}
        if target is not None:
            q = np.asarray(target, float)
            d["trans_candidate"] = _trans_bits(q[te], c[te])
            d["trans_projection"] = _trans_bits(q[te], np.clip(proj, 0.0, 1.0))
            d["channel_gap_bits"] = float(max(0.0, d["trans_candidate"] - d["trans_projection"]))
        out_dirs.append(d)
    out = {"directions": out_dirs,
           "r2_heldout_mean": float(np.mean([d["r2_heldout"] for d in out_dirs])),
           "certified": False}
    if target is not None:
        out["trans_candidate_mean"] = float(np.mean([d["trans_candidate"] for d in out_dirs]))
        out["trans_projection_mean"] = float(np.mean([d["trans_projection"] for d in out_dirs]))
        out["channel_gap_bits_mean"] = float(np.mean([d["channel_gap_bits"] for d in out_dirs]))
    return out


# --------------------------------------------------------------------------------------------
# assembly — one bracket row per (metric, executor)
# --------------------------------------------------------------------------------------------

def bracket_report(sigs: np.ndarray, target_passes: np.ndarray, *,
                   achieved_pool: Optional[Sequence[float]] = None,
                   candidates: Optional[Dict[str, np.ndarray]] = None) -> dict:
    """Assemble the full bracket for one metric x executor from saved artifacts.

    ``target_passes``: (N,) soft q or (N, F) per-form/pass panel (soft preferred — requirement #1).
    ``achieved_pool``: species-level achieved recoveries from the attack portfolio (feeds EVT).
    ``candidates``: named per-item behavior vectors (e.g. {'gepa_final': ..., 'seed_desc': ...})
    for span location. Every sub-object keeps its own scope flags; nothing here certifies.
    """
    A = np.asarray(target_passes, float)
    q = np.nanmean(A, axis=1) if A.ndim == 2 else A
    rel = reliability_ceiling(A)
    row = {
        "reliability": rel,
        "frozen_head": frozen_head(sigs, q),
        "joint_ladder": joint_combiner_ceiling(sigs, q),
        "certified": False,
    }
    if achieved_pool is not None:
        row["evt"] = evt_endpoint(achieved_pool)
    if candidates:
        row["span"] = {name: span_residual(sigs, vec, target=q) for name, vec in candidates.items()}
    lo = row["frozen_head"]["R_frozen_mean"]
    hi = rel["T_soft"]
    row["bracket_bits"] = {"achieved_lower": lo, "dpi_upper": hi,
                           "headroom_upper_bound": float(max(0.0, hi - lo)),
                           "note": "headroom is an upper bound on the fixed-target gap, not an "
                                   "estimate of attainable content (audit: tightness not established)"}
    return row
