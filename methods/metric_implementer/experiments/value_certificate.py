"""Value-weighted checklist discovery diagnostics on the quotient partition.
Theory: ``notes/2026-06-18__prompt-optimality-theory.md`` §12.6 (derivations D1–D3 + the bridge).

VALIDATION STATUS (2026-07-10): this module does not currently issue a proved upper bound. The head
is an adaptively selected achieved value; OSW/Good-Toulmin supplies a horizon point prediction, not
the high-probability horizon bound assembled below; and ``gamma_tail`` is a measured point estimate,
not a lower confidence bound for unseen synergy. Historical field names are retained for artifact
compatibility, but outputs carry ``upper_bound_valid=False``. The all-prompt fixed-target ceiling is
implemented separately in ``vinfo.fixed_target_channel_certificate``.

Per metric M_i, CPU post-hoc on a frozen ``…_metric{gi}_sigs.npz`` checkpoint (sigs + M_i):

  HEAD (partition-FREE, §12.6.2)   pool-level greedy on the RAW criteria vs M_i → an achieved gain sequence
                                   OPT_Ω = Σ g_k (bits), tail_frac, Hill ξ. Duplicates land in the tail
                                   with gain ≈ 0, so species explosion cannot corrupt the head.
  TAIL / FLUX (quotient, §12.6.3)  value spectrum w_j = Σ_{n_s=j} v(s|S_g) over the QUOTIENT species of
                                   the frozen-iid stream;  flux V̇ = w_1/N  (D1, value-weighted
                                   Good–Turing/Robbins);  McDiarmid slack B·√(2·ln(1/δ_eff)/N) + B/N
                                   (D2 diagnostic: B = H(M), fixed before head selection; the formal
                                   bounded-difference argument additionally requires the head, partition,
                                   and value map to be frozen independently, which this pipeline does not do);
                                   Good–Toulmin
                                   horizon Ĝ(c) = −Σ_{j≤k0} (−c)^j w_j  (D3; Efron–Thisted truncation
                                   at k0=4 for c ≤ 1; for 1 < c ≤ ln N the OSW binomial smoothing
                                   ``osw_horizon_value`` — §12.8.7 I1, clamped and flagged past ln N;
                                   neither route supplies a one-sided weighted-horizon confidence bound).
  BRIDGE (§12.6.4)                 ε = (1/γ̂)·[ max(Ĝ(c), 0) + slack ],  γ̂ = measured tail-submodularity
                                   ratio; backstopped by the γ-free
                                   ``orthogonalize.adversarial_saturation`` (pass precomputed; the probe
                                   set MUST include holistic/composed prompts) and the unconditional DPI
                                   ceiling ≤ T(m̄_ω).
  ORDER-ADVERSE (§12.6.6)          the flux/ε sensitivity re-run under ``n_orders`` core-build permutations of
                                   the strict partition; ``eps_bits_adv`` = the max — the reported ε.
                                   The error allocation is unioned across these statistics, but the result
                                   remains a sensitivity analysis because the freezing premise is unmet.
  STOPPING (§12.8.0 T1)            ``stopping='anytime'`` (default) makes the band TIME-UNIFORM over
                                   the doubling checkpoint grid N_j = n0·2^{j−1}: the §12.6.6 loop
                                   (UNDERSAMPLED → draw more → re-issue) is optional continuation, and
                                   a fixed-N band read at a data-chosen N is unsound. ``'fixed'`` is
                                   valid ONLY for a budget declared in advance and a single issuance.
  VERDICT (§12.6.6)                ``alpha_probe.decide``: UNDERSAMPLED / FORM-DOMINATED /
                                   CODIFIABLE / DEEP / INDETERMINATE (reads ``eps_bits_adv``).

SCOPE (corrected 2026-07-10). The measured object is achieved checklist recovery under a named combiner
class F (``combiner``: 'linear' = F₁ additive-logistic, parity-blind — the XOR
planted control documents it; 'pairs' = F₂ adds pairwise interactions on the head), relative to the
named proposal process. It is not a bound on prompt-space OPT: a composed prompt is not a function of
unit verdicts, so no unit-to-composition DPI applies. The separate fixed-target DPI still bounds that
prompt directly. The composition residual is measured, never assumed zero —
``composition_gap.delta_comp`` (Δ_comp arm) + holistic prompts in the adversarial probe set.

GUARDS. Value is CONDITIONAL on the FROZEN post-run head S_g (a fixed function ⇒ stationary stream —
GV7 satisfied; marginal/additive values are reported only as the conservative upper read, never mixed
into the primary flux). The flux accounting uses ONLY the frozen-iid draws (tags ∉ {children, gepa} —
guard G1). Over-split is flux-SAFE (a fake singleton carries v|S_g ≈ 0); over-merge is the ONE
anti-conservative direction ⇒ merge-precision is the binding gate and w_1 is reported at two probe
sizes (§12.6.2). Units: Shannon bits throughout (H(M_i) ≤ 1 for a binary target).

Scaling mode (§12.6.5): per-executor achieved-value/heuristic tables. Any gap using ``OPT_Ω+ε`` is
descriptive only. No fitted asymptotes.

Examples (CPU):
    python -m methods.metric_implementer.experiments.value_certificate --dir /…/crc_scaling/llama8b
    python -m methods.metric_implementer.experiments.value_certificate \\
        --scaling llama3b:/…/llama3b,llama8b:/…/llama8b,qwen122b:/…/qwen122b --ceiling 0.79
    # judge-quotient (GLM quota — prune hard):
    python -m methods.metric_implementer.experiments.value_certificate --dir … --quotient judge
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import vinfo
from . import alpha_probe as ap
from .orthogonalize import _oof_ce
from .value_census import hill_tail_index, i_binary


# --------------------------------------------------------------------------------------------
# surrogate CMI blocks  (CE-drop; the chain-rule-friendly Shannon proxy — same engine as orthogonalize)
# --------------------------------------------------------------------------------------------

def _cmi_block(y: np.ndarray, X_base: np.ndarray, X_new: np.ndarray, *, seed: int = 0) -> float:
    """I(X_new ; y | X_base) in bits via the 5-fold OOF surrogate: CE(y|X_base) − CE(y|X_base, X_new),
    clamped ≥ 0. ``X_new`` may be a single column or a BLOCK (the joint-block form ``tail_gamma`` needs,
    which ``shannon_cmi_surrogate`` does not expose)."""
    y = np.asarray(y, int)
    Xb = np.asarray(X_base, float)
    if Xb.ndim == 1:
        Xb = Xb[:, None]
    if Xb.size == 0:
        Xb = np.empty((len(y), 0))
    Xn = np.asarray(X_new, float)
    if Xn.ndim == 1:
        Xn = Xn[:, None]
    ce_base = _oof_ce(Xb, y, seed=seed)
    ce_full = _oof_ce(np.hstack([Xb, Xn]), y, seed=seed)
    return float(max(0.0, ce_base - ce_full))


# --------------------------------------------------------------------------------------------
# the HEAD — pool-level greedy on raw criteria (partition-free; §12.6.2)
# --------------------------------------------------------------------------------------------

def greedy_head(binmat: np.ndarray, M: np.ndarray, *, top_k: int = 30, prerank: int = 120,
                eps: float = 0.005, k_head: int = 3, seed: int = 0, combiner: str = "linear",
                pair_prerank: int = 24, max_pairs: int = 4) -> dict:
    """Pool-level greedy selection of RAW criteria by conditional value v(e | selected) — the
    partition-free head of the §12.6 certificate. ``binmat`` (n_criteria, n_probes) binarized
    signatures; ``M`` (n_probes,) the binarized target (m̄_ω / M_i).

    Duplicates/paraphrases of already-selected criteria have conditional gain ≈ 0, so they fall into
    the tail without moving the head — no dedup decision can corrupt OPT_Ω (§12.6.2 robustness).
    Tractability: candidates pre-ranked by ADDITIVE value (safe under submodularity: low-additive
    cannot be high-conditional), greedy stops at gain < ``eps`` or ``top_k`` picks.

    **Combiner class (§12.6.4 combiner-class escape — the certificate is F-indexed).**
    ``combiner='linear'`` (default) = F₁, the additive-logistic readout: BLIND to purely conjunctive
    value (a parity/XOR metric gives every unit AND every raw pair v ≈ 0 — the XOR planted control
    documents this limitation). ``combiner='pairs'`` = F₂: after the singles greedy stalls, a pair
    stage evaluates candidates ``[x_i, x_j, x_i·x_j]`` (the product term makes XOR/AND linearly
    representable) over pairs among the top ``pair_prerank`` ADDITIVE units — exhaustive when
    ``pair_prerank ≥ n``; otherwise parity among low-additive units is found only if the prerank
    reaches it (state the sampling cap when reporting). Deeper conjunctions remain outside F₂.

    Returns: ``selected`` (ints; pair picks appear as ``('pair', i, j)``), ``selected_units`` (flat
    ints incl. pair members — for tail exclusion), ``S_cols`` (n_probes × d conditioning matrix —
    the head's actual feature set incl. product columns), gains g_1 ≥ g_2 ≥ …, ``opt_omega_bits`` =
    Σ g_k (bits), ``frac_H`` = OPT_Ω/H(M), ``tail_frac`` = share of OPT_Ω beyond the first ``k_head``
    gains (the DEEP signal), Hill ξ on additive values, and ``combiner``."""
    B = np.asarray(binmat, int)
    M = np.asarray(M, int)
    h_m = float(vinfo._h_bits(float(M.mean()))) if len(np.unique(M)) > 1 else 0.0
    v_add = np.array([i_binary(M, B[i]) for i in range(len(B))])
    cand = list(np.argsort(-v_add)[: min(prerank, len(B))])
    chosen: List = []
    gains: List[float] = []
    cur = np.empty((len(M), 0), float)
    while cand and len(chosen) < top_k:
        best_e, best_g = None, -1.0
        for e in cand:
            g = _cmi_block(M, cur, B[e], seed=seed)
            if g > best_g:
                best_e, best_g = e, g
        if best_e is None or best_g < eps:
            break
        chosen.append(int(best_e))
        gains.append(float(best_g))
        cur = np.hstack([cur, B[best_e][:, None].astype(float)])
        cand.remove(best_e)
    selected_units = [int(e) for e in chosen]
    if combiner == "pairs":
        cand_p = [int(x) for x in np.argsort(-v_add)[: min(pair_prerank, len(B))]]
        pairs = [(cand_p[a], cand_p[b]) for a in range(len(cand_p))
                 for b in range(a + 1, len(cand_p))]
        n_pairs = 0
        while pairs and n_pairs < max_pairs and len(chosen) < top_k:
            best_p, best_g = None, -1.0
            for (i, j) in pairs:
                blk = np.stack([B[i], B[j], B[i] * B[j]], axis=1)
                g = _cmi_block(M, cur, blk, seed=seed)
                if g > best_g:
                    best_p, best_g = (i, j), g
            if best_p is None or best_g < eps:
                break
            i, j = best_p
            chosen.append(("pair", int(i), int(j)))
            gains.append(float(best_g))
            cur = np.hstack([cur, np.stack([B[i], B[j], B[i] * B[j]], axis=1).astype(float)])
            selected_units += [int(i), int(j)]
            pairs.remove(best_p)
            n_pairs += 1
    elif combiner != "linear":
        raise ValueError(f"unknown combiner {combiner!r}")
    opt = float(sum(gains))
    tail_frac = float(sum(gains[k_head:]) / opt) if opt > 1e-12 and len(gains) > k_head else 0.0
    return {"selected": chosen, "selected_units": sorted(set(selected_units)), "S_cols": cur,
            "gains": gains, "opt_omega_bits": opt,
            "H_M": h_m, "frac_H": float(opt / h_m) if h_m > 1e-9 else float("nan"),
            "tail_frac": tail_frac, "hill": float(hill_tail_index(v_add)),
            "v_additive": v_add, "k_head": int(k_head), "combiner": combiner}


# --------------------------------------------------------------------------------------------
# D1–D3 — value-weighted Good–Turing flux, McDiarmid slack, Good–Toulmin horizon
# --------------------------------------------------------------------------------------------

def value_spectrum(n_s: np.ndarray, v_s: np.ndarray) -> Dict[int, float]:
    """w_j = Σ_{s : n_s = j} v_s — the value-weighted multiplicity spectrum (§12.6.3)."""
    n_s = np.asarray(n_s, int)
    v_s = np.asarray(v_s, float)
    w: Dict[int, float] = {}
    for j in np.unique(n_s):
        w[int(j)] = float(v_s[n_s == j].sum())
    return w


def flux_certificate(w: Dict[int, float], N: int, B: float, *, delta: float = 0.05) -> dict:
    """D1+D2 flux diagnostic: Vdot = w_1/N (Robbins/value-Good–Turing — expected new-species value
    of the NEXT draw), with the one-sided McDiarmid bound (bounded differences 2B/N) plus the
    Good–Turing bias rider B/N. The concentration statement requires a species map, head, and value
    function frozen independently of the iid capture stream.

    ``B`` must be an a-priori cap on any realizable species value; this pipeline passes ``H(M)``.
    ``delta`` may be the per-readout allocation from :func:`anytime_delta`, but that allocation does not
    repair adaptive head/value estimation. The caller marks the assembled bound invalid."""
    N = max(int(N), 1)
    v_dot = float(w.get(1, 0.0) / N)
    slack = float(B * np.sqrt(2.0 * np.log(1.0 / delta) / N) + B / N)
    return {"flux": v_dot, "slack": slack, "flux_hi": v_dot + slack, "B": float(B),
            "N": N, "delta": float(delta)}


def anytime_delta(N: int, *, delta: float = 0.05, n0: int = 16, n_stats: int = 1) -> dict:
    """§12.8.0 T1 — the per-readout confidence level that makes the D2 band (a) TIME-UNIFORM over the
    deterministic doubling checkpoint grid ``N_j = n0·2^{j−1}`` and (b) SIMULTANEOUS over the
    ``n_stats`` flux statistics whose max is reported (canonical partition + n_orders permutations).

    Why (a): the §12.6.6 decision loop invites continuation (UNDERSAMPLED → draw more → re-issue);
    reading a fixed-N McDiarmid band at a data-chosen N is optional stopping — the pre-2026-07-01
    band was unsound there. The checkpoint index ``j(N) = ⌊log2(max(N,n0)/n0)⌋ + 1`` is a function of
    N ALONE, so ANY adaptive rule that re-issues at checkpoints is covered: allocating
    ``δ_j = δ/(j(j+1))`` spends Σ_j δ_j = δ over ALL checkpoints (telescoping sum), and by union bound
    the level-``δ_j/n_stats`` fixed-N bound holds at every (checkpoint, statistic) pair simultaneously
    w.p. ≥ 1−δ. Price: ln(1/δ) grows by ≈ 2·ln(j+1) + ln(n_stats) — the iterated-logarithm-style cost.

    Why (b): ``eps_bits_adv = max_k ε_k`` is a claim about ALL n_stats bands at once; without the
    union it holds only at level n_stats·δ.

    A statistic issued once at a probe budget declared in advance may use the fixed regime (no checkpoint
    union). This helper allocates error only; it does not establish the freezing assumptions of the bound."""
    j = int(np.floor(np.log2(max(int(N), int(n0)) / float(n0)))) + 1
    n_stats = max(int(n_stats), 1)
    return {"checkpoint": j, "delta_eff": float(delta) / (j * (j + 1) * n_stats),
            "n_stats": n_stats, "n0": int(n0), "delta": float(delta)}


def good_toulmin_value(w: Dict[int, float], c: float = 1.0, *, k0: Optional[int] = 4) -> float:
    """D3: truncated Good–Toulmin point predictor for expected new value in the next c*N draws.

    The infinite Poissonized series is unbiased under its model; truncation at ``j <= k0`` (default 4)
    introduces bias to reduce variance. A plug-in calculation suggests high-count species have small
    unseen probability, while their raw alternating terms contribute O(w_j) sign-oscillation
    variance (on a saturated spectrum like w = {5,6,7} the raw series returns w_5 − w_6 + w_7 ≈ w_jmax,
    manufacturing horizon mass out of parity). Truncation is the hard-cutoff member of the ET/OSW
    smoothing family; pass ``k0=None`` for the raw series. c ≤ 1 only (c is clamped) — for horizons
    c > 1 use :func:`osw_horizon_value` (§12.8.7 I1, implemented 2026-07-01)."""
    c = float(min(max(c, 0.0), 1.0))
    hi = float("inf") if k0 is None else int(k0)
    return float(-sum(((-c) ** j) * wj for j, wj in w.items() if 1 <= j <= hi))


def osw_horizon_value(w: Dict[int, float], t: float, N: int, *, q: Optional[float] = None) -> dict:
    """OSW-style smoothed Good–Toulmin point predictor for horizons ``t > 1``.

    Orlitsky–Suresh–Wu analyze unseen species counts. This value-weighted adaptation is linear but does
    not inherit a one-sided deviation guarantee merely by linearity.

    Estimator:  Ĝ_L(t) = −Σ_{j≥1} (−t)^j · P(L ≥ j) · w_j   with L ~ Binomial(k, q). The binomial
    tail P(L ≥ j) tapers the raw alternating coefficients t^j — whose variance explodes
    exponentially for t > 1 (a ±1 fluctuation in f_5 swings the raw estimate by t^5) — smoothly to
    zero beyond j ≈ k. Parameters per OSW: ``q = 2/(t+2)`` (their optimal-rate smoothing family) and
    k set by their bias–variance balance ``E[t^L]² ≈ N·t²/(t−1)``, i.e.
    ``k = ⌈ ln(N·t²/(t−1)) / (2·ln(1+q(t−1))) ⌉`` (capped at 400 — the cap only binds as t → 1⁺,
    where the taper covers all realized multiplicities anyway and the estimator ≈ raw GT).

    Guarantee for OSW's unseen-count target (Thms 1–3): vanishing normalized MSE for all t up to
    Θ(log N), and prediction is
    provably IMPOSSIBLE beyond — so t is clamped at ln N and the clamp is flagged (right-censor past
    the boundary; never extrapolate). The weighted statistic below is a linear marked-species
    adaptation, but this module does not prove that OSW's worst-case constants transfer when values
    are estimated adaptively from the same probes.

    SCOPE: this is the horizon POINT estimate (bias-controlled per OSW); the certificate's 1−δ band
    remains the T1 flux band (§12.8.0) — no deviation constants are invented here. t ≤ 1 delegates
    to :func:`good_toulmin_value` (raw/ET regime; smoothing unnecessary)."""
    t = float(t)
    N = max(int(N), 1)
    if t <= 1.0:
        return {"estimate": good_toulmin_value(w, t), "t": float(max(t, 0.0)), "t_requested": t,
                "k": None, "q": None, "clamped": False, "family": "et"}
    t_max = max(1.0, float(np.log(N)))
    t_eff = float(min(t, t_max))
    qq = 2.0 / (t_eff + 2.0) if q is None else float(q)
    growth = float(np.log1p(qq * (t_eff - 1.0)))
    k = int(np.ceil(np.log(N * t_eff ** 2 / max(t_eff - 1.0, 1e-9)) / max(2.0 * growth, 1e-9)))
    k = int(min(max(k, 1), 400))
    # Bin(k, q) pmf via the stable multiplicative recurrence (exact combs overflow for large k)
    pmf = np.empty(k + 1)
    pmf[0] = (1.0 - qq) ** k
    for i in range(k):
        pmf[i + 1] = pmf[i] * (k - i) / (i + 1.0) * qq / (1.0 - qq)
    tail = np.concatenate([np.cumsum(pmf[::-1])[::-1], [0.0]])      # tail[j] = P(L ≥ j), j = 0..k+1
    est = 0.0
    for j, wj in sorted(w.items()):
        if j < 1:
            continue
        h_j = float(tail[min(int(j), k + 1)])
        est -= ((-t_eff) ** int(j)) * h_j * wj
    return {"estimate": float(est), "t": t_eff, "t_requested": t, "k": k, "q": float(qq),
            "clamped": bool(t_eff < t), "family": "osw-binomial"}


# --------------------------------------------------------------------------------------------
# the BRIDGE — tail submodularity γ̂ and the ε-gap assembly (§12.6.4)
# --------------------------------------------------------------------------------------------

def tail_gamma(binmat: np.ndarray, M: np.ndarray, selected: Sequence[int], *,
               n_samples: int = 10, block: int = 3, seed: int = 0) -> float:
    """γ̂ — the measured tail-submodularity ratio on the DISCOVERED tail (§6.2 lower-tail estimate,
    conditional on the frozen head S_g): for sampled tail blocks Ω′,
        ratio = Σ_{e∈Ω′} v(e|S_g)  /  v(Ω′ jointly | S_g),
    γ̂ = the lower tail (10th pct) of the ratios, clipped to (0, 1]. ratio ≥ 1 ⇔ submodular direction
    (sum of singles ≥ joint); ratio < 1 ⇔ synergy (the escape the adversarial probe covers). Returns 1.0
    when the tail is empty (nothing unseen to bound)."""
    B = np.asarray(binmat, int)
    M = np.asarray(M, int)
    rng = np.random.default_rng(seed)
    sel = list(selected)
    tail = [i for i in range(len(B)) if i not in set(sel)]
    if not tail:
        return 1.0
    S_g = B[sel].T.astype(float) if sel else np.empty((len(M), 0))
    ratios = []
    for _ in range(n_samples):
        blk = list(rng.choice(tail, size=min(block, len(tail)), replace=False))
        joint = _cmi_block(M, S_g, B[blk].T, seed=seed)
        singles = float(sum(_cmi_block(M, S_g, B[e], seed=seed) for e in blk))
        if joint > 1e-9:
            ratios.append(singles / joint)
    if not ratios:
        return 1.0
    g = float(np.percentile(ratios, 10))
    return float(min(max(g, 1e-3), 1.0))


def epsilon_gap(G_c: float, slack: float, gamma_hat: float) -> float:
    """Historical bridge statistic ``(max(G,0)+slack)/gamma_hat``.

    This is a process-horizon diagnostic, not a confidence upper bound: ``G`` has no implemented
    one-sided horizon deviation bound and ``gamma_hat`` is not a certified lower bound.
    """
    return float((max(G_c, 0.0) + max(slack, 0.0)) / max(gamma_hat, 1e-3))


# --------------------------------------------------------------------------------------------
# per-species conditional values on the quotient partition (the tail's fragile object, §12.6.2)
# --------------------------------------------------------------------------------------------

def species_conditional_values(binmat: np.ndarray, M: np.ndarray, labels: np.ndarray,
                               S_g_cols: np.ndarray, *, idx: Optional[Sequence[int]] = None,
                               seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-species v(s|S_g) (primary, GV7-stationary: S_g is FROZEN) and additive v(s) (the conservative
    upper read — under submodularity v(s|S_g) ≤ v(s), so additive-weighted flux upper-bounds the
    conditional one). Species rep = majority-vote of member rows over ``idx`` (default: all rows).
    Returns (n_s, v_cond, v_add) aligned over the species present in ``idx``."""
    B = np.asarray(binmat, int)
    M = np.asarray(M, int)
    labels = np.asarray(labels)
    idx = list(range(len(B))) if idx is None else list(idx)
    members: Dict[int, List[int]] = {}
    for i in idx:
        if int(labels[i]) >= 0:
            members.setdefault(int(labels[i]), []).append(i)
    species = sorted(members)
    n_s = np.array([len(members[s]) for s in species], int)
    v_cond, v_add = [], []
    for s in species:
        rep = (B[members[s]].mean(axis=0) > 0.5).astype(int)
        v_add.append(i_binary(M, rep))
        v_cond.append(_cmi_block(M, S_g_cols, rep, seed=seed))
    return n_s, np.asarray(v_cond, float), np.asarray(v_add, float)


# --------------------------------------------------------------------------------------------
# orchestrator — one metric checkpoint → the §12.6 certificate report
# --------------------------------------------------------------------------------------------

def certificate(sigs: np.ndarray, M_i: np.ndarray, tags: Sequence, *,
                labels: Optional[np.ndarray] = None, prompts: Optional[Sequence[str]] = None,
                quotient: str = "behavioral", judge_fn=None, cmi_thresh: float = 0.15,
                hi_nmi: float = 0.5, max_judge_pairs: int = 300, c: float = 1.0,
                delta: float = 0.05, top_k: int = 30, adv_saturated: Optional[bool] = None,
                probe_subfrac: float = 0.8, combiner: str = "linear", n_orders: int = 8,
                stopping: str = "anytime", seed: int = 0) -> dict:
    """Historical §12.6 diagnostic for one metric: achieved head, flux, horizon prediction, and bridge.

    The returned payload explicitly marks the assembled upper bound invalid. ``quotient``: 'legacy' = conditional_species at the 0.15 significance threshold
    (status quo — over-merge-prone); 'behavioral' = strict identity-level partition (hi_nmi, no judge);
    'judge' = full §12.6.2 quotient (strict + judge-nominated merges; pass ``judge_fn`` or one is built
    from GLM — quota!). ``adv_saturated`` = precomputed ``orthogonalize.adversarial_saturation``
    verdict if available (None = not run and therefore not a passing gate) — the probe set
    MUST include holistic/composed prompts (``composition_gap.holistic_probe_prompts``) or the
    composition escape stays open.

    ``combiner`` names the readout class F used by the achieved checklist: 'linear' = F₁
    additive-logistic (parity-blind — the documented XOR limitation); 'pairs' = F₂ adds a
    pairwise-interaction stage on the head. This is a measured checklist value under F — never
    prompt-space OPT (composition is measured separately: ``composition_gap.delta_comp``).

    ``n_orders`` > 0 adds the ORDER-ADVERSE band: the flux/ε leg is recomputed under ``n_orders``
    core-build permutations of the strict behavioral partition (the head is order-free by
    construction) and ``eps_bits_adv`` = the max — permutations QUANTIFY the partition's
    order-arbitrariness, so verdicts (``alpha_probe.decide``) read the adverse end. Under
    ``quotient='judge'`` the band is still computed on the strict behavioral partition (re-judging
    per order would multiply GLM quota); the judge-vs-behavioral Δε is reported by running both
    quotient modes.

    ``stopping`` controls the intended delta allocation over checkpoints and order statistics. The
    unconditional H(M) is the only fixed per-species information cap used below. Because the same pool
    selects the head and values, this allocation does not establish coverage; it also supplies no horizon
    deviation theorem."""
    S = np.asarray(sigs, float)
    M = (np.asarray(M_i, float) > 0.5).astype(int)
    B = (np.nan_to_num(S, nan=0.5) > 0.5).astype(int)
    tags = [str(t) for t in tags]
    iid = [i for i, t in enumerate(tags) if t not in ("children", "gepa")]

    audit = None
    if labels is None:
        if quotient == "legacy":
            labels = ap.conditional_species(S, cmi_thresh=cmi_thresh)
        elif quotient == "behavioral":
            labels = ap.conditional_species(S, cmi_thresh=hi_nmi)
        elif quotient == "judge":
            labels, audit = ap.quotient_species(S, prompts, judge_fn=judge_fn,
                                                cmi_thresh=cmi_thresh, hi_nmi=hi_nmi,
                                                max_judge_pairs=max_judge_pairs, seed=seed)
        else:
            raise ValueError(f"unknown quotient mode {quotient!r}")

    # HEAD — partition-free greedy on the raw pool (all sources: Ω is Ω)
    head = greedy_head(B, M, top_k=top_k, seed=seed, combiner=combiner)
    S_g_cols = head["S_cols"]

    # FLUX — quotient species over the frozen-iid stream only (guard G1)
    n_s, v_cond, v_add = species_conditional_values(B, M, labels, S_g_cols, idx=iid, seed=seed)
    N_iid = len(iid)

    # δ-spend (§12.8.0 T1): union over the n_orders+1 simultaneous statistics, plus (anytime) the
    # doubling-checkpoint grid so the §12.6.6 continuation loop cannot invalidate the band.
    n_stats = 1 + (int(n_orders) if n_orders and n_orders > 0 else 0)
    if stopping == "anytime":
        ad = anytime_delta(N_iid, delta=delta, n_stats=n_stats)
    elif stopping == "fixed":
        ad = {"checkpoint": None, "delta_eff": float(delta) / n_stats, "n_stats": n_stats,
              "n0": None, "delta": float(delta)}
    else:
        raise ValueError(f"unknown stopping regime {stopping!r} (anytime|fixed)")
    d_eff = ad["delta_eff"]

    w_cond = value_spectrum(n_s, v_cond)
    w_add = value_spectrum(n_s, v_add)
    # H(M) is fixed before head selection and bounds every conditional/additive information value.
    # H(M)-estimated_head is data-dependent and can understate the required bounded-differences cap.
    resid_cap = float(max(0.0, head["H_M"] - head["opt_omega_bits"]))
    B_cond = float(head["H_M"])
    B_add = max(float(v_add.max()) if v_add.size else 0.0, float(head["H_M"]))
    fc = flux_certificate(w_cond, N_iid, B_cond, delta=d_eff)
    fa = flux_certificate(w_add, N_iid, B_add, delta=d_eff)
    # Horizon point-predictor router: truncated series for c <= 1; OSW-style smoothing beyond 1.
    osw_meta = None
    if c > 1.0:
        osw_meta = osw_horizon_value(w_cond, c, N_iid)
        c_eff = osw_meta["t"]
        _horizon = lambda w_: osw_horizon_value(w_, c, N_iid)["estimate"]
        G_cond = osw_meta["estimate"]
    else:
        c_eff = float(min(max(c, 0.0), 1.0))
        _horizon = lambda w_: good_toulmin_value(w_, c)
        G_cond = good_toulmin_value(w_cond, c)
    G_add = _horizon(w_add)

    # probe-subsample sensitivity of w_1 (the over-merge / merge-precision knob, §12.6.2)
    rng = np.random.default_rng(seed)
    npr = B.shape[1]
    sub = sorted(rng.choice(npr, max(2, int(probe_subfrac * npr)), replace=False))
    n_s2, v_cond2, _ = species_conditional_values(B[:, sub], M[sub], labels,
                                                  S_g_cols[sub], idx=iid, seed=seed)
    flux_sub = float(value_spectrum(n_s2, v_cond2).get(1, 0.0) / max(N_iid, 1))

    # BRIDGE
    gamma_hat = tail_gamma(B, M, head["selected_units"], seed=seed)
    eps_cond = epsilon_gap(G_cond, fc["slack"], gamma_hat)
    eps_add = epsilon_gap(G_add, fa["slack"], gamma_hat)

    # ORDER-ADVERSE band (§12.6.6): the partition-dependent leg under permuted core-build orders.
    # The head/S_g is fixed (order-free); only the species labeling — hence (w_j, flux, Ĝ, ε) — moves.
    order_band = None
    eps_adv = eps_cond
    if n_orders and n_orders > 0:
        rng_o = np.random.default_rng(seed + 1)
        eps_k, flux_k, D_k = [], [], []
        for _ in range(int(n_orders)):
            lab_k = ap.conditional_species(S, cmi_thresh=hi_nmi,
                                           order=list(rng_o.permutation(len(S))))
            n_k, v_k, _ = species_conditional_values(B, M, lab_k, S_g_cols, idx=iid, seed=seed)
            w_k = value_spectrum(n_k, v_k)
            B_k = float(head["H_M"])
            fc_k = flux_certificate(w_k, N_iid, B_k, delta=d_eff)
            eps_k.append(epsilon_gap(_horizon(w_k), fc_k["slack"], gamma_hat))
            flux_k.append(fc_k["flux_hi"])
            D_k.append(int(len(n_k)))
        eps_adv = float(max([eps_cond] + eps_k))
        order_band = {"n_orders": int(n_orders), "eps_min": float(min(eps_k)),
                      "eps_max": float(max(eps_k)), "flux_hi_min": float(min(flux_k)),
                      "flux_hi_max": float(max(flux_k)), "D_min": int(min(D_k)),
                      "D_max": int(max(D_k)),
                      "note": "band on the strict behavioral partition (judge merges not re-run)"}

    f1_over_N = float((n_s == 1).sum() / max(N_iid, 1)) if n_s.size else float("nan")
    rep = {
        "H_M": head["H_M"], "opt_omega_bits": head["opt_omega_bits"], "frac_H": head["frac_H"],
        "n_head": len(head["selected"]), "gains": head["gains"], "tail_frac": head["tail_frac"],
        "hill": head["hill"], "gamma_tail": gamma_hat, "combiner": head["combiner"],
        "head_selected": head["selected"],
        "N_iid": N_iid, "D_iid": int(len(n_s)), "f1_over_N": f1_over_N,
        "w_spectrum_cond": {str(k): v for k, v in sorted(w_cond.items())},
        "flux_cond": fc["flux"], "slack": fc["slack"], "flux_cond_hi": fc["flux_hi"],
        "flux_cond_probe_sub": flux_sub, "G_c_cond": G_cond,
        "flux_add_hi": fa["flux_hi"], "G_c_add": G_add,
        "eps_bits": eps_cond, "eps_bits_adv": eps_adv, "order_band": order_band,
        "eps_bits_add": eps_add, "c": c_eff, "c_requested": float(c),
        "horizon_estimator": "osw" if c > 1.0 else "et", "osw": osw_meta,
        "delta": float(delta), "quotient": quotient, "quotient_audit": audit,
        "adv_saturated": adv_saturated,
        "head_value_bits": head["opt_omega_bits"],
        "head_is_achieved_lower_bound": True,
        "upper_bound_valid": False,
        "bound_status": "HEURISTIC_PROCESS_HORIZON",
        "upper_bound_invalid_reasons": [
            "head selection is adaptive on the value-estimation probes",
            "horizon estimator has no implemented one-sided deviation bound",
            "gamma_tail is not a certified lower bound for unseen synergy",
            "adversarial saturation must be explicitly true",
        ],
        # Error-allocation metadata; it does not validate the assembled bridge.
        "stopping": stopping, "checkpoint": ad["checkpoint"], "delta_effective": float(d_eff),
        "n_union": int(n_stats), "B_flux_cap": float(B_cond), "resid_cap": resid_cap,
    }
    return rep


# --------------------------------------------------------------------------------------------
# CLI — per-checkpoint tables + the §12.6.5 scaling read
# --------------------------------------------------------------------------------------------

def _run_dir(d: str, a, judge_fn=None) -> List[dict]:
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*_metric*_sigs.npz"))):
        try:
            z = np.load(f, allow_pickle=True)
            S = np.asarray(z["sigs"], float)
            tags = list(z["tags"])
            prompts = list(z["prompts"]) if "prompts" in z.files else None
            name = str(z["name"]) if "name" in z.files else os.path.basename(f)
            if "M_i" not in z.files:
                print(f"  skip {os.path.basename(f)}: no M_i in checkpoint")
                continue
            M_i = np.asarray(z["M_i"], float)
        except Exception as e:                                       # unreadable ckpt
            print(f"  skip {os.path.basename(f)}: {e}")
            continue
        cert = certificate(S, M_i, tags, prompts=prompts, quotient=a.quotient, judge_fn=judge_fn,
                           cmi_thresh=a.cmi_thresh, hi_nmi=a.hi_nmi,
                           max_judge_pairs=a.max_judge_pairs, c=a.c, delta=a.delta,
                           top_k=a.top_k, combiner=a.combiner, n_orders=a.n_orders,
                           stopping=a.stopping, seed=a.seed)
        fi_path = f.replace("_sigs.npz", "_forminv.json")
        fi = json.load(open(fi_path)) if os.path.exists(fi_path) else None
        verdict = ap.decide({"f1_over_N": cert["f1_over_N"]}, form_invariance=fi, certificate=cert)
        rows.append({"file": os.path.basename(f), "name": name, "verdict": verdict,
                     "form_invariant": (fi or {}).get("form_invariant"), **cert})
    return rows


def _print_rows(rows: List[dict]):
    print("%-32s %5s %6s %6s %5s %5s %5s %7s %7s %7s %-16s" % (
        "metric", "H_M", "OPT_Ω", "%H", "head", "tailF", "γ̂", "f1/N", "flux_hi", "ε_adv", "verdict"))
    print("-" * 118)
    for r in sorted(rows, key=lambda x: -(x["frac_H"] if np.isfinite(x["frac_H"]) else -1)):
        print("%-32s %5.2f %6.3f %6.2f %5d %5.2f %5.2f %7.3f %7.4f %7.4f %-16s" % (
            r["name"][:32], r["H_M"], r["opt_omega_bits"],
            r["frac_H"] if np.isfinite(r["frac_H"]) else float("nan"), r["n_head"],
            r["tail_frac"], r["gamma_tail"], r["f1_over_N"], r["flux_cond_hi"],
            r.get("eps_bits_adv", r["eps_bits"]), r["verdict"]))


def _scaling(specs: List[Tuple[str, str]], a, ceiling: Optional[float]):
    """Descriptive per-tier achieved ``OPT_Omega`` plus heuristic epsilon sensitivity."""
    per: Dict[str, Dict[str, dict]] = {}
    tier_names = []
    for name, d in specs:
        tier_names.append(name)
        print(f"\n### tier = {name}  ({d})")
        rows = _run_dir(d, a)
        _print_rows(rows)
        for r in rows:
            per.setdefault(r["name"], {})[name] = r
    print("\n" + "=" * 100)
    print("SCALING DIAGNOSTIC (achieved OPT_Ω + heuristic ε per tier%s)" %
          ("; descriptive contrast C - [OPT_Ω+ε]" if ceiling else ""))
    print("=" * 100)
    slopes = []
    hdr = "%-30s" % "metric" + "".join("%14s" % t for t in tier_names) + ("%10s" % "slope" if ceiling else "")
    print(hdr)
    for mname in sorted(per):
        tiers = per[mname]
        if len(tiers) < 2:
            continue
        vals, row = [], "%-30s" % mname[:30]
        for t in tier_names:
            r = tiers.get(t)
            if r is None:
                row += "%14s" % "-"
                vals.append(np.nan)
            else:
                e_adv = r.get("eps_bits_adv", r["eps_bits"])       # adverse end (§12.6.6)
                ub = r["opt_omega_bits"] + e_adv
                row += "%14s" % ("%.3f+%.3f" % (r["opt_omega_bits"], e_adv))
                vals.append(ceiling - ub if ceiling else ub)
        vals = np.asarray(vals, float)
        fin = np.isfinite(vals)
        if ceiling and fin.sum() >= 2:
            x = np.arange(len(vals))[fin]
            sl = float(np.polyfit(x, vals[fin], 1)[0])
            slopes.append(sl)
            row += "%10.4f" % sl
        print(row)
    if ceiling and slopes:
        sl = np.asarray(slopes)
        boot = [np.mean(np.random.default_rng(i).choice(sl, len(sl))) for i in range(500)]
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        verdict = ("FLAT descriptive trend" if lo <= 0.0 <= hi
                   else "SHRINKING — right-censor: 'not yet articulated at this capability'"
                   if np.mean(sl) < 0 else "WIDENING (check estimator: gap should not grow)")
        print(f"\nheuristic contrast slope: mean={float(np.mean(sl)):+.4f}  "
              f"bootstrap95%=[{lo:+.4f},{hi:+.4f}]  -> {verdict}")
        print("(descriptive only; epsilon is not a bound and fitted asymptotes are not certificates)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="value_certificate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=None, help="one executor's checkpoint dir")
    p.add_argument("--scaling", default=None, help="name:dir,name:dir,… capability-ORDERED tiers")
    p.add_argument("--ceiling", type=float, default=None,
                   help="task ceiling C lowerCI in BITS (same units as OPT_Ω) for the Δ(E) read")
    p.add_argument("--quotient", default="behavioral", choices=("legacy", "behavioral", "judge"))
    p.add_argument("--glm-model", default="glm-4.7")
    p.add_argument("--max-judge-pairs", type=int, default=300)
    p.add_argument("--cmi-thresh", type=float, default=0.15)
    p.add_argument("--hi-nmi", type=float, default=0.5)
    p.add_argument("--c", type=float, default=1.0,
                   help="horizon: c ≤ 1 = ET-truncated Good–Toulmin; 1 < c ≤ ln N = OSW binomial "
                        "smoothing (§12.8.7 I1; clamped + flagged at ln N)")
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--combiner", default="linear", choices=("linear", "pairs"),
                   help="readout class F used by the achieved checklist diagnostic")
    p.add_argument("--n-orders", type=int, default=8,
                   help="order-adverse band size (0 = skip; §12.6.6)")
    p.add_argument("--stopping", default="anytime", choices=("anytime", "fixed"),
                   help="delta-allocation regime for the flux sensitivity; this does not validate the bridge")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)

    judge_fn = None
    if a.quotient == "judge":
        judge_fn = ap.make_glm_semantic_judge(a.glm_model)       # GLM quota — candidates are capped

    if a.scaling:
        specs = [s.split(":", 1) for s in a.scaling.split(",") if ":" in s]
        _scaling(specs, a, a.ceiling)
        return
    if not a.dir:
        p.error("need --dir or --scaling")
    rows = _run_dir(a.dir, a, judge_fn=judge_fn)
    print(f"=== §12.6 value diagnostic (upper_bound_valid=False): {a.dir}  "
          f"(quotient={a.quotient}, c={a.c}, delta={a.delta}) ===")
    _print_rows(rows)
    from collections import Counter
    print("\nverdicts:", dict(Counter(r["verdict"] for r in rows)))
    if a.json:
        json.dump([_no_np(r) for r in rows], open(a.json, "w"), indent=1)
        print(f"json → {a.json}")


def _no_np(o):
    if isinstance(o, dict):
        return {k: _no_np(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_no_np(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _no_np(o.tolist())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


if __name__ == "__main__":
    main()
