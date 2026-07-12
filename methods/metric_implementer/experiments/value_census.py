"""§12.3 VALUE-CENSUS — population dynamics on RECOVERING M_i, per metric (NOT the aggregate Y).
Theory: ``notes/2026-06-18__prompt-optimality-theory.md`` §12.3 (reframed metric-level + reconstruction-
based 2026-06-26).

Per metric M_i (one R2 cluster — §1). The behavior census (§12.1a) puts the COUNTING measure on M_i's
criteria and asks how fast new ones appear (α_i) — SILENT on value: α_i≈1 says M_i has inexhaustibly many
EXPRESSIBLE criteria, NOT that adding them keeps recovering M_i. The value census swaps in the VALUE
measure — mass = each criterion's contribution to RECOVERING M_i — and re-runs the same machinery. Where
α_i≈1 destroyed the behavior map, α_{V,i} can be small. That contrast describes this realized sample;
it does not show that ``T_i`` is reachable or that unseen value is exhausted.

ANCHOR-FREE / UNSUPERVISED: value is measured against M_i's OWN verdict pattern, **never the aggregate**
``Y = f(M_1,…,M_j)`` (the original §12.3 used I(Y;σ) — a slip that conflated all j metrics and contradicted
§1's label-free loop; corrected here). ``M_i = alpha_probe.metric_verdict`` = the executor's soft P(YES)
on the probe items using the cluster ``merged_description`` as the rubric (the reconstruction TARGET);
full-strength recovery uses ``recon_channel.run_metric`` → ``iv_transmission``.

UNITS — Shannon BITS. ``v(s) = I(M_i; bin σ(s))`` is a standalone-value diagnostic that can
over-count redundancy and under-count synergy. ``v(s|selected)`` is a greedy conditional-value
diagnostic. Mutual information is not submodular in general, so neither curve is a certified
upper or lower bound without an additional structural assumption.

CPU-ONLY: runs on the persisted per-metric ``…_metric{idx}_sigs.npz`` (which stores M_i alongside the
criterion signatures). No GPU, no new draws, no Y anywhere.

GUARDS (theory §12.3 E):
  GV2  additive and conditional-greedy values are descriptive. Redundancy can make the additive
       sum too large; synergy can make it too small. Report both without an ordering claim.
  GV3  no certificate is issued from the same-sample value map. A separate independently frozen map can
       support a bounded next-draw flux statement, but not an optimum-gap claim.
  GV4  SAME frozen sample, SAME τ as M_i's behavior census (collide at the run's τ) so α_i and α_{V,i}
       read the SAME species partition — else the breadth gap α_i − α_{V,i} is meaningless.
  GV5  METRIC SCOPE: M_i, Ω_i, the breadth sample, and the value target are ALL for ONE R2 cluster.
       Task-level pooling (the α≈0.9 sweep) is the WRONG level — never run this on a task-pooled sample.

Reuse (§12.3 C): species partition ``alpha_probe.collide``; Heaps ``alpha_probe.heaps_alpha`` /
``_terminal_alpha``; Shannon CMI ``orthogonalize.shannon_cmi_surrogate``; binary entropy ``vinfo._h_bits``;
M_i verdict ``alpha_probe.metric_verdict`` (saved in the checkpoint). The value rarefaction is
``alpha_probe._rarefy``'s gammaln form with per-species v_s weights instead of the count.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import vinfo
from .alpha_probe import collide, heaps_alpha, _terminal_alpha
from .orthogonalize import shannon_cmi_surrogate


# --------------------------------------------------------------------------------------------
# additive marginal value  v(s) = I(M_i ; bin sigma(s)) (closed-form binary Shannon MI, bits)
# --------------------------------------------------------------------------------------------

def i_binary(target: np.ndarray, predictor: np.ndarray) -> float:
    """Closed-form Shannon MI I(target; predictor) in BITS for two binary vectors (exact — no surrogate
    needed for a single binary predictor). The additive per-species value: how much the species'
    binarized verdict reduces uncertainty about M_i. 0 when the predictor is constant or independent of
    M_i. (Generic in `target`; called with M_i as the recovery target — never the aggregate Y.)"""
    y = np.asarray(target, int)
    x = np.asarray(predictor, int)
    n = len(y)
    if n == 0 or len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return 0.0
    h_y = float(vinfo._h_bits(float(y.mean())))
    h_y_given_x = 0.0
    for xv in (0, 1):
        m = x == xv
        if not m.any():
            continue
        h_y_given_x += float(m.mean()) * float(vinfo._h_bits(float(y[m].mean())))
    return float(max(0.0, h_y - h_y_given_x))


def _species_bin_signatures(sigs: np.ndarray, labels: np.ndarray) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Collapse to one binarized verdict vector per species (σ̄_s = nanmean of member σ over probes,
    binarized > 0.5). Returns (species ids, n_s, sp_binmat (n_probes, n_species))."""
    labels = np.asarray(labels)
    species = sorted(set(int(l) for l in labels))
    n_s = np.array([int(np.sum(labels == s)) for s in species])
    cols = []
    for s in species:
        sigbar = np.nanmean(sigs[labels == s], axis=0)        # (n_probes,)
        cols.append((np.nan_to_num(sigbar, nan=0.5) > 0.5).astype(int))
    sp_bin = np.array(cols).T if cols else np.zeros((sigs.shape[1], 0), int)   # (n_probes, n_species)
    return species, n_s, sp_bin


def additive_value(sigs: np.ndarray, labels: np.ndarray, M_i: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-species additive value v_s = I(M_i; bin sigma_bar_s) in bits.

    This is standalone recovery. Summing it can over-count redundancy or under-count synergy, so it has
    no bound direction. Returns (species ids, n_s, v_s).
    """
    M_i = np.asarray(M_i, int)
    species, n_s, sp_bin = _species_bin_signatures(sigs, labels)
    v_s = np.array([i_binary(M_i, sp_bin[:, i]) for i in range(len(species))])
    return np.asarray(species), n_s, v_s


# --------------------------------------------------------------------------------------------
# value rarefaction (EXCHANGEABLE — the clean parallel to behavior's S(m))  +  value Heaps α_V
# --------------------------------------------------------------------------------------------

def value_rarefaction(n_s: np.ndarray, v_s: np.ndarray, N: int, *, n_points: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """E[V(m)] = Σ_s v_s · (1 − C(N−n_s, m)/C(N, m)) — the expected ADDITIVE value a RANDOM m-subsample
    captures (the value-axis analog of the behavior rarefaction E[S(m)]). Computed via log-gamma for
    numerical stability; the per-species weight ``v_s`` rides in place of the count. This is the
    EXCHANGEABLE (random-subset) value curve — the apples-to-apples partner of behavior's S(m), NOT the
    greedy best-subset returns curve (that is ``submodular_value`` below)."""
    from scipy.special import gammaln
    lg = gammaln
    n_s = np.asarray(n_s, float)
    v_s = np.asarray(v_s, float)
    ms = np.unique(np.linspace(1, N, min(n_points, N)).astype(int))
    V = np.zeros(len(ms))
    for idx, m in enumerate(ms):
        if m >= N:
            V[idx] = float(v_s.sum())
            continue
        log_CN_m = lg(N + 1) - lg(m + 1) - lg(N - m + 1)
        total = 0.0
        for j, vs in zip(n_s, v_s):
            if vs <= 0:
                continue
            nrem = N - j
            if nrem < m:                                       # species always captured in any m-subset
                total += vs
                continue
            log_ratio = (lg(nrem + 1) - lg(m + 1) - lg(nrem - m + 1)) - log_CN_m   # log C(N−j,m)/C(N,m)
            total += vs * (1.0 - float(np.exp(log_ratio)))
        V[idx] = total
    return ms, V


# --------------------------------------------------------------------------------------------
# conditional marginal-value diagnostic (greedy Shannon-CMI surrogate)
# --------------------------------------------------------------------------------------------

def submodular_value(sigs: np.ndarray, labels: np.ndarray, M_i: np.ndarray, *, top_k: int = 150,
                     eps: float = 0.005, seed: int = 0) -> dict:
    """Greedy Shannon-CMI selection of species by marginal value v(s|selected) = I(M_i; σ(s) | σ(selected))
    — each species' CONDITIONAL recovery of M_i. This is a descriptive greedy read: it discounts
    redundancy, but synergy can make conditional value exceed standalone value. Returns the greedy
    ordered marginal-gain sequence, the heuristic returns curve V_best(m) = sum_{k<=m} g_k, and the
    diagnostic total R_full = sum g_k. The gains need not be monotone.

    SCALING: greedy is O(steps × candidates × CMI-fit); full-species greedy is infeasible at breadth
    scale. We restrict to the TOP-``top_k`` species by additive value as a computational heuristic.
    This can miss synergistic low-marginal species and must never be used for bound semantics."""
    M_i = np.asarray(M_i, int)
    species, n_s, sp_bin = _species_bin_signatures(sigs, labels)
    _sp, _ns, v_add = additive_value(sigs, labels, M_i)
    candidates = list(np.argsort(-v_add)[: top_k])                 # rank by additive value, take top-K
    chosen: List[int] = []
    gains: List[float] = []
    cur_X = np.empty((len(M_i), 0), float)
    remaining = list(candidates)
    while remaining:
        best_e, best_g = None, -1.0
        for e in remaining:
            g = float(shannon_cmi_surrogate(M_i, cur_X, x_new=sp_bin[:, e], seed=seed))
            if g > best_g:
                best_e, best_g = e, g
        if best_e is None or best_g < eps:                          # descriptive threshold stop
            break
        chosen.append(int(best_e))
        gains.append(float(best_g))
        cur_X = np.hstack([cur_X, sp_bin[:, [best_e]]])
        remaining.remove(best_e)
        if len(chosen) >= top_k:
            break
    r_full = float(sum(gains))
    v_best = np.cumsum([0.0] + gains).tolist()                      # V_best(m), m = 0..len(gains)
    add_sum_selected = float(v_add[chosen].sum()) if chosen else 0.0
    return {"order": chosen, "marginal_gains": gains, "V_best": v_best, "R_full": r_full,
            "n_selected": len(chosen), "additive_sum_selected": add_sum_selected,
            "redundancy_gap_selected": float(add_sum_selected - r_full)}


# --------------------------------------------------------------------------------------------
# behavioral redundancy trace — the I(M_i ; new | Ω) overlap measure (continuous "recapture")
# --------------------------------------------------------------------------------------------

def redundancy_trace(sigs: np.ndarray, M_i: np.ndarray, *, eps: float = 0.005,
                     max_core: int = 80, seed: int = 0) -> dict:
    """Per-criterion redundancy via the conditional-MI novelty ``I(M_i; σ(s) | Ω_core)``.

    Walks criteria in DRAW ORDER and, for each, measures how much of its M_i-value is ALREADY covered
    by the non-redundant core accumulated so far — a continuous "overlap with the existing pool" read
    that complements binary species-recapture (``collide`` at τ). For each criterion s:

      v(s)        = I(M_i; bin σ(s))                  standalone value
      v(s|core)   = I(M_i; bin σ(s) | bin σ(core))    NEW info beyond the core (the novelty)
      overlap(s)  = 1 − v(s|core)/v(s)                fraction of s's value already spanned (redundancy)

    The core grows draw-order-greedy (add s iff v(s|core) ≥ ε), so it stays low-dimensional (~ the
    thresholded ``n_selected``), keeping the CMI fit stable. ``core_size`` and ``saturation_index`` are
    historical field names for a descriptive draw-order threshold trace. They are not estimates of the
    support size and do not certify saturation.

    No ``collide`` / no τ: works at the criterion level, so paraphrase-splitting cannot inflate it.
    """
    M_i = np.asarray(M_i, int)
    sigs = np.asarray(sigs, float)
    binmat = (np.nan_to_num(sigs, nan=0.5) > 0.5).astype(int)          # (n_criteria, n_probes)
    core_X = np.empty((len(M_i), 0), int)
    core_idx: List[int] = []
    overlaps, novelties, standalones = [], [], []
    cum = 0.0
    for i in range(len(sigs)):
        v_s = float(i_binary(M_i, binmat[i]))
        v_cond = float(shannon_cmi_surrogate(M_i, core_X, x_new=binmat[i], seed=seed))
        overlap = 1.0 - (v_cond / v_s) if v_s > 1e-9 else 1.0
        standalones.append(v_s); novelties.append(v_cond); overlaps.append(overlap)
        cum += v_cond
        if v_cond >= eps and len(core_idx) < max_core:
            core_idx.append(i)
            core_X = np.hstack([core_X, binmat[i:i + 1].T])
    return {"n": len(sigs), "core_size": len(core_idx), "core_idx": core_idx,
            "saturation_index": (core_idx[-1] if core_idx else -1),
            "cum_novelty": float(cum),
            "overlaps": np.asarray(overlaps), "novelties": np.asarray(novelties),
            "standalones": np.asarray(standalones)}


# --------------------------------------------------------------------------------------------
# value singleton-flux diagnostic and conditional split-sample bound
# --------------------------------------------------------------------------------------------

def value_missing_mass(n_s: np.ndarray, v_s: np.ndarray, N: int, *, b_cap: float,
                       delta: float = 0.05, frozen_value_map: bool = False) -> dict:
    """Value-weighted singleton flux and, conditionally, a fixed-N upper bound.

    ``sum(singleton values)/N`` is descriptive unless the species partition and
    per-species values were frozen independently of these iid capture draws. Under
    that freezing premise and a predeclared cap, McDiarmid gives

      E[next unseen-value flux] <= w1/N + B sqrt(2 log(1/delta)/N) + B/N.

    The high-level ``value_census`` does not meet the freezing premise and never
    issues this certificate.
    """
    n_s = np.asarray(n_s)
    v_s = np.asarray(v_s, float)
    N = max(int(N), 1)
    mv0 = float(v_s[n_s == 1].sum() / N)
    B = float(b_cap)
    if not np.isfinite(B) or B < 0.0:
        raise ValueError("b_cap must be finite and nonnegative")
    if v_s.size and float(v_s.max()) > B + 1e-12:
        raise ValueError("b_cap is smaller than an observed species value")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    cert_hi = None
    if frozen_value_map:
        cert_hi = min(B, mv0 + B * float(np.sqrt(2.0 * np.log(1.0 / delta) / N)) + B / N)
    return {"MV0": mv0, "B": B, "b_cap_predeclared": True,
            "cert_hi": None if cert_hi is None else float(cert_hi),
            "certificate_valid": bool(frozen_value_map), "delta": float(delta),
            "conditional_on": "iid captures and an independently frozen species/value map"}


# --------------------------------------------------------------------------------------------
# Hill tail-index on per-species value  (the value-axis analog of α for the 0≪α_V<α case, §12.3 B)
# --------------------------------------------------------------------------------------------

def hill_tail_index(v_s: np.ndarray, *, top_frac: float = 0.1) -> float:
    """Hill estimator ξ on the upper tail of the per-species additive-value distribution. SMALL ξ ⇒
    light tail (DESCRIPTIVE only — review 2026-07-11: Hill estimates a tail index and does NOT
    establish a finite endpoint or a stopping rule); LARGE ξ describes a heavier fitted tail.
    Unstable for tiny tails → NaN."""
    v = np.sort(np.asarray(v_s, float))[::-1]                       # descending
    v = v[v > 0]
    if len(v) < 20:
        return float("nan")
    k = max(5, int(len(v) * top_frac))
    if k >= len(v):
        return float("nan")
    return float(np.mean(np.log(v[:k]) - np.log(v[k])))


# --------------------------------------------------------------------------------------------
# orchestrator  — the full §12.3 value report on a frozen per-metric breadth sample (+ M_i)
# --------------------------------------------------------------------------------------------

def value_census(sigs: np.ndarray, M_i: np.ndarray, *, tau: float, delta: float = 0.05,
                 cmi_seed: int = 0, submodular_top_k: int = 150,
                 prompts: Optional[Sequence[str]] = None) -> dict:
    """Run the full §12.3 value census on ONE metric's frozen breadth sample + its OWN verdict M_i.

    Inputs: ``sigs`` (n_criteria, n_probes) criterion signatures for M_i, from the SAME frozen sample as
    M_i's behavior census; ``M_i`` (n_probes,) the metric's OWN verdict (anchor-free reconstruction
    target — ``metric_verdict`` on the cluster ``merged_description``, NEVER the aggregate Y); ``tau`` the
    collide threshold (GV4 — MUST match the behavior run's τ). Returns α_{V,i}, the breadth-gap partner of
    α_i, a descriptive singleton-flux estimate, additive/conditional-greedy diagnostics, the Hill
    index, and the top-value species (the criteria that actually recover M_i)."""
    sigs = np.asarray(sigs, float)
    M_i = np.asarray(M_i, int)
    labels = collide(sigs, tau)                                     # SAME partition as the behavior census (GV4)
    species, n_s, v_add = additive_value(sigs, labels, M_i)
    N = int(len(labels))
    ms, V = value_rarefaction(n_s, v_add, N)
    alpha_V_traj = heaps_alpha(ms, V)
    alpha_V = _terminal_alpha(ms, alpha_V_traj)
    h_mi = float(vinfo._h_bits(float(M_i.mean()))) if np.unique(M_i).size > 1 else 0.0
    mm = value_missing_mass(n_s, v_add, N, b_cap=h_mi, delta=delta, frozen_value_map=False)
    sub = submodular_value(sigs, labels, M_i, top_k=submodular_top_k, seed=cmi_seed)
    hill = hill_tail_index(v_add)

    # the value MAP: top species by additive v, with a representative criterion string (the modal prompt)
    top_idx = list(np.argsort(-v_add)[: min(20, len(species))])
    top_species = []
    for si in top_idx:
        sp = int(species[si])
        members = [i for i, l in enumerate(labels) if int(l) == sp]
        rep = ""
        if prompts is not None:
            cands = [prompts[m] for m in members if prompts and m < len(prompts) and prompts[m]]
            rep = max(set(cands), key=cands.count) if cands else ""
        top_species.append({"species": sp, "n_captures": int(n_s[si]), "v_additive": float(v_add[si]),
                            "representative_criterion": rep})

    n_singleton = int((n_s == 1).sum())
    f_spectrum = dict(sorted(((int(j), int(c)) for j, c in
                              zip(*np.unique(n_s, return_counts=True))))) if len(n_s) else {}
    report = {
        "N": N, "D": len(species), "tau": float(tau),
        "n_singleton_species": n_singleton, "f_spectrum": f_spectrum,
        "alpha_V_trajectory": [float(x) for x in alpha_V_traj],
        "alpha_V_terminal": float(alpha_V),
        "value_rarefaction_m": [int(x) for x in ms],
        "value_rarefaction_V": [float(x) for x in V],
        "MV0": mm["MV0"], "value_cert_hi": mm["cert_hi"], "value_B": mm["B"],
        "value_certificate_valid": mm["certificate_valid"],
        "additive_value_sum": float(v_add.sum()),
        "submodular": {"R_full": sub["R_full"], "n_selected": sub["n_selected"],
                       "marginal_gains": sub["marginal_gains"], "V_best": sub["V_best"],
                       "redundancy_gap_selected": sub["redundancy_gap_selected"]},
        "hill_tail_index": hill,
        "top_value_species": top_species,
    }
    return report
