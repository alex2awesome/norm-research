"""Information-theoretic orthogonalization of a mined criterion pool -> a behaviorally-atomic Ω,
plus the two-pronged "magic word" stopping criterion and the order-independence permutation test.

This is the constructive machinery the prompt-optimality theory (notes/2026-06-18) asks for in
§6.5/§6.6 (build Ω so the conditional-submodularity theorem's CI-given-M structure holds *by
construction*, not by assumption), §6.7c/§6.9 (replace the Good–Turing missing-MASS defense, which
cannot see missing IMPACT, with a submodular tail-bound + adversarial behavioral saturation), and
§6.8 (the permutation test that justifies the set-abstraction f(S)=R(C(S))).

WHY SHANNON HERE, TVD ELSEWHERE.  The orthogonalization filter needs high-dimensional CONDITIONAL
mutual information I(X_e ; X_Ω) — and TVD-MI has no chain rule, so a plug-in over the 2^|Ω| joint is
noise. We therefore estimate the filter's Shannon CMI as the cross-entropy REDUCTION of a surrogate
classifier (a chain-rule-friendly, well-sampled proxy). Once Ω is FROZEN, the recovery certificates
(Rung-1 cap, Rung-2 within-class) stay in the gaming-robust TVD-MI (vinfo.tvd_*). Two f's for two
jobs, exactly as §6 prescribes — not a conflict to collapse.

numpy core; sklearn imported lazily (mirrors external_aggregator._oof_logreg); TVD pieces reuse
vinfo + external_aggregator.tvd_mi_joint. ZERO GPU.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .. import vinfo
from .external_aggregator import tvd_mi_joint

_EPS = 1e-12


# --------------------------------------------------------------------------------------------
# Shannon CMI via a surrogate classifier (the chain-rule-friendly filter proxy)
# --------------------------------------------------------------------------------------------

def _oof_ce(X: np.ndarray, y: np.ndarray, *, seed: int = 0) -> float:
    """Cross-entropy CE(y | X) in BITS from 5-fold out-of-fold logistic-regression probabilities
    (mirrors external_aggregator._oof_logreg). X may have 0 columns -> the surrogate predicts the
    train-fold class prior, so CE(y | ∅) ≈ H(y). y binary."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    X = np.asarray(X, float)
    y = np.asarray(y, int)
    if X.ndim == 1:
        X = X[:, None]
    n = len(y)
    if np.unique(y).size < 2:                    # degenerate target -> zero entropy
        return 0.0
    oof = np.full(n, np.nan)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        if np.unique(y[tr]).size < 2 or X.shape[1] == 0:
            oof[te] = float(np.mean(y[tr]))
        else:
            lr = LogisticRegression(C=1.0, max_iter=2000)
            lr.fit(X[tr], y[tr])
            oof[te] = lr.predict_proba(X[te])[:, 1]
    p = np.clip(oof, _EPS, 1.0 - _EPS)
    return float(-np.mean(y * np.log2(p) + (1 - y) * np.log2(1 - p)))


def shannon_cmi_surrogate(target: Sequence[float], X_cond: np.ndarray,
                          x_new: Optional[Sequence[float]] = None, *, seed: int = 0) -> float:
    """Shannon CMI I(x_new ; target | X_cond) in bits, as the cross-entropy reduction of a 5-fold
    OOF surrogate:  CE(target | X_cond) − CE(target | X_cond, x_new),  clamped ≥ 0.

    `target` binary (N,); `X_cond` (N,d) binary conditioning signals (d may be 0); `x_new` (N,)
    binary or None. With ``x_new is None`` returns the UNCONDITIONAL surrogate MI
    I(X_cond ; target) = H(target) − CE(target | X_cond). This is the chain-rule-friendly proxy the
    orthogonalization filter needs (TVD-MI has none); units are bits."""
    y = np.asarray(target, int)
    Xc = np.asarray(X_cond, float)
    if Xc.ndim == 1:
        Xc = Xc[:, None]
    if Xc.size == 0:
        Xc = np.empty((len(y), 0))
    ce_cond = _oof_ce(Xc, y, seed=seed)
    if x_new is None:
        h_target = float(vinfo._h_bits(float(np.mean(y))))
        return float(max(0.0, h_target - ce_cond))
    xn = np.asarray(x_new, float)[:, None]
    ce_full = _oof_ce(np.hstack([Xc, xn]), y, seed=seed)
    return float(max(0.0, ce_cond - ce_full))


# --------------------------------------------------------------------------------------------
# the orthogonalization filter  (theory §6.5 step 3 / §6.6 — build a behaviorally-atomic Ω)
# --------------------------------------------------------------------------------------------

def orthogonalization_filter(signals: np.ndarray, *, cmi_thresh: float = 0.02,
                             seed: int = 0) -> dict:
    """Iteratively build Ω from a mined candidate pool, keeping only behaviorally-ORTHOGONAL units.

    `signals` is (N_items, n_candidates) binary: column e is candidate criterion e's per-item
    behavior X_e (does it fire on item i). We process candidates in a deterministic order — most
    discriminating first (descending balance −|mean−½|, ties by index) — and add the first
    unconditionally. For each next candidate e we measure how much of its behavior the current Ω
    already explains, info_already = I(X_e ; X_Ω) via `shannon_cmi_surrogate`:

      * info_already ≥ cmi_thresh · H(X_e)  -> e is a redundant PARAPHRASE of what Ω already does
                                               -> DISCARD;
      * info_already <  cmi_thresh · H(X_e)  -> e induces a NEW, orthogonal partition -> KEEP.

    This is the BEHAVIORAL individuation of an "atomic unit" (theory §6.5/§6.6): a unit is the same
    element as another iff it induces the same per-item verdict vector, regardless of wording — so
    near-duplicate phrasings collapse, and we do NOT split a composite the executor processes as one
    behavioral signal ("green and round" stays one unit if it fires as one). Making Ω orthogonal is
    what lets the §6.6 conditional theorem (CI-given-M) approximately hold by construction, so the
    γ it certifies is trustworthy — replacing the spectral lower bound, which assumed a linear-
    regression value function that LLM attention interactions violate.
    """
    S = np.asarray(signals, float)
    n, m = S.shape
    means = S.mean(axis=0)
    order = sorted(range(m), key=lambda e: (abs(means[e] - 0.5), e))   # most balanced first
    kept: List[int] = []
    trace = []
    for e in order:
        xe = S[:, e]
        h_e = float(vinfo._h_bits(float(means[e])))
        if h_e < _EPS:                                    # constant candidate carries no partition
            trace.append((e, 0.0, False))
            continue
        if not kept:                                      # anchor Ω with the first real signal
            kept.append(e)
            trace.append((e, 0.0, True))
            continue
        info_already = shannon_cmi_surrogate(xe, S[:, kept], seed=seed)
        keep = info_already < cmi_thresh * h_e
        if keep:
            kept.append(e)
        trace.append((e, info_already, bool(keep)))
    kept_sorted = sorted(kept)
    dropped = [e for e in range(m) if e not in kept]
    return {"kept": kept_sorted, "dropped": dropped, "trace": trace,
            "n_kept": len(kept_sorted), "n_candidates": m}


# --------------------------------------------------------------------------------------------
# the submodular tail-bound  (theory §6.7c / §6.9 — bounds missing IMPACT, not missing mass)
# --------------------------------------------------------------------------------------------

def submodular_tail_bound(M: Sequence[float], X_Omega: np.ndarray,
                          order: Optional[Sequence[int]] = None) -> dict:
    """Greedily accumulate the orthogonalized Ω by maximal marginal TVD-recovery gain and return the
    certified tail-bound on missing IMPACT.

    Running recovery R(S) = I_TVD(M ; X_S) = `tvd_mi_joint(M, X_S)`. We add columns in greedy order
    (each step the column with the largest Δ(e | Ω_{<i}); pass an explicit `order` to override),
    recording the greedy marginal-gain sequence, and we compute every added element's LEAVE-ONE-OUT
    marginal Δ(e_i | Ω∖{e_i}) = R(Ω) − R(Ω∖{e_i}).

        certified_bound := min_i (greedy marginal gain)   <- the VALID bound under submodularity
        tail_bound       := min_{e_i ∈ Ω} Δ(e_i | Ω∖{e_i}) <- tighter diagnostic (loo ≤ greedy gain)

    Under tail-submodularity (γ≈1 in the discovered tail), the marginal of ANY unseen unit e∉Ω is
    bounded by the SMALLEST GREEDY marginal gain: max_{e∉Ω} Δ(e | Ω) ≤ certified_bound (greedy chose
    e_last over every other candidate, so by submodularity no unseen candidate beats it). The
    leave-one-out `tail_bound` is ≤ certified_bound — a tighter, more aggressive reading, NOT itself a
    guaranteed upper bound — but both vanish together as the tail saturates. Once they decay to ~0,
    even a "magic word" outside Ω cannot raise recovery by more than the bound — an upper bound on
    missing *impact*, which Good–Turing's missing-*mass* provably cannot give (a magic word is rare-to-
    sample by definition). HONEST CAVEAT: this leg is CONDITIONAL on tail-submodularity; it must be
    paired with `adversarial_saturation`, which tests the same closure empirically WITHOUT relying
    on submodularity.
    """
    Mv = np.asarray(M, int)
    X = np.asarray(X_Omega, float)
    if X.ndim == 1:
        X = X[:, None]
    m = X.shape[1]

    def R(cols):
        cols = list(cols)
        if not cols:
            return 0.0
        return float(tvd_mi_joint(Mv, X[:, cols].astype(int)))

    chosen: List[int] = []
    marginal_gains: List[float] = []
    remaining = list(range(m)) if order is None else list(order)
    if order is None:
        cur = 0.0
        while remaining:
            best_e, best_gain = None, -np.inf
            for e in remaining:
                g = R(chosen + [e]) - cur
                if g > best_gain:
                    best_e, best_gain = e, g
            chosen.append(best_e)
            marginal_gains.append(float(best_gain))
            cur = R(chosen)
            remaining.remove(best_e)
    else:
        cur = 0.0
        for e in remaining:
            g = R(chosen + [e]) - cur
            chosen.append(e)
            marginal_gains.append(float(g))
            cur = R(chosen)

    r_full = R(chosen)
    loo = [float(r_full - R([c for c in chosen if c != e])) for e in chosen]
    # The CERTIFIED bound (valid under submodularity): greedy chose e_last over every other candidate, so
    # by submodularity no unseen candidate's marginal exceeds the smallest GREEDY marginal gain. The
    # leave-one-out min is <= that (loo <= greedy-gain since Ω∖{e_i} ⊇ Ω_{<i}), i.e. a TIGHTER diagnostic,
    # not a guaranteed upper bound — both vanish together as the tail saturates.
    certified = float(min(marginal_gains)) if marginal_gains else float("nan")
    return {"order": chosen, "marginal_gains": marginal_gains, "loo_marginals": loo,
            "certified_bound": certified, "tail_bound": float(min(loo)) if loo else float("nan"),
            "R_full": r_full}


# --------------------------------------------------------------------------------------------
# adversarial behavioral saturation  (theory §6.7c — the empirical leg, no submodularity assumed)
# --------------------------------------------------------------------------------------------

def adversarial_saturation(M: Sequence[float], X_Omega: np.ndarray, X_probes: np.ndarray, *,
                           cmi_thresh: float = 0.02, seed: int = 0,
                           probe_kinds: Optional[Sequence[str]] = None) -> dict:
    """For each adversarial probe column, I(probe ; M | X_Ω) via `shannon_cmi_surrogate`.

    The probes are rare / out-of-distribution fragments (random tokens, "use XML schema", "ignore
    previous instructions"). If every conditional CMI ≈ 0, no probe induces an M-relevant partition
    that Ω does not already carry -> the behavior space is SATURATED. This is the EMPIRICAL leg of
    the two-pronged magic-word defense (theory §6.7c): unlike the submodular tail-bound it makes NO
    submodularity assumption, so it catches a high-impact rare unit the tail-bound would miss.
    Mining halts only when tail_bound < δ AND `saturated`.

    **Composition escape (§12.6.4, 2026-07-01):** unit-only probes leave the COMPOSED-prompt channel
    untested — a whole prompt is not a function of unit verdicts, so saturation over criteria does
    not imply saturation over prompts. The probe set must therefore include holistic/composed/persona
    prompts (``composition_gap.holistic_probe_prompts``; GEPA-optimized whole prompts = the strongest
    attacker). Pass ``probe_kinds`` (one label per probe column: 'unit'/'holistic'/'composed'/
    'persona'/'gepa'): ``covers_composition`` reports whether any non-unit kind is present — a
    saturation verdict with ``covers_composition=False`` certifies the CHECKLIST channel only."""
    Xo = np.asarray(X_Omega, float)
    if Xo.ndim == 1:
        Xo = Xo[:, None]
    P = np.asarray(X_probes, float)
    if P.ndim == 1:
        P = P[:, None]
    per_probe = [(j, shannon_cmi_surrogate(M, Xo, P[:, j], seed=seed)) for j in range(P.shape[1])]
    max_cmi = max((c for _, c in per_probe), default=0.0)
    kinds = [str(k).lower() for k in probe_kinds] if probe_kinds is not None else None
    covers = (any(k in ("holistic", "composed", "persona", "gepa") for k in kinds)
              if kinds else False)
    return {"per_probe": per_probe, "max_cmi": float(max_cmi),
            "saturated": bool(max_cmi < cmi_thresh),
            "probe_kinds": kinds, "covers_composition": bool(covers)}


# --------------------------------------------------------------------------------------------
# the order-independence permutation test  (theory §6.8)
# --------------------------------------------------------------------------------------------

def permutation_order_test(R_by_subset: Sequence[float], R_by_perm: Sequence[float]) -> dict:
    """Justify the set-abstraction f(S)=R(C(S)) by comparing two variances (theory §6.8):

      σ²_subset = Var of R across DISTINCT subsets S        (the dominant factor — *which* criteria)
      σ²_perm   = Var of R across PERMUTATIONS of one fixed S (the order-sensitivity residual)

    The canonical compiler fixes one order so f is a set function; that abstraction is justified iff
    σ²_subset ≫ σ²_perm — subset selection dominates and order is a bounded second-order residual.
    Large σ²_perm means order is a real coordinate and must be lifted to an outer finite-set form
    coordinate instead."""
    a = np.asarray(R_by_subset, float)
    b = np.asarray(R_by_perm, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    vs = float(np.var(a)) if a.size > 1 else float("nan")
    vp = float(np.var(b)) if b.size > 1 else 0.0
    return {"var_subset": vs, "var_perm": vp,
            "ratio": float(vs / vp) if vp > _EPS else float("inf"),
            "set_abstraction_justified": bool(vs > vp)}


# --------------------------------------------------------------------------------------------
# family coherence  (R2-level diagnostic — is a "family" really ONE metric?)
# --------------------------------------------------------------------------------------------

def family_coherence(X_children: np.ndarray, *, tau_corr: float = 0.10,
                     tau_pr: float = 2.0) -> dict:
    """Coherence of a metric family: do its children measure ONE underlying construct, or are they a
    grab-bag of unrelated criteria lumped together? This is a SEPARATE diagnostic from reconstruction
    (which asks whether a subset can stand in for the whole); here we ask whether the whole hangs
    together at all.

    `X_children` is (N_items, K) per-item signals (binary or continuous) for the family's candidate
    criteria — the SAME matrix `_score_criteria_signals` already produces for the certificate, so this
    is a post-hoc numpy summary with NO new model scoring. A coherent family (one construct) has
    children that move together: items passing one child tend to pass the others, so the child-columns
    are positively correlated and the signal matrix is ~1-dimensional.

      * mean_pairwise_corr = mean upper-triangular Pearson r between child-columns (high & positive
        ⇒ one construct; near 0 / negative ⇒ independent or conflicting criteria).
      * participation_ratio = (Σλ)²/Σ(λ²) over the child-correlation eigenvalues (trace=K, so PR∈[1,K]:
        ≈1 ⇒ unidimensional; ≈K ⇒ all independent) — the standard 'effective number of dimensions'.

    Constant (collapsed) columns are dropped first — a criterion that never varies carries no construct.
    `coherent` = mean_pairwise_corr > tau_corr AND participation_ratio < tau_pr. Thresholds are starting
    points; retune on the v1 per-family distribution."""
    X = np.asarray(X_children, float)
    if X.ndim == 1:
        X = X[:, None]
    _, k = X.shape
    if k < 2:
        return {"mean_pairwise_corr": float("nan"), "participation_ratio": float("nan"),
                "n_criteria": int(k), "n_usable": int(k), "coherent": None,
                "note": "k<2: coherence undefined (need >=2 varying children)"}
    std = X.std(axis=0)
    usable = [j for j in range(k) if std[j] > _EPS]
    nu = len(usable)
    if nu < 2:
        return {"mean_pairwise_corr": float("nan"), "participation_ratio": float("nan"),
                "n_criteria": int(k), "n_usable": int(nu), "coherent": None,
                "note": f"<2 varying children ({nu}); coherence undefined"}
    C = np.corrcoef(X[:, usable], rowvar=False)      # nu×nu (binary ⇒ phi-coefficients)
    C = np.nan_to_num(C, nan=0.0)
    iu = np.triu_indices(nu, 1)
    mean_r = float(C[iu].mean())
    evals = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    denom = float((evals ** 2).sum())
    pr = float((evals.sum() ** 2) / denom) if denom > _EPS else float("nan")
    return {"mean_pairwise_corr": mean_r, "participation_ratio": pr,
            "n_criteria": int(k), "n_usable": int(nu),
            "coherent": bool(mean_r > tau_corr and pr < tau_pr)}


# --------------------------------------------------------------------------------------------
# self-check (planted ground truth; ZERO GPU)
# --------------------------------------------------------------------------------------------

def _selfcheck() -> bool:
    rng = np.random.default_rng(0)
    N = 300
    ok = True

    # (a)+(b) filter: a base signal + 3 EXACT redundant copies + one orthogonal signal.
    base = (rng.random(N) < 0.5).astype(float)
    orth = (rng.random(N) < 0.5).astype(float)
    signals = np.column_stack([base, base.copy(), base.copy(), base.copy(), orth])
    filt = orthogonalization_filter(signals, seed=0)
    drops_copies = set(filt["dropped"]) == {1, 2, 3}
    keeps_base_orth = set(filt["kept"]) == {0, 4}
    print(f"[a/b] filter kept={filt['kept']} dropped={filt['dropped']} "
          f"-> {'PASS' if drops_copies and keeps_base_orth else 'FAIL'}")
    ok &= drops_copies and keeps_base_orth

    # (c)+(d) tail-bound: M from 3 latent criteria of DECREASING weight (in Ω) + a hidden magic
    #         signal h (OUTSIDE Ω). Greedy marginals must decay; tail_bound is the small last one.
    k1 = (rng.random(N) < 0.5).astype(float)
    k2 = (rng.random(N) < 0.5).astype(float)
    k3 = (rng.random(N) < 0.5).astype(float)
    h = (rng.random(N) < 0.5).astype(float)
    w = 0.70 * k1 + 0.40 * k2 + 0.18 * k3 + 0.60 * h + 0.05 * rng.standard_normal(N)
    M = (w > np.median(w)).astype(int)
    Omega = np.column_stack([k1, k2, k3])
    tb = submodular_tail_bound(M, Omega)
    gains = tb["marginal_gains"]
    decays = gains[0] >= gains[-1] - 1e-9 and tb["tail_bound"] <= max(gains) + 1e-9 and tb["R_full"] > 0
    print(f"[c/d] greedy gains={[round(g,3) for g in gains]} tail_bound={tb['tail_bound']:.3f} "
          f"R_full={tb['R_full']:.3f} -> {'PASS' if decays else 'FAIL'}")
    ok &= decays

    # (e) adversarial saturation: a pure-noise probe -> saturated; the hidden M-driver h -> NOT.
    noise = (rng.random(N) < 0.5).astype(float)
    sat_noise = adversarial_saturation(M, Omega, noise[:, None], seed=0)
    sat_hidden = adversarial_saturation(M, Omega, h[:, None], seed=0)
    e_ok = sat_noise["saturated"] and not sat_hidden["saturated"]
    print(f"[e]   noise-probe saturated={sat_noise['saturated']} (cmi={sat_noise['max_cmi']:.3f}) | "
          f"hidden-probe saturated={sat_hidden['saturated']} (cmi={sat_hidden['max_cmi']:.3f}) "
          f"-> {'PASS' if e_ok else 'FAIL'}")
    ok &= e_ok

    # (f) permutation test: σ²_subset (spread across subsets) ≫ σ²_perm (order jitter).
    R_subset = np.array([0.05, 0.18, 0.31, 0.42, 0.49, 0.27, 0.11])
    R_perm = 0.42 + 0.005 * rng.standard_normal(20)
    pt = permutation_order_test(R_subset, R_perm)
    f_ok = pt["set_abstraction_justified"] and pt["ratio"] > 1.0
    print(f"[f]   var_subset={pt['var_subset']:.4f} var_perm={pt['var_perm']:.5f} "
          f"ratio={pt['ratio']:.1f} -> {'PASS' if f_ok else 'FAIL'}")
    ok &= f_ok

    # (g) family coherence: one-construct family (high corr, PR≈1) vs grab-bag (low corr, PR≈K).
    latent = (rng.random(N) < 0.5).astype(float)
    flips = (rng.random(N) < 0.1)
    coherent_fam = np.column_stack([latent, latent.copy(),
                                    (latent.astype(int) ^ flips.astype(int)).astype(float), latent.copy()])
    grab_bag = np.column_stack([(rng.random(N) < 0.5).astype(float) for _ in range(4)])
    coh_c = family_coherence(coherent_fam)
    coh_g = family_coherence(grab_bag)
    g_ok = (coh_c["mean_pairwise_corr"] > coh_g["mean_pairwise_corr"]
            and coh_c["participation_ratio"] < coh_g["participation_ratio"])
    print(f"[g]   coherent: r={coh_c['mean_pairwise_corr']:.2f} PR={coh_c['participation_ratio']:.2f} | "
          f"grab-bag: r={coh_g['mean_pairwise_corr']:.2f} PR={coh_g['participation_ratio']:.2f} "
          f"-> {'PASS' if g_ok else 'FAIL'}")
    ok &= g_ok

    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selfcheck() else 1)
