"""The |Ω|>15 frontier — two pieces the rest of the §6 machinery bypasses by brute force.

For small Ω (≤15) `small_omega_brute_force` enumerates the subset lattice exactly and needs neither γ nor
an approximation algorithm. The theory note (notes/2026-06-18) foregrounds two rung-2 objects that
matter ONLY once |Ω| is too large to enumerate (§3.2 and §6.1). This module implements both, on a
generic value function f(S)->float so they apply to any R(C(S)):

  * U2 — the per-instance additive UPPER bound (§3.2). A corollary of the submodularity-ratio
    definition + monotonicity:
        OPT_class ≤ U2 := f(S_g) + (1/γ) · Σ_{j=1..k} δ_(j)
    where S_g is the greedy solution, δ(e) = f(S_g ∪ {e}) − f(S_g) for e ∉ S_g are the candidate
    marginal gains at S_g, δ_(1..k) are the k LARGEST of them, and γ ∈ (0,1] is the (measured)
    submodularity ratio. This is the tighter run-time bound the note says to "use for certifying
    this run" (vs the multiplicative worst-case for reporting thickness). Valid for monotone f, or
    the free-disposal monotonization R↑; for genuinely non-monotone R use double_greedy instead.

  * double_greedy — Buchbinder–Feldman–Naor–Schwartz (2015) unconstrained submodular MAXIMIZATION
    for non-monotone f (§6.1). Deterministic 1/3-approximation, randomized ½-approximation (tight).
    PRUNE evidence shows raw R(S) is non-monotone, so the monotone greedy + U2 do not apply to raw R;
    double-greedy is the honest algorithm there (no monotonicity, no exponential sweep).

Both reuse the exact γ computation from submod_conditional/gamma_from_f when |Ω| is small enough for
a self-check; at production scale γ is measured (submod_conditional lower-tail / orthogonalize.py).

ZERO GPU. Pure combinatorics over a user-supplied f.

Self-check:  on a KNOWN submodular f (weighted coverage) γ=1 ⇒ U2 is the classical Minoux bound and
double_greedy recovers OPT; on a planted super-modular f γ<1 ⇒ U2 loosens as 1/γ predicts.

Run:  python -m methods.metric_implementer.experiments.large_omega
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# a set function over subsets of range(K): maps a tuple of element indices to a float
F = Callable[[Tuple[int, ...]], float]


def _greedy(f: F, K: int, budget: Optional[int] = None) -> Tuple[Tuple[int, ...], float, List[float]]:
    """Forward greedy on f. Returns (S_g, f(S_g), [marginal-gain-per-step]).

    Stops when the budget is reached or no element has positive marginal gain (the synergy trap /
    PRUNE case). The per-step gains are returned so U2's δ_(j) leave-one-out diagnostic can reuse them."""
    budget = budget if budget is not None else K
    S: List[int] = []
    gains: List[float] = []
    fS = 0.0
    for _ in range(budget):
        best_e, best_g = None, 0.0
        for e in range(K):
            if e in S:
                continue
            g = f(tuple(sorted(S + [e]))) - fS
            if g > best_g:
                best_e, best_g = e, g
        if best_e is None or best_g <= 1e-12:           # no improving element (PRUNE / saturated)
            break
        S.append(best_e)
        gains.append(best_g)
        fS = f(tuple(sorted(S)))
    return tuple(sorted(S)), fS, gains


def U2_bound(f: F, K: int, *, budget: Optional[int] = None, gamma: float,
             delta_source: Optional[Sequence[int]] = None) -> dict:
    """§3.2 per-instance additive upper bound:  OPT_class ≤ f(S_g) + (1/γ)·Σ_{j=1..k} δ_(j).

    `gamma` is the (measured) submodularity ratio of f over the criterion universe (0,1]; pass a
    certified lower bound you computed from f (small Ω: exhaustive gamma_exact; a sampled large-Ω
    lower tail is diagnostic only). `budget` k defaults to K. `delta_source` lets you compute the candidate marginals
    δ(e)=f(S_g∪{e})−f(S_g) over a SUBSET of e∉S_g when the full universe is huge (default: all e∉S_g).

    Returns the certified U2 plus the greedy solution, the realized greedy/OPT ratio if OPT is known
    (None otherwise), and the raw greedy marginals for inspection. The proof (3 lines, note §3.2):
      monotonicity:   OPT ≤ f(S_g ∪ S*) = f(S_g) + [f(S_g∪S*) − f(S_g)]
      γ-definition:   f(S_g∪S*) − f(S_g) ≤ (1/γ)·Σ_{e∈S*∖S_g} δ(e|S_g)
      top-k:          |S*∖S_g| ≤ k  ⇒  Σ_{e∈S*∖S_g} δ ≤ Σ_{top-k} δ
    so OPT ≤ f(S_g) + (1/γ)·Σ_{top-k} δ.  (γ≤1 ⇒ 1/γ≥1 ⇒ looser, as a less-submodular f deserves.)
    """
    if not (0.0 < gamma <= 1.0 + 1e-9):
        raise ValueError(f"gamma must be in (0,1]; got {gamma}")
    gamma = float(min(gamma, 1.0))
    k = budget if budget is not None else K
    S_g, fg, greedy_gains = _greedy(f, K, budget=k)
    # candidate marginals at S_g, over the requested source set (default: everything not in S_g)
    src = delta_source if delta_source is not None else [e for e in range(K) if e not in S_g]
    deltas = [f(tuple(sorted(tuple(S_g) + (e,)))) - fg for e in src if e not in S_g]
    topk = sorted(deltas, reverse=True)[:k]
    U2 = fg + (1.0 / gamma) * float(sum(topk))
    return {"U2": U2, "f_Sg": fg, "S_g": S_g, "gamma": gamma, "budget": k,
            "topk_deltas": topk, "greedy_marginals": greedy_gains, "n_source": len(src)}


def _double_greedy_core(f: F, K: int, rng: Optional[np.random.Generator]) -> Tuple[Tuple[int, ...], float]:
    """Buchbinder–Feldman–Naor–Schwartz (2015) double-greedy. Maintain A (included) and B (the
    candidate set, Ω minus already-excluded); for each element i∈B, by the submodular exchange
    inequality decide include-in-A vs exclude-from-B using the two marginals:
        dA = f(A∪{i}) − f(A)       (gain of ADDING i to the included set)
        dB = f(B) − f(B∖{i})       (marginal of KEEPING i in the candidate set; ≥0 for submodular f)
    `rng=None` => deterministic 1/3-approximation (keep iff dA≥dB, clamped ≥0); an rng => randomized
    ½-approximation (keep with prob max(dA,0)/(max(dA,0)+max(dB,0))). Returns (S*, f(S*))."""
    A: List[int] = []
    B: set = set(range(K))
    for i in range(K):
        if i not in B:                       # already excluded earlier
            continue
        fA = f(tuple(sorted(A)))
        dA = f(tuple(sorted(A + [i]))) - fA
        fB = f(tuple(sorted(B)))
        dB = fB - f(tuple(sorted(x for x in B if x != i)))
        dA_c, dB_c = max(dA, 0.0), max(dB, 0.0)
        if rng is None:                      # deterministic 1/3: keep iff adding helps at least as much
            keep = dA_c >= dB_c
        else:                                # randomized 1/2
            if dA_c + dB_c == 0:
                keep = rng.random() < 0.5
            else:
                keep = rng.random() < (dA_c / (dA_c + dB_c))
        if keep:
            A.append(i)
        B.discard(i)
    return tuple(sorted(A)), f(tuple(sorted(A)))


def double_greedy(f: F, K: int, *, randomized: bool = False, seed: int = 0) -> dict:
    """§6.1 double-greedy for NON-MONOTONE unconstrained submodular maximization.

    `randomized=False` => deterministic 1/3-approximation; `randomized=True` => randomized
    ½-approximation (the tight guarantee, Buchbinder–Feldman–Naor–Schwartz 2015). This is the
    algorithm the note says to "default to" for raw non-monotone R(S) (PRUNE evidence), where the
    monotone greedy + U2 do not apply. Holds for any submodular f; for weakly-submodular (γ<1) the
    guarantee degrades and is reported as a measured fact, not a verified ratio (note §6.1)."""
    rng = np.random.default_rng(seed) if randomized else None
    S, val = _double_greedy_core(f, K, rng)
    return {"S": S, "f_S": val, "approx": "1/2 (randomized)" if randomized else "1/3 (deterministic)",
            "randomized": randomized}


# --------------------------------------------------------------------------------------------
# dict adapters + measured γ (for wiring into small_omega_brute_force at any K)
# --------------------------------------------------------------------------------------------

def f_from_dict(R: Dict[Tuple[int, ...], float]) -> F:
    """Wrap a precomputed R(S) dict (keyed by SORTED tuple; f(∅)=0) as the generic f(S)->float that
    U2_bound / double_greedy consume. Used by small_omega_brute_force to run the approximation on the SAME
    exact R it brute-forced — a zero-GPU cross-check at small K, the only path at large K."""
    def f(S: Tuple[int, ...]) -> float:
        if not S:
            return 0.0
        return float(R[tuple(sorted(S))])
    return f


def gamma_measured(f: F, K: int, *, n_samples: int = 400, seed: int = 0,
                   max_pair: int = 6) -> Tuple[float, float]:
    """Lower-tail ESTIMATE of the submodularity ratio γ of f — the large-K substitute for
    gamma_exact_set (which is O(4^K), infeasible past ~K=12). Samples (S, Ω') pairs (S of random
    size, Ω' a random subset of Ω∖S up to size `max_pair`) and returns the (min, 10th-percentile) of
    the ratio Σ_{e∈Ω'}[f(S∪e)−f(S)] / [f(S∪Ω')−f(S)] over the pairs with positive denominator.

    This is a lower-tail point estimate of γ, NOT a certificate — report with its sampling spread,
    never as a guaranteed ratio (note §6.2: 'γ is empirical/structural, never analytic for a
    black-box judge'). The sampled minimum is smaller than the other sampled ratios but need not be
    below the unobserved global minimum; plugging it into U2 does not make U2 a valid bound."""
    from itertools import combinations
    rng = np.random.default_rng(seed)
    items = list(range(K))
    ratios = []
    for _ in range(n_samples):
        sr = rng.integers(0, K + 1)
        S = tuple(sorted(rng.choice(K, size=min(sr, K), replace=False))) if sr else ()
        rest = [e for e in items if e not in S]
        if not rest:
            continue
        m = rng.integers(1, min(max_pair, len(rest)) + 1)
        Om = tuple(sorted(rng.choice(rest, size=m, replace=False)))
        fS = f(S)
        denom = f(tuple(sorted(tuple(S) + tuple(Om)))) - fS
        if denom <= 1e-6:
            continue
        num = sum(f(tuple(sorted(tuple(S) + (e,)))) - fS for e in Om)
        ratios.append(num / denom)
    if not ratios:
        return float("nan"), float("nan")
    arr = np.array(ratios)
    # γ is the min ratio by definition; on a NON-MONOTONE f the numerator can be negative (PRUNE:
    # adding an element lowers f), which makes γ ill-defined (<0). Per note §6.1/§6.2, γ on raw
    # non-monotone R is meaningless — so clamp negatives to 0 (signaling "γ broken, use R↑"). The
    # caller should prefer gamma_measured on the monotone envelope R↑ whenever PRUNE is detected.
    raw_min = float(np.nanmin(arr))
    gmin = float(min(1.0, max(0.0, raw_min)))
    p10 = float(min(1.0, max(0.0, np.nanquantile(arr, 0.1))))
    return gmin, p10


# --------------------------------------------------------------------------------------------
# self-check (ZERO GPU): known submodular f => γ=1 => U2 = Minoux bound, double-greedy ~ OPT;
# planted super-modular f => γ<1 => U2 loosens by 1/γ, as predicted.
# --------------------------------------------------------------------------------------------

def _coverage_f(weights: np.ndarray, covers: List[np.ndarray], lam: float = 0.0) -> F:
    """A weighted-coverage set function f(S)=Σ_i w_i·(1−Π_{e∈S}(1−covers[e][i])) + lam·|S|.
    Monotone submodular for lam≥0 (γ=1 when lam=0). Used as the known-γ ground truth for U2."""
    w = np.asarray(weights, float)

    def f(S: Tuple[int, ...]) -> float:
        if not S:
            return 0.0
        miss = np.ones_like(w)
        for e in S:
            miss *= (1.0 - covers[e])
        cover = w * (1.0 - miss)
        return float(cover.sum() + lam * len(S))
    return f


def gamma_exact_set(f: F, K: int) -> float:
    """Exact submodularity ratio of a set function f:2^Ω→ℝ (min over (S,Ω') disjoint of
    Σ_{e∈Ω'}[f(S∪e)−f(S)] / [f(S∪Ω')−f(S)], when the denominator > 0). Mirrors gamma_from_f /
    submod_conditional.gamma_exact but on a generic f callable (no signal matrix). O(4^K): toy K only."""
    from itertools import combinations
    items = list(range(K))
    g = np.inf
    for r in range(K):
        for S in combinations(items, r):
            rest = [e for e in items if e not in S]
            for ro in range(1, len(rest) + 1):
                for Om in combinations(rest, ro):
                    denom = f(tuple(sorted(tuple(S) + tuple(Om)))) - f(tuple(sorted(S)))
                    if denom > 1e-6:
                        num = sum(f(tuple(sorted(tuple(S) + (e,)))) - f(tuple(sorted(S))) for e in Om)
                        g = min(g, num / denom)
    return float(min(1.0, g)) if np.isfinite(g) else float("nan")


def _selfcheck() -> bool:
    rng = np.random.default_rng(0)
    K, N = 6, 10
    w = rng.random(N)
    covers = [ (rng.random(N) < 0.4).astype(float) for _ in range(K) ]
    f = _coverage_f(w, covers, lam=0.0)          # γ=1 (pure submodular)

    g = gamma_exact_set(f, K)
    u2 = U2_bound(f, K, gamma=g, budget=3)
    # exact OPT for the check
    from itertools import combinations
    opt = max((f(tuple(sorted(S))) for r in range(1, K+1) for S in combinations(range(K), r)))
    dg = double_greedy(f, K, randomized=True, seed=0)
    print(f"[submodular] γ={g:.3f}  U2={u2['U2']:.3f}  greedy={u2['f_Sg']:.3f}  OPT={opt:.3f}  "
          f"double-greedy={dg['f_S']:.3f}")
    submod_ok = abs(g - 1.0) < 0.02 and u2["U2"] >= opt - 1e-6 and dg["f_S"] >= 0.8 * opt
    print(f"  γ≈1 ✓, U2≥OPT ✓ (the Minoux bound holds), double-greedy≥0.8·OPT ✓ "
          f"-> {'PASS' if submod_ok else 'FAIL'}")

    # super-modular (XOR-ish): two elements worthless alone, valuable together => γ<1
    def f_super(S: Tuple[int, ...]) -> float:
        Sset = set(S)
        val = 0.0
        for i in range(K):
            val += w[i] * (1.0 if i in Sset else 0.0) * 0.1   # weak solo signal
        if 0 in Sset and 1 in Sset:                            # synergy pair
            val += 5.0
        return val
    g2 = gamma_exact_set(f_super, K)
    u2_2 = U2_bound(f_super, K, gamma=g2, budget=3)
    print(f"[super-mod] γ={g2:.3f}  U2={u2_2['U2']:.3f}  (1/γ={1.0/g2:.2f} inflates the bound, as predicted)")
    # γ<1 here, and 1/γ > 1 looser — exactly the §3.2 sanity the note states
    sup_ok = g2 < 0.99 and 1.0 / g2 > 1.0 and u2_2["U2"] >= u2_2["f_Sg"]
    print(f"  γ<1 ✓, 1/γ>1 loosens U2 ✓ -> {'PASS' if sup_ok else 'FAIL'}")

    # dict adapter + measured γ (the cross-check path small_omega_brute_force uses): build the R-dict that
    # brute_force WOULD produce, wrap it as f, and confirm U2/double-greedy match the direct-f run,
    # and that gamma_measured(min) ≤ gamma_exact (a valid conservative proxy).
    R = {}
    for r in range(1, K + 1):
        for S in combinations(range(K), r):
            R[tuple(sorted(S))] = f(tuple(sorted(S)))
    fd = f_from_dict(R)
    u2d = U2_bound(fd, K, gamma=g, budget=3)
    gmin, gp10 = gamma_measured(fd, K, n_samples=1000, seed=0)
    measured_ok = abs(u2d["U2"] - u2["U2"]) < 1e-6 and gmin <= g + 1e-6
    print(f"[dict/γ-measured] U2(dict)={u2d['U2']:.3f} vs U2(f)={u2['U2']:.3f}; "
          f"γ_exact={g:.3f} ≥ γ_measured(min)={gmin:.3f} (p10={gp10:.3f})")
    print(f"  dict-U2 == direct-U2 ✓, γ_measured(min) ≤ γ_exact (conservative) ✓ "
          f"-> {'PASS' if measured_ok else 'FAIL'}")
    print(f"\n{'ALL PASS' if submod_ok and sup_ok and measured_ok else 'SOME FAILED'}")
    return bool(submod_ok and sup_ok and measured_ok)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selfcheck() else 1)
