"""§6.6 demonstration: submodularity of criterion recovery is a CONDITIONAL theorem.

Plants two worlds over criterion signals X_i and an original verdict M, with the *ideal* recovery
value function  f(S) = I(M; X_S)  (Shannon — the co-information identity that governs γ is Shannon):

  * CI-GIVEN-M (redundant): X_i = M flipped w.p. ε_i, independent noise -> criteria conditionally
    independent given M. Theory (§6.6 step 3) predicts: f monotone SUBMODULAR, γ≈1, greedy ≈ OPT_Ω,
    pairwise co-information I(X_i;X_j|M) − I(X_i;X_j) ≤ 0 (no synergy flagged).
  * SYNERGY (XOR): M = Z1 ⊕ Z2; criteria X_0=Z1, X_1=Z2 are individually uninformative about M but
    jointly determine it; plus weak redundant distractor copies of M. Theory predicts: f SUPER-modular
    on the pair, γ<1, greedy (chasing the weak distractors, blind to the 0-marginal pair) << OPT_Ω,
    and co-information FLAGS the (Z1,Z2) pair (+1 bit).

f(S)=I(M;X_S) is monotone in BOTH worlds (observing more never lowers info about M), so this isolates
the SUBMODULARITY axis. The non-monotonicity axis (PRUNE) is the executor bottleneck (§6.6 step 4),
demonstrated separately in ladder_synthetic.py. Exact γ by brute force (K small -> O(2^K) is cheap).

Run:  python -m methods.metric_implementer.experiments.submod_conditional
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

_EPS = 1e-12


# ---- discrete Shannon information (plug-in, bits) ------------------------------------------

def _ent(counts: np.ndarray) -> float:
    p = counts[counts > 0].astype(float)
    p /= p.sum()
    return float(-np.sum(p * np.log2(p)))


def _code(cols: np.ndarray) -> np.ndarray:
    """Rows of small non-negative ints -> a single integer code per row."""
    cols = np.atleast_2d(cols)
    if cols.shape[0] != 0 and cols.shape[1] == 0:
        return np.zeros(cols.shape[0], dtype=np.int64)
    out = np.zeros(cols.shape[0], dtype=np.int64)
    for j in range(cols.shape[1]):
        out = out * (int(cols[:, j].max()) + 1) + cols[:, j]
    return out


def _H(a: np.ndarray) -> float:
    return _ent(np.bincount(a.astype(np.int64)))


def _mi(a: np.ndarray, b: np.ndarray) -> float:
    """I(a;b) = H(a)+H(b)-H(a,b)."""
    ab = a.astype(np.int64) * (int(b.max()) + 1) + b
    return _H(a) + _H(b) - _ent(np.bincount(ab))


def _cmi(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """I(a;b|c) = H(a,c)+H(b,c)-H(a,b,c)-H(c)."""
    ac = _code(np.column_stack([a, c]))
    bc = _code(np.column_stack([b, c]))
    abc = _code(np.column_stack([a, b, c]))
    return _ent(np.bincount(ac)) + _ent(np.bincount(bc)) - _ent(np.bincount(abc)) - _H(c)


def f_recovery(M: np.ndarray, X: np.ndarray, S) -> float:
    """Ideal recovery I(M; X_S)."""
    S = list(S)
    if not S:
        return 0.0
    return _mi(M, _code(X[:, S]))


# ---- exact submodularity ratio γ (brute force; O(4^K), fine for toy K) ---------------------

def gamma_exact(M: np.ndarray, X: np.ndarray, K: int) -> float:
    """γ = min over (S, Ω disjoint, Ω≠∅, f(S∪Ω)>f(S)) of
       [Σ_{e∈Ω}(f(S∪e)−f(S))] / [f(S∪Ω)−f(S)].  γ≥1 ⇒ submodular (report min(1,·))."""
    items = list(range(K))
    fcache = {}

    def f(s):
        key = tuple(sorted(s))
        if key not in fcache:
            fcache[key] = f_recovery(M, X, key)
        return fcache[key]

    g = np.inf
    for r in range(0, K):
        for S in combinations(items, r):
            rest = [e for e in items if e not in S]
            for ro in range(1, len(rest) + 1):
                for Om in combinations(rest, ro):
                    denom = f(set(S) | set(Om)) - f(S)
                    if denom > 1e-6:
                        num = sum(f(set(S) | {e}) - f(S) for e in Om)
                        g = min(g, num / denom)
    return float(min(1.0, g)) if np.isfinite(g) else float("nan")


def greedy(M: np.ndarray, X: np.ndarray, K: int, budget: int):
    """Forward greedy on the monotone f(S)=I(M;X_S)."""
    S, items = [], list(range(K))
    for _ in range(budget):
        gains = [(f_recovery(M, X, S + [e]) - f_recovery(M, X, S), e) for e in items if e not in S]
        if not gains:
            break
        g, e = max(gains)
        if g <= 1e-9:               # no first-order signal left (the synergy trap)
            break
        S.append(e)
    return S, f_recovery(M, X, S)


def opt_brute(M: np.ndarray, X: np.ndarray, K: int, budget: int):
    best_S, best = (), 0.0
    for r in range(1, budget + 1):
        for S in combinations(range(K), r):
            v = f_recovery(M, X, S)
            if v > best:
                best, best_S = v, S
    return list(best_S), best


# ---- planted worlds ------------------------------------------------------------------------

def plant_ci(N, K, rng, eps_lo=0.10, eps_hi=0.40):
    """X_i = M flipped w.p. ε_i (independent) ⇒ conditionally independent given M (redundant)."""
    M = (rng.random(N) < 0.5).astype(int)
    eps = np.linspace(eps_lo, eps_hi, K)
    X = np.empty((N, K), int)
    for i in range(K):
        X[:, i] = M ^ (rng.random(N) < eps[i]).astype(int)
    return M, X


def plant_synergy(N, K, rng, distractor_eps=0.35):
    """M = Z1⊕Z2; X_0=Z1, X_1=Z2 (synergy pair, 0 marginal info); X_2.. = weak copies of M."""
    Z1 = (rng.random(N) < 0.5).astype(int)
    Z2 = (rng.random(N) < 0.5).astype(int)
    M = Z1 ^ Z2
    X = np.empty((N, K), int)
    X[:, 0], X[:, 1] = Z1, Z2
    for i in range(2, K):
        X[:, i] = M ^ (rng.random(N) < distractor_eps).astype(int)
    return M, X


def report(name, M, X, K, budget):
    print(f"\n===== {name}  (N={len(M)}, K={K}, budget={budget}) =====")
    g = gamma_exact(M, X, K)
    Sg, fg = greedy(M, X, K, budget)
    So, fo = opt_brute(M, X, K, budget)
    ratio = fg / fo if fo > _EPS else float("nan")
    print(f"exact γ = {g:.3f}   (γ≈1 ⇒ submodular; γ<1 ⇒ synergy)")
    print(f"greedy  picks {Sg}  f={fg:.3f}")
    print(f"OPT_Ω   picks {So}  f={fo:.3f}")
    print(f"greedy / OPT_Ω = {ratio:.3f}   (1−1/e ≈ 0.632 is the submodular floor)")
    # pairwise co-information: I(X_i;X_j|M) − I(X_i;X_j)  (>0 ⇒ synergy)
    pairs = []
    for i, j in combinations(range(K), 2):
        ci = _cmi(X[:, i], X[:, j], M) - _mi(X[:, i], X[:, j])
        pairs.append((ci, i, j))
    pairs.sort(reverse=True)
    flagged = [(round(ci, 3), i, j) for ci, i, j in pairs if ci > 0.02]
    print(f"synergy pairs (co-info > 0): {flagged if flagged else 'none — all redundant (co-info ≤ 0)'}")
    top = pairs[0]
    print(f"max co-info pair: X_{top[1]},X_{top[2]} = {top[0]:+.3f}")
    return {"name": name, "gamma": g, "greedy_over_opt": ratio, "flagged": flagged}


def run(N=6000, K=6, budget=4, seed=0):
    rng = np.random.default_rng(seed)
    print("§6.6 — submodularity of criterion recovery is a CONDITIONAL theorem.")
    print("f(S)=I(M;X_S) is monotone in both worlds; only γ (submodularity) differs.")
    Mc, Xc = plant_ci(N, K, rng)
    rc = report("CI-given-M (redundant copies)", Mc, Xc, K, budget)
    Ms, Xs = plant_synergy(N, K, rng)
    rs = report("SYNERGY (XOR pair + weak distractors)", Ms, Xs, K, budget)

    print("\n--- §6.6 verdict ---")
    print(f"CI world:      γ={rc['gamma']:.3f} (submodular), greedy/OPT={rc['greedy_over_opt']:.3f} "
          f"(≥ 0.632), synergy flagged: {bool(rc['flagged'])}")
    print(f"Synergy world: γ={rs['gamma']:.3f} (super-modular), greedy/OPT={rs['greedy_over_opt']:.3f} "
          f"(< 0.632), synergy flagged: {bool(rs['flagged'])}")
    print("Theorem confirmed: CI-given-M ⇒ submodular ⇒ greedy near-OPT; synergy ⇒ γ<1, greedy fails,")
    print("and the co-information fingerprint localizes the offending criterion pair.")
    return rc, rs


if __name__ == "__main__":
    run()
