"""Granularity search over units (user algorithm, 2026-07-06): adaptively SPLIT and MERGE units to
find the granularity at which a metric's verdict is best recovered by a compiled checklist.

  unit i           : detectability gate (U1) — if a unit carries no verdict signal, try refining it
  split(i -> a,b)  : keep the parts iff held-out recovery improves
  merge(i,j -> ij) : keep the composite iff held-out recovery improves

Reframes OPT_Omega as granularity-indexed OPT_Omega(G); the search walks G locally. Two disciplines
keep it honest: (1) proposals are SIGNATURE-GUIDED (merge only high-|interaction| pairs, split only
low-solo-contribution units) so the move space is O(K) per round, not O(n^2); (2) accept/reject uses
EVEN probes only, all reported numbers use ODD probes only (adaptive selection cannot inflate OPT).
The degenerate endpoint (one unit spanning the whole description = self-recovery) is excluded by a
max-span budget. score_fn-agnostic: drivers plug vLLM; tests plug synthetic executors.

Interpretation: merge-accepts localize CONFIGURAL structure (executor integrates an interaction the
linear compiler cannot); split-accepts localize CONFLATION (compiler weighting beats the executor's
internal integration). The accepted-move ledger is itself the finding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ------------------------------------------------------------------------------------------------
@dataclass
class Unit:
    text: str
    sig: np.ndarray                       # P(YES) over the probe set
    origin: str = "atomic"                # atomic | split | merge
    parents: Tuple[str, ...] = ()
    span_words: int = 0

    def __post_init__(self):
        if not self.span_words:
            self.span_words = len(self.text.split())


def _cv_recovery(S: np.ndarray, m: np.ndarray, idx: np.ndarray, k: int = 5,
                 ridge: float = 1.0, seed: int = 0) -> float:
    """Recovery proxy: k-fold CV correlation of ridge(S[idx]) -> m[idx]. Consistent, cheap; final
    apples-to-apples numbers must be re-run through value_certificate on the refined pool."""
    S, m = S[:, idx], m[idx]
    n = len(m)
    if S.shape[0] == 0 or n < 2 * k:
        return 0.0
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    preds = np.zeros(n)
    X = S.T                                                     # (n, K)
    for f in range(k):
        te = order[f::k]
        tr = np.setdiff1d(order, te)
        Xt = X[tr] - X[tr].mean(0)
        A = Xt.T @ Xt + ridge * np.eye(X.shape[1])
        w = np.linalg.solve(A, Xt.T @ (m[tr] - m[tr].mean()))
        preds[te] = (X[te] - X[tr].mean(0)) @ w + m[tr].mean()
    sd = preds.std() * m.std()
    return float(np.corrcoef(preds, m)[0, 1]) if sd > 1e-12 else 0.0


def _interaction(si: np.ndarray, sj: np.ndarray, m: np.ndarray, idx: np.ndarray) -> float:
    """Cheap configurality score for a pair: does the product term predict the verdict residual
    beyond the linear parts? (screening only — the accept test is the real arbiter)."""
    si, sj, m = si[idx], sj[idx], m[idx]
    X = np.column_stack([si, sj, np.ones_like(si)])
    w, *_ = np.linalg.lstsq(X, m, rcond=None)
    resid = m - X @ w
    prod = (si - si.mean()) * (sj - sj.mean())
    sd = prod.std() * resid.std()
    return abs(float(np.corrcoef(prod, resid)[0, 1])) if sd > 1e-12 else 0.0


# ------------------------------------------------------------------------------------------------
def refine(units: List[Unit],
           m: np.ndarray,
           score_fn: Callable[[List[str]], np.ndarray],       # texts -> sigs (len(texts), n_probes)
           splitter: Callable[[str], Optional[Tuple[str, str]]],
           merger: Callable[[str, str], str],
           *,
           rounds: int = 4,
           k_merge: int = 4,
           k_split: int = 4,
           eps_accept: float = 0.01,
           max_span_frac: float = 0.5,
           host_words: Optional[int] = None,
           seed: int = 0) -> Dict:
    """Local search. Returns {units, ledger, opt_curve}. All accepted moves re-verified on ODD probes."""
    n_probes = len(m)
    even = np.arange(0, n_probes, 2)                          # search set
    odd = np.arange(1, n_probes, 2)                           # report set
    host_words = host_words or sum(u.span_words for u in units)
    pool = list(units)
    ledger: List[Dict] = []

    def S(p): return np.vstack([u.sig for u in p])

    opt_even = _cv_recovery(S(pool), m, even, seed=seed)
    opt_curve = [(_cv_recovery(S(pool), m, odd, seed=seed), len(pool), "init")]

    for rd in range(rounds):
        improved = False

        # ---- merge proposals: top-k pairs by interaction screen ----------------------------------
        pairs = []
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                if (pool[i].span_words + pool[j].span_words) > max_span_frac * host_words:
                    continue
                pairs.append((_interaction(pool[i].sig, pool[j].sig, m, even), i, j))
        pairs.sort(reverse=True)
        for inter, i, j in pairs[:k_merge]:
            text = merger(pool[i].text, pool[j].text)
            sig = score_fn([text])[0]
            cand = [u for k_, u in enumerate(pool) if k_ not in (i, j)]
            cand.append(Unit(text, sig, "merge", (pool[i].text, pool[j].text)))
            gain = _cv_recovery(S(cand), m, even, seed=seed) - opt_even
            if gain > eps_accept:
                ledger.append({"round": rd, "move": "merge", "gain_even": gain,
                               "a": pool[i].text[:60], "b": pool[j].text[:60],
                               "interaction_screen": inter})
                pool, opt_even, improved = cand, opt_even + gain, True
                break                                          # one accepted merge per round

        # ---- split proposals: lowest solo-contribution units --------------------------------------
        contrib = []
        base = _cv_recovery(S(pool), m, even, seed=seed)
        for i, u in enumerate(pool):
            without = [v for k_, v in enumerate(pool) if k_ != i]
            loo = base - _cv_recovery(S(without), m, even, seed=seed) if len(without) else base
            contrib.append((loo, i))
        contrib.sort()
        for _loo, i in contrib[:k_split]:
            parts = splitter(pool[i].text)
            if not parts:
                continue
            sigs = score_fn(list(parts))
            cand = [v for k_, v in enumerate(pool) if k_ != i]
            cand += [Unit(t, s, "split", (pool[i].text,)) for t, s in zip(parts, sigs)]
            gain = _cv_recovery(S(cand), m, even, seed=seed) - opt_even
            if gain > eps_accept:
                ledger.append({"round": rd, "move": "split", "gain_even": gain,
                               "parent": pool[i].text[:60]})
                pool, opt_even, improved = cand, opt_even + gain, True
                break

        opt_curve.append((_cv_recovery(S(pool), m, odd, seed=seed), len(pool),
                          ledger[-1]["move"] if improved and ledger else "none"))
        if not improved:
            break

    return {"units": pool, "ledger": ledger, "opt_curve": opt_curve,
            "opt_heldout_init": opt_curve[0][0], "opt_heldout_final": opt_curve[-1][0],
            "n_merges": sum(1 for l in ledger if l["move"] == "merge"),
            "n_splits": sum(1 for l in ledger if l["move"] == "split")}
