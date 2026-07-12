"""§5.5 V-corner verification — code metrics as the instrument's calibrated zero. **ZERO GPU.**

The executor is a COMPILER (deterministic Python predicate over the item), so transmission `T = 1` by
construction (no within-item noise). On REAL items we verify the three checks the verifiable corner gives
the pipeline (§5.5 / §6.8 gating):

  C5  cap + estimator sanity. A balanced code metric, PERFECTLY recovered (`M̂ = M`, the compiler is
      faithful), must hit `R̂ = cap = 2π(1−π)` = Gini(M) (½ at balance, never above). Degrade the
      recovery with label noise p → `R̂` falls monotonically. And the cap RISES with verdict granularity
      K (binary ½ → K-ary `1−1/K`) — the headroom a finer readout buys (the §3.1 derivation, on real data).

  C6  planted-γ submodularity. Using REAL code predicates as the ground set Ω:
        * REDUNDANT world  — M and Ω are nested length-thresholds of ONE feature (correlated through M)
          ⇒ diminishing returns ⇒ submodular, γ≈1, greedy≈OPT, no synergy pair flagged.
        * SYNERGY world    — M = (has_digit ⊕ has_def) of two ~independent real predicates; the pair
          jointly determines M but neither alone ⇒ super-modular, γ<1, greedy fails, co-information
          FLAGS exactly the (has_digit, has_def) pair.
      This is the calibration gate for §6.8: trust a measured γ̂ on an opaque Ω only after it recovers a
      KNOWN planted γ on real items.

Run (sk3, NO GPU):
  HOME=/lfs/skampere3/0/alexspan python -m methods.metric_implementer.experiments.code_metric_calibration \
      --pool datasets/competition_unified/problems.parquet --n-items 4000
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from .. import vinfo
from .submod_conditional import _cmi, _mi, gamma_exact, greedy, opt_brute


def _load_texts(pool, n_items):
    if pool.endswith(".parquet"):
        df = pd.read_parquet(pool)
    elif pool.endswith(".csv") or pool.endswith(".csv.gz"):
        df = pd.read_csv(pool)
    else:
        df = pd.read_json(pool, lines=True, compression="gzip" if pool.endswith(".gz") else None)
    # auto-pick the column with the longest mean string length (the actual content)
    str_cols = [c for c in df.columns if df[c].dtype == object]
    best = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())
    texts = df[best].astype(str).tolist()
    texts = [t for t in texts if len(t) > 30][:n_items]
    return texts, best


def _features(texts):
    wc = np.array([len(t.split()) for t in texts], float)
    cc = np.array([len(t) for t in texts], float)
    has_digit = np.array([bool(re.search(r"\d", t)) for t in texts], int)
    has_def = np.array([bool(re.search(r"\b(def|class|function|return)\b", t)) for t in texts], int)
    return wc, cc, has_digit, has_def


def _thr(x, q):
    return (x > np.quantile(x, q)).astype(int)


# ----------------------------------------------------------------------------- C5
def c5(texts):
    wc, cc, has_digit, has_def = _features(texts)
    M = _thr(wc, 0.5)                                    # balanced code metric (word-count > median)
    pi = M.mean()
    cap = 2 * pi * (1 - pi)                              # Gini(M) = perfect-recovery ceiling
    R_perfect = vinfo.tvd_recovery(M.astype(float), M)["tvd_recovery"]
    print(f"\n===== C5  cap + estimator sanity (N={len(M)}) =====")
    print(f"M = (word_count > median): base-rate π={pi:.3f}  ⇒  cap = 2π(1−π) = {cap:.3f}")
    print(f"perfect compiler recovery (M̂=M): R̂ = {R_perfect:.3f}   "
          f"[{'OK — hits cap' if abs(R_perfect - cap) < 1e-6 else 'MISMATCH'}; never exceeds ½]")
    rng = np.random.default_rng(0)
    print("label-noise sweep (R̂ must fall monotonically):")
    prev = 1.0
    mono = True
    for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        flip = rng.random(len(M)) < p
        Mhat = np.where(flip, 1 - M, M).astype(float)
        r = vinfo.tvd_recovery(Mhat, M)["tvd_recovery"]
        mono &= r <= prev + 1e-9
        prev = r
        print(f"   p={p:.1f}  R̂={r:.3f}")
    print(f"   monotone-decreasing: {mono}")
    # K-ary cap headroom: perfect-recovery value = Gini of a balanced K-ary split = 1−1/K
    print("verdict-granularity headroom (perfect-recovery Gini vs cap 1−1/K):")
    for K in [2, 3, 5, 10]:
        edges = np.quantile(wc, np.linspace(0, 1, K + 1)[1:-1])
        Mk = np.digitize(wc, edges)
        _, cnt = np.unique(Mk, return_counts=True)
        p = cnt / cnt.sum()
        gini = 1 - np.sum(p ** 2)
        print(f"   K={K:<3d} Gini={gini:.3f}   cap=1−1/K={1 - 1/K:.3f}")
    return {"cap": cap, "R_perfect": R_perfect, "mono": mono}


# ----------------------------------------------------------------------------- C6
def _report_world(name, M, X, colnames, budget=4):
    K = X.shape[1]
    g = gamma_exact(M, X, K)
    Sg, fg = greedy(M, X, K, budget)
    So, fo = opt_brute(M, X, K, budget)
    ratio = fg / fo if fo > 1e-12 else float("nan")
    from itertools import combinations
    pairs = sorted(((_cmi(X[:, i], X[:, j], M) - _mi(X[:, i], X[:, j]), i, j)
                    for i, j in combinations(range(K), 2)), reverse=True)
    flagged = [(round(c, 3), colnames[i], colnames[j]) for c, i, j in pairs if c > 0.02]
    print(f"\n----- {name}  (N={len(M)}, K={K}, budget={budget}) -----")
    print(f"Ω = {colnames}")
    print(f"exact γ = {g:.3f}   (γ≈1 ⇒ submodular; γ<1 ⇒ synergy)")
    print(f"greedy {[colnames[i] for i in Sg]} f={fg:.3f}   OPT {[colnames[i] for i in So]} f={fo:.3f}   "
          f"greedy/OPT={ratio:.3f}  (1−1/e≈0.632 floor)")
    print(f"synergy pairs (co-info>0): {flagged if flagged else 'none — all redundant'}")
    return {"name": name, "gamma": g, "greedy_over_opt": ratio, "flagged": flagged}


def c6(texts):
    wc, cc, has_digit, has_def = _features(texts)
    print(f"\n===== C6  planted-γ on REAL code predicates (N={len(wc)}) =====")
    print(f"feature base-rates: has_digit={has_digit.mean():.2f} has_def={has_def.mean():.2f} "
          f"corr(digit,def)={np.corrcoef(has_digit, has_def)[0,1]:+.2f}")
    # REDUNDANT: M and Ω are nested thresholds of ONE feature (word count) -> correlated through M
    M_red = _thr(wc, 0.5)
    Xr = np.column_stack([_thr(wc, 0.3), _thr(wc, 0.5), _thr(wc, 0.7), _thr(cc, 0.5)])
    red = _report_world("REDUNDANT (nested length thresholds)", M_red, Xr,
                        ["wc>q30", "wc>q50", "wc>q70", "cc>q50"])
    # SYNERGY: M = has_digit XOR has_def; the pair determines M, neither alone; + distractors
    M_syn = (has_digit ^ has_def)
    Xs = np.column_stack([has_digit, has_def, _thr(cc, 0.5), _thr(wc, 0.5)])
    syn = _report_world("SYNERGY (has_digit ⊕ has_def)", M_syn, Xs,
                        ["has_digit", "has_def", "cc>q50", "wc>q50"])
    return red, syn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--n-items", type=int, default=4000)
    a = ap.parse_args()
    texts, col = _load_texts(a.pool, a.n_items)
    print(f"loaded {len(texts)} items from {a.pool} (text col '{col}')")
    r5 = c5(texts)
    red, syn = c6(texts)
    print("\n===== V-corner verdict =====")
    print(f"C5: estimator hits cap ({abs(r5['R_perfect']-r5['cap'])<1e-6}), monotone under noise "
          f"({r5['mono']}), cap ≤ ½ — verifiable corner behaves.")
    ok = red["gamma"] > 0.8 and syn["gamma"] < 0.8 and bool(syn["flagged"]) and not red["flagged"]
    print(f"C6: redundant γ={red['gamma']:.2f} (submodular), synergy γ={syn['gamma']:.2f} (<1), "
          f"co-info flags the synergy pair only ⇒ γ̂ machinery CALIBRATED on real items: {ok}")
    print("→ submodularity certificate on an opaque Ω may be trusted (gate passed)." if ok
          else "→ GATE NOT PASSED — do not trust γ̂ on opaque Ω yet.")


if __name__ == "__main__":
    main()
