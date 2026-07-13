"""Seam certificates — theoretical guarantees for hybrid metric channels.

Implements the certificate ladder of the seam proposal (§5bis, 2026-07-02):

  S1  Attenuation ceiling (seam Rung 1): no implementation can correlate with a noisy
      judge channel beyond the judge's own reliability.
        Model: M_p(i) = tau(i) + eps_p(i), eps uncorrelated across passes and with tau
        and with any deterministic f(X). For the K-pass mean M_bar:
          corr(f, M_bar) = corr(f, tau) * sqrt(rel_K)  <=  sqrt(rel_K)
        with rel_K = K*rel_1 / (1 + (K-1) rel_1)  (Spearman-Brown), rel_1 = corr(M_1, M_2).
        The bound needs only Cov(f, eps)=0 (Cauchy-Schwarz) — no homoscedasticity.
        Stochastic implementations: corr(f, M_bar) <= sqrt(rel_f * rel_K); code has rel_f=1.
        Rank-correlation version is approximate (exact for Pearson under CTT) — validated
        empirically in tests_certificates.py.

  S2  Codability bracket (seam Rung 2): kappa*(C) = best achievable fidelity by class C.
        lower edge:  exhibited witness, held-out estimate minus bootstrap CI
        upper edge:  min( attenuation ceiling , exact enumeration bound for small formal
                     classes ).  Open-ended classes get NO upper edge — search-saturation
                     is evidence, not certificate (the honest wall: circuit lower bounds).

  S3  Matroid-U2 (within-class rung over criteria x implementations): the Minoux/(1/gamma)
      instance bound with a partition-matroid constraint (<=1 implementation/criterion).
      Bound uses top-k marginals over ALL elements (superset of feasible) => still valid,
      slightly loose.

  S4  Tightening decomposition: with a FIXED linear aggregator, per-channel residuals add;
      migrated (code) channels have rel=1 and contribute a pure learnability term (1-kappa)
      with CI only — the uncertified residual is confined to LLM channels.

  S5  Op-value: evidence ops raise the outer channel ceiling I(M;X,Z)>=I(M;X); computation
      ops enlarge the program class => kappa* monotone in the op set; U2 over op subsets
      gives a certified best-op-set and certified prunes.

  Rung 3 is statistical: gate_certificate() — bootstrap best-in-set at confidence 1-delta.
"""
import math, random, statistics as st


# ---------------------------------------------------------------- basics
def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


# ---------------------------------------------------------------- S1 attenuation
def spearman_brown(rel1, k=2):
    """Reliability of the k-pass mean given single-pass reliability rel1."""
    if not (0 <= rel1 <= 1):
        raise ValueError("rel1 must be in [0,1]")
    return k * rel1 / (1 + (k - 1) * rel1)


def attenuation_ceiling(rel1, k=2, rel_f=1.0):
    """Max |corr| any implementation with reliability rel_f can reach vs the k-pass mean."""
    return math.sqrt(max(0.0, spearman_brown(rel1, k)) * rel_f)


def ceiling_normalized(rho, rel1, k=2, rel_f=1.0):
    """Fidelity as a fraction of the attenuation ceiling (estimates corr(f, tau))."""
    c = attenuation_ceiling(rel1, k, rel_f)
    return rho / c if c > 0 else float("nan")


# ---------------------------------------------------------------- Rung 3 gate
def bootstrap_gate(hybrid, baseline, judge, ids, margin=0.10, floor=0.60,
                   B=2000, seed=0, corr=spearman, skip_undefined=False):
    """P over item-bootstrap that: corr(hybrid,judge) >= max(corr(base,judge)+margin, floor).

    Returns dict with rho CIs, P(gate), P(hybrid>baseline). ids must index all three maps.
    skip_undefined (2026-07-10, opt-in — default preserves all frozen results): resamples
    where either correlation is NaN (e.g. a near-constant column resampled to constant) are
    EXCLUDED from the P denominators instead of counted as losses; n_undefined is reported.
    Without it, P_beats_baseline is biased downward against near-constant baselines.
    """
    sel = [d for d in ids if d in judge and hybrid.get(d) is not None
           and baseline.get(d) is not None]
    if len(sel) < 20:
        raise ValueError("too few items")
    rng = random.Random(seed)
    rh, gate_ok, dpos, n_undef = [], 0, 0, 0
    for _ in range(B):
        s = [sel[rng.randrange(len(sel))] for _ in sel]
        h = corr([hybrid[d] for d in s], [judge[d] for d in s])
        b = corr([baseline[d] for d in s], [judge[d] for d in s])
        if skip_undefined and (h != h or b != b):
            n_undef += 1
            continue
        rh.append(h)
        if h >= max(b + margin, floor):
            gate_ok += 1
        if h > b:
            dpos += 1
    denom = max(1, B - n_undef) if skip_undefined else B
    rh.sort()
    out = {"n": len(sel), "rho_mean": round(st.mean(rh), 3),
           "rho_ci": [round(rh[int(.025 * len(rh))], 3), round(rh[int(.975 * len(rh))], 3)],
           "P_gate": round(gate_ok / denom, 3), "P_beats_baseline": round(dpos / denom, 3)}
    if skip_undefined:
        out["n_undefined_resamples"] = n_undef
    return out


# ---------------------------------------------------------------- S2 bracket
def enumerate_stump_class(features, target, n_thresholds=16):
    """Best agreement over the GRIDDED stump class {sign(x_j - t), both polarities,
    t in the tested threshold grid}.

    features: {name: {id: float}}, target: {id: 0/1}. Agreement = accuracy vs target.
    Returns (best_acc, best_desc). This is a CERTIFICATE for the gridded class only:
    with `n_thresholds` below the number of distinct values, thresholds BETWEEN grid
    points are untested and a stump there can exceed best_acc (lemma-note Gap 4). For
    the exact all-stumps certificate pass n_thresholds >= max #distinct values per
    feature; `class_size` fed to hoeffding_term must count the same grid either way.
    """
    ids = [d for d in target if all(d in col for col in features.values())]
    y = [target[d] for d in ids]
    base = max(st.mean(y), 1 - st.mean(y))  # constant classifier
    best, desc = base, "constant"
    for name, col in features.items():
        vals = sorted({col[d] for d in ids})
        if len(vals) < 2:
            continue
        step = max(1, len(vals) // n_thresholds)
        for t in vals[::step]:
            pred = [1 if col[d] > t else 0 for d in ids]
            acc = sum(p == yy for p, yy in zip(pred, y)) / len(y)
            acc = max(acc, 1 - acc)  # both polarities
            if acc > best:
                best, desc = acc, f"stump({name} > {t:.4g})"
    return best, desc


def hoeffding_term(n, delta=0.05, class_size=None, vc_dim=None):
    """Uniform-convergence half-width for the bracket edges."""
    if class_size is not None:
        return math.sqrt(math.log(2 * class_size / delta) / (2 * n))
    if vc_dim is not None:
        return math.sqrt((vc_dim * math.log(2 * n) + math.log(4 / delta)) / n)
    return math.sqrt(math.log(2 / delta) / (2 * n))


def codability_bracket(witness_rho_ci_lo, rel1, k=2, enum_bound=None, enum_n=None,
                       enum_class_size=None, delta=0.05):
    """[lower, upper] bracket on kappa*(C) (fidelity scale of the witness estimate).

    lower = witness bootstrap-CI lower edge (constructive certificate).
    upper = attenuation ceiling, AND-ed with an exact small-class enumeration bound
            (+ its uniform-convergence term) when one is supplied.
    """
    upper = attenuation_ceiling(rel1, k)
    if enum_bound is not None:
        eps = hoeffding_term(enum_n, delta, class_size=enum_class_size)
        upper = min(upper, enum_bound + eps)
    return [round(witness_rho_ci_lo, 3), round(upper, 3)]


# ---------------------------------------------------------------- S3 matroid-U2
def u2_matroid_bound(best_val, marginals, k, gamma=1.0, parts=None):
    """OPT_class <= best_val + (1/gamma) * sum(top-k marginals).

    marginals: marginal gains delta(e | S_greedy) for ALL e not in S_greedy (any
    implementation variant; the superset makes the bound valid under the partition
    matroid, slightly loose). Requires gamma in (0,1]; monotonized-R reading (PO §3.2).

    parts: optional {key: part_id} with `marginals` a {key: delta} mapping — reduces
    to per-part maxima before the top-k (any feasible set holds <=1 implementation per
    criterion), a strictly tighter, still-valid bound (lemma B1 remark R2).

    CAUTION (lemma-note Gap 5a): a plug-in lattice gamma-hat from
    op_submodularity_ratio is >= gamma_pop, so 1/gamma-hat makes this bound TOO SMALL —
    an invalid certificate direction. Quote u2_matroid_bound_curve (bound as a curve in
    gamma) or a gamma lower confidence bound, never a bare plug-in point.
    """
    if not 0 < gamma <= 1:
        raise ValueError("gamma in (0,1]")
    if parts is not None:
        by_part = {}
        for key, d in dict(marginals).items():
            p = parts[key]
            by_part[p] = max(by_part.get(p, float("-inf")), d)
        marginals = by_part.values()
    elif isinstance(marginals, dict):
        marginals = marginals.values()
    top = sorted((d for d in marginals if d > 0), reverse=True)[:k]
    return best_val + sum(top) / gamma


def u2_matroid_bound_curve(best_val, marginals, k, gammas=(1.0, 0.8, 0.6, 0.4, 0.2),
                           parts=None):
    """Gap-5a-safe U2 reporting: the bound as a curve in gamma. The certificate is
    conditional on the (unknown) population gamma; the reader picks their prior."""
    return {g: round(u2_matroid_bound(best_val, marginals, k, g, parts), 4)
            for g in gammas}


# ---------------------------------------------------------------- S4 tightening
def tightening_decomposition(channels):
    """channels: [{name, kind: 'code'|'llm', weight, fidelity, rel(optional)}] with a
    FIXED linear aggregator. Returns certified/uncertified residual split:
      code channel residual  = w * (1 - fidelity)          [pure learnability, CI-only]
      llm  channel residual  = w * (1 - fidelity/ceiling)  [articulation headroom]
    """
    out, certified, uncertified = [], 0.0, 0.0
    for c in channels:
        w, f = c["weight"], c["fidelity"]
        if c["kind"] == "code":
            r = w * (1 - f)
            certified += r
            out.append({**c, "residual": round(r, 4), "class": "learnability(CI-only)"})
        else:
            ceil = attenuation_ceiling(c.get("rel", 1.0))
            r = w * (1 - min(1.0, f / ceil))
            uncertified += r
            out.append({**c, "residual": round(r, 4), "class": "articulation-headroom"})
    return {"channels": out, "certified_residual": round(certified, 4),
            "uncertified_residual": round(uncertified, 4)}


# ---------------------------------------------------------------- S5 op value
def shapley_2(v):
    """Exact Shapley for two components given v: {(): a, ('L',): b, ('T',): c, ('L','T'): d}."""
    e, L, T, LT = v[()], v[("L",)], v[("T",)], v[("L", "T")]
    return {"L": round(0.5 * ((L - e) + (LT - T)), 4),
            "T": round(0.5 * ((T - e) + (LT - L)), 4)}


def op_monotonicity_violations(kappa_by_opset):
    """kappa_by_opset: {frozenset: kappa*}. Computation-op theory says kappa* is monotone
    in the op set. Returns list of (subset, superset) violations beyond tolerance."""
    bad = []
    sets_ = list(kappa_by_opset)
    for a in sets_:
        for b in sets_:
            if a < b and kappa_by_opset[a] > kappa_by_opset[b] + 1e-9:
                bad.append((sorted(a), sorted(b),
                            round(kappa_by_opset[a] - kappa_by_opset[b], 4)))
    return bad


def op_submodularity_ratio(kappa_by_opset, ops):
    """Empirical weak-submodularity ratio gamma over the observed op lattice:
    gamma = min over (S, T disjoint) of sum_e delta(e|S) / (v(S+T) - v(S)), for
    positive denominators. Needs kappa for all subsets of `ops` (small op sets)."""
    v = kappa_by_opset
    full = frozenset(ops)
    if any(frozenset(s) not in v for s in _powerset(ops)):
        raise ValueError("need kappa for every op subset")
    gamma = 1.0
    for S in _powerset(ops):
        S = frozenset(S)
        rest = full - S
        for T in _powerset(sorted(rest)):
            T = frozenset(T)
            if len(T) < 2:
                continue
            denom = v[S | T] - v[S]
            if denom <= 1e-9:
                continue
            num = sum(v[S | {e}] - v[S] for e in T)
            gamma = min(gamma, num / denom)
    return max(0.0, round(gamma, 4))


def _powerset(xs):
    xs = list(xs)
    for m in range(1 << len(xs)):
        yield [xs[i] for i in range(len(xs)) if m >> i & 1]
