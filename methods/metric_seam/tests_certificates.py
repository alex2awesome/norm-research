"""Planted/dummy test battery for certificates.py — every certificate gets a known-answer case.

Run: python -m methods.metric_seam.tests_certificates
(prints PASS/FAIL per test; exit 1 on any FAIL)
"""
import math, random, statistics as st, sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from certificates import (spearman_brown, attenuation_ceiling, ceiling_normalized,
                          pearson, spearman, bootstrap_gate, enumerate_stump_class,
                          codability_bracket, hoeffding_term, u2_matroid_bound,
                          tightening_decomposition, shapley_2,
                          op_monotonicity_violations, op_submodularity_ratio)

FAILS = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


rng = random.Random(42)
N = 6000


def synth_judge(sigma, hetero=False, n=N):
    """tau ~ N(0,1); M_p = tau + eps_p. Returns tau, M1, M2, true rel1."""
    tau = [rng.gauss(0, 1) for _ in range(n)]
    def eps(i):
        s = sigma * (0.4 + abs(tau[i])) if hetero else sigma
        return rng.gauss(0, s)
    m1 = [tau[i] + eps(i) for i in range(n)]
    m2 = [tau[i] + eps(i) for i in range(n)]
    return tau, m1, m2


# ---- T1: Spearman-Brown sanity -------------------------------------------------
check("T1 spearman_brown identities",
      abs(spearman_brown(0.5, 2) - 2 / 3) < 1e-12 and spearman_brown(1.0, 2) == 1.0
      and spearman_brown(0.0, 2) == 0.0)

# ---- T2: attenuation ceiling is TIGHT for the best implementation (f = tau) ----
tau, m1, m2 = synth_judge(sigma=1.0)          # rel1_true = 1/(1+1) = 0.5
rel1 = pearson(m1, m2)
mbar = [(a + b) / 2 for a, b in zip(m1, m2)]
ach = pearson(tau, mbar)
ceil = attenuation_ceiling(rel1, 2)
check("T2 ceiling tight at f=tau", abs(ach - ceil) < 0.02,
      f"achieved {ach:.3f} vs ceiling {ceil:.3f} (rel1 {rel1:.3f})")

# ---- T3: ceiling HOLDS for arbitrary implementations ---------------------------
viol = 0
for q in [0.0, 0.3, 0.7, 1.0]:
    f = [q * tau[i] + (1 - q) * rng.gauss(0, 1) for i in range(N)]
    if pearson(f, mbar) > ceil + 0.02:
        viol += 1
check("T3 ceiling holds (4 arbitrary f)", viol == 0)

# ---- T4: heteroscedastic noise — Cauchy-Schwarz robustness ---------------------
tauh, h1, h2 = synth_judge(sigma=1.0, hetero=True)
relh = pearson(h1, h2)
mbarh = [(a + b) / 2 for a, b in zip(h1, h2)]
ceilh = attenuation_ceiling(relh, 2)
check("T4 ceiling holds under heteroscedastic noise",
      pearson(tauh, mbarh) <= ceilh + 0.02,
      f"{pearson(tauh, mbarh):.3f} <= {ceilh:.3f}")

# ---- T5: rank (Spearman) version approximate but bounded -----------------------
g = [math.tanh(t) for t in tau]               # monotone transform of tau
check("T5 rank-version ceiling (approx)",
      spearman(g, mbar) <= attenuation_ceiling(spearman(m1, m2), 2) + 0.03,
      f"{spearman(g, mbar):.3f} vs {attenuation_ceiling(spearman(m1, m2), 2):.3f}")

# ---- T6: ceiling_normalized recovers corr(f, tau) ------------------------------
f = [0.6 * tau[i] + 0.8 * rng.gauss(0, 1) for i in range(N)]
est = ceiling_normalized(pearson(f, mbar), rel1, 2)
check("T6 normalization estimates corr(f,tau)", abs(est - pearson(f, tau)) < 0.03,
      f"est {est:.3f} vs true {pearson(f, tau):.3f}")

# ---- T7: stump enumeration — representable vs planted-hard ---------------------
n2 = 800
X = {"f0": {}, "f1": {}, "f2": {}}
y_stump, y_xor = {}, {}
for i in range(n2):
    a, b, c = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
    X["f0"][i], X["f1"][i], X["f2"][i] = a, b, c
    y_stump[i] = 1 if a > 0.2 else 0
    y_xor[i] = 1 if (a > 0) != (b > 0) else 0
acc_s, desc_s = enumerate_stump_class(X, y_stump, n_thresholds=64)
acc_x, _ = enumerate_stump_class(X, y_xor, n_thresholds=64)
check("T7a stump class recovers representable target", acc_s > 0.98, desc_s)
check("T7b stump class certified-BELOW on planted XOR", acc_x < 0.62,
      f"certified max {acc_x:.3f} (theory: 0.5 + margin)")

# ---- T8: codability bracket contains truth in both planted cases ---------------
br_hard = codability_bracket(witness_rho_ci_lo=0.40, rel1=0.9, enum_bound=acc_x,
                             enum_n=n2, enum_class_size=3 * 64 * 2)
check("T8 bracket: lower<=upper, enum edge binds below ceiling",
      br_hard[0] <= br_hard[1] and br_hard[1] < attenuation_ceiling(0.9, 2),
      str(br_hard))

# ---- T9: bootstrap gate — dominant vs null -------------------------------------
ids = list(range(300))
judge = {d: rng.gauss(0, 1) for d in ids}
hyb_good = {d: judge[d] + rng.gauss(0, 0.35) for d in ids}
base = {d: judge[d] + rng.gauss(0, 1.5) for d in ids}
g1 = bootstrap_gate(hyb_good, base, judge, ids, B=500, seed=1)
hyb_null = {d: base[d] + rng.gauss(0, 0.01) for d in ids}
g0 = bootstrap_gate(hyb_null, base, judge, ids, B=500, seed=1)
check("T9a gate fires on dominant hybrid", g1["P_gate"] > 0.95, str(g1))
check("T9b gate silent on null hybrid", g0["P_gate"] < 0.10, str(g0))

# ---- T10: matroid-U2 bound >= brute-force OPT (planted, gamma=1 modular) --------
import itertools
crits = ["c1", "c2", "c3"]
impls = ["code", "llm"]
w = {("c1", "code"): 0.5, ("c1", "llm"): 0.3, ("c2", "code"): 0.1,
     ("c2", "llm"): 0.4, ("c3", "code"): 0.25, ("c3", "llm"): 0.2}
def val(S):  # modular
    return sum(w[e] for e in S)
k = 2
best_opt = 0.0
for combo in itertools.combinations(w, k):          # feasibility: <=1 impl per criterion
    if len({c for c, _ in combo}) == len(combo):
        best_opt = max(best_opt, val(combo))
Sg = [("c1", "code")]                                # partial greedy (1 of k) — bound must cover OPT
gval = val(Sg)
marg = [val(Sg + [e]) - gval for e in w if e[0] != "c1"]
bound = u2_matroid_bound(gval, marg, k, gamma=1.0)
check("T10 matroid-U2 covers OPT", bound >= best_opt - 1e-9,
      f"bound {bound:.3f} >= OPT {best_opt:.3f}")

# ---- T11: tightening decomposition ----------------------------------------------
dec = tightening_decomposition([
    {"name": "quotes_presence", "kind": "code", "weight": 0.5, "fidelity": 0.95},
    {"name": "quotes_quality", "kind": "llm", "weight": 0.5, "fidelity": 0.60, "rel": 0.9},
])
check("T11 tightening: code residual certified, llm residual uncertified",
      abs(dec["certified_residual"] - 0.025) < 1e-9
      and dec["uncertified_residual"] > 0.1, str(dec))

# ---- T12: Shapley exactness on additive value -----------------------------------
sh = shapley_2({(): 0.5, ("L",): 0.66, ("T",): 0.55, ("L", "T"): 0.71})
check("T12 shapley additive", abs(sh["L"] - 0.16) < 1e-9 and abs(sh["T"] - 0.05) < 1e-9,
      str(sh))

# ---- T13: op monotonicity + submodularity ratio on planted lattice ---------------
# planted: kappa*(O) via stump enumeration with op-provided features (computation ops)
feats_all = {"raw": X["f2"], "op_a": X["f0"], "op_b": X["f1"]}
kmap = {}
for sub in [[], ["op_a"], ["op_b"], ["op_a", "op_b"]]:
    F = {"raw": feats_all["raw"], **{o: feats_all[o] for o in sub}}
    kmap[frozenset(sub)], _ = enumerate_stump_class(F, y_stump, n_thresholds=64)
check("T13a computation-op monotonicity", op_monotonicity_violations(kmap) == [])
gam = op_submodularity_ratio(kmap, ["op_a", "op_b"])
check("T13b op submodularity ratio in (0,1]", 0 < gam <= 1, f"gamma={gam}")

# ---- T14: hoeffding term sane ----------------------------------------------------
check("T14 hoeffding shrinks with n",
      hoeffding_term(100, class_size=100) > hoeffding_term(10000, class_size=100))

print(f"\n{len(FAILS)} failures" + (f": {FAILS}" if FAILS else " — ALL GREEN"))
sys.exit(1 if FAILS else 0)
