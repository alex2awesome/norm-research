"""Definitive algorithm soundness check under CW-like conditions (many covariates).
Plants a coefficient break and asks: does GapTree split on the right variable when m is large
(like CW's ~26 z-covariates), at n_perm=999 vs 199? Offline synthetic — no GLM."""
import sys; sys.path.insert(0, "methods")
import numpy as np
from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.mob.glmtree import GapTree


def make_break_data(n=800, seed=0, beta=2.5, m_null=25):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z_break = rng.uniform(size=n)
    sign = np.where(z_break >= 0.5, -1.0, 1.0)
    y = (rng.uniform(size=n) < 1/(1+np.exp(-(sign*beta*x)))).astype(float)
    Z = {"z_break": (z_break, "numeric")}
    for j in range(m_null):
        Z[f"z_null_{j}"] = (rng.uniform(size=n), "numeric")
    return x, y, Z


def fit_and_report(n_perm, bonferroni, n=800, m_null=25):
    x, y, Z = make_break_data(n=n, m_null=m_null)
    cfg = InfillConfig(n_permutations=n_perm, min_node_size=40, max_depth=3,
                       random_seed=0, bonferroni=bonferroni, alpha=0.05)
    tree = GapTree(cfg).fit(x.reshape(-1, 1), y, Z, feature_names=["x"])
    sp = tree.root.split
    # min achievable adj_p under this config
    m = len(Z)
    min_p = 1.0/(1+n_perm); min_adjp = min_p*m
    status = (f"split={sp.variable}@{sp.threshold:.2f}" if sp else "STUMP")
    print(f"  n_perm={n_perm} bonferroni={bonferroni} m={m}: {status}  "
          f"(min possible adj_p={min_adjp:.3f} vs alpha=0.05)", flush=True)
    # show top-3 candidates by adj_p
    if tree.root.fluct:
        top = sorted(tree.root.fluct, key=lambda r: r.adj_pvalue)[:3]
        for r in top:
            print(f"      {r.variable:12} stat={r.statistic:.2f} p={r.pvalue:.4f} adj_p={r.adj_pvalue:.4f}", flush=True)
    return sp


print("== planted break, CW-like m=26 ==", flush=True)
print("n_perm=999 (the fix):", flush=True)
sp999 = fit_and_report(999, True)
print("n_perm=199 (the bug):", flush=True)
sp199 = fit_and_report(199, True)
print("\n== stable data, m=26 (should be STUMP) ==", flush=True)
rng = np.random.default_rng(1)
x = rng.normal(size=800); y = (rng.uniform(size=800) < 1/(1+np.exp(-(2.5*x)))).astype(float)
Z = {f"z_{j}": (rng.uniform(size=800), "numeric") for j in range(26)}
cfg = InfillConfig(n_permutations=999, min_node_size=40, max_depth=3, random_seed=0, bonferroni=True)
tree = GapTree(cfg).fit(x.reshape(-1,1), y, Z, feature_names=["x"])
print(f"  stable m=26: {'STUMP (correct)' if tree.root.is_terminal else 'SPLIT (wrong!)'}", flush=True)

print("\n== VERDICT ==", flush=True)
print(f"  n_perm=999 splits on planted break under m=26: {'YES (algorithm sound)' if (sp999 and sp999.variable=='z_break') else 'NO (bug!)'}", flush=True)
print(f"  n_perm=199 failed under m=26: {'YES (confirms config pathology)' if sp199 is None else 'no'}", flush=True)
