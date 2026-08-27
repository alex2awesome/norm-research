#!/usr/bin/env python3
"""Analysis for lake_lint_results.csv: per-feature AUC vs y on the built
subset + failure-class tables. Prints markdown tables."""
import sys

import numpy as np
import pandas as pd

PATH = sys.argv[1] if len(sys.argv) > 1 else (
    "/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib/"
    "lake_lint_results.csv")


def rank_auc(y, x):
    """AUC of x predicting y=1 (ties handled via average ranks)."""
    y = np.asarray(y)
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    y, x = y[ok], x[ok]
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return np.nan, n1 + n0
    r = pd.Series(x).rank().values
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return auc, n1 + n0


df = pd.read_csv(PATH)
print(f"rows: {len(df)}; status counts:\n{df.status.value_counts()}\n")

ok = df[df.status == "ok"].copy()
ok["warnings_per_file"] = ok.n_warn_total / ok.n_files
ok["log_t_rebuild"] = np.log1p(ok.t_rebuild)
ok["log_t_rebuild_per_file"] = np.log1p(ok.t_rebuild / ok.n_files)
lint_ok = ok[ok.lint_available == 1].copy()
lint_ok["lint_errors_per_file"] = (
    lint_ok.n_lint_errors.astype(float) / lint_ok.n_lint_modules.clip(lower=1))

print("## Per-feature AUC vs y (built subset)\n")
print("| feature | AUC | n | base rate y=1 |")
print("|---|---|---|---|")
feats = [
    ("warnings_per_file", ok, "warnings_per_file"),
    ("n_warn_total", ok, "n_warn_total"),
    ("n_warn_deprecated", ok, "n_warn_deprecated"),
    ("n_warn_unused_variable", ok, "n_warn_unused_variable"),
    ("n_warn_linter", ok, "n_warn_linter"),
    ("n_lint_errors", lint_ok, "n_lint_errors"),
    ("lint_errors_per_module", lint_ok, "lint_errors_per_file"),
    ("log_t_rebuild", ok, "log_t_rebuild"),
    ("log_t_rebuild_per_file", ok, "log_t_rebuild_per_file"),
]
for name, d, col in feats:
    x = pd.to_numeric(d[col], errors="coerce")
    auc, n = rank_auc(d.y.values, x.values)
    print(f"| {name} | {auc:.3f} | {n} | {d.y.mean():.3f} |")

print("\n## Feature prevalence (built subset)\n")
print(f"any warning: {(ok.n_warn_total > 0).mean():.3f} "
      f"(mean {ok.n_warn_total.mean():.2f})")
print(f"lint_available: {(ok.lint_available == 1).mean():.3f} "
      f"({len(lint_ok)} PRs)")
if len(lint_ok):
    print(f"any lint error: "
          f"{(pd.to_numeric(lint_ok.n_lint_errors) > 0).mean():.3f} "
          f"(mean {pd.to_numeric(lint_ok.n_lint_errors).mean():.2f})")
    from collections import Counter
    c = Counter()
    for s in lint_ok.lint_linters.dropna():
        for part in str(s).split(";"):
            if ":" in part:
                k, v = part.rsplit(":", 1)
                c[k] += int(v)
    print(f"top linters: {c.most_common(10)}")
print(f"lint timeouts: {pd.to_numeric(df.n_lint_timeouts, errors='coerce').fillna(0).gt(0).sum()} PRs")

print("\n## Warning kinds by year (built subset)\n")
g = ok.groupby("year")[["n_warn_total", "n_warn_deprecated",
                        "n_warn_unused_variable", "n_warn_linter",
                        "n_warn_other"]].mean().round(2)
print(g.to_string())

fails = df[df.status.isin(["build_failed", "build_timeout"])]
print("\n## Failure classes by year\n")
if len(fails):
    t = fails.groupby(["year", "failure_class"]).size().unstack(fill_value=0)
    print(t.to_string())
    print("\nfirst error lines (lean_error):")
    for _, r in fails[fails.failure_class == "lean_error"].iterrows():
        print(f"  PR {r.number} ({r.year}, y={r.y}): "
              f"{str(r.first_error_line)[:140]}")
    print("\nfirst error lines (infra):")
    for _, r in fails[fails.failure_class == "infra"].iterrows():
        print(f"  PR {r.number} ({r.year}, y={r.y}): "
              f"{str(r.first_error_line)[:140]}")

print("\n## Revised headline: genuine first-draft breakage\n")
done = df[df.status.isin(["ok", "build_failed", "build_timeout"])]
n_lean = (done.failure_class == "lean_error").sum()
n_lint_as = (done.failure_class == "lint_as_error").sum()
n_infra = (done.failure_class == "infra").sum()
print(f"of {len(done)} probed: lean_error {n_lean} "
      f"({n_lean/len(done):.3f}), lint_as_error {n_lint_as}, "
      f"infra {n_infra} ({n_infra/len(done):.3f})")
print("\nlean_error rate by y:")
for yv in (0, 1):
    sub = done[done.y == yv]
    print(f"  y={yv}: {(sub.failure_class == 'lean_error').mean():.3f} "
          f"(n={len(sub)})")
print("\nlean_error rate by year:")
for yr, sub in done.groupby("year"):
    print(f"  {yr}: {(sub.failure_class == 'lean_error').mean():.3f} "
          f"(n={len(sub)})")
