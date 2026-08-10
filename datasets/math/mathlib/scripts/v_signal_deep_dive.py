#!/usr/bin/env python3
"""Conditional / concentrated V-signal deep dive for mathlib review friction.

Pre-stated hypotheses (PI, 2026-06-12):
  H1 process-not-outcome: first-push CI failure -> more review rounds / days_open
     among merged, conditional on size.
  H2 precision-not-AUC: risk ratio P(friction | CI-failed) / P(friction | clean)
     + coverage.
  H3 dose-response: friction by dose of fix iterations (pre-review force pushes),
     lint-error count, rebuild time; conditional on size.
  H4 area heterogeneity: does the V signal concentrate in tactic/meta code?
  H5 V as entry gate: among closed-unmerged PRs, what fraction of closures are
     V-attributable (CI failing) vs V-clean?

Data: friction_full_v2.csv.gz (32,787 merged), pr_reviews_mathlib4.jsonl (37,249
closed PRs incl. first-commit statusCheckRollup, force-push timeline, review
times), pr_thread_comments.jsonl (thread-comment timestamps), lake_probe_results
.csv (504 first-push lake builds), lake_lint_results.csv (393 lint/rebuild rows).

Everything conditions on PR size (log additions+deletions, log changed_files)
plus year and author_association where regression-based.
"""
import json
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib"
RNG = np.random.default_rng(0)


def sec(title):
    print("\n" + "=" * 78)
    print("## " + title)
    print("=" * 78)


def rr_ci(k1, n1, k0, n0, label=""):
    """Risk ratio with Wald CI on log scale."""
    if min(k1, k0) == 0:
        print(f"  {label}: zero cell (k1={k1}, k0={k0}) — RR undefined")
        return None
    p1, p0 = k1 / n1, k0 / n0
    rr = p1 / p0
    se = np.sqrt(1 / k1 - 1 / n1 + 1 / k0 - 1 / n0)
    lo, hi = np.exp(np.log(rr) - 1.96 * se), np.exp(np.log(rr) + 1.96 * se)
    print(f"  {label}: P1={p1:.3f} (k={k1}/n={n1})  P0={p0:.3f} (k={k0}/n={n0})  "
          f"RR={rr:.3f} [{lo:.3f}, {hi:.3f}]")
    return rr, lo, hi


def partial_spearman(df, x, y, controls):
    """Spearman partial correlation: rank-transform, residualize, Pearson."""
    d = df[[x, y] + controls].dropna()
    R = d.rank()
    Xc = sm.add_constant(R[controls])
    rx = sm.OLS(R[x], Xc).fit().resid
    ry = sm.OLS(R[y], Xc).fit().resid
    r, p = stats.pearsonr(rx, ry)
    return r, p, len(d)


def poisson_rr(df, formula, term):
    """Poisson GLM with robust (HC1) SEs -> rate ratio for `term`."""
    m = smf.glm(formula, data=df, family=sm.families.Poisson()).fit(cov_type="HC1")
    b, se = m.params[term], m.bse[term]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues[term], int(m.nobs)


def logit_term(df, formula, term):
    m = smf.logit(formula, data=df).fit(disp=0)
    b, se = m.params[term], m.bse[term]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues[term], int(m.nobs)


# ============================================================================
# LOAD
# ============================================================================
sec("LOAD + JOIN")

full = pd.read_csv(f"{BASE}/friction_full_v2.csv.gz", low_memory=False)
full["friction"] = 1 - full["y"]          # y=1 frictionless (zero threads)
full["log_size"] = np.log1p(full["additions"] + full["deletions"])
full["log_files"] = np.log1p(full["changed_files"])
merged_numbers = set(full["number"])
print(f"friction_full_v2: {len(full)} merged PRs, P(friction)={full['friction'].mean():.3f}")

# --- per-PR GraphQL record scan ---
recs = []
with open(f"{BASE}/pr_reviews_mathlib4.jsonl") as f:
    for line in f:
        r = json.loads(line)
        cn = r.get("commits", {}).get("nodes") or []
        scr = cn[0]["commit"].get("statusCheckRollup") if cn else None
        revs = r["reviews"]["nodes"] or []
        rev_times = [x["submittedAt"] for x in revs if x.get("submittedAt")]
        tl = r["timelineItems"]
        fp_times = sorted(n["createdAt"] for n in (tl["nodes"] or [])
                          if n and n.get("createdAt"))
        labels = [l["name"] for l in (r["labels"]["nodes"] or [])]
        recs.append(dict(
            number=r["number"],
            gh_state=r["state"],
            created=r["createdAt"],
            closed=r["closedAt"],
            yr=int(r["createdAt"][:4]),
            ci_state=(scr or {}).get("state"),
            first_rev_time=min(rev_times) if rev_times else None,
            # NOTE: timelineItems.totalCount IGNORES the itemTypes filter
            # (counts all timeline items) — verified empirically. True
            # force-push count = len(nodes), capped at 20 by `first: 20`.
            fp_total=len(fp_times),
            fp_times=fp_times,
            labels_list=labels,
            assoc_j=r["authorAssociation"],
            adds_j=r["additions"], dels_j=r["deletions"], files_j=r["changedFiles"],
            title_j=r["title"],
        ))
J = pd.DataFrame(recs)
print(f"pr_reviews jsonl: {len(J)} PRs; ci_state dist: "
      f"{J['ci_state'].value_counts(dropna=False).to_dict()}")

# --- first thread-comment time per PR ---
first_thread = {}
with open(f"{BASE}/pr_thread_comments.jsonl") as f:
    for line in f:
        r = json.loads(line)
        times = [c["createdAt"]
                 for th in (r["reviewThreads"]["nodes"] or [])
                 for c in ((th.get("comments") or {}).get("nodes") or [])
                 if c.get("createdAt")]
        if times:
            n = int(r["number"])
            t = min(times)
            if n not in first_thread or t < first_thread[n]:
                first_thread[n] = t
J["first_thread_time"] = J["number"].map(first_thread)
print(f"thread-comment times for {J['first_thread_time'].notna().sum()} PRs")

# t_first_review = earliest human review activity (formal review or line thread)
def _tmin(a, b):
    ts = [t for t in (a, b) if isinstance(t, str)]
    return min(ts) if ts else None

J["t_first_review"] = [_tmin(a, b) for a, b in zip(J["first_rev_time"], J["first_thread_time"])]

# pre-review force pushes (the author's fix-iteration loop before any human review)
def _prereview_fp(fp_times, t):
    if not fp_times:
        return 0
    if t is None:
        return len(fp_times)        # never reviewed: all pushes are self/CI-driven
    return sum(1 for x in fp_times if x < t)

J["fp_pre"] = [_prereview_fp(ft, t) for ft, t in zip(J["fp_times"], J["t_first_review"])]
J["fp_trunc"] = J["fp_total"] >= 20
print(f"force-push timeline truncated (>=20 events, capped) for {J['fp_trunc'].sum()} PRs")

# topic areas from labels
TOPIC_RE = re.compile(r"^t-[a-z0-9-]+$")
J["topics"] = J["labels_list"].apply(lambda ls: [l for l in ls if TOPIC_RE.match(l)])

# --- merged master table M ---
M = full.merge(
    J[["number", "ci_state", "fp_pre", "fp_total", "fp_trunc", "t_first_review",
       "topics", "labels_list", "yr"]],
    on="number", how="left")
print(f"M (merged + jsonl): {len(M)} rows")

# CI cohort CC: merged, first-commit rollup present and decisive
CC = M[M["ci_state"].isin(["SUCCESS", "FAILURE"])].copy()
CC["ci_failed"] = (CC["ci_state"] == "FAILURE").astype(int)
CC["year_f"] = CC["year"].astype(str)
print(f"CC (merged w/ first-commit CI rollup): {len(CC)}  "
      f"years: {CC['year'].value_counts().to_dict()}  "
      f"P(ci_failed)={CC['ci_failed'].mean():.3f}")

# size <-> ci_failed sanity (the universal confound, must be conditioned)
r_sz, p_sz = stats.spearmanr(CC["ci_failed"], CC["log_size"])
print(f"sanity: spearman(ci_failed, log_size) = {r_sz:.3f} (p={p_sz:.2g})")

# probe cohorts
P = pd.read_csv(f"{BASE}/lake_probe_results.csv")
P = P.merge(full[["number", "friction", "y", "log_size", "log_files",
                  "n_review_threads", "days_open", "year"]],
            on="number", how="left", suffixes=("_probe", ""))
P["build_failed"] = (P["status"] == "build_failed").astype(int)
L = pd.read_csv(f"{BASE}/lake_lint_results.csv")
L = L.merge(full[["number", "friction", "log_size", "log_files",
                  "n_review_threads", "days_open"]], on="number", how="left",
            suffixes=("_lint", ""))
print(f"P (lake build probe): {len(P)}, failures={P['build_failed'].sum()}")
print(f"L (lake lint probe): {len(L)}, status: {L['status'].value_counts().to_dict()}")

# ============================================================================
# H1 — PROCESS NOT OUTCOME
# ============================================================================
sec("H1: first-push CI failure -> review rounds / days_open among merged, | size")

print("\n[H1a] CI cohort CC (2025-26, n=%d). Outcome: n_review_threads" % len(CC))
rr, lo, hi, p, n = poisson_rr(
    CC, "n_review_threads ~ ci_failed + log_size + log_files + C(year_f) "
       "+ C(author_association)", "ci_failed")
print(f"  Poisson rate ratio (robust): {rr:.3f} [{lo:.3f}, {hi:.3f}]  p={p:.2g}  n={n}")
raw1 = CC.groupby("ci_failed")["n_review_threads"].agg(["mean", "median", "count"])
print("  raw means:\n", raw1.to_string())

print("\n[H1b] Outcome: days_open (log1p OLS, same controls)")
m = smf.ols("np.log1p(days_open) ~ ci_failed + log_size + log_files + C(year_f)"
            " + C(author_association)", data=CC).fit(cov_type="HC1")
b, se = m.params["ci_failed"], m.bse["ci_failed"]
print(f"  multiplicative effect on (1+days): {np.exp(b):.3f} "
      f"[{np.exp(b-1.96*se):.3f}, {np.exp(b+1.96*se):.3f}]  p={m.pvalues['ci_failed']:.2g}")
raw2 = CC.groupby("ci_failed")["days_open"].agg(["mean", "median", "count"])
print("  raw days_open:\n", raw2.to_string())
r, p2, n2 = partial_spearman(CC, "ci_failed", "days_open", ["log_size", "log_files"])
print(f"  partial Spearman(ci_failed, days_open | size): r={r:.3f} p={p2:.2g} n={n2}")
r, p2, n2 = partial_spearman(CC, "ci_failed", "n_review_threads", ["log_size", "log_files"])
print(f"  partial Spearman(ci_failed, n_threads | size):  r={r:.3f} p={p2:.2g} n={n2}")

print("\n[H1c] Probe cohort P (our own lake build, 2022-25, n with outcome=%d)"
      % P["friction"].notna().sum())
Pq = P.dropna(subset=["friction"])
rr, lo, hi, p, n = poisson_rr(
    Pq, "n_review_threads ~ build_failed + log_size + log_files", "build_failed")
print(f"  Poisson rate ratio threads ~ build_failed: {rr:.3f} [{lo:.3f}, {hi:.3f}] p={p:.2g} n={n}")
r, p2, n2 = partial_spearman(Pq, "build_failed", "days_open", ["log_size", "log_files"])
print(f"  partial Spearman(build_failed, days_open | size): r={r:.3f} p={p2:.2g} n={n2}")

# ============================================================================
# H2 — PRECISION NOT AUC
# ============================================================================
sec("H2: risk ratio + coverage (V fires on k%; when it fires, friction risk xX)")

print("\n[H2a] CI cohort CC: friction = >=1 review thread")
cov = CC["ci_failed"].mean()
print(f"  coverage: V fires (first-commit CI failed) on {cov*1e2:.1f}% of merged PRs "
      f"(n={CC['ci_failed'].sum()}/{len(CC)})")
k1 = int(CC.loc[CC.ci_failed == 1, "friction"].sum()); n1 = int((CC.ci_failed == 1).sum())
k0 = int(CC.loc[CC.ci_failed == 0, "friction"].sum()); n0 = int((CC.ci_failed == 0).sum())
rr_ci(k1, n1, k0, n0, "unadjusted RR")
# size-adjusted RR via Poisson-robust on binary outcome
rr, lo, hi, p, n = poisson_rr(
    CC, "friction ~ ci_failed + log_size + log_files + C(year_f) + C(author_association)",
    "ci_failed")
print(f"  size/year/assoc-adjusted RR: {rr:.3f} [{lo:.3f}, {hi:.3f}]  p={p:.2g}")
# Mantel-Haenszel over size quartiles x year
CC["size_q"] = pd.qcut(CC["log_size"], 4, labels=False, duplicates="drop")
num = den = 0.0
for _, g in CC.groupby(["size_q", "year"]):
    nn1 = (g.ci_failed == 1).sum(); nn0 = (g.ci_failed == 0).sum()
    if nn1 == 0 or nn0 == 0:
        continue
    a = int(g.loc[g.ci_failed == 1, "friction"].sum())
    c = int(g.loc[g.ci_failed == 0, "friction"].sum())
    N = len(g)
    num += a * nn0 / N
    den += c * nn1 / N
print(f"  Mantel-Haenszel RR (size-quartile x year strata): {num/den:.3f}")
# implied discrimination of the binary feature, for context vs pooled-AUC framing
p_f1 = k1 / n1; p_f0 = k0 / n0
auc_bin = 0.5 + 0.5 * (p_f1 - p_f0) * 1  # = (sens - (1-spec))/2 + 0.5 of the binary
sens = k1 / (k1 + k0)
spec = (n0 - k0) / ((n1 - k1) + (n0 - k0))
print(f"  context: as a binary classifier this is sens={sens:.3f}, "
      f"spec-ish split — implied AUC = {0.5*(sens + spec):.3f} (why pooled AUC hides it)")

print("\n[H2b] Probe cohort P (our lake build, balanced 50/50 sample)")
Pq = P.dropna(subset=["friction"])
cov = Pq["build_failed"].mean()
print(f"  coverage: build fails at first push on {cov*1e2:.1f}% "
      f"(n={int(Pq['build_failed'].sum())}/{len(Pq)})")
k1 = int(Pq.loc[Pq.build_failed == 1, "friction"].sum()); n1 = int((Pq.build_failed == 1).sum())
k0 = int(Pq.loc[Pq.build_failed == 0, "friction"].sum()); n0 = int((Pq.build_failed == 0).sum())
rr_ci(k1, n1, k0, n0, "unadjusted RR (NOTE: balanced sample, P(friction)=0.5 design)")

print("\n[H2c] Probe, genuine lean errors only (failure_class from lint re-probe)")
Lq = L.dropna(subset=["friction"])
gen = Lq[Lq["failure_class"] == "lean_error"]
cln = Lq[Lq["status"] == "ok"]
if len(gen) >= 5:
    k1, n1 = int(gen["friction"].sum()), len(gen)
    k0, n0 = int(cln["friction"].sum()), len(cln)
    rr_ci(k1, n1, k0, n0, f"lean_error (n={n1}) vs built (n={n0})")
else:
    print(f"  only {len(gen)} genuine lean errors with outcome — too small")

# ============================================================================
# H3 — DOSE-RESPONSE
# ============================================================================
sec("H3: dose-response of fix iterations / lint errors / rebuild time")

print("\n[H3a] Dose = pre-review force pushes (fix loop before any human review),")
print("       all-years merged cohort M")
Mq = M.dropna(subset=["fp_pre", "friction"]).copy()
Mq["fp_cat"] = pd.cut(Mq["fp_pre"], [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5, np.inf],
                      labels=["0", "1", "2", "3", "4-5", "6+"])
tab = Mq.groupby("fp_cat").agg(
    n=("friction", "size"), P_friction=("friction", "mean"),
    mean_threads=("n_review_threads", "mean"),
    med_days=("days_open", "median"), med_size=("log_size", "median"))
print(tab.to_string(float_format=lambda x: f"{x:.3f}"))
Mq["year_f"] = Mq["year"].astype(str)
Mq["fp_pre_c"] = Mq["fp_pre"].clip(upper=10)
orr, lo, hi, p, n = logit_term(
    Mq, "friction ~ fp_pre_c + log_size + log_files + C(year_f) + C(author_association)",
    "fp_pre_c")
print(f"  trend (logit, OR per pre-review push, | size/year/assoc): "
      f"OR={orr:.3f} [{lo:.3f}, {hi:.3f}]  p={p:.2g}  n={n}")
r, p2, n2 = partial_spearman(Mq, "fp_pre", "n_review_threads", ["log_size", "log_files"])
print(f"  partial Spearman(fp_pre, n_threads | size): r={r:.3f} p={p2:.2g} n={n2}")

print("\n[H3a'] same dose within CI cohort CC, by ci_failed (does the loop mediate?)")
Cq = CC.dropna(subset=["fp_pre"]).copy()
Cq["fp_pre_c"] = Cq["fp_pre"].clip(upper=10)
for v, g in Cq.groupby("ci_failed"):
    r, p2, n2 = partial_spearman(g, "fp_pre", "n_review_threads", ["log_size", "log_files"])
    print(f"  ci_failed={v}: partial Spearman(fp_pre, threads | size) r={r:.3f} p={p2:.2g} n={n2}")

print("\n[H3b] Dose = n_lint_errors (lint probe L)")
Lb = L[(L["lint_available"] == 1) & L["friction"].notna()].copy()
Lb["lint_cat"] = pd.cut(Lb["n_lint_errors"], [-0.5, 0.5, 5.5, np.inf],
                        labels=["0", "1-5", "6+"])
tab = Lb.groupby("lint_cat").agg(n=("friction", "size"), P_friction=("friction", "mean"))
print(tab.to_string(float_format=lambda x: f"{x:.3f}"))
r, p2, n2 = partial_spearman(Lb, "n_lint_errors", "friction", ["log_size", "log_files"])
print(f"  partial Spearman(n_lint_errors, friction | size): r={r:.3f} p={p2:.2g} n={n2}")

print("\n[H3c] Dose = t_rebuild (cache-hot rebuild wall time, proof-weight proxy)")
Lc = L[(L["build_exit"] == 0) & L["friction"].notna()].copy()
Lc["log_trebuild"] = np.log1p(Lc["t_rebuild"])
r, p2, n2 = partial_spearman(Lc, "log_trebuild", "friction", ["log_size", "log_files"])
print(f"  partial Spearman(log t_rebuild, friction | size): r={r:.3f} p={p2:.2g} n={n2}")
Lc["tr_q"] = pd.qcut(Lc["t_rebuild"], 5, labels=False, duplicates="drop")
tab = Lc.groupby("tr_q").agg(n=("friction", "size"), P_friction=("friction", "mean"),
                             med_t=("t_rebuild", "median"))
print(tab.to_string(float_format=lambda x: f"{x:.3f}"))

# ============================================================================
# H4 — AREA HETEROGENEITY
# ============================================================================
sec("H4: does the V signal concentrate in tactic/meta code?")

Cx = CC.explode("topics")
areas = Cx["topics"].value_counts()
print("areas with n>=200 in CI cohort:\n", areas[areas >= 200].to_string())
print("\nper-area: coverage P(ci_failed) and RR(friction | ci_failed vs clean):")
rows = []
for a, na in areas[areas >= 200].items():
    g = Cx[Cx["topics"] == a]
    k1 = int(g.loc[g.ci_failed == 1, "friction"].sum()); n1 = int((g.ci_failed == 1).sum())
    k0 = int(g.loc[g.ci_failed == 0, "friction"].sum()); n0 = int((g.ci_failed == 0).sum())
    if min(k1, k0, n1 - k1, n0 - k0) > 0:
        rr = (k1 / n1) / (k0 / n0)
        se = np.sqrt(1 / k1 - 1 / n1 + 1 / k0 - 1 / n0)
        rows.append(dict(area=a, n=na, cov=n1 / (n1 + n0), rr=rr,
                         lo=np.exp(np.log(rr) - 1.96 * se),
                         hi=np.exp(np.log(rr) + 1.96 * se),
                         base_friction=k0 / n0))
A = pd.DataFrame(rows).sort_values("rr", ascending=False)
print(A.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

CC["is_meta"] = CC["topics"].apply(lambda ts: int(any(t in ("t-meta",) for t in (ts or []))))
print(f"\nis_meta (t-meta label) n={CC['is_meta'].sum()}")
m = smf.logit("friction ~ ci_failed * is_meta + log_size + log_files + C(year_f)",
              data=CC).fit(disp=0)
ib, ise = m.params["ci_failed:is_meta"], m.bse["ci_failed:is_meta"]
print(f"  interaction ci_failed x is_meta (logit OR): {np.exp(ib):.3f} "
      f"[{np.exp(ib-1.96*ise):.3f}, {np.exp(ib+1.96*ise):.3f}] "
      f"p={m.pvalues['ci_failed:is_meta']:.2g}")

# probe cohort, descriptive only
Pt = P.merge(J[["number", "topics"]], on="number", how="left")
Pt["is_meta"] = Pt["topics"].apply(lambda ts: int(any(t == "t-meta" for t in (ts or []))))
print("\nprobe cohort build failures x t-meta (descriptive, tiny n):")
print(pd.crosstab(Pt["is_meta"], Pt["build_failed"]).to_string())

# ============================================================================
# H5 — V AS ENTRY GATE
# ============================================================================
sec("H5: closed-without-merge — V-attributable vs V-clean closures")

U = J[~J["number"].isin(merged_numbers)].copy()
print(f"unmerged closed PRs in fetch: {len(U)} ({len(U)/len(J)*1e2:.1f}% of all closed)")

ABANDON = ["merge-conflict", "awaiting-author", "WIP", "blocked-by-other-PR",
           "please-adopt", "stale", "awaiting-zulip", "too-late", "duplicate",
           "awaiting-CI", "LLM-generated", "easy", "help-wanted"]
print("\nclosure-state labels among unmerged (multi-label, all years):")
for lab in ABANDON:
    n = U["labels_list"].apply(lambda ls: lab in ls).sum()
    if n > 0:
        print(f"  {lab:<22} {n:>5}  ({n/len(U)*1e2:.1f}%)")
n_none = U["labels_list"].apply(
    lambda ls: not any(l in ABANDON for l in ls)).sum()
print(f"  {'<none of the above>':<22} {n_none:>5}  ({n_none/len(U)*1e2:.1f}%)")

Ur = U[U["ci_state"].isin(["SUCCESS", "FAILURE"])].copy()
Cr = CC  # merged baseline same era
print(f"\nrollup-present unmerged (2025-26 era): n={len(Ur)}")
print(f"  P(first-commit CI FAILURE | closed-unmerged) = "
      f"{(Ur['ci_state']=='FAILURE').mean():.3f} (n={len(Ur)})")
print(f"  P(first-commit CI FAILURE | merged, same era) = "
      f"{Cr['ci_failed'].mean():.3f} (n={len(Cr)})")
k1, n1 = int((Ur["ci_state"] == "FAILURE").sum()), len(Ur)
k0, n0 = int(Cr["ci_failed"].sum()), len(Cr)
rr_ci(k1, n1, k0, n0, "RR of first-commit CI failure, unmerged vs merged")

Ur["has_abandon_label"] = Ur["labels_list"].apply(
    lambda ls: any(l in ABANDON for l in ls))
ct = pd.crosstab(Ur["ci_state"], Ur["has_abandon_label"], margins=True)
print("\nCI state x has-abandonment-label among rollup-present unmerged:")
print(ct.to_string())
vc = Ur[(Ur["ci_state"] == "FAILURE") & (~Ur["has_abandon_label"])]
print(f"\n'V-attributable, no other stated reason' closures: {len(vc)} "
      f"= {len(vc)/len(Ur)*1e2:.1f}% of rollup-present unmerged")
print(f"'V-clean (CI SUCCESS) closures': "
      f"{(Ur['ci_state']=='SUCCESS').sum()} = "
      f"{(Ur['ci_state']=='SUCCESS').mean()*1e2:.1f}%")

# size comparison: are CI-failed unmerged just bigger?
Ur["log_size"] = np.log1p(Ur["adds_j"] + Ur["dels_j"])
print("\nmedian log_size by CI state (unmerged):")
print(Ur.groupby("ci_state")["log_size"].agg(["median", "count"]).to_string())

print("\nDONE")
