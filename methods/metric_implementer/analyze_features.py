"""Stage-0 scaling-axis diagnostic (2026-06-18).

Given the assembled per-version table (mining.assemble_all), examine which prompt features and
provenance covariates *lend themselves to a scaling-law axis*. For each candidate axis it reports:

  (1) VARIATION       -- does it move at all? (n_distinct, range, CV). A constant cannot be an axis.
  (2) CO-VARIATION    -- WITHIN-METRIC rank-normalized Spearman vs a recovery outcome, so the signal
                         is not the apples-to-apples confound (easy metrics score high AND happen to
                         look a certain way) [[feedback_apples_to_apples_dense_vs_baseline]].
  (3) CONFOUND        -- how much of the feature is explained by the GEPA operator / optimizer
                         (eta^2); a feature that just tracks the operator is not an independent axis.

It is deliberately DESCRIPTIVE: it reports variation, correlations, and flags, and ranks *candidate*
axes -- it does NOT declare scaling laws [[feedback_report_results_not_conclusions]]. Correlational
signal here is a hypothesis; turning a candidate into a lever needs the intervention/ablation pass.

CAUTION baked into the output: recon_behavioral / fidelity_scalar were (proxies for) the optimizer's
SELECTION objective, so features correlating with them partly reflect what was selected, not
necessarily a causal articulability lever. oracle_spearman (when present) is the more independent
outcome.

CLI:  python -m methods.metric_implementer.analyze_features [outputs_dir] [--outcome recon_behavioral]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import REPO_ROOT
from .features import COVARIATE_COLUMNS, FEATURE_GROUPS, all_feature_names
from .mining import assemble_all

NUMERIC_COVARIATES = ["data_budget_n", "optimizer_round", "lineage_depth", "token_cap"]
CATEGORICAL_CONFOUNDERS = ["operator", "optimizer", "judge_model"]
RECOVERY_OUTCOMES = ["recon_behavioral", "oracle_spearman", "cf_validity", "fidelity_scalar"]
GROUP = ["task", "metric_id"]


def _eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Correlation ratio eta^2: fraction of a numeric feature's variance explained by a
    categorical confounder (operator/optimizer). High => the feature mostly tracks that label."""
    m = np.isfinite(values)
    values, groups = values[m], groups[m]
    if values.size < 4 or np.std(values) < 1e-12:
        return float("nan")
    grand = values.mean()
    ss_tot = float(np.sum((values - grand) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    ss_between = 0.0
    for g in pd.unique(groups):
        v = values[groups == g]
        if v.size:
            ss_between += v.size * (v.mean() - grand) ** 2
    return float(ss_between / ss_tot)


def variation_report(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in columns:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        nn = s.notna()
        vals = s[nn]
        nd = int(vals.nunique())
        std = float(vals.std()) if len(vals) > 1 else 0.0
        mean = float(vals.mean()) if len(vals) else float("nan")
        cv = float(std / abs(mean)) if mean not in (0.0,) and np.isfinite(mean) and abs(mean) > 1e-9 else float("nan")
        flag = ("CONSTANT" if nd <= 1 else "few-levels" if nd <= 3 else "varies")
        rows.append({"column": c, "n_nonnull": int(nn.sum()), "n_distinct": nd,
                     "min": float(vals.min()) if len(vals) else float("nan"),
                     "max": float(vals.max()) if len(vals) else float("nan"),
                     "std": std, "cv": cv, "pct_present": round(100 * nn.sum() / max(n, 1), 1),
                     "variation": flag})
    return pd.DataFrame(rows).sort_values("n_distinct", ascending=False).reset_index(drop=True)


def _within_metric_corr(df: pd.DataFrame, feat: str, outcome: str, min_n: int = 5):
    """Per-(task,metric) Spearman(feat, outcome), then aggregate + a pooled rank-normalized rho.
    Rank-normalizing feat and outcome WITHIN each group before pooling removes between-metric
    level differences (the apples-to-apples control) while keeping all the within-group signal."""
    per_group, pooled_f, pooled_y = [], [], []
    for _, g in df.groupby(GROUP):
        f = pd.to_numeric(g[feat], errors="coerce")
        y = pd.to_numeric(g[outcome], errors="coerce")
        m = f.notna() & y.notna()
        if m.sum() < min_n or f[m].nunique() < 2 or y[m].nunique() < 2:
            continue
        rho = spearmanr(f[m], y[m]).statistic
        if np.isfinite(rho):
            per_group.append(rho)
        # rank-normalize within group to [0,1] then pool
        fr = f[m].rank(pct=True).to_numpy()
        yr = y[m].rank(pct=True).to_numpy()
        pooled_f.extend(fr.tolist())
        pooled_y.extend(yr.tolist())
    if not per_group:
        return None
    pg = np.array(per_group)
    pooled = (spearmanr(pooled_f, pooled_y).statistic
              if len(pooled_f) >= min_n else float("nan"))
    return {"feature": feat, "outcome": outcome, "n_groups": len(pg),
            "mean_within_rho": float(np.mean(pg)),
            "frac_same_sign": float(np.mean(np.sign(pg) == np.sign(np.mean(pg)))),
            "pooled_rank_rho": float(pooled), "n_pooled": len(pooled_f)}


def covariation_report(df: pd.DataFrame, columns: List[str], outcome: str) -> pd.DataFrame:
    rows = []
    for c in columns:
        if c not in df.columns:
            continue
        r = _within_metric_corr(df, c, outcome)
        if r is None:
            continue
        # confound: how much of this feature does the GEPA operator explain?
        if "operator" in df.columns:
            r["operator_eta2"] = _eta_squared(pd.to_numeric(df[c], errors="coerce").to_numpy(),
                                              df["operator"].astype(str).to_numpy())
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["abs_pooled"] = out["pooled_rank_rho"].abs()
    return out.sort_values("abs_pooled", ascending=False).reset_index(drop=True)


def candidate_axes(variation: pd.DataFrame, cov: pd.DataFrame,
                   min_distinct: int = 4, min_rho: float = 0.2,
                   min_groups: int = 2) -> pd.DataFrame:
    """Features that (a) vary enough, (b) co-vary with recovery within-metric with a
    consistent sign across groups, (c) appear in >= min_groups metric-tasks. Ranked by |rho|.
    These are HYPOTHESES to probe with intervention, not established axes."""
    if cov.empty:
        return cov
    varmap = variation.set_index("column")["n_distinct"].to_dict()
    rows = []
    for _, r in cov.iterrows():
        nd = varmap.get(r["feature"], 0)
        if (nd >= min_distinct and abs(r["pooled_rank_rho"]) >= min_rho
                and r["n_groups"] >= min_groups and r["frac_same_sign"] >= 0.6):
            rows.append({**r, "n_distinct": nd})
    return (pd.DataFrame(rows).sort_values("abs_pooled", ascending=False).reset_index(drop=True)
            if rows else pd.DataFrame())


def run(outputs_dir: str, outcome: str = "recon_behavioral",
        tasks: Optional[List[str]] = None, write: bool = True) -> dict:
    df = assemble_all(outputs_dir, tasks=tasks)
    if df.empty:
        print("no versions found")
        return {}
    feat_cols = all_feature_names() + NUMERIC_COVARIATES
    feat_cols = [c for c in feat_cols if c in df.columns]
    smoke = bool(df["judge_model"].astype(str).str.contains("fake").any())

    var = variation_report(df, feat_cols + [c for c in RECOVERY_OUTCOMES if c in df.columns])
    present_outcomes = [o for o in RECOVERY_OUTCOMES
                        if o in df.columns and df[o].notna().sum() >= 8]
    primary = outcome if outcome in present_outcomes else (
        present_outcomes[0] if present_outcomes else outcome)
    cov = covariation_report(df, feat_cols, primary)
    cand = candidate_axes(var, cov)

    lines = []
    lines.append("# Stage-0 scaling-axis diagnostic — prompt features vs recovery\n")
    lines.append(f"*Generated by `analyze_features.py`. {len(df)} prompt versions, "
                 f"{df.groupby(GROUP).ngroups} metric-task groups, {df['task'].nunique()} tasks "
                 f"({', '.join(sorted(df['task'].unique()))}).*\n")
    if smoke:
        lines.append("> **SMOKE DATA.** judge_model contains `fake/*` — these are plumbing "
                     "smoke-test runs, not real scoring. Numbers below validate the *pipeline*; "
                     "the real population is the sk3 registry. Re-run there for science.\n")
    lines.append(f"**Primary recovery outcome:** `{primary}` "
                 f"(non-null on {int(df[primary].notna().sum())} versions). "
                 f"Outcomes available: {', '.join(present_outcomes) or 'none'}.\n")

    lines.append("## 1. Variation — can each thing even be an axis?\n")
    lines.append("A constant has no axis. `cv` = std/|mean|; `n_distinct` = distinct values.\n")
    lines.append(var.to_markdown(index=False, floatfmt=".3f"))
    dead = var[var["variation"] == "CONSTANT"]["column"].tolist()
    if dead:
        lines.append(f"\n**No variation (cannot be an axis here):** {', '.join(dead)}.\n")

    lines.append(f"\n## 2. Co-variation with `{primary}` (within-metric, rank-normalized)\n")
    lines.append("`mean_within_rho` = avg per-metric Spearman; `pooled_rank_rho` = Spearman on "
                 "within-metric rank-normalized values pooled across metrics (the headline, "
                 "apples-to-apples); `frac_same_sign` = sign consistency across metrics; "
                 "`operator_eta2` = fraction of the feature explained by the GEPA operator "
                 "(high ⇒ the feature just tracks the operator, not an independent axis).\n")
    if not cov.empty:
        lines.append(cov[["feature", "n_groups", "mean_within_rho", "pooled_rank_rho",
                          "frac_same_sign", "operator_eta2"]].to_markdown(index=False, floatfmt=".3f"))
    else:
        lines.append("_no feature had enough within-metric variation + non-null outcome to correlate._")

    lines.append("\n\n## 3. Candidate scaling axes (hypotheses to probe)\n")
    lines.append("Filter: varies (n_distinct≥4) · |pooled rank rho|≥0.2 · ≥2 metric-groups · "
                 "sign-consistent ≥0.6. **Correlational — a candidate is a hypothesis, not a "
                 "lever; confirm by intervention (ablate/inject the feature, re-measure).**\n")
    if not cand.empty:
        lines.append(cand[["feature", "n_distinct", "pooled_rank_rho", "mean_within_rho",
                           "n_groups", "frac_same_sign", "operator_eta2"]].to_markdown(
                               index=False, floatfmt=".3f"))
    else:
        lines.append("_no feature cleared the candidate-axis bar on this population._")

    lines.append("\n\n## 4. Caveats\n")
    lines.append("- **Selection confound:** `recon_behavioral`/`fidelity_scalar` were the "
                 "optimizer's (proxy) selection target — features correlating with them may "
                 "reflect *what was selected*, not a causal articulability lever. `oracle_spearman` "
                 "is the more independent outcome; prefer it when present.\n")
    lines.append("- **Operator confound:** features with high `operator_eta2` move *because* a "
                 "GEPA operator (e.g. MECHANIZE lengthens + decomposes) produced them — separate "
                 "the operator effect before claiming the feature is the axis.\n")
    lines.append("- **Correlation ≠ lever:** every candidate needs the intervention pass.\n")
    lines.append("- **Population breadth:** GEPA converges; the bad tail is under-sampled. Add the "
                 "weak/random generator (Phase 1) to widen the range before trusting absolute rhos.\n")

    report = "\n".join(lines)
    if write:
        rpt = REPO_ROOT / "notes" / f"2026-06-18__prompt-feature-scaling-axes__{primary}.md"
        rpt.write_text(report)
        csv = Path(outputs_dir) / "feature_table.csv"
        df.to_csv(csv, index=False)
        print(f"wrote report -> {rpt}\nwrote table  -> {csv} ({len(df)} rows)")
    print("\n" + report[:2000])
    return {"df": df, "variation": var, "covariation": cov, "candidates": cand,
            "primary_outcome": primary}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    outc = next((a.split("=", 1)[1] for a in sys.argv[1:]
                 if a.startswith("--outcome")), "recon_behavioral")
    out = args[0] if args else str(REPO_ROOT / "outputs" / "metric_implementer")
    run(out, outcome=outc)
