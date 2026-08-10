"""
De-confounded thanks label for AoPS forum solutions + re-run of the two key
editorial-similarity analyses.

Confounds on raw thanks (established in editorial_similarity_pilot.md):
  - post_number / within-thread order (within-problem rho ~ -0.55)
  - post age (older posts accumulate thanks; also era effects)
  - thread popularity (topic_num_views)

Recipe (Math.SE v3.3 spirit):
  1. Fit log1p(thanks) ~ f(confounds) with HistGradientBoosting, using
     5-fold OUT-OF-FOLD predictions (folds grouped by problem so a thread
     never predicts itself). Confound features only:
       log1p(post_number), within-thread solution rank (raw + fraction),
       post age (years to corpus max post_time), days since first retained
       solution in the thread, log1p(topic_num_views),
       log1p(n solutions in thread).
  2. thanks_resid = log1p(thanks) - oof_pred  (the de-confounded label)
  3. Validation: Spearman(thanks_resid, each confound), pooled + mean
     within-problem; should be ~0 vs the raw -0.55.
  4. Robustness: propensity-decile matching - decile-bin oof_pred, compute
     Spearman(sim, raw thanks) within each decile, average.

Re-tests (raw vs de-confounded side by side):
  (a) editorial similarity (sim_word / sim_char) vs thanks, pooled +
      within-problem; AUC of sim for top-vs-bottom quartile of label.
  (b) taste-within-correct: same among is_correct solutions; plus simple
      style features (length, latex density, num_edits); plus correctness
      vs thanks (the raw -0.076 artifact check).

Outputs:
  thanks_deconfounded.parquet  (ids + confound features + oof_pred +
                                thanks_resid)
  stdout tables for notes/thanks_deconfounded.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parent.parent
KEY = ["contest", "year", "variant", "problem_n"]

CONFOUNDS = ["post_number", "sol_rank", "sol_rank_frac", "post_age_years",
             "days_since_first_sol", "log_views", "log_n_sols"]


def per_problem_spearman(df, xcol, ycol, min_n=3):
    out = {}
    for k, g in df.groupby(KEY):
        if len(g) < min_n or g[xcol].nunique() < 2 or g[ycol].nunique() < 2:
            continue
        rho = spearmanr(g[xcol], g[ycol]).statistic
        if np.isfinite(rho):
            out[k] = rho
    return pd.Series(out)


def summarize_rhos(rhos, label):
    if not len(rhos):
        print(f"{label}: no eligible problems")
        return
    w = wilcoxon(rhos, alternative="two-sided")
    print(f"{label}: n_problems={len(rhos)} mean_rho={rhos.mean():.3f} "
          f"median_rho={rhos.median():.3f} frac>0={(rhos > 0).mean():.3f} "
          f"wilcoxon_p={w.pvalue:.2e}")


def quartile_auc(df, score_col, label_col):
    """AUC of score for top-vs-bottom quartile of label (pooled)."""
    lo, hi = df[label_col].quantile([0.25, 0.75])
    sub = df[(df[label_col] <= lo) | (df[label_col] >= hi)]
    y = (sub[label_col] >= hi).astype(int)
    if y.nunique() < 2:
        return np.nan, len(sub)
    return roc_auc_score(y, sub[score_col]), len(sub)


def block(df, sims, label, tag):
    """Pooled + within-problem Spearman and quartile AUC for each sim."""
    for c in sims:
        d = df.dropna(subset=[c, label])
        r = spearmanr(d[c], d[label])
        auc, n_auc = quartile_auc(d, c, label)
        print(f"[{tag}] pooled {c} vs {label}: rho={r.statistic:+.3f} "
              f"p={r.pvalue:.2e} (n={len(d)}) | "
              f"AUC top-vs-bottom-quartile={auc:.3f} (n={n_auc})")
        summarize_rhos(per_problem_spearman(d, c, label),
                       f"[{tag}] within-problem {c} vs {label}")


def main() -> int:
    sol = pd.read_parquet(REPO / "editorial_similarity.parquet")
    sol["thanks"] = sol.thanks_received.fillna(0)
    sol["log_thanks"] = np.log1p(sol.thanks)

    # ----- confound features ---------------------------------------------
    ref = sol.post_time.max()
    sol["post_age_years"] = (ref - sol.post_time).dt.days / 365.25
    sol["sol_rank"] = sol.groupby("topic_id").post_number.rank(method="first")
    n_sols = sol.groupby("topic_id").post_id.transform("size")
    sol["sol_rank_frac"] = sol.sol_rank / n_sols
    first_t = sol.groupby("topic_id").post_time.transform("min")
    sol["days_since_first_sol"] = (sol.post_time - first_t).dt.days
    sol["log_views"] = np.log1p(sol.topic_num_views)
    sol["log_n_sols"] = np.log1p(n_sols)
    sol["log_post_number"] = np.log1p(sol.post_number)

    feats = ["log_post_number", "sol_rank", "sol_rank_frac", "post_age_years",
             "days_since_first_sol", "log_views", "log_n_sols"]
    X = sol[feats].values
    y = sol.log_thanks.values
    groups = sol[KEY].astype(str).agg("|".join, axis=1).values

    # ----- out-of-fold confound model -------------------------------------
    oof = np.full(len(sol), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            min_samples_leaf=50, random_state=0)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    sol["oof_pred"] = oof
    sol["thanks_resid"] = sol.log_thanks - sol.oof_pred
    ss_res = ((sol.log_thanks - sol.oof_pred) ** 2).sum()
    ss_tot = ((sol.log_thanks - sol.log_thanks.mean()) ** 2).sum()
    print(f"confound model (OOF, problem-grouped folds): "
          f"R^2={1 - ss_res / ss_tot:.3f} on log1p(thanks), n={len(sol)}")

    # ----- validation: confound removed? ----------------------------------
    print("\n=== validation: residual vs confounds (raw label in parens) ===")
    for c in ["post_number", "sol_rank", "post_age_years", "topic_num_views",
              "days_since_first_sol"]:
        rr = spearmanr(sol[c], sol.thanks_resid)
        rw = spearmanr(sol[c], sol.log_thanks)
        print(f"pooled {c}: resid rho={rr.statistic:+.3f}  "
              f"(raw {rw.statistic:+.3f})")
    rhos_r = per_problem_spearman(sol, "post_number", "thanks_resid")
    rhos_raw = per_problem_spearman(sol, "post_number", "log_thanks")
    summarize_rhos(rhos_r, "within-problem post_number vs thanks_resid")
    summarize_rhos(rhos_raw, "within-problem post_number vs log_thanks (raw)")

    sims = ["sim_word", "sim_char"]

    # ----- (a) similarity vs thanks, raw vs de-confounded ------------------
    print("\n=== (a) editorial similarity vs thanks ===")
    block(sol, sims, "log_thanks", "raw")
    block(sol, sims, "thanks_resid", "deconf")

    # robustness: propensity-decile matching on raw thanks
    print("\n--- robustness: decile-matched (bins of oof_pred) ---")
    sol["pdecile"] = pd.qcut(sol.oof_pred, 10, labels=False, duplicates="drop")
    for c in sims:
        rhos = []
        for d, g in sol.groupby("pdecile"):
            if g[c].nunique() < 2 or g.log_thanks.nunique() < 2:
                continue
            rhos.append((spearmanr(g[c], g.log_thanks).statistic, len(g)))
        mean_rho = np.average([r for r, n in rhos], weights=[n for r, n in rhos])
        print(f"{c} vs raw thanks within oof-decile: "
              f"weighted mean rho={mean_rho:+.3f} over {len(rhos)} deciles")

    # ----- (b) within-correct ----------------------------------------------
    print("\n=== (b) taste-within-correct (AIME/AMC, is_correct==True) ===")
    cor = sol[sol.is_correct == True].copy()  # noqa: E712
    print(f"n correct solutions: {len(cor)}")
    block(cor, sims, "log_thanks", "raw|correct")
    block(cor, sims, "thanks_resid", "deconf|correct")

    # style features among correct
    print("\n--- style features vs thanks_resid | correct ---")
    cor["len_chars"] = cor.body_noquote.str.len()
    cor["latex_density"] = cor.body_noquote.str.count(r"\$") / cor.len_chars
    cor["n_display_math"] = cor.body_noquote.str.count(r"\\\[|\$\$")
    for c in ["len_chars", "latex_density", "n_display_math", "num_edits"]:
        d = cor.dropna(subset=[c])
        r_raw = spearmanr(d[c], d.log_thanks)
        r_res = spearmanr(d[c], d.thanks_resid)
        auc, n_auc = quartile_auc(d, c, "thanks_resid")
        summarize = per_problem_spearman(d, c, "thanks_resid")
        wp = (f"mean within-problem rho={summarize.mean():+.3f} "
              f"(n_prob={len(summarize)})" if len(summarize) else "n/a")
        print(f"{c}: pooled resid rho={r_res.statistic:+.3f} "
              f"p={r_res.pvalue:.2e} (raw {r_raw.statistic:+.3f}) | "
              f"AUC={auc:.3f} | {wp}")

    # correctness vs thanks (artifact check)
    print("\n--- correctness vs thanks (raw artifact check) ---")
    sub = sol[sol.has_answer == True].copy()  # noqa: E712
    sub["corr_int"] = (sub.is_correct == True).astype(int)  # noqa: E712
    for label in ["log_thanks", "thanks_resid"]:
        rhos = per_problem_spearman(sub, "corr_int", label)
        summarize_rhos(rhos, f"within-problem correctness vs {label}")

    # ----- persist ----------------------------------------------------------
    out = REPO / "thanks_deconfounded.parquet"
    sol[["topic_id", "post_id"] + KEY + CONFOUNDS +
        ["log_post_number", "thanks", "log_thanks", "oof_pred",
         "thanks_resid", "sim_word", "sim_char", "has_answer", "is_correct"]
        ].to_parquet(out, index=False)
    print(f"\nwrote {out} ({len(sol)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
