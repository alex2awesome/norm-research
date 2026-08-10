#!/usr/bin/env python3
"""FREEZE ADDENDUM 4 -- POSITION-IN-CONTAINER audit for the AoPS CURATION cell,
as an OBSERVED-COVARIATE line (never a bank feature, never judged, never fit into
anything that feeds the closure curve).

THE BRIEF'S TWO AXES, both recovered as OBSERVED ordinals rather than guessed at
from text (`build_position_covariates.py` joins the raw crawl
`datasets/math/aops/forum_solutions.parquet` at 100% coverage):

  (1) POST POSITION WITHIN THE THREAD -- `post_number`, the true 1-based ordinal
      of the post inside its AoPS topic (post 1 is the problem statement, so
      solutions start at 2), plus `sol_rank`, the 0-based rank of this row among
      the POPULATION's solutions to the same problem, and `position_pct`.
  (2) PROBLEM-THREAD AGE -- `thread_age_days`, how long after the topic's first
      post this solution was written, and `years_after_contest`, how long after
      the competition itself.

Additional observed covariates carried:
  n_sols_group      solutions to this problem in the population   [GROUP-LEVEL]
  n_posts_topic     posts in the whole thread                     [TOPIC-LEVEL]
  num_edits         how many times the poster edited it
  topic_num_views   thread traffic                                [TOPIC-LEVEL]
  poster_n_posts    posts by this author across the crawl -- an UPSTREAM
                    AUTHOR-STANDING proxy, reported beside the position family
                    and kept OUT of the joint position model, which is a
                    position model
  post_year         calendar year of the post (site-era control)

NEIGHBOURING OUTCOMES, reported for context and NEVER modelled or stratified on:
  thanks_received / nothanks_received (crowd approval on the post).

STRUCTURAL CONTRAST WITH THE math.SE CELLS, stated before the readouts.  There
the label was a within-question median split or a one-per-question accept, so
group-level covariates were arithmetically barred from predicting y.  Here y is
"does this solution take the editorial's approach", which is a property of the
solution against an external reference: a problem whose canonical approach is
the obvious one can have EVERY solution positive, and 606 problems differ
enormously in that. Group-level covariates therefore CAN predict y here, and a
high pooled AUC for `n_sols_group` or `topic_num_views` is a real between-problem
effect rather than an artifact -- which is exactly why the within-problem tier is
reported alongside every pooled number.

CPU only.  Usage: python3 position_line.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import cells as C
import closure_core as L

HERE = Path(__file__).resolve().parent

DIRECTION = {
    "post_number": "UPSTREAM, within-thread -- the true thread ordinal. The brief's first "
                   "axis. Exogenous to solution quality and NOT constant within a problem.",
    "sol_rank": "UPSTREAM, within-problem -- rank of this solution among the population's "
                "solutions to the same problem, by thread order. The container the closure "
                "cell actually groups on.",
    "is_first_solution": "UPSTREAM, within-problem -- the sharp form of the same channel",
    "position_pct": "UPSTREAM, within-problem -- arrival order normalised by how many "
                    "solutions the problem drew",
    "thread_age_days": "UPSTREAM, within-thread -- the brief's second axis: how long after "
                       "the thread opened this solution arrived. Separates 'wrote it during "
                       "the contest rush' from 'revived a decade-old thread'.",
    "years_after_contest": "UPSTREAM, mostly within-thread -- how long after the competition "
                           "itself. Confounded with site era; reported beside post_year.",
    "n_sols_group": "GROUP-LEVEL -- how many solutions the problem drew. NOT structurally "
                    "neutralised on this y (see module docstring); a real between-problem "
                    "effect if it predicts.",
    "n_posts_topic": "TOPIC-LEVEL -- thread length including non-solution posts",
    "topic_num_views": "TOPIC-LEVEL -- thread traffic; a popularity/eyeballs proxy",
    "num_edits": "ITEM-LEVEL, upstream -- polishing effort visible only in metadata",
    "post_year": "ERA -- site conventions and notation drift",
    "poster_n_posts": "UPSTREAM AUTHOR STANDING -- how prolific this poster is across the "
                      "whole crawl. NOT a position variable; reported beside the family and "
                      "excluded from the joint position model.",
}

POSITION_FAMILY = ["post_number", "sol_rank", "is_first_solution", "position_pct",
                   "thread_age_days", "years_after_contest", "post_year"]
WITHIN_GROUP = ["post_number", "sol_rank", "is_first_solution", "position_pct",
                "thread_age_days"]
CONTEXT_ONLY = ["n_sols_group", "n_posts_topic", "topic_num_views", "num_edits",
                "poster_n_posts"]
NEIGHBOURING_OUTCOMES = ["thanks_received", "nothanks_received"]


def build_covariates(d):
    cov = d["cov"]
    assert cov is not None, "run build_position_covariates.py first"
    out = {k: cov[k].astype(float) for k in
           ("post_number", "sol_rank", "position_pct", "thread_age_days",
            "years_after_contest", "post_year", "n_sols_group", "n_posts_topic",
            "topic_num_views", "num_edits", "poster_n_posts")}
    out["is_first_solution"] = (cov["sol_rank"] == 0).astype(float)
    return out


def joint_oof(X, y, groups, seeds=(0, 1, 2)):
    folds = list(GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(
        np.zeros(len(y)), groups=groups))
    preds = []
    for s in seeds:
        oof = np.zeros(len(y))
        for tr, te in folds:
            m = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.06,
                                               max_iter=400, early_stopping=True,
                                               validation_fraction=0.1, n_iter_no_change=20,
                                               random_state=s)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        preds.append(oof)
    return np.mean(preds, axis=0)


def within_group_auc(y, p, groups, min_n=2):
    """n-weighted mean of within-PROBLEM AUCs.  Problems with one row or one
    class are dropped and their rows counted."""
    y, p = np.asarray(y), np.asarray(p)
    g = np.asarray([str(x) for x in groups])
    num, tot, used, drop = 0.0, 0, 0, 0
    for q in np.unique(g):
        m = g == q
        n = int(m.sum())
        if n < min_n or len(set(y[m].tolist())) < 2:
            drop += n
            continue
        num += n * roc_auc_score(y[m], p[m])
        tot += n
        used += 1
    if tot == 0:
        return float("nan"), {"n_groups_used": 0, "n_rows_used": 0, "n_rows_dropped": drop}
    return float(num / tot), {"n_groups_used": used, "n_rows_used": tot,
                              "n_rows_dropped": drop}


# name kept for the sibling cells' call sites
within_question_auc = within_group_auc


def auc_ok(y, p):
    m = np.isfinite(p)
    if m.sum() < 20 or len(set(y[m].tolist())) < 2:
        return None, int(m.sum())
    return float(roc_auc_score(y[m], p[m])), int(m.sum())


def main():
    sk = C.sklearn_guard()
    d = C.load()
    y, g = d["y"], d["groups"]
    z = np.load(HERE / "aops_curation_r0_preds.npz", allow_pickle=True)
    va = z["va_nl"]
    dense = d["dense"]

    pos = build_covariates(d)
    all_names = POSITION_FAMILY + CONTEXT_ONLY

    out = {"cell": "aops_curation", "sklearn": sk, "n": int(len(y)),
           "note": "OBSERVED COVARIATE AUDIT ONLY -- no position variable is ever added to "
                   "V or A, judged by any LLM, or fit into anything feeding the closure "
                   "curve. Every ordinal here is RECOVERED from the raw crawl, not inferred "
                   "from text.",
           "structural_note": "y is a match against an external editorial solution, NOT a "
                              "within-group split, so group-level covariates are not "
                              "structurally barred from predicting it; pooled and "
                              "within-problem readouts are both reported for every variable.",
           "variables": {}}
    for n in all_names:
        col = pos[n]
        a_full, n_full = auc_ok(y, col)
        m = np.isfinite(col)
        wq, wqi = within_group_auc(y[m], col[m], g[m])
        out["variables"][n] = {
            "family": "POSITION" if n in POSITION_FAMILY else "CONTEXT",
            "coverage": float(m.mean()),
            "alone_AUC_pooled": a_full, "n": n_full,
            "alone_AUC_within_problem": wq, **{f"wp_{k}": v for k, v in wqi.items()},
            "spearman_with_T": float(spearmanr(col[m], dense[m]).statistic),
            "spearman_with_VA_nl": float(spearmanr(col[m], va[m]).statistic),
            "direction": DIRECTION[n]}

    # neighbouring outcomes: reported, never modelled
    out["neighbouring_outcomes_context_only"] = {}
    for n in NEIGHBOURING_OUTCOMES:
        col = d["cov"][n].astype(float)
        a, nn = auc_ok(y, col)
        out["neighbouring_outcomes_context_only"][n] = {
            "alone_AUC_pooled": a, "n": nn,
            "alone_AUC_within_problem": within_group_auc(y, col, g)[0],
            "WARNING": "crowd-approval outcome on the same post; never a feature, never a "
                       "stratifier, never in any model here"}

    # headline: label rate by thread ordinal and by rank within the problem
    for key, cap in (("post_number", 20), ("sol_rank", 8)):
        p = pos[key]
        rows = []
        for v in range(0, cap + 1):
            m = p == v
            if m.sum() < 25:
                continue
            rows.append({key: v, "n": int(m.sum()), "label_rate": float(y[m].mean()),
                         "mean_thanks": float(np.nanmean(d["cov"]["thanks_received"][m]))})
        m = p > cap
        if m.sum() >= 25:
            rows.append({key: f"{cap}+", "n": int(m.sum()), "label_rate": float(y[m].mean()),
                         "mean_thanks": float(np.nanmean(d["cov"]["thanks_received"][m]))})
        out[f"label_rate_by_{key}"] = rows

    # thread-age quintiles (the brief's second axis)
    ta = pos["thread_age_days"]
    st = L.decile_strata(ta, q=5)
    out["label_rate_by_thread_age_quintile"] = [
        {"quintile": int(s), "n": int((st == s).sum()),
         "days_median": float(np.nanmedian(ta[st == s])),
         "label_rate": float(y[st == s].mean())} for s in np.unique(st)]

    Xpos = np.column_stack([pos[n] for n in POSITION_FAMILY])
    Xwg = np.column_stack([pos[n] for n in WITHIN_GROUP])
    Xall = np.column_stack([pos[n] for n in all_names])
    jp = joint_oof(np.nan_to_num(Xpos, nan=-1), y, g)
    jw = joint_oof(np.nan_to_num(Xwg, nan=-1), y, g)
    ja = joint_oof(np.nan_to_num(Xall, nan=-1), y, g)

    for label, vec, vs in (("joint_position_model", jp, POSITION_FAMILY),
                           ("joint_within_problem_only", jw, WITHIN_GROUP),
                           ("joint_position_plus_context", ja, all_names)):
        out[label] = {
            "variables": vs,
            "grouped_OOF_AUC_pooled": float(roc_auc_score(y, vec)),
            "AUC_within_problem": within_group_auc(y, vec, g)[0],
            "spearman_with_T": float(spearmanr(vec, dense).statistic),
            "spearman_with_VA_nl": float(spearmanr(vec, va).statistic)}

    # discounted readouts on HONEST (= the full population)
    for label, score in (("joint_position_model", jp),
                         ("joint_within_problem_only", jw),
                         ("joint_position_plus_context", ja),
                         ("post_number_raw", pos["post_number"]),
                         ("thread_age_days_raw", ta)):
        st = (L.decile_strata(score, q=10) if not label.endswith("_raw")
              else L.decile_strata(np.nan_to_num(score, nan=-1), q=10))
        tA, iA = L.stratified_auc(y, dense, st, min_n=20)
        vA, _ = L.stratified_auc(y, va, st, min_n=20)
        out[f"Delta_stratified_on_{label}"] = {
            "pooled_T": float(roc_auc_score(y, dense)),
            "pooled_VA": float(roc_auc_score(y, va)),
            "pooled_Delta": float(roc_auc_score(y, dense) - roc_auc_score(y, va)),
            "T_adj": tA, "VA_adj": vA, "Delta_adj": tA - vA, **iA}

    # stacked increment (FREEZE ADDENDUM 1)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def stack_auc(cols, mask):
        Xs = np.nan_to_num(np.column_stack(cols), nan=0.0)[mask]
        yy, gg = y[mask], g[mask]
        folds = list(GroupKFold(n_splits=5).split(np.zeros(mask.sum()), groups=gg))
        oof = np.zeros(mask.sum())
        for tr, te in folds:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            clf.fit(Xs[tr], yy[tr])
            oof[te] = clf.predict_proba(Xs[te])[:, 1]
        return float(roc_auc_score(yy, oof))

    sp = json.loads((HERE / "aops_curation_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    for popname, mask in (("HONEST", np.ones(len(y), bool)), ("MONITOR", split == "monitor")):
        rec = {"position_alone": stack_auc([jp], mask),
               "position_plus_dense": stack_auc([jp, dense], mask),
               "position_plus_bank": stack_auc([jp, va], mask),
               "position_plus_dense_plus_bank": stack_auc([jp, dense, va], mask),
               "n": int(mask.sum())}
        rec["dense_increment"] = rec["position_plus_dense"] - rec["position_alone"]
        rec["bank_increment"] = rec["position_plus_bank"] - rec["position_alone"]
        rec["dense_increment_over_position_plus_bank"] = (
            rec["position_plus_dense_plus_bank"] - rec["position_plus_bank"])
        out[f"stacked_increment_over_position_{popname}"] = rec
    out["stacked_increment_note"] = (
        "grouped 5-fold OOF logistic stack; the increment the dense arm adds over the whole "
        "named position family in one scalar (FREEZE ADDENDUM 1). Does not degenerate as "
        "the nuisance set grows.")

    np.savez(HERE / "aops_curation_position.npz", X=Xpos, joint=jp, joint_within=jw,
             joint_all=ja, Xall=Xall,
             names=np.array(POSITION_FAMILY, dtype=object),
             names_all=np.array(all_names, dtype=object))
    (HERE / "position_line.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
