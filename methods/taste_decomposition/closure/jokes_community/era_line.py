#!/usr/bin/env python3
"""FREEZE ADDENDUM 4 -- POSITION-IN-CONTAINER audit for the reddit-jokes community cell,
as an OBSERVED-COVARIATE line (never a bank feature, never judged, never fit into
anything that feeds the closure curve).

WHAT THE CONTAINER IS ON THIS CORPUS.  A forum joke has no sibling-set the way a math.SE
answer has its question.  The container is THE SUBREDDIT'S OWN POSTING STREAM and the
ordinal is WHEN the post entered it -- which era of the forum's conventions, running
gags, current-events references and repost cycle the item comes from.  That is the same
family as the programme's other strongest spurious findings (patents claim ordinal, code
repo-recency): an ordinal in a container that a text-only model can read a fingerprint
of.

  created_utc      raw POSIX timestamp of the post          (86.1% matched)
  era_rank_pct     rank of created_utc within the matched population, in [0, 1]
  post_year        calendar year
  post_month_of_y  month within the year (seasonality / holiday-joke cycle)
  post_hour_utc    hour of day (posting-time-of-day effects on forum visibility)
  post_dow         day of week

STRUCTURAL FACTS THAT GOVERN EVERY NUMBER HERE, stated before the readouts.
1. y is a top-quartile-versus-bottom-quartile split taken inside
   (length_bin x format x topic) strata, NOT inside time strata.  Era is therefore NOT
   matched away by the y-definition: unlike math.SE's question-level covariates, an era
   effect here is free to show up, and a null is a real null.
2. Coverage is 86.1%: the timestamp is recovered by an exact sha1 join against the raw
   scrape and 2,228 rows do not match.  Unmatched rows are carried as NaN and DROPPED
   from each readout (never imputed); the matched/unmatched label rates are reported so
   a coverage confound is visible.

CPU only.  Usage: python3 era_line.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import cells as C
import closure_core as L

HERE = Path(__file__).resolve().parent

DIRECTION = {
    "created_utc": "UPSTREAM, container-ordinal -- absolute position in the subreddit's "
                   "posting stream. The FREEZE ADDENDUM 4 channel for this corpus. Not "
                   "matched away by the y-definition (which strata on length/format/topic, "
                   "not time), so a null here is informative.",
    "era_rank_pct": "UPSTREAM -- the same ordinal, rank-normalised so the readout is "
                    "scale-free and robust to the scrape's calendar coverage.",
    "post_year": "UPSTREAM, coarse era -- the level at which forum conventions and meme "
                 "stock actually turn over.",
    "post_month_of_y": "UPSTREAM, seasonal -- holiday and event joke cycles.",
    "post_hour_utc": "UPSTREAM, submission timing -- when a post enters the stream governs "
                     "how much of the audience ever sees it; a classic vote-mechanics "
                     "nuisance with no plausible craft component.",
    "post_dow": "UPSTREAM, submission timing -- same family as post_hour_utc.",
}


def build_covariates(d):
    t = d["created_utc"].astype(float)
    fin = np.isfinite(t)
    rank = np.full(len(t), np.nan)
    if fin.sum():
        order = np.argsort(np.argsort(t[fin]))
        rank[fin] = order / max(fin.sum() - 1, 1)
    yr = np.full(len(t), np.nan)
    mo = np.full(len(t), np.nan)
    hr = np.full(len(t), np.nan)
    dw = np.full(len(t), np.nan)
    for i in np.where(fin)[0]:
        dt = datetime.fromtimestamp(t[i], tz=timezone.utc)
        yr[i], mo[i], hr[i], dw[i] = dt.year, dt.month, dt.hour, dt.weekday()
    return {"created_utc": t, "era_rank_pct": rank, "post_year": yr,
            "post_month_of_y": mo, "post_hour_utc": hr, "post_dow": dw}


def joint_oof(X, y, groups, seeds=(0, 1, 2)):
    Xf = np.nan_to_num(X, nan=-1.0)
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
            m.fit(Xf[tr], y[tr])
            oof[te] = m.predict_proba(Xf[te])[:, 1]
        preds.append(oof)
    return np.mean(preds, axis=0)


def within_topic_auc(y, p, groups, min_n=25):
    """n-weighted mean of within-TOPIC AUCs -- this cell's TIER-2 readout.

    y is a quartile split taken inside (length_bin x format x topic) strata, so the
    within-topic readout is the one that matches the y-definition; it is reported every
    round and never substituted for the pooled tier."""
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
        return float("nan"), {"n_topics_used": 0, "n_rows_used": 0, "n_rows_dropped": drop}
    return float(num / tot), {"n_topics_used": used, "n_rows_used": tot,
                              "n_rows_dropped": drop}


def auc_ok(y, p, min_n=50):
    m = np.isfinite(p)
    if m.sum() < min_n or len(set(y[m].tolist())) < 2:
        return None, int(m.sum())
    return float(roc_auc_score(y[m], p[m])), int(m.sum())


def main():
    sk = C.sklearn_guard()
    d = C.load()
    y, g = d["y"], d["groups"]
    held = np.isin(d["dense_split"], ["eval", "test"])
    z = np.load(HERE / "jokes_community_r0_preds.npz", allow_pickle=True)
    va = z["va_nl"]

    cov = build_covariates(d)
    names = list(cov)
    X = np.column_stack([cov[n] for n in names])
    fin = np.isfinite(cov["created_utc"])

    out = {"cell": "jokes_community", "sklearn": sk, "n": int(len(y)),
           "n_HONEST": int(held.sum()),
           "note": "OBSERVED COVARIATE AUDIT ONLY -- no era variable is ever added to V or "
                   "A, judged by any LLM, or fit into anything feeding the closure curve.",
           "container": "the subreddit's own posting stream; the ordinal is when the post "
                        "entered it (FREEZE ADDENDUM 4 analogue for a corpus whose items "
                        "have no sibling-set container)",
           "coverage": {
               "matched": int(fin.sum()), "rate": float(fin.mean()),
               "label_rate_matched": float(y[fin].mean()),
               "label_rate_unmatched": float(y[~fin].mean()) if (~fin).any() else None,
               "note": "unmatched rows are DROPPED from each readout, never imputed"},
           "structural_note": "y strata on (length_bin x format x topic), NOT on time, so "
                              "era is free to predict y; a null here is a real null.",
           "T_convention_in_this_file": "every T here is the AUC of the seed-MEAN dense "
                                        "prediction vector (the seed ENSEMBLE), because the "
                                        "stratified and stacked readouts need ONE score "
                                        "column. The campaign's quoted T is the "
                                        "mean-over-seeds-of-the-AUC in round0.py / "
                                        "cells.T_by_seed; the two are never mixed in one "
                                        "figure. On HONEST they read .7469 (ensemble) and "
                                        ".7377 (mean-of-AUCs).",
           "variables": {}}

    for k, n in enumerate(names):
        col = X[:, k]
        a_full, n_full = auc_ok(y, col)
        a_hon, n_hon = auc_ok(y[held], col[held])
        m = np.isfinite(col) & held
        wt, wti = within_topic_auc(y[np.isfinite(col)], col[np.isfinite(col)],
                                   g[np.isfinite(col)])
        out["variables"][n] = {
            "coverage": float(np.isfinite(col).mean()),
            "alone_AUC_pooled_full": a_full, "n_full": n_full,
            "alone_AUC_pooled_HONEST": a_hon, "n_HONEST": n_hon,
            "alone_AUC_within_topic_full": wt, **{f"wt_{k2}": v for k2, v in wti.items()},
            "spearman_with_T_HONEST": float(spearmanr(col[m], d["dense"][m]).statistic)
            if m.sum() > 20 else None,
            "spearman_with_VA_nl_HONEST": float(spearmanr(col[m], va[m]).statistic)
            if m.sum() > 20 else None,
            "direction": DIRECTION[n]}

    # headline table: label rate by year
    rows = []
    for v in sorted({int(v) for v in cov["post_year"][fin]}):
        m = fin & (cov["post_year"] == v)
        if m.sum() < 50:
            continue
        rows.append({"post_year": v, "n": int(m.sum()),
                     "label_rate": float(y[m].mean()),
                     "mean_dense_HONEST": float(np.nanmean(d["dense"][m & held]))
                     if (m & held).sum() > 20 else None})
    out["label_rate_by_post_year"] = rows

    jp = joint_oof(X, y, g)
    out["joint_era_model"] = {
        "variables": names,
        "grouped_OOF_AUC_pooled_full": float(roc_auc_score(y, jp)),
        "AUC_pooled_HONEST": float(roc_auc_score(y[held], jp[held])),
        "AUC_within_topic_full": within_topic_auc(y, jp, g)[0],
        "spearman_with_T_HONEST": float(spearmanr(jp[held], d["dense"][held]).statistic),
        "spearman_with_VA_nl_HONEST": float(spearmanr(jp[held], va[held]).statistic),
        "na_encoding": "missing timestamps encoded as -1 so the tree can isolate them; the "
                       "per-variable readouts above drop them instead."}

    # discounted readouts on HONEST, stratified by the era model and by raw era rank
    yh, dh, vh, gh = y[held], d["dense"][held], va[held], g[held]
    for label, score in (("joint_era_model", jp[held]),
                         ("era_rank_pct_raw", cov["era_rank_pct"][held])):
        s = np.nan_to_num(score, nan=-1.0)
        st = L.decile_strata(s, q=10)
        tA, iA = L.stratified_auc(yh, dh, st, min_n=20)
        vA, _ = L.stratified_auc(yh, vh, st, min_n=20)
        out[f"Delta_stratified_on_{label}"] = {
            "pooled_T": float(roc_auc_score(yh, dh)),
            "pooled_VA": float(roc_auc_score(yh, vh)),
            "pooled_Delta": float(roc_auc_score(yh, dh) - roc_auc_score(yh, vh)),
            "T_adj": tA, "VA_adj": vA, "Delta_adj": tA - vA, **iA}

    def stack_auc(base_cols, extra=None):
        Xs = np.column_stack(base_cols + ([extra] if extra is not None else []))
        Xs = np.nan_to_num(Xs, nan=0.0)
        folds = list(GroupKFold(n_splits=min(5, len(np.unique(gh)))).split(
            np.zeros(held.sum()), groups=gh))
        oof = np.zeros(held.sum())
        for tr, te in folds:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            clf.fit(Xs[tr], yh[tr])
            oof[te] = clf.predict_proba(Xs[te])[:, 1]
        return float(roc_auc_score(yh, oof))

    base = [jp[held]]
    inc = {"era_alone": stack_auc(base),
           "era_plus_dense": stack_auc(base, dh),
           "era_plus_bank": stack_auc(base, vh),
           "note": "grouped 5-fold OOF logistic stack on HONEST; the increment the dense arm "
                   "adds over the whole named era family in one scalar (FREEZE ADDENDUM 1 "
                   "stacked readout; does not degenerate as the nuisance set grows)."}
    inc["dense_increment"] = inc["era_plus_dense"] - inc["era_alone"]
    inc["bank_increment"] = inc["era_plus_bank"] - inc["era_alone"]
    out["stacked_increment_over_era"] = inc

    np.savez(HERE / "jokes_community_era.npz", X=X, joint=jp,
             names=np.array(names, dtype=object))
    (HERE / "era_line.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
