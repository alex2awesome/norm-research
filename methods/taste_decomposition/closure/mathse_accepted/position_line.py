#!/usr/bin/env python3
"""FREEZE ADDENDUM 4 -- POSITION-IN-CONTAINER audit for the math.SE VOTE-SCORE
cell, as an OBSERVED-COVARIATE line (never a bank feature, never judged, never
fit into anything that feeds the closure curve).

The container is the QUESTION and the order is answer arrival order, which the
population carries directly as `answer_position` (0-based).  This is the cell's
single most-named upstream prior: the FIRST-ANSWER ADVANTAGE.

Variables (all from datasets/math-stackexchange/v2_va/population.csv.gz):

  answer_position     0-based arrival order of the answer under its question
  is_first            answer_position == 0
  position_pct        answer_position / (n_answers - 1)
  n_answers           how many answers the question drew  [QUESTION-LEVEL]
  answer_year         year the answer was posted          [QUESTION-LEVEL-ish]
  answer_id_rank_pct  global rank of the numeric answer_id -- a corpus/time
                      ordinal, included so absolute time can be told apart from
                      within-question order

STRUCTURAL FACT THAT GOVERNS EVERY NUMBER HERE, stated before the readouts.
y is a WITHIN-QUESTION median split of the raw score, so it is (near-)balanced
inside every question.  A covariate that is CONSTANT WITHIN A QUESTION therefore
cannot predict y at all, no matter how strongly it drives raw scores: `n_answers`
and `answer_year` are question-level and their pooled AUC must sit at ~.50 by
construction, and that is a property of the y-definition, not evidence that the
upstream factor is absent.  Only WITHIN-question covariates -- answer_position
above all -- can move this label.  The y-definition has, in effect, already
matched away the entire question-popularity / era family; the audit's job is the
remaining within-question ordinal.

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
    "answer_position": "UPSTREAM, within-question -- the first-answer advantage the brief "
                       "names. Genuinely exogenous to answer quality and NOT constant "
                       "within a question, so it is the one position variable this y can "
                       "see. A nuisance if the dense arm reads a textual fingerprint of it.",
    "is_first": "UPSTREAM, within-question -- the sharp form of the same channel",
    "position_pct": "UPSTREAM, within-question -- arrival order normalised by how many "
                    "answers the question drew",
    "n_answers": "QUESTION-LEVEL -- constant within the grouping unit, so structurally "
                 "unable to predict a within-question median split. Reported to show the "
                 "y-definition has already matched this family away, never as a null result "
                 "about question popularity.",
    "answer_year": "QUESTION-LEVEL-ish (answers to one question cluster in time) -- same "
                   "structural caveat as n_answers",
    "answer_id_rank_pct": "MIXED -- monotone in absolute posting time (question-level "
                          "component) AND in within-question arrival order (the exogenous "
                          "component); included as the control that separates them",
}
WITHIN_Q = ["answer_position", "is_first", "position_pct", "answer_id_rank_pct"]


def build_covariates(d):
    pos = d["answer_position"].astype(float)
    na = d["n_answers"].astype(float)
    aid = d["answer_id"]
    aidn = np.array([float(a) if str(a).isdigit() else np.nan for a in aid])
    denom = np.where(na > 1, na - 1, np.nan)
    return {
        "answer_position": pos,
        "is_first": (pos == 0).astype(float),
        "position_pct": pos / denom,
        "n_answers": na,
        "answer_year": d["answer_year"].astype(float),
        "answer_id_rank_pct": (np.argsort(np.argsort(np.nan_to_num(aidn, nan=-1)))
                               / max(len(aidn) - 1, 1)),
    }


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


def within_question_auc(y, p, groups, min_n=2):
    """n-weighted mean of within-question AUCs (the readout that respects the
    y-definition). Questions with only one kept answer or one class are dropped."""
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
        return float("nan"), {"n_questions_used": 0, "n_rows_used": 0, "n_rows_dropped": drop}
    return float(num / tot), {"n_questions_used": used, "n_rows_used": tot,
                              "n_rows_dropped": drop}


def auc_ok(y, p):
    m = np.isfinite(p)
    if m.sum() < 20 or len(set(y[m].tolist())) < 2:
        return None, int(m.sum())
    return float(roc_auc_score(y[m], p[m])), int(m.sum())


def main():
    sk = C.sklearn_guard()
    d = C.load()
    y, g = d["y"], d["groups"]
    held = np.isin(d["dense_split"], ["eval", "test"])
    z = np.load(HERE / "mathse_accepted_r0_preds.npz", allow_pickle=True)
    va = z["va_nl"]

    pos = build_covariates(d)
    names = list(pos)
    X = np.column_stack([pos[n] for n in names])

    out = {"cell": "mathse_accepted", "sklearn": sk, "n": int(len(y)),
           "n_HONEST": int(held.sum()),
           "note": "OBSERVED COVARIATE AUDIT ONLY -- no position variable is ever added to "
                   "V or A, judged by any LLM, or fit into anything feeding the closure curve.",
           "structural_caveat": ("y is a WITHIN-QUESTION median split, so question-level "
                                 "covariates (n_answers, answer_year) are structurally unable "
                                 "to predict it; their ~.50 is the y-definition working, not "
                                 "an absence of the upstream factor."),
           "variables": {}}
    for k, n in enumerate(names):
        col = X[:, k]
        a_full, n_full = auc_ok(y, col)
        a_hon, n_hon = auc_ok(y[held], col[held])
        m = np.isfinite(col) & held
        wq, wqi = within_question_auc(y[np.isfinite(col)], col[np.isfinite(col)],
                                      g[np.isfinite(col)])
        out["variables"][n] = {
            "coverage": float(np.isfinite(col).mean()),
            "alone_AUC_pooled_full": a_full, "n_full": n_full,
            "alone_AUC_pooled_HONEST": a_hon, "n_HONEST": n_hon,
            "alone_AUC_within_question_full": wq, **{f"wq_{k2}": v for k2, v in wqi.items()},
            "spearman_with_T_HONEST": float(spearmanr(col[m], d["dense"][m]).statistic)
            if m.sum() > 20 else None,
            "spearman_with_VA_nl_HONEST": float(spearmanr(col[m], va[m]).statistic)
            if m.sum() > 20 else None,
            "direction": DIRECTION[n]}

    # the headline table: label rate by arrival order
    p = pos["answer_position"]
    rows = []
    for v in range(0, 8):
        m = p == v
        if m.sum() < 25:
            continue
        rows.append({"answer_position": v, "n": int(m.sum()),
                     "label_rate": float(y[m].mean()),
                     "mean_raw_score": float(np.nanmean(d["score_raw"][m])),
                     "accepted_rate": float(np.nanmean(d["accepted"][m]))})
    m = p >= 8
    if m.sum() >= 25:
        rows.append({"answer_position": "8+", "n": int(m.sum()),
                     "label_rate": float(y[m].mean()),
                     "mean_raw_score": float(np.nanmean(d["score_raw"][m])),
                     "accepted_rate": float(np.nanmean(d["accepted"][m]))})
    out["label_rate_by_answer_position"] = rows

    jp = joint_oof(X, y, g)
    jw = joint_oof(X[:, [names.index(n) for n in WITHIN_Q]], y, g)
    out["joint_position_model"] = {
        "variables": names,
        "grouped_OOF_AUC_pooled_full": float(roc_auc_score(y, jp)),
        "AUC_pooled_HONEST": float(roc_auc_score(y[held], jp[held])),
        "AUC_within_question_full": within_question_auc(y, jp, g)[0],
        "spearman_with_T_HONEST": float(spearmanr(jp[held], d["dense"][held]).statistic),
        "spearman_with_VA_nl_HONEST": float(spearmanr(jp[held], va[held]).statistic)}
    out["joint_position_model_WITHIN_QUESTION_VARS_ONLY"] = {
        "variables": WITHIN_Q,
        "grouped_OOF_AUC_pooled_full": float(roc_auc_score(y, jw)),
        "AUC_pooled_HONEST": float(roc_auc_score(y[held], jw[held])),
        "AUC_within_question_full": within_question_auc(y, jw, g)[0]}

    # discounted readouts on HONEST, stratified by the position model and by raw position
    yh, dh, vh, gh = y[held], d["dense"][held], va[held], g[held]
    for label, score in (("joint_position_model", jp[held]),
                         ("within_question_vars_only", jw[held]),
                         ("answer_position_raw", p[held])):
        st = (L.decile_strata(score, q=10) if label != "answer_position_raw"
              else np.clip(np.nan_to_num(score, nan=-1), 0, 6).astype(int))
        tA, iA = L.stratified_auc(yh, dh, st, min_n=20)
        vA, _ = L.stratified_auc(yh, vh, st, min_n=20)
        out[f"Delta_stratified_on_{label}"] = {
            "pooled_T": float(roc_auc_score(yh, dh)), "pooled_VA": float(roc_auc_score(yh, vh)),
            "pooled_Delta": float(roc_auc_score(yh, dh) - roc_auc_score(yh, vh)),
            "T_adj": tA, "VA_adj": vA, "Delta_adj": tA - vA, **iA}

    # stacked increment (FREEZE ADDENDUM 1): does the dense score add over position alone?
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def stack_auc(base_cols, extra=None):
        Xs = np.column_stack(base_cols + ([extra] if extra is not None else []))
        Xs = np.nan_to_num(Xs, nan=0.0)
        folds = list(GroupKFold(n_splits=5).split(np.zeros(held.sum()), groups=gh))
        oof = np.zeros(held.sum())
        for tr, te in folds:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            clf.fit(Xs[tr], yh[tr])
            oof[te] = clf.predict_proba(Xs[te])[:, 1]
        return float(roc_auc_score(yh, oof))

    base = [jp[held]]
    out["stacked_increment_over_position"] = {
        "position_alone": stack_auc(base),
        "position_plus_dense": stack_auc(base, dh),
        "position_plus_bank": stack_auc(base, vh),
        "note": "grouped 5-fold OOF logistic stack on HONEST; the increment the dense arm "
                "adds over the whole named position family in one scalar (FREEZE ADDENDUM 1 "
                "stacked readout; does not degenerate as the nuisance set grows).",
    }
    out["stacked_increment_over_position"]["dense_increment"] = (
        out["stacked_increment_over_position"]["position_plus_dense"]
        - out["stacked_increment_over_position"]["position_alone"])
    out["stacked_increment_over_position"]["bank_increment"] = (
        out["stacked_increment_over_position"]["position_plus_bank"]
        - out["stacked_increment_over_position"]["position_alone"])

    # can the TEXT recover arrival order?  the fingerprint question, asked of the two
    # instruments we already have (never a new model): correlation is above; here the
    # tag composition check.
    tags = np.array([str(t) for t in d["primary_tag"]])
    tt, cnt = np.unique(tags, return_counts=True)
    top = tt[np.argsort(-cnt)][:12]
    out["label_rate_by_primary_tag_top12"] = [
        {"tag": str(t) if t else "(blank)", "n": int((tags == t).sum()),
         "label_rate": float(y[tags == t].mean()),
         "mean_answer_position": float(np.nanmean(p[tags == t]))} for t in top]

    np.savez(HERE / "mathse_accepted_position.npz", X=X, joint=jp, joint_within=jw,
             names=np.array(names, dtype=object))
    (HERE / "position_line.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
