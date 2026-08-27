#!/usr/bin/env python3
"""FREEZE ADDENDUM 4 -- position-in-container audit for the two humor cells.

Addendum 4 (2026-08-08) requires Track-B proposers to consider POSITION or ORDER
within the item's container, and -- where the position is recoverable
programmatically -- requires it to be AUDITED as an observed covariate, the way
the N&C campaign audited docket sequence
(notes/2026-08-06__closure_nc_responded.md §7A).  Both corpora here have an
observable position:

  HashtagWars       the tweet id is a Twitter snowflake (monotone in submission
                    time), so within-contest rank by tweet id is SUBMISSION ORDER;
                    plus contest size and raw position in the scraped stream.
  Style Invitational  the published results list a week's entries best-first, so
                    within-week entry index and week size are recoverable; week_id
                    is the contest's ordinal in time.

THIS IS AN AUDIT OF THE CORPUS, NOT A BANK METRIC.  No position variable is ever
added to V or A, judged by any LLM, or fit into anything that feeds the closure
curve.

Readouts (all on the FULL population unless stated):
  1. alone-AUC of each position variable, and of a grouped-OOF joint model;
  2. label rate by container-size quintile;
  3. Spearman correlation of each variable with T and with VA_nl (the question
     that decides whether position can bias Delta or only lowers the ceiling);
  4. Delta on the HONEST population, stratified on position deciles;
  5. DIRECTION TEST: whether the position variable is plausibly upstream of the
     label or is a downstream artifact of how the corpus was published.

CPU only.  Usage: python position_audit.py [cell ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

import cells as C
import closure_core as L

HERE = Path(__file__).resolve().parent

# Whether the variable can be upstream of the outcome, or is a downstream artifact
# of the way the corpus was published.  Stated per variable, not inferred.
DIRECTION = {
    "hashtagwars_verdict": {
        "submission_time_snowflake": "upstream-plausible (absolute submission time)",
        "within_container_rank": "DOWNSTREAM ARTIFACT -- verified: within EVERY contest the "
                                 "10 labelled positives occupy the lowest tweet-id ranks "
                                 "(within-contest AUC .000 in 30/40 contests, mean .022), i.e. "
                                 "the release retrieved the top-10/winner tweets as a separate, "
                                 "earlier batch",
        "within_container_pct": "DOWNSTREAM ARTIFACT -- same retrieval-batch signature",
        "container_size": "upstream-plausible (contest popularity / entry count)",
        "stream_ix": "corpus-construction artifact (order of the scraped TSV stream)",
    },
    "style_inv_toptier": {
        "entry_ix": "DOWNSTREAM ARTIFACT -- the published results list winners first",
        "within_container_rank": "DOWNSTREAM ARTIFACT -- publication order is the ranking",
        "within_container_pct": "DOWNSTREAM ARTIFACT -- publication order is the ranking",
        "container_size": "upstream-plausible (how many entries the column printed)",
        "week_num": "upstream-plausible (the contest's ordinal in time)",
    },
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
                                               validation_fraction=0.1,
                                               n_iter_no_change=20, random_state=s)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        preds.append(oof)
    return np.mean(preds, axis=0)


def run(cell):
    d = C.load(cell)
    sp = json.loads((HERE / f"{cell}_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    y, g = d["y"], d["groups"]
    held = np.isin(d["dense_split"], ["eval", "test"])
    pos = d["position"]
    names = sorted(pos)
    X = np.column_stack([pos[n] for n in names])

    z = np.load(HERE / f"{cell}_r0_preds.npz")
    va = z["va_nl"]

    out = {"cell": cell, "n": int(len(y)), "n_HONEST": int(held.sum()),
           "variables": {}, "note": "OBSERVED COVARIATE AUDIT ONLY -- never added to "
                                    "V or A, never judged, never fit into the closure curve"}
    for k, n in enumerate(names):
        col = X[:, k]
        out["variables"][n] = {
            "alone_AUC_full": L.auc(y, col),
            "alone_AUC_HONEST": L.auc(y[held], col[held]),
            "spearman_with_T_HONEST": float(spearmanr(col[held], d["dense"][held]).statistic),
            "spearman_with_VA_nl_HONEST": float(spearmanr(col[held], va[held]).statistic),
            "direction": DIRECTION[cell].get(n, "unstated"),
        }

    jp = joint_oof(X, y, g)
    out["joint_position_model"] = {
        "variables": names,
        "grouped_OOF_AUC_full": L.auc(y, jp),
        "AUC_HONEST": L.auc(y[held], jp[held]),
        "spearman_with_T_HONEST": float(spearmanr(jp[held], d["dense"][held]).statistic),
        "spearman_with_VA_nl_HONEST": float(spearmanr(jp[held], va[held]).statistic),
    }
    # upstream-only joint model (drops the publication-order artifacts)
    up = [k for k, n in enumerate(names) if not DIRECTION[cell].get(n, "").startswith("DOWNSTREAM")]
    if len(up) < len(names):
        jp_up = joint_oof(X[:, up], y, g)
        out["joint_position_model_UPSTREAM_ONLY"] = {
            "variables": [names[k] for k in up],
            "grouped_OOF_AUC_full": L.auc(y, jp_up),
            "AUC_HONEST": L.auc(y[held], jp_up[held]),
        }
        np.savez(HERE / f"{cell}_position.npz", joint=jp, joint_upstream=jp_up,
                 names=np.array(names, dtype=object), X=X)
    else:
        np.savez(HERE / f"{cell}_position.npz", joint=jp,
                 names=np.array(names, dtype=object), X=X)

    # label rate by container-size quintile
    sz = pos["container_size"]
    q = L.decile_strata(sz, q=5)
    out["label_rate_by_container_size_quintile"] = [
        {"quintile": int(v), "n": int((q == v).sum()),
         "mean_container_size": float(sz[q == v].mean()),
         "label_rate": float(y[q == v].mean())} for v in sorted(set(q.tolist()))]

    # Delta stratified on position deciles (HONEST)
    yh, dh, vh = y[held], d["dense"][held], va[held]
    for label, score in (("joint", jp[held]),) + (
            (("joint_upstream", out.get("joint_position_model_UPSTREAM_ONLY")
              and np.load(HERE / f"{cell}_position.npz")["joint_upstream"][held]),)
            if len(up) < len(names) else ()):
        if score is None:
            continue
        st = L.decile_strata(score, q=10)
        tA, iA = L.stratified_auc(yh, dh, st, min_n=20)
        vA, _ = L.stratified_auc(yh, vh, st, min_n=20)
        out[f"Delta_stratified_on_position_{label}"] = {
            "pooled_T": L.auc(yh, dh), "pooled_VA": L.auc(yh, vh),
            "pooled_Delta": L.auc(yh, dh) - L.auc(yh, vh),
            "T_adj": tA, "VA_adj": vA, "Delta_adj": tA - vA, **iA}

    (HERE / f"{cell}_position_audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    todo = sys.argv[1:] or C.CELLS
    allr = {c: run(c) for c in todo}
    (HERE / "position_audit_all.json").write_text(json.dumps(allr, indent=1))
