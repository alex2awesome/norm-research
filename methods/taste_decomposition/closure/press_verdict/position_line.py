#!/usr/bin/env python3
"""FREEZE ADDENDUM 4 -- position-in-container audit for the press/journalism cell,
as an OBSERVED-COVARIATE line (never a bank feature, never judged, never fit into the
closure curve).

The container here is the ISSUING ORGANISATION's stream of releases, and the order is
calendar time.  `press_release_date` is recoverable for 2,528 / 2,956 rows (85.5%) from
`datasets/press-releases/press_release_modeling_dataset.csv.gz`, keyed by
press_release_id.  Variables built:

  within_company_rank    0-based index of the release in its company's date-sorted
                         stream, computed WITHIN THE SAMPLED POPULATION
  within_company_pct     the same as a fraction of the company's stream length
  company_size           how many population rows the company contributes
  days_since_start       calendar position in the corpus timeline
  year, month, dow       calendar covariates (dow = ISO weekday, the embargo/timing
                         convention the brief names)
  id_rank_pct            rank of the numeric press_release_id -- a corpus-construction
                         ordinal, included precisely so it can be told apart from
                         genuine calendar time

DIRECTION IS STATED PER VARIABLE, not inferred.  The load-bearing one is
`company_size`: this cell's population was built as ALL releases with >=3 tracked
outlet pickups plus a topic-matched downsample of zero-pickup releases, so a company's
row count in the population is a direct function of how often it got picked up.
company_size is therefore LABEL-COUPLED BY CONSTRUCTION and is reported as a
corpus-construction artifact, not as an upstream factor.

CPU only.  Usage: python position_line.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

import cells as C
import closure_core as L

HERE = Path(__file__).resolve().parent

DIRECTION = {
    "within_company_rank": "upstream-plausible but PARTIAL -- the order is taken over the "
                           "SAMPLED population, not the organisation's true release stream, "
                           "so it is a noisy proxy for real release order",
    "within_company_pct": "same as within_company_rank, normalised",
    "company_size": "CORPUS-CONSTRUCTION ARTIFACT / LABEL-COUPLED -- the population keeps ALL "
                    "of a company's >=3-outlet releases and only a topic-matched sample of its "
                    "zero-pickup ones, so a company's row count is a function of its pickup rate",
    "days_since_start": "upstream-plausible (era of the media environment and of the "
                        "corpus's own outlet tracking)",
    "year": "upstream-plausible (same)",
    "month": "upstream-plausible (news-cycle seasonality)",
    "dow": "upstream-plausible (embargo and day-of-week release conventions)",
    "id_rank_pct": "CORPUS-CONSTRUCTION ORDINAL -- position in the scraped id space, included "
                   "as a control so genuine calendar time can be told apart from it",
}


def build_covariates(d):
    dates = pd.to_datetime(pd.Series([s or None for s in d["dates"]]), utc=True, errors="coerce")
    comp = pd.Series(d["groups"])
    idnum = pd.to_numeric(pd.Series(d["ids"]), errors="coerce")
    df = pd.DataFrame({"comp": comp, "date": dates, "idnum": idnum})
    df["order_key"] = df["date"].astype("int64").where(df["date"].notna(), np.nan)
    df["rank"] = df.groupby("comp")["order_key"].rank(method="first")
    df["size"] = df.groupby("comp")["comp"].transform("size")
    df["pct"] = (df["rank"] - 1) / (df.groupby("comp")["order_key"].transform("count") - 1).replace(0, np.nan)
    t0 = df["date"].min()
    pos = {
        "within_company_rank": (df["rank"] - 1).to_numpy(float),
        "within_company_pct": df["pct"].to_numpy(float),
        "company_size": df["size"].to_numpy(float),
        "days_since_start": ((df["date"] - t0).dt.total_seconds() / 86400).to_numpy(float),
        "year": df["date"].dt.year.to_numpy(float),
        "month": df["date"].dt.month.to_numpy(float),
        "dow": df["date"].dt.dayofweek.to_numpy(float),
        "id_rank_pct": df["idnum"].rank(pct=True).to_numpy(float),
    }
    return pos, float(df["date"].notna().mean())


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


def auc_ok(y, p):
    m = np.isfinite(p)
    if m.sum() < 20 or len(set(y[m].tolist())) < 2:
        return None, int(m.sum())
    return L.auc(y[m], p[m]), int(m.sum())


def main():
    C.sklearn_guard()
    d = C.load()
    y, g = d["y"], d["groups"]
    held = np.isin(d["dense_split"], ["eval", "test"])
    z = np.load(HERE / "press_verdict_r0_preds.npz", allow_pickle=True)
    va = z["va_nl"]

    pos, date_cov = build_covariates(d)
    names = list(pos)
    X = np.column_stack([pos[n] for n in names])

    out = {"cell": "press_verdict", "n": int(len(y)), "n_HONEST": int(held.sum()),
           "date_coverage": date_cov,
           "note": "OBSERVED COVARIATE AUDIT ONLY -- no position variable is ever added to V "
                   "or A, judged by any LLM, or fit into anything feeding the closure curve.",
           "variables": {}}
    for k, n in enumerate(names):
        col = X[:, k]
        a_full, n_full = auc_ok(y, col)
        a_hon, n_hon = auc_ok(y[held], col[held])
        m = np.isfinite(col) & held
        out["variables"][n] = {
            "coverage": float(np.isfinite(col).mean()),
            "alone_AUC_full": a_full, "n_full": n_full,
            "alone_AUC_HONEST": a_hon, "n_HONEST": n_hon,
            "spearman_with_T_HONEST": float(spearmanr(col[m], d["dense"][m]).statistic)
            if m.sum() > 20 else None,
            "spearman_with_VA_nl_HONEST": float(spearmanr(col[m], va[m]).statistic)
            if m.sum() > 20 else None,
            "direction": DIRECTION[n]}

    jp = joint_oof(X, y, g)
    out["joint_position_model"] = {
        "variables": names,
        "grouped_OOF_AUC_full": L.auc(y, jp), "AUC_HONEST": L.auc(y[held], jp[held]),
        "spearman_with_T_HONEST": float(spearmanr(jp[held], d["dense"][held]).statistic),
        "spearman_with_VA_nl_HONEST": float(spearmanr(jp[held], va[held]).statistic)}

    clean = [k for k, n in enumerate(names)
             if not DIRECTION[n].startswith(("CORPUS-CONSTRUCTION", "LABEL"))]
    jp_clean = joint_oof(X[:, clean], y, g)
    out["joint_position_model_ARTIFACT_FREE"] = {
        "variables": [names[k] for k in clean],
        "grouped_OOF_AUC_full": L.auc(y, jp_clean), "AUC_HONEST": L.auc(y[held], jp_clean[held])}

    cal = [k for k, n in enumerate(names) if n in ("days_since_start", "year", "month", "dow")]
    ordv = [k for k, n in enumerate(names) if n in ("within_company_rank", "within_company_pct")]
    out["calendar_only"] = {"variables": [names[k] for k in cal],
                            "grouped_OOF_AUC_full": L.auc(y, joint_oof(X[:, cal], y, g))}
    out["within_company_order_only"] = {
        "variables": [names[k] for k in ordv],
        "grouped_OOF_AUC_full": L.auc(y, joint_oof(X[:, ordv], y, g))}

    sz = pos["company_size"]
    q = L.decile_strata(sz, q=5)
    out["label_rate_by_company_size_quintile"] = [
        {"quintile": int(v), "n": int((q == v).sum()),
         "mean_company_size": float(sz[q == v].mean()),
         "label_rate": float(y[q == v].mean())} for v in sorted(set(q.tolist()))]

    yh, dh, vh = y[held], d["dense"][held], va[held]
    for label, score in (("joint", jp[held]), ("artifact_free", jp_clean[held])):
        st = L.decile_strata(score, q=10)
        tA, iA = L.stratified_auc(yh, dh, st, min_n=20)
        vA, _ = L.stratified_auc(yh, vh, st, min_n=20)
        out[f"Delta_stratified_on_position_{label}"] = {
            "pooled_T": L.auc(yh, dh), "pooled_VA": L.auc(yh, vh),
            "pooled_Delta": L.auc(yh, dh) - L.auc(yh, vh),
            "T_adj": tA, "VA_adj": vA, "Delta_adj": tA - vA, **iA}
    st = L.decile_strata(np.nan_to_num(pos["year"][held], nan=-1), q=10)
    tA, iA = L.stratified_auc(yh, dh, st, min_n=20)
    vA, _ = L.stratified_auc(yh, vh, st, min_n=20)
    out["Delta_stratified_on_year"] = {"T_adj": tA, "VA_adj": vA, "Delta_adj": tA - vA, **iA}

    np.savez(HERE / "press_verdict_position.npz", X=X, joint=jp, joint_clean=jp_clean,
             names=np.array(names, dtype=object))
    (HERE / "press_verdict_position_line.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
