#!/usr/bin/env python3
"""Decorrelation weights for DECORRELATED TRAINING (debias instrument #3).

Importance-reweighting of the dense training distribution so that y is
INDEPENDENT of a named nuisance score s under the reweighted measure.  No text
edits, no row deletion, no adversarial dynamics -- weights only.

Spec (notes/2026-08-05__taste-decomposition-design.md S12, binding):
    w_i  proportional to  1 / P_hat(y_i | s_i)
fit on TRAIN rows only, logistic, group(docket)-grouped CV, clipped at the 99th
percentile; report clip rate and n_eff = (sum w)^2 / sum w^2.

DECLARED IMPLEMENTATION CHOICE (stabilized weights).  We use
    w_i = P_hat(y_i) / P_hat(y_i | s_i)
i.e. the spec's 1/P_hat(y|s) multiplied by a PER-CLASS constant (stabilized
inverse-propensity weights).  Under either form the reweighted joint factorizes
P~(y,s) = P(y)P(s), which is exactly the "y independent of s" target; the
stabilized form additionally PRESERVES THE y MARGINAL (raw 1/P_hat(y|s) forces
the weighted positive rate to .50, which on a .78-positive cell would confound
the V2' gate (c) task-AUC comparison with a class-rebalancing intervention the
vanilla arm never received).  Both n_eff values are reported.

The fitted logistic P_hat(y|s) uses OOF (out-of-fold) predictions under
GroupKFold on the grouping unit, TRAIN rows only; eval/test rows get w = 1 and
never touch the fit (they are only ever scored unweighted).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

P_FLOOR = 1e-3          # numerical floor on P_hat(y_obs|s) before clipping
CLIP_PCT = 99.0         # spec: clip weights at the 99th percentile
N_FOLDS = 5


def _lin():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def assemble_s(nz, channels, pos):
    """Nuisance matrix for the named channels, rows aligned to the corpus.
    Mirrors train_grl.py's channel assembly exactly."""
    groups = json.loads(str(nz["groups_json"]))
    names = [str(s) for s in nz["names"]]
    extra = {"plant": nz["plant"].astype(float), "realtok": nz["realtok"].astype(float)}
    blocks, col_names = [], []
    for ch in channels:
        if ch in groups:
            blocks.append(nz["Z"][np.ix_(pos, groups[ch])])
            col_names += [names[j] for j in groups[ch]]
        elif ch in extra:
            blocks.append(extra[ch][pos][:, None])
            col_names.append(ch)
        else:
            raise ValueError(f"unknown channel {ch}")
    return np.hstack(blocks), col_names


def weighted_auc(y, s, w):
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, s, sample_weight=w))


def fit_oof_prob(S, y, groups):
    """OOF P_hat(y=1|s) via grouped-CV logistic on TRAIN rows."""
    n_grp = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(N_FOLDS, n_grp))
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(S, y, groups):
        m = _lin()
        m.fit(S[tr], y[tr])
        oof[te] = m.predict_proba(S[te])[:, 1]
    assert not np.isnan(oof).any()
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--nuisance", required=True)
    ap.add_argument("--channels", required=True, help="comma list: plant|realtok|length|format|date|topic|standard")
    ap.add_argument("--out", required=True, help="output weights .npz")
    ap.add_argument("--subset_ids_from", default=None, help="restrict to this population (V3b)")
    ap.add_argument("--s_npz", default=None,
                    help="OPTIONAL: explicit nuisance-score matrix npz (keys: doc_id, S) "
                         "used INSTEAD of --nuisance channels (cap_crowd joint-B arm)")
    ap.add_argument("--id_col", default="doc_id")
    ap.add_argument("--group_col", default="docket")
    ap.add_argument("--split_col", default="split")
    args = ap.parse_args()

    df = pd.read_csv(args.corpus)
    if args.subset_ids_from:
        keep = set(pd.read_csv(args.subset_ids_from)[args.id_col].astype(str))
        df = df[df[args.id_col].astype(str).isin(keep)].reset_index(drop=True)
    ids = df[args.id_col].astype(str).values
    y = df["judgement"].astype(int).values
    grp = df[args.group_col].astype(str).values
    split = df[args.split_col].astype(str).values
    tr = split == "train"

    channels = [c for c in args.channels.split(",") if c]
    if args.s_npz:
        z = np.load(args.s_npz, allow_pickle=True)
        order = {str(d): i for i, d in enumerate(z["doc_id"])}
        pos = np.array([order[d] for d in ids])
        S = np.asarray(z["S"], dtype=float)
        if S.ndim == 1:
            S = S[:, None]
        S = S[pos]
        col_names = [str(s) for s in z["names"]] if "names" in z else [f"s{i}" for i in range(S.shape[1])]
    else:
        nz = np.load(args.nuisance, allow_pickle=True)
        order = {str(d): i for i, d in enumerate(nz["doc_id"])}
        pos = np.array([order[d] for d in ids])
        S, col_names = assemble_s(nz, channels, pos)

    # ---- fit P_hat(y|s) on TRAIN only, grouped OOF ---------------------------
    p1 = fit_oof_prob(S[tr], y[tr], grp[tr])
    p_obs = np.where(y[tr] == 1, p1, 1.0 - p1)
    n_floored = int((p_obs < P_FLOOR).sum())
    p_obs = np.clip(p_obs, P_FLOOR, None)

    marg = y[tr].mean()
    p_marg = np.where(y[tr] == 1, marg, 1.0 - marg)

    w_stab_raw = p_marg / p_obs                    # stabilized (used for training)
    w_unstab_raw = 1.0 / p_obs                     # spec-literal (reported)

    def finalize(w_raw):
        cut = np.percentile(w_raw, CLIP_PCT)
        w = np.minimum(w_raw, cut)
        clip_rate = float((w_raw > cut).mean())
        w = w / w.mean()
        n_eff = float(w.sum() ** 2 / (w ** 2).sum())
        return w, clip_rate, n_eff, float(cut)

    w, clip_rate, n_eff, cut = finalize(w_stab_raw)
    w_u, clip_rate_u, n_eff_u, cut_u = finalize(w_unstab_raw)

    # ---- independence diagnostics (TRAIN, weighted vs unweighted) -----------
    def col_corr(wv):
        out = {}
        yc = y[tr] - np.average(y[tr], weights=wv)
        for k, nm in enumerate(col_names[: min(len(col_names), 30)]):
            sc = S[tr, k] - np.average(S[tr, k], weights=wv)
            denom = np.sqrt(np.average(yc ** 2, weights=wv) * np.average(sc ** 2, weights=wv))
            out[nm] = float(np.average(yc * sc, weights=wv) / denom) if denom > 1e-12 else None
        return out

    diag = {
        "auc_shat_vs_y_train_unweighted": weighted_auc(y[tr], p1, np.ones(tr.sum())),
        "auc_shat_vs_y_train_weighted": weighted_auc(y[tr], p1, w),
        "corr_y_s_unweighted": col_corr(np.ones(tr.sum())),
        "corr_y_s_weighted": col_corr(w),
        "weighted_pos_rate_train_stabilized": float(np.average(y[tr], weights=w)),
        "weighted_pos_rate_train_unstabilized": float(np.average(y[tr], weights=w_u)),
        "unweighted_pos_rate_train": float(marg),
    }
    if "plant" in channels or "realtok" in channels:
        b = S[tr, col_names.index("plant" if "plant" in channels else "realtok")]
        for lab, wv in (("unweighted", np.ones(tr.sum())), ("weighted", w)):
            diag[f"P(tok|y=1)_{lab}"] = float(np.average(b[y[tr] == 1], weights=wv[y[tr] == 1]))
            diag[f"P(tok|y=0)_{lab}"] = float(np.average(b[y[tr] == 0], weights=wv[y[tr] == 0]))

    # ---- write ---------------------------------------------------------------
    w_all = np.ones(len(df))
    w_all[tr] = w
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, doc_id=ids, w=w_all.astype(np.float32),
                        is_train=tr.astype(bool))
    rep = {
        "corpus": args.corpus, "channels": channels, "s_npz": args.s_npz,
        "n": int(len(df)), "n_train": int(tr.sum()),
        "estimator": "GroupKFold(<=5) logistic (StandardScaler+LogisticRegression C=1) OOF, train only",
        "weight_form": "STABILIZED  w = P_hat(y)/P_hat(y|s)  (declared; see module docstring)",
        "p_floor": P_FLOOR, "n_floored": n_floored,
        "clip_percentile": CLIP_PCT,
        "stabilized": {"clip_rate": clip_rate, "clip_value": cut, "n_eff": n_eff,
                       "n_eff_frac": n_eff / tr.sum(),
                       "w_min": float(w.min()), "w_max": float(w.max())},
        "spec_literal_unstabilized": {"clip_rate": clip_rate_u, "clip_value": cut_u,
                                      "n_eff": n_eff_u, "n_eff_frac": n_eff_u / tr.sum()},
        "diagnostics": diag,
    }
    out.with_suffix(".report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
