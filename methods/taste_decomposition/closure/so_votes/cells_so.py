#!/usr/bin/env python3
"""SO answer-votes closure campaign: cell loader + estimators.

Contract mirrors `closure/code_v3/cells_code.py` (the current-standards template)
so this campaign's numbers are comparable to code's. Everything estimator-side is
imported from the frozen Layer-1 modules; nothing is reimplemented.

WHAT THIS CELL IS. Software-code field, VOTE column. y = 1 iff an answer's raw
net vote Score is strictly above the median answer Score on its own question
(ties dropped). n = 12,202 over 5,972 questions. Build note:
notes/2026-08-08__v6_stackoverflow_build.md. Independent audit (corrections
folded in below): notes/2026-08-11__so_votes_audit.md.

T CONVENTION — the audit's ruling, and why (both conventions carried):

  The headline dense arm (`title` view) never saw the question body, while the A
  judge did. The build's own item-view control (`abank` view) was
  DISPLACEMENT-BLINDED: on 6.9% of rows prepending the question pushed the
  ANSWER out of the 1024-token window, and on those rows the control collapses
  to chance (.48-.52) while the bank reads .75. Averaging over the whole leg hid
  a real +.014-.016 context gain.

  `qtrunc` is the honest convention: question-inclusive AND answer-preserving
  (the question body is token-truncated so the answer is never cut -- the
  judge's own convention). It is therefore the PRIMARY T reference here, with
  `title` (3 seeds) carried as the labelled secondary. Both are reported in
  every readout; neither is ever quoted without its label.

MONITOR ⊂ DENSE-HELD-OUT (prereg AMENDMENT 1, binding). The dense arm holds out
eval+test = 2,440 rows; MONITOR and the mining slice M are both carved from
those, so every dense score read by the campaign is honest and none is an
in-sample training prediction.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "methods/taste_decomposition"))
import layer1_gemma_cells as L  # noqa: E402
import scaleupC_layer1 as SC  # noqa: E402

BANK_OUT = REPO / "outputs/va_gemma_banks_so_votes"
SO = REPO / "datasets/stackoverflow-votes/va"
RESULTS = REPO / "methods/taste_decomposition/results"
DENSE = {"title": SO / "dense_standard_so_votes",
         "qtrunc": SO / "dense_standard_so_votes_qtrunc",
         "abank": SO / "dense_standard_so_votes_abankview"}
T_PRIMARY = "qtrunc"
COLLAPSE_MODAL_MAX = 0.98
MONITOR_FRAC = 0.20
SALT = "so-votes-closure-v1|"

CELL_META = {
    "so_votes": {
        "item": "StackOverflow answer",
        "corpus": "answers to [python] questions on StackOverflow, 2016-2023",
        "construct": ("whether the crowd voted this answer above the median "
                      "answer on its own question"),
        "group": "question_id",
    }
}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def modal_share(col):
    v = np.asarray(col, dtype=float).copy()
    fin = np.isfinite(v)
    v[~fin] = float(np.nanmedian(v)) if fin.any() else 0.5
    _, c = np.unique(v, return_counts=True)
    return float(c.max() / len(v))


def _load_preds(view: str, seed: int):
    """Positional join of a seed's preds onto its split file's row_ids, GATED on
    row-by-row (judgement, group) equality. The preds CSVs carry no ids, so the
    gate is what makes the join trustworthy (audit §5 discipline)."""
    d = DENSE[view]
    out = {}
    gates = []
    for leg in ("eval", "test"):
        p = d / f"rm_out_seed{seed}" / f"preds_{leg}.csv"
        s = d / "split" / f"{leg}.csv"
        if not p.exists():
            return None, [{"view": view, "seed": seed, "leg": leg, "status": "MISSING"}]
        pr = pd.read_csv(p)
        sp = pd.read_csv(s)
        ok = (len(pr) == len(sp)
              and bool((pr.judgement.values == sp.judgement.values).all())
              and bool((pr.group.astype(str).values == sp.group.astype(str).values).all()))
        gates.append({"view": view, "seed": seed, "leg": leg, "n": int(len(pr)),
                      "gate_pass": bool(ok)})
        if not ok:
            raise AssertionError(f"alignment gate FAILED {view} seed{seed} {leg}")
        for rid, prob in zip(sp.row_id.astype(str), pr.prob.astype(float)):
            out[rid] = prob
    return out, gates


def load(collapse_gate=True):
    meta, A, V, groups, shard, ids = SC.load_scaleupC_bank("so_votes", out=BANK_OUT)
    y = np.array(meta["ys"]["vote_score"], dtype=float)
    keep = np.isfinite(y)
    A, V, y = A[keep], V[keep], y[keep].astype(int)
    groups, ids, shard = groups[keep], ids[keep], shard[keep]
    a_names = list(meta["a_names"])
    v_names = list(meta["v_names"])

    gate = {"modal_max": COLLAPSE_MODAL_MAX, "dropped": []}
    if collapse_gate:
        sh = np.array([modal_share(A[:, c]) for c in range(A.shape[1])])
        k = sh <= COLLAPSE_MODAL_MAX
        gate["dropped"] = [{"criterion": n, "modal_share": round(float(s), 4)}
                           for n, s in zip(a_names, sh) if s > COLLAPSE_MODAL_MAX]
        A = A[:, k]
        a_names = [n for n, kk in zip(a_names, k) if kk]
    gate["kept_n"] = int(A.shape[1])

    rub = [json.loads(l) for l in open(SO / "rubrics.jsonl") if l.strip()]
    track = {r["name"]: r["track"] for r in rub}

    pop = pd.read_csv(SO / "population.csv.gz")
    pop["row_id"] = pop.row_id.astype(str)
    P = pop.set_index("row_id")
    order = [str(i) for i in ids]
    position = P.loc[order, "position"].values.astype(float)
    n_ans = P.loc[order, "n_answers_q"].values.astype(float)
    y_acc = P.loc[order, "y_accepted"].values.astype(int)
    dense_split = P.loc[order, "split"].values.astype(object)
    char_len = P.loc[order, "body"].astype(str).str.len().values.astype(float)
    q_view = P.loc[order, "q_viewcount"].values.astype(float)

    dense, gates = {}, []
    for view in ("title", "qtrunc", "abank"):
        for seed in (42, 1, 2):
            pr, g = _load_preds(view, seed)
            gates += g
            if pr is None:
                continue
            dense[f"{view}_s{seed}"] = np.array(
                [pr.get(r, np.nan) for r in order], dtype=float)
    for view in ("title", "qtrunc", "abank"):
        cols = [v for k, v in dense.items() if k.startswith(view + "_s")]
        if cols:
            dense[f"{view}_mean"] = np.nanmean(np.vstack(cols), axis=0)

    return dict(A=A, V=V, y=y, groups=groups, ids=ids, shard_of=shard,
                a_names=a_names, v_names=v_names, track=track, meta=meta,
                position=position, n_answers_q=n_ans, y_accepted=y_acc,
                dense_split=dense_split, char_len=char_len, q_viewcount=q_view,
                dense=dense, alignment_gates=gates, collapse_gate=gate)


def build_splits(d, monitor_frac=MONITOR_FRAC):
    """FIT+MINE / MONITOR with MONITOR ⊂ dense-held-out (prereg AMENDMENT 1).

    The dense arm's held-out rows (eval+test) are the only rows where its scores
    are honest. Both MONITOR and the mining pool are carved from those by a
    stable hash of the QUESTION id, so no question straddles the boundary."""
    held = np.isin(d["dense_split"], ["eval", "test"])
    g = np.array([str(x) for x in d["groups"]])
    uq = sorted(set(g[held]))
    mon_q = {q for q in uq if int(sha(SALT + q)[:8], 16) / 0xFFFFFFFF < monitor_frac}
    split = np.array(["fit_mine"] * len(g), dtype=object)
    split[held & np.isin(g, list(mon_q))] = "monitor"
    rep = {"n_total": int(len(g)),
           "n_dense_heldout": int(held.sum()),
           "n_monitor": int((split == "monitor").sum()),
           "n_monitor_questions": len(mon_q),
           "n_mining_pool": int((held & (split == "fit_mine")).sum()),
           "n_fit_mine_total": int((split == "fit_mine").sum()),
           "monitor_pos_rate": float(d["y"][split == "monitor"].mean()),
           "mining_pool_pos_rate": float(d["y"][held & (split == "fit_mine")].mean()),
           "rule": "MONITOR = dense-held-out questions with "
                   f"sha256('{SALT}'+question_id) < {monitor_frac}; "
                   "mining pool = the remaining dense-held-out rows"}
    return split, rep


def within_group_auc(y, groups, pred):
    from sklearn.metrics import roc_auc_score
    tot = w = 0.0
    n = 0
    for q in np.unique(groups):
        m = groups == q
        yy = y[m]
        if yy.min() == yy.max():
            continue
        np_ = int(yy.sum() * (len(yy) - yy.sum()))
        tot += np_ * roc_auc_score(yy, pred[m])
        w += np_
        n += 1
    return (float(tot / w), n) if w else (float("nan"), 0)


def fit_va(d, cols_A, fit_mask, seeds=(0, 1, 2)):
    """Grouped-OOF VA stack fitted INSIDE fit_mine, applied everywhere.
    Returns (oof_within_fit, pred_all, per-seed AUC list)."""
    from sklearn.metrics import roc_auc_score
    X = np.column_stack([d["V"], d["A"][:, cols_A]]) if len(cols_A) else d["V"]
    y, g = d["y"], d["groups"]
    idx = np.flatnonzero(fit_mask)
    folds = L.outer_folds(len(idx), g[idx], n_splits=5)
    lin_auc, lin_oof = L.linear_oof_family1(X[idx], y[idx], g[idx], folds)
    nl = [L.gbm_oof_family1(X[idx], y[idx], g[idx], folds, s) for s in seeds]
    return {"lin_auc": lin_auc, "lin_oof": lin_oof,
            "nl_auc_mean": float(np.mean([r["auc"] for r in nl])),
            "nl_aucs": [r["auc"] for r in nl],
            "nl_oof": np.mean([r["oof"] for r in nl], axis=0),
            "fit_idx": idx}


if __name__ == "__main__":
    d = load()
    print(f"n={len(d['y'])} pos={d['y'].mean():.4f} "
          f"questions={len(set(map(str, d['groups'])))} "
          f"A={d['A'].shape[1]}c V={d['V'].shape[1]}c")
    print("collapse gate dropped:", d["collapse_gate"]["dropped"])
    print("dense columns:", sorted(d["dense"]))
    bad = [g for g in d["alignment_gates"] if not g.get("gate_pass", True)]
    print(f"alignment gates: {len(d['alignment_gates'])} checked, {len(bad)} failed")
    sp, rep = build_splits(d)
    print(json.dumps(rep, indent=1))
