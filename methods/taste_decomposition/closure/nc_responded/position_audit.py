#!/usr/bin/env python3
"""FREEZE ADDENDUM 4 (2026-08-07): POSITION-IN-CONTAINER audit for N&C RESPONDED.

The addendum's motivation: the patents audit found claim-ordinal alone-AUC .754 and a
code-recency channel, and BOTH were found by manual audit -- no proposer in the whole
program has ever named the position-in-container family unprompted. So the family gets
two treatments here:

  (a) THIS SCRIPT -- a direct programmatic audit of the real thing. N&C `doc_id` is
      `<AGENCY>-<YEAR>-<DOCKET#>-<SEQ>`, where SEQ is the comment's sequence number
      within its docket on regulations.gov. That is the container position, available
      exactly and for free. It is an AUDIT of the data, not a bank metric -- it is
      never added to V or A and never scored by a judge (the standing rule that all
      MEASUREMENT is by LLM judges governs metrics, not diagnostics of the corpus).

  (b) round 5's Track-B brief -- proposers are required to consider TEXTUAL
      fingerprints of position (deadline language, references to other comments or to
      the record so far, "supplemental"/"late-filed"/"timely" markers), which is what
      a judge could actually score.

Readouts here:
  * alone-AUC of raw sequence number, of within-docket rank, and of within-docket
    percentile rank (the scale-free one -- dockets differ hugely in size);
  * the same restricted to the honest dense-held-out population, so it is directly
    comparable to T and VA_nl;
  * whether the dense model or the bank already carries it: correlation of each
    position variable with T and with VA_nl, plus the stacked increment of position
    over the joint Track-B model;
  * a within-docket check: does position predict INSIDE a docket (it must, by
    construction of the rank) and does it survive as a between-docket effect too.

CPU only. Usage: python position_audit.py --state 4
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nc_closure_lib as L
from readout import load_dense

HERE = Path(__file__).resolve().parent
SEQ_RE = re.compile(r"-(\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=int, default=4)
    a = ap.parse_args()

    pop = L.load_population()
    summary, split, dsplit, mining, monitor_full = L.load_splits()
    y, docket, doc_id = pop["y"], pop["docket"], pop["doc_id"]
    dense = load_dense()
    heldout = np.isin(dsplit, ["eval", "test"])
    va = np.load(HERE / f"state{a.state}_preds.npz", allow_pickle=True)["nl_mean"]

    seq = np.array([int(m.group(1)) if (m := SEQ_RE.search(d)) else -1 for d in doc_id],
                   dtype=float)
    ok = seq >= 0
    df = pd.DataFrame({"seq": seq, "docket": docket})
    rank = df.groupby("docket")["seq"].rank(method="average").values
    pct = df.groupby("docket")["seq"].rank(method="average", pct=True).values
    dsize = df.groupby("docket")["seq"].transform("size").values

    variables = {
        "raw_sequence_number": seq,
        "within_docket_rank": rank,
        "within_docket_percentile": pct,
        "docket_size": dsize.astype(float),
    }

    out = {"cell": "nc_responded", "state": a.state,
           "n_parsed": int(ok.sum()), "n_total": int(len(seq)),
           "doc_id_example": str(doc_id[0]),
           "note": "AUDIT of the corpus, not a bank metric; never added to V/A, never judged.",
           "variables": {}}

    for name, v in variables.items():
        rec = {}
        for label, mask in (("full_population", ok),
                            ("honest_dense_heldout", ok & heldout)):
            m = mask
            try:
                auc = float(roc_auc_score(y[m], v[m]))
            except ValueError:
                auc = float("nan")
            rec[label] = {"n": int(m.sum()), "alone_auc": auc,
                          "alone_auc_abs_signal": abs(auc - 0.5)}
        h = ok & heldout
        rec["spearman_with_T"] = float(pd.Series(v[h]).corr(pd.Series(dense[h]), method="spearman"))
        rec["spearman_with_VA_nl"] = float(pd.Series(v[h]).corr(pd.Series(va[h]), method="spearman"))
        out["variables"][name] = rec

    # grouped-OOF joint model of all position variables (honest population)
    h = ok & heldout
    X = np.column_stack([variables[k][h] for k in variables])
    folds = list(GroupKFold(n_splits=5).split(np.zeros(h.sum()), groups=docket[h]))
    oof = np.zeros(h.sum())
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(X[tr], y[h][tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    out["joint_position_model"] = {
        "n": int(h.sum()),
        "alone_auc": float(roc_auc_score(y[h], oof)),
    }

    # does the existing Track-B map already carry it?
    tb = HERE / f"round{a.state}_track_b_discount.json"
    if tb.exists():
        d = json.loads(tb.read_text())
        out["comparison"] = {
            "joint_TrackB_alone_auc_histgb": d["variants"]["all"]["spurious_alone_histgb"],
            "n_TrackB_channels": d["n_channels"],
            "best_named_TrackB_channel": max(
                d["variants"]["all"]["per_channel_alone_auc"].items(),
                key=lambda kv: abs(kv[1] - 0.5)),
        }

    # stacked increment: does position add over the whole named Track-B map?
    st = HERE / f"state{a.state}_preds.npz"
    if tb.exists() and st.exists():
        import track_b_discount as TB
        XB, meta = TB.load_b_blocks(a.state)
        if XB is not None:
            keep, meds = L.clean_fit(XB[split == "fit_mine"])
            XBc = L.clean_apply(XB, keep, meds)
            bl = TB.oof_score(XBc[h], y[h], docket[h], TB._gb)
            s1 = float(roc_auc_score(y[h], bl))
            X2 = np.column_stack([bl, oof])
            oof2 = np.zeros(h.sum())
            for tr, te in folds:
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
                clf.fit(X2[tr], y[h][tr])
                oof2[te] = clf.predict_proba(X2[te])[:, 1]
            out["stacked_increment_position_over_named_TrackB"] = {
                "auc_named_B_only": s1,
                "auc_named_B_plus_position": float(roc_auc_score(y[h], oof2)),
                "increment": float(roc_auc_score(y[h], oof2)) - s1,
            }

    (HERE / "position_in_container_audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
