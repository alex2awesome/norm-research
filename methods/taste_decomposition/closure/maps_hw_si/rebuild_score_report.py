#!/usr/bin/env python3
"""Rebuild <tag>_score_report.json from <tag>_scores.npz.

Needed once: the round-d HashtagWars scoring process was killed between writing the
score matrix and writing its report (coordinator error, recorded in the note). The
npz carries the population matrix, the anchor matrix and the anchor tags, so the
report is fully recoverable -- byte-identical logic to score_gemma_maps.py.
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
for tag in sys.argv[1:]:
    z = np.load(HERE / f"{tag}_scores.npz", allow_pickle=True)
    Xpop, Xanc = z["X"], z["Xanchor"]
    cids = [str(s) for s in z["crit_ids"]]
    names = [str(s) for s in z["crit_names"]]
    tags = np.array([str(s) for s in z["anchor_tags"]])
    rep = {"tag": tag, "n_rows": int(Xpop.shape[0]), "n_criteria": len(cids),
           "per_criterion": {}, "rebuilt_from_npz": True}
    for k, cid in enumerate(cids):
        col = Xpop[:, k]
        ok = col[~np.isnan(col)]
        vals, counts = (np.unique(ok, return_counts=True) if len(ok)
                        else (np.array([]), np.array([])))
        rep["per_criterion"][cid] = {
            "name": names[k], "na_rate": float(np.isnan(col).mean()),
            "mean": float(np.mean(ok)) if len(ok) else None,
            "std": float(np.std(ok)) if len(ok) else None,
            "n_distinct": int(len(vals)),
            "modal_frac": float(counts.max() / len(ok)) if len(ok) else None,
            "value_counts": {str(v): int(c) for v, c in zip(vals, counts)},
            "collapsed": bool(len(ok) == 0 or len(vals) <= 1
                              or counts.max() / len(ok) > 0.98)}
    item = np.nanmean(Xanc, axis=1)
    anc = {"k_per_class": int((tags == "anchor_pos").sum())}
    for t in ("anchor_pos", "anchor_neg", "anchor_scram"):
        v = item[tags == t]
        anc[t] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1))}
    pv, nv, sv = (item[tags == "anchor_pos"], item[tags == "anchor_neg"],
                  item[tags == "anchor_scram"])
    anc["pos_vs_neg_auc"] = float(roc_auc_score([1]*len(pv)+[0]*len(nv), np.concatenate([pv, nv])))
    anc["coherent_vs_scrambled_auc"] = float(roc_auc_score(
        [1]*(len(pv)+len(nv))+[0]*len(sv), np.concatenate([pv, nv, sv])))
    anc["pass_scrambled"] = bool(anc["coherent_vs_scrambled_auc"] >= 0.70)
    rep["anchors"] = anc
    rep["n_collapsed"] = int(sum(v["collapsed"] for v in rep["per_criterion"].values()))
    rep["overall_na_rate"] = float(np.isnan(Xpop).mean())
    (HERE / f"{tag}_score_report.json").write_text(json.dumps(rep, indent=2))
    print(tag, "anchors", json.dumps(anc), "collapsed", rep["n_collapsed"],
          "NA", round(rep["overall_na_rate"], 4))
