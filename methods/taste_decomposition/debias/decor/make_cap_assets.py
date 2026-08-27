#!/usr/bin/env python3
"""cap_crowd bonus-arm assets for DECORRELATED TRAINING.

Builds, from the closure maps_batch1 artifacts (read-only):
  1. corpus_cap.csv    -- id/docket(group=contest)/judgement/split(=dense_split)/text
  2. cap_B.npz         -- the cell's MINED NUISANCE SET: all B-routed,
                          non-collapsed criterion score columns from rounds 1+2
                          (r1: 10 B minus 1 collapsed; r2: 16 B; no retirements
                          recorded for this cell), keys doc_id / S / names
  3. archived same-rows dense preds copied alongside for the readout.

The joint-B nuisance score used for the weights is then P_hat(y|B) from
fit_weights.py's grouped-CV logistic on this matrix (train rows only) -- i.e.
exactly the "mined joint-B score" of the production recipe.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MAPS = HERE.parents[1] / "closure" / "maps_batch1"
OUT = HERE.parent / "build" / "cap"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    pop = pd.read_csv(MAPS / "cap_crowd_population.csv")
    corpus = pd.DataFrame({
        "doc_id": pop["id"].astype(str),
        "docket": pop["group"].astype(str),          # contest; column named docket for tool reuse
        "judgement": pop["judgement"].astype(int),
        "split": pop["dense_split"].astype(str),     # the cell's canonical dense split (contest-disjoint, verified)
        "text": pop["text"].astype(str),
    })
    assert set(corpus["split"]) == {"train", "eval", "test"}
    spans = corpus.groupby("docket")["split"].nunique()
    assert (spans == 1).all(), "dense_split is not contest-disjoint"
    corpus.to_csv(OUT / "corpus_cap.csv", index=False)

    cols, names = [], []
    for r in (1, 2):
        z = np.load(MAPS / f"cap_crowd_r{r}_scores.npz", allow_pickle=True)
        assert (np.array([str(s) for s in z["row_id"]]) == corpus["doc_id"].values).all(), \
            f"r{r} score rows misaligned"
        cids = [str(s) for s in z["crit_ids"]]
        cnames = [str(s) for s in z["crit_names"]]
        routing = json.loads((MAPS / f"cap_crowd_r{r}_routing_final.json").read_text())
        rep = json.loads((MAPS / f"cap_crowd_r{r}_score_report.json").read_text())
        b_ids = [x["blind_id"] for x in routing["final"] if x["final_route"] == "B"]
        for k, cid in enumerate(cids):
            if cid not in b_ids or rep["per_criterion"][cid]["collapsed"]:
                continue
            cols.append(np.asarray(z["X"][:, k], dtype=float))
            names.append(f"r{r}:{cid} {cnames[k]}")
    S = np.column_stack(cols)

    # degeneracy screen + NaN impute, TRAIN statistics only
    tr = (corpus["split"] == "train").values
    med = np.nanmedian(S[tr], axis=0)
    S = np.where(np.isnan(S), med[None, :], S)
    keep = S[tr].std(0) > 1e-9
    dropped = [n for n, k in zip(names, keep) if not k]
    S, names = S[:, keep], [n for n, k in zip(names, keep) if k]

    np.savez_compressed(OUT / "cap_B.npz", doc_id=corpus["doc_id"].values,
                        S=S, names=np.array(names, dtype=object))
    report = {"n_rows": int(len(corpus)), "n_B_columns": int(S.shape[1]),
              "dropped_degenerate": dropped, "names": names,
              "split_counts": corpus["split"].value_counts().to_dict(),
              "pos_rate": float(corpus["judgement"].mean())}
    (OUT / "cap_assets_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
