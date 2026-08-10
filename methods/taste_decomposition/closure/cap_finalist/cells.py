#!/usr/bin/env python3
"""Cell adapter for the Layer-3 articulation-closure campaign on CAP_FINALIST
(New Yorker cartoon caption contest; y = EDITOR finalist selection, hard-negative pool).

Same contract as jokes_community/cells.py and maps_batch1/cells.py: return the cell's
Layer-1 A/V population in a REPRODUCIBLE row order together with ids / groups / y /
raw A,V blocks / texts / dense.  The loader body is INHERITED verbatim from
maps_batch1/cells.py::_load_caption -- rounds 1 and 2 of this campaign ran on it and
continuity requires the identical row order -- with three additions, each recorded:

  1. `descs`   the CARTOON DESCRIPTION for each row (canny + uncanny, from
               datasets/humor/newyorker_cartoon_descriptions.csv.gz, 225/227 contests).
               See ITEM-VIEW DEFECT below: this is the fix for it.
  2. `crowd`   observed non-text covariates (crowd_mean, crowd_votes, within-contest
               crowd percentile rank) -- NEVER features in any bank, used only by
               contest_line.py as the observed-covariate line.
  3. `contest_no` the contest's position in the New Yorker's own contest series
               (530..895) -- this cell's FREEZE-ADDENDUM-4 position-in-container ordinal.

POPULATION.  5,218 rows = the finalist-B hard-negative pool (role in {finalist,
neg_hard}), 227 contests, pos-rate .1299.  Row order = sorted(hardneg_ids), verbatim
from layer1_gemma_cells._caption_pools(), which is what results/cap_finalist_layer1.json
was computed on.

GROUPING UNIT = CONTEST.  Each contest contributes exactly 3 finalists and ~20 hard
negatives, so the pos-rate is CONSTANT at .1304 in all but two of the 227 contests.
Two structural consequences, registered before any new round:
  * the contest ordinal cannot carry label information (measured: alone-AUC .5022), so
    the Addendum-4 position-in-container family is STRUCTURALLY NULL on this cell -- not
    "searched for and not found".  Recorded so no round wastes budget re-deriving it.
  * within-contest AUC (TIER 2 here) is the readout that matches the y-definition, since
    y is a within-container selection.

ITEM-VIEW DEFECT (found 2026-08-09, this campaign, and FIXED here).  The A bank was
scored by Gemma-4-31B on `CARTOON: <description>\\n\\nCAPTION: "<text>"`
(datasets/humor/caption_multiy/score_va_gemma_captions.py:190).  maps_batch1's
score_gemma_maps.py scored MINED criteria on `CAPTION:\\n<text>` with NO cartoon.  A New
Yorker caption is close to ungradeable without the drawing it captions, so every round-1
and round-2 mined criterion was measured on a strictly weaker view than the bank it was
being added to.  Round 3 of this campaign is a VIEW-REPAIR pass (TIER R) that re-scores
those criteria on the matched view; every round from 3 on uses the matched view.

DENSE (T).  closure/samerows_preds/cap_finalist_dense_preds_slim.csv, one probability
column per row plus its dense_split.  ONE T convention exists on this cell (the master
ledger's .6124 on the 1,055 dense-held-out rows reproduces from this column exactly), so
unlike jokes_community there is no meanAUC/ensemble pair to keep apart.

CPU only.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLOSURE = HERE.parent
TD = CLOSURE.parent
REPO = TD.parents[1]
sys.path.insert(0, str(TD))
sys.path.insert(0, str(CLOSURE))

CAP_POOL = REPO / "datasets" / "humor" / "caption_multiy" / "caption_contest_v2.jsonl"
DESC_CSV = REPO / "datasets" / "humor" / "newyorker_cartoon_descriptions.csv.gz"
SAMEROWS = CLOSURE / "samerows_preds"
RESULTS = TD / "results"

CELL_META = {
    "cap_finalist": dict(
        group_column="contest", item="caption",
        corpus="reader-submitted entries to a weekly cartoon caption contest, each one "
               "written for a specific published cartoon (the cartoon is described "
               "alongside every entry below)",
        construct="how good the caption is as a contest entry for THAT cartoon",
        text_trunc=1200,
        layer1="cap_finalist_layer1.json",
    ),
}


def _caption_meta():
    meta = {}
    for line in open(CAP_POOL):
        if not line.strip():
            continue
        r = json.loads(line)
        did = f"{r['contest']}_{hashlib.sha1(r['text'].encode()).hexdigest()[:12]}"
        meta[did] = r
    return meta


def load_descriptions():
    """canny + uncanny description per contest number.  225 of the 227 contests in the
    hard-negative pool are covered; the two uncovered contests get "" and are recorded
    (never imputed with another contest's cartoon)."""
    descs = {}
    with gzip.open(DESC_CSV, "rt") as fh:
        for row in csv.DictReader(fh):
            c = str(int(row["contest_number"]))
            canny = (row.get("canny") or "").strip()
            uncanny = (row.get("uncanny") or "").strip()
            descs[c] = f"{canny} {uncanny}".strip()
    return descs


def _load_cap_finalist():
    import layer1_gemma_cells as L1G

    c = L1G._caption_pools()
    ids = sorted(x for x in c["hardneg_ids"] if x in c["X_by_id"])
    y = np.array([c["y_fin"][d] for d in ids])
    A = np.array([c["X_by_id"][d] for d in ids], dtype=float)
    V = np.array([c["V_by_id"][d] for d in ids], dtype=float)
    groups = np.array([str(c["contest_by_id"][d]) for d in ids])
    meta = _caption_meta()
    texts = [meta[i]["text"] if i in meta else "" for i in ids]

    descs = load_descriptions()
    row_desc = [descs.get(g, "") for g in groups]

    def _f(key):
        out = []
        for i in ids:
            v = meta.get(i, {}).get(key)
            out.append(float(v) if v is not None else np.nan)
        return np.array(out, dtype=float)

    crowd_mean, crowd_votes = _f("crowd_mean"), _f("crowd_votes")
    cdf = pd.DataFrame({"g": groups, "cm": crowd_mean})
    crowd_pct = cdf.groupby("g")["cm"].rank(pct=True).values
    contest_no = np.array([float(g) for g in groups])

    d = pd.read_csv(SAMEROWS / "cap_finalist_dense_preds_slim.csv")
    pr = dict(zip(d["id"].astype(str), d["dense_prob"].astype(float)))
    ds = dict(zip(d["id"].astype(str), d["dense_split"].astype(str)))
    dense = np.array([pr.get(k, np.nan) for k in ids])
    dsplit = np.array([ds.get(k, "unmapped") for k in ids])

    return dict(tag="cap_finalist", ids=ids, y=y, groups=groups, A=A, V=V,
                a_names=[str(s) for s in c["a_names"]],
                v_names=[str(s) for s in c["v_names"]],
                texts=texts, descs=row_desc, years=None,
                crowd_mean=crowd_mean, crowd_votes=crowd_votes,
                crowd_pct_in_contest=crowd_pct, contest_no=contest_no,
                dense=dense, dense_split=dsplit)


LOADERS = {"cap_finalist": _load_cap_finalist}
CELLS = list(LOADERS)


def load(cell="cap_finalist"):
    d = LOADERS[cell]()
    d["meta"] = CELL_META[cell]
    led = json.loads((RESULTS / CELL_META[cell]["layer1"]).read_text())
    d["layer1"] = led
    assert len(d["y"]) == led["n"], f"{cell}: n={len(d['y'])} != layer1 n={led['n']}"

    # ------------------------------------------------- ALIGNMENT GATE (mandatory)
    # Version-independent: refit is NOT attempted here; instead the loaded A/V blocks
    # are asserted against the ledger's own published LINEAR V/A/VA AUCs, which are
    # deterministic given the row order and do not depend on GBM fold assignments.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import closure_core as L

    gate = {}
    for name, M in (("V", d["V"]), ("A", d["A"])):
        keep, med = L.clean_fit(M)
        X = L.clean_apply(M, keep, med)
        folds = list(GroupKFold(n_splits=5).split(np.zeros(len(d["y"])), groups=d["groups"]))
        oof = np.zeros(len(d["y"]))
        for tr, te in folds:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            clf.fit(X[tr], d["y"][tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        gate[name] = {"reproduced": float(roc_auc_score(d["y"], oof)),
                      "published": float(led["linear"][name])}
        gate[name]["abs_diff"] = abs(gate[name]["reproduced"] - gate[name]["published"])
        # The ledger's screen ran globally under sklearn 1.8.0; this campaign runs 1.7.2
        # AND enforces the collapse gate inside clean_fit, so an exact match is not
        # expected.  The gate is a ROW-ORDER assertion: a misaligned y would move these
        # by >> .02, a screen/version difference moves them by < .01.
        gate[name]["pass"] = bool(gate[name]["abs_diff"] < 0.02)
    d["alignment_gate"] = gate
    for name, g in gate.items():
        assert g["pass"], (f"ALIGNMENT GATE FAIL on {name}: reproduced {g['reproduced']:.4f} "
                           f"vs published {g['published']:.4f} (diff {g['abs_diff']:.4f})")
    return d


if __name__ == "__main__":
    d = load("cap_finalist")
    print(f"cap_finalist n={len(d['y'])} groups={len(set(d['groups']))} "
          f"pos={d['y'].mean():.4f} A={d['A'].shape[1]} V={d['V'].shape[1]} "
          f"heldout={int(np.isin(d['dense_split'], ['eval','test']).sum())}")
    print("alignment gate:", json.dumps(d["alignment_gate"], indent=1))
    nod = sum(1 for x in d["descs"] if not x)
    print(f"cartoon description: {len(d['descs']) - nod}/{len(d['descs'])} rows covered "
          f"({nod} rows in the 2 uncovered contests)")
    print(f"crowd_mean coverage {np.isfinite(d['crowd_mean']).mean():.4f}")


def sklearn_guard():
    """Record the sklearn release every number in this campaign was computed under.

    The Layer-1 ledger was produced under a different release; fold ASSIGNMENTS move
    across sklearn versions, so a LEVEL is only reproducible within one version. This
    campaign recomputes every quantity it quotes under the version reported here, and
    load()'s alignment gate (a row-order assertion against the ledger's published linear
    AUCs) is what guarantees the rows themselves have not moved."""
    import sklearn
    return {"sklearn": sklearn.__version__,
            "note": "all campaign numbers recomputed under this one release; the "
                    "alignment gate in load() is the version-independent row-order check"}
