#!/usr/bin/env python3
"""Cell adapter for the Layer-3 articulation-closure campaign on CAP_CROWD
(New Yorker cartoon caption contest; y = CROWD vote, above/below the contest median).

SISTER CELL.  cap_crowd is the same corpus with the EDITOR label; the two terminal maps
are meant to be read together as the humour curation-vs-community contrast, so this loader
is the cap_crowd loader with three lines changed (crowd_ids / y_crowd / the layer-1
file) and one thing ADDED (the second dense arm, below).

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
from layer1_gemma_cells._caption_pools(), which is what results/cap_crowd_layer1.json
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

DENSE (T) -- TWO ARMS, AND THE RE-BASE CAVEAT (registry 2026-08-08).  This cell is the
one place in the programme where the dense standard is known to be trainer-dependent:

  T_archived        closure/samerows_preds/cap_crowd_dense_preds_slim.csv
                    AUC .5554 on the 2,190 dense-held-out rows. This is the number the
                    master ledger carries and the number rounds 1-2 were read against.
  T_matched_vanilla methods/taste_decomposition/debias/runs/D20_cap_vanilla/preds_slim.csv
                    AUC .6047 on the SAME 2,190 rows and the SAME train/eval/test split
                    -- a fresh vanilla dense model trained by the debias pilot's trainer
                    (lambda_adv = 0, i.e. no debiasing), built precisely to check whether
                    the archived arm was undertrained. It was: the bank's lead over the
                    dense standard collapses from -.110 to -.066 when the dense arm is
                    re-based.

CAMPAIGN T CONVENTION, declared here and never mixed: **T_matched_vanilla is PRIMARY.**
The registry instruction is "RE-BASE before quoting", the archived arm is a different
pipeline from every other cell's, and a Delta_beyond computed against an undertrained
dense model overstates the bank. T_archived is reported BESIDE it in every table and in
the terminal ledger, never dropped and never quoted alone. `d["dense"]` is the
matched-vanilla column (it is the score VECTOR the swap algebra, the stacks and the strata
use); `d["dense_archived"]` is the archived column, carried for the paired readout.

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
    "cap_crowd": dict(
        group_column="contest", item="caption",
        corpus="reader-submitted entries to a weekly cartoon caption contest, each one "
               "written for a specific published cartoon (the cartoon is described "
               "alongside every entry below)",
        construct="how good the caption is as a contest entry for THAT cartoon",
        text_trunc=1200,
        layer1="cap_crowd_layer1.json",
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


def _load_cap_crowd():
    import layer1_gemma_cells as L1G

    c = L1G._caption_pools()
    ids = sorted(x for x in c["crowd_ids"] if x in c["X_by_id"])
    y = np.array([c["y_crowd"][d] for d in ids])
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

    d = pd.read_csv(SAMEROWS / "cap_crowd_dense_preds_slim.csv")
    pr = dict(zip(d["id"].astype(str), d["dense_prob"].astype(float)))
    ds = dict(zip(d["id"].astype(str), d["dense_split"].astype(str)))
    dense_arch = np.array([pr.get(k, np.nan) for k in ids])
    dsplit = np.array([ds.get(k, "unmapped") for k in ids])

    # matched fresh vanilla arm (the RE-BASE; see the T block in the docstring)
    vp = (TD / "debias" / "runs" / "D20_cap_vanilla" / "preds_slim.csv")
    v = pd.read_csv(vp)
    vpr = dict(zip(v["doc_id"].astype(str), v["prob"].astype(float)))
    vsp = dict(zip(v["doc_id"].astype(str), v["split"].astype(str)))
    dense_van = np.array([vpr.get(k, np.nan) for k in ids])
    # the two arms MUST agree on the split assignment, or "same rows" is false
    vsplit = np.array([vsp.get(k, "unmapped") for k in ids])
    assert (vsplit == dsplit).all(), (
        "matched-vanilla arm does not share the archived arm's train/eval/test split; "
        "the two T values would not be on the same rows")
    dense = dense_van

    return dict(tag="cap_crowd", ids=ids, y=y, groups=groups, A=A, V=V,
                a_names=[str(s) for s in c["a_names"]],
                v_names=[str(s) for s in c["v_names"]],
                texts=texts, descs=row_desc, years=None,
                crowd_mean=crowd_mean, crowd_votes=crowd_votes,
                crowd_pct_in_contest=crowd_pct, contest_no=contest_no,
                dense=dense, dense_archived=dense_arch, dense_split=dsplit,
                dense_arm="matched fresh vanilla (D20_cap_vanilla, lambda_adv=0) -- "
                          "CAMPAIGN PRIMARY; d['dense_archived'] is the archived arm")


LOADERS = {"cap_crowd": _load_cap_crowd}
CELLS = list(LOADERS)


def load(cell="cap_crowd"):
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
    d = load("cap_crowd")
    print(f"cap_crowd n={len(d['y'])} groups={len(set(d['groups']))} "
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
