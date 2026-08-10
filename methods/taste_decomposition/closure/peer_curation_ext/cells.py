#!/usr/bin/env python3
"""Unified cell adapter for the MAP-FOCUSED Layer-3 dual-track batch 1.

Cells: peer_curation, peer_revealed, cap_crowd, cap_finalist, nc_outcome.

Each loader returns the cell's Layer-1 A/V population in a REPRODUCIBLE row order
(never Python `set` iteration -- nc_outcome sorts `valid_out`, exactly as
layer2_robustness.load_nc_cell does), together with:

    ids     stable per-row key used to join dense predictions
    groups  the cell's canonical grouping unit (ntitle / contest / docket)
    y       0/1 label  (NEVER shown to a proposer)
    A, V    RAW score blocks (no degeneracy screen -- clean_fit is applied inside
            FIT+MINE only, per the closure protocol)
    texts   raw item text
    dense   per-row dense probability + dense_split, or None if not yet rescored

The A/V matrices are deliberately raw: closure_lib.clean_fit/clean_apply learn the
screen and the imputation medians on FIT+MINE only.

CPU only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLOSURE = HERE.parent
TD = CLOSURE.parent
REPO = TD.parents[1]
sys.path.insert(0, str(TD))
sys.path.insert(0, str(CLOSURE))

PEER = REPO / "datasets" / "peer-review" / "vat_3y"
NC_DIR = REPO / "datasets" / "notice-and-comment" / "v4"
CAP_POOL = REPO / "datasets" / "humor" / "caption_multiy" / "caption_contest_v2.jsonl"
SAMEROWS = CLOSURE / "samerows_preds"
RESULTS = TD / "results"

CELL_META = {
    "peer_curation": dict(
        group_column="ntitle", item="abstract",
        corpus="abstracts of machine-learning conference paper submissions",
        construct="research quality",
        text_trunc=6000,
        layer1="peer_curation_layer1.json",
    ),
    "peer_revealed": dict(
        group_column="ntitle", item="abstract",
        corpus="abstracts of machine-learning conference paper submissions",
        construct="research quality",
        text_trunc=6000,
        layer1="peer_revealed_layer1.json",
    ),
    "cap_crowd": dict(
        group_column="contest", item="caption",
        corpus="reader-submitted captions for a weekly cartoon caption contest",
        construct="how good the caption is as a contest entry",
        text_trunc=1200,
        layer1="cap_crowd_layer1.json",
    ),
    "cap_finalist": dict(
        group_column="contest", item="caption",
        corpus="reader-submitted captions for a weekly cartoon caption contest",
        construct="how good the caption is as a contest entry",
        text_trunc=1200,
        layer1="cap_finalist_layer1.json",
    ),
    "nc_outcome": dict(
        group_column="docket", item="public comment",
        corpus="public comments submitted to United States federal agency rulemaking dockets",
        construct="the substantive quality of the comment",
        text_trunc=3000,
        layer1="nc_outcome_layer1.json",
    ),
    "nc_agree": dict(
        group_column="docket", item="public comment",
        corpus="public comments submitted to United States federal agency rulemaking dockets",
        construct="the substantive quality of the comment",
        text_trunc=3000,
        layer1="nc_agree_layer1.json",
    ),
}


# ------------------------------------------------------------------ peer ----
def _load_peer(cell_jsonl, tag):
    import layer1_stack as L1

    z = np.load(L1.NPZ, allow_pickle=True)
    X, V, nt = z["X"], z["V"], z["ntitle"]
    a_names = [str(s) for s in z["a_names"]]
    v_names = [str(s) for s in z["v_names"]]
    X_by = {nt[i]: X[i] for i in range(len(nt))}
    V_by = {nt[i]: V[i] for i in range(len(nt))}

    rows = [json.loads(l) for l in open(PEER / f"{cell_jsonl}.jsonl") if l.strip()]
    R = [r for r in rows if r.get("ntitle") in X_by and L1._valid_y(r)]
    ids = [str(r["ntitle"]) for r in R]
    y = np.array([int(float(r["judgement"])) for r in R])
    A = np.array([X_by[k] for k in ids], dtype=float)
    Vm = np.array([V_by[k] for k in ids], dtype=float)
    texts = [r.get("text", "") for r in R]
    years = np.array([float(r["year"]) if r.get("year") is not None else np.nan for r in R])

    # dense split membership (group column == ntitle)
    dense_of = {}
    for s in ("train", "eval", "test"):
        p = PEER / "dense_llama" / cell_jsonl / "split" / f"{s}.csv"
        for g in pd.read_csv(p)["group"].astype(str):
            dense_of[g] = s
    dsplit = np.array([dense_of.get(k, "unmapped") for k in ids])

    dense = None
    dp = SAMEROWS / f"{tag}_dense_preds_slim.csv"
    if dp.exists():
        d = pd.read_csv(dp)
        pr = dict(zip(d["id"].astype(str), d["dense_prob"].astype(float)))
        dense = np.array([pr.get(k, np.nan) for k in ids])
    return dict(tag=tag, ids=ids, y=y, groups=np.array(ids), A=A, V=Vm,
                a_names=a_names, v_names=v_names, texts=texts, years=years,
                dense=dense, dense_split=dsplit)


# --------------------------------------------------------------- caption ----
def _caption_meta():
    meta = {}
    for line in open(CAP_POOL):
        if not line.strip():
            continue
        r = json.loads(line)
        did = f"{r['contest']}_{hashlib.sha1(r['text'].encode()).hexdigest()[:12]}"
        meta[did] = r
    return meta


def _load_caption(slug):
    import layer1_gemma_cells as L1G

    c = L1G._caption_pools()
    id_key = "crowd_ids" if slug == "cap_crowd" else "hardneg_ids"
    y_key = "y_crowd" if slug == "cap_crowd" else "y_fin"
    ids = sorted(x for x in c[id_key] if x in c["X_by_id"])
    y = np.array([c[y_key][d] for d in ids])
    A = np.array([c["X_by_id"][d] for d in ids], dtype=float)
    V = np.array([c["V_by_id"][d] for d in ids], dtype=float)
    groups = np.array([str(c["contest_by_id"][d]) for d in ids])
    meta = _caption_meta()
    texts = [meta[i]["text"] if i in meta else "" for i in ids]

    d = pd.read_csv(SAMEROWS / f"{slug}_dense_preds_slim.csv")
    pr = dict(zip(d["id"].astype(str), d["dense_prob"].astype(float)))
    ds = dict(zip(d["id"].astype(str), d["dense_split"].astype(str)))
    dense = np.array([pr.get(k, np.nan) for k in ids])
    dsplit = np.array([ds.get(k, "unmapped") for k in ids])
    return dict(tag=slug, ids=ids, y=y, groups=groups, A=A, V=V,
                a_names=[str(s) for s in c["a_names"]],
                v_names=[str(s) for s in c["v_names"]],
                texts=texts, years=None, dense=dense, dense_split=dsplit)


# ------------------------------------------------------------------- N&C ----
def _load_nc(cell):
    """cell in {"outcome","agree"}: same matched pool, different expert label.
    Row order is `sorted(...)` so it is reproducible across processes (NCData's
    valid_out / valid_agr are Python sets -- never iterate them directly)."""
    import nc_layer1_stack as NCL

    data = NCL.NCData()
    if cell == "outcome":
        ids = sorted(data.valid_out)
        y_by = data.y_out_by_id
        tag, preds = "nc_outcome", "nc_outcome_dense_preds_slim.csv"
    elif cell == "agree":
        ids = sorted(data.valid_agr)
        y_by = data.y_agr_by_id
        tag, preds = "nc_agree", "nc_agree_dense_preds_slim.csv"
    else:
        raise ValueError(cell)
    y = np.array([y_by[d] for d in ids])
    A = np.array([data.X_m[d].astype(float) for d in ids], dtype=float)
    texts = [data.text_m.get(d, "") for d in ids]
    V = np.array([[NCL.v_features(t)[n] for n in NCL.V_NAMES] for t in texts], dtype=float)
    groups = np.array([str(data.docket_m[d]) for d in ids])

    year_by_id = {}
    for p in (NC_DIR / "nc_vat_sample.jsonl", NC_DIR / "nc_unmatched_sample.jsonl"):
        for line in open(p):
            if line.strip():
                rr = json.loads(line)
                year_by_id.setdefault(rr["doc_id"], rr.get("year"))
    years = np.array([float(year_by_id[d]) if year_by_id.get(d) is not None else np.nan
                      for d in ids])

    d = pd.read_csv(SAMEROWS / preds)
    pr = dict(zip(d["doc_id"].astype(str), d["dense_prob"].astype(float)))
    ds = dict(zip(d["doc_id"].astype(str), d["dense_split"].astype(str)))
    dense = np.array([pr.get(k, np.nan) for k in ids])
    dsplit = np.array([ds.get(k, "unmapped") for k in ids])
    return dict(tag=tag, ids=ids, y=y, groups=groups, A=A, V=V,
                a_names=[str(s) for s in data.a_names],
                v_names=list(NCL.V_NAMES), texts=texts, years=years,
                dense=dense, dense_split=dsplit)


LOADERS = {
    "peer_curation": lambda: _load_peer("curation", "peer_curation"),
    "peer_revealed": lambda: _load_peer("revealed", "peer_revealed"),
    "cap_crowd": lambda: _load_caption("cap_crowd"),
    "cap_finalist": lambda: _load_caption("cap_finalist"),
    "nc_outcome": lambda: _load_nc("outcome"),
    "nc_agree": lambda: _load_nc("agree"),
}
CELLS = list(LOADERS)


def load(cell):
    d = LOADERS[cell]()
    d["meta"] = CELL_META[cell]
    led = json.loads((RESULTS / CELL_META[cell]["layer1"]).read_text())
    d["layer1"] = led
    n_expect = led["n"]
    assert len(d["y"]) == n_expect, f"{cell}: n={len(d['y'])} != layer1 n={n_expect}"
    return d


if __name__ == "__main__":
    for c in CELLS:
        d = load(c)
        dn = d["dense"]
        print(f"{c:14s} n={len(d['y']):6d} groups={len(set(d['groups'])):5d} "
              f"pos={d['y'].mean():.3f} A={d['A'].shape[1]:3d} V={d['V'].shape[1]:3d} "
              f"dense={'yes' if dn is not None and np.isfinite(dn).all() else 'NO'} "
              f"heldout={int(np.isin(d['dense_split'], ['eval','test']).sum())}")
