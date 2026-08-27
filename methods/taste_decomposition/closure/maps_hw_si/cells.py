#!/usr/bin/env python3
"""Cell adapter for the map-focused Layer-3 dual-track batch on the two HUMOR
cells that now have dense bounds: HashtagWars verdict and Style Invitational
top-tier.

Same contract as maps_batch1/cells.py: each loader returns the cell's Layer-1
A/V population in the bank's own row order (meta["item_ids"]), together with

    ids          bank item id (sha1) -- the stable per-row key
    groups       hashtag contest / week_id
    y            0/1 label (NEVER shown to a proposer)
    A, V         RAW Gemma-4-31B criterion scores + programmatic V features
    texts        the SAME context block the Gemma A-judge and the dense reader saw
    dense        per-row dense probability (seed-ensemble; NaN on dense-train rows)
    dense_split  train / eval / test
    dense_seeds  per-seed probability matrix (n, 3) for the T-definition band
    position     per-row position-in-container covariates (FREEZE ADDENDUM 4)

RECORDED DECISION (dense preds).  Unlike the batch-1 cells there is no
"same-rows rescore" for these two: the dense-standard chain wrote per-row
probabilities only on its own eval/test rows (3 seeds x {eval,test}).  The
HONEST population is exactly those rows, which is the only population the
closure protocol reads T on, so nothing is lost -- but `dense` is NaN on the
80% dense-train rows and code must never assume it is finite everywhere.

RECORDED DECISION (which seed is T).  The registry T for these cells
(HW .6642, SI .6343) is the MEAN OVER SEEDS of the per-seed eval AUC.  A
row-level readout needs one score per row, so the row-level dense score used
by every fit/stratification/stack here is the SEED-ENSEMBLE (mean of the three
seeds' probabilities).  Both are reported everywhere: `T_registry_mean_of_seed_AUC`
and `T_ensemble`.  No claim rests on the difference.

POSITION (FREEZE ADDENDUM 4).  Both corpora have an observable container
position and it is recovered here as an OBSERVED COVARIATE, exactly as N&C did
with docket sequence -- never added to V or A, never judged, never fit into the
closure curve.
  * HashtagWars: the tweet id is a Twitter snowflake, monotone in submission
    time, so within-contest rank by tweet id IS submission order; plus contest
    size and raw stream order.
  * Style Invitational: the published results list entries best-first, so
    within-week entry order and week size are recoverable; week_id is the
    contest's ordinal in time.

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

VA_OUT = REPO / "outputs" / "va_gemma_banks"
HW_DIR = REPO / "datasets" / "humor" / "hashtagwars"
SI_DIR = REPO / "datasets" / "humor" / "style_invitational"
RESULTS = TD / "results"
DENSE_JSON = CLOSURE / "maps_hw_si_dense_preds.json"

SEED_COLS = ["p_seed1", "p_seed2", "p_seed42"]

CELL_META = {
    "hashtagwars_verdict": dict(
        group_column="hashtag contest", item="tweet",
        corpus="reader-submitted one-line tweets entered into a nightly comedy "
               "hashtag contest on Twitter",
        construct="how good the tweet is as an entry in that hashtag contest",
        text_trunc=600,
        layer1="hashtagwars_verdict_layer1.json",
        bank="hashtagwars",
    ),
    "style_inv_toptier": dict(
        group_column="week_id", item="contest entry",
        corpus="reader-submitted entries to a weekly newspaper humour contest "
               "(the Washington Post Style Invitational)",
        construct="how good the entry is as an entry in that week's contest",
        text_trunc=900,
        layer1="style_inv_toptier_layer1.json",
        bank="style_invitational",
    ),
}


def _bank(name):
    meta = json.loads((VA_OUT / f"{name}_meta.json").read_text())
    Xs, Vs, ids = [], [], []
    si = 0
    while (VA_OUT / f"{name}_shard{si}.npz").exists():
        z = np.load(VA_OUT / f"{name}_shard{si}.npz", allow_pickle=True)
        Xs.append(z["X"])
        Vs.append(z["V"])
        ids += z["ids"].tolist()
        si += 1
    X = np.vstack(Xs)
    V = np.vstack(Vs)
    order = {d: i for i, d in enumerate(ids)}
    idx = np.array([order[d] for d in meta["item_ids"]])
    return meta, X[idx].astype(float), V[idx].astype(float)


def _dense_records(cell):
    return json.loads(DENSE_JSON.read_text())[cell]


# ------------------------------------------------------------ HashtagWars ---
def _load_hw():
    meta, A, V = _bank("hashtagwars")
    ids = [str(s) for s in meta["item_ids"]]
    groups = np.array([str(s) for s in meta["item_groups"]], dtype=object)
    y = np.array(meta["ys"]["top10_or_winner"], dtype=int)

    # texts + tweet ids, rebuilt from the same TSVs the bank was built from
    tw, order_in_stream = {}, {}
    seen = set()
    npos = 0
    for sd in ("train_data", "trial_data", "gold_labels"):
        for path in sorted((HW_DIR / sd).glob("*.tsv")):
            hashtag = path.stem
            with path.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    tid, text = parts[0], parts[1]
                    rid = hashlib.sha1(f"{hashtag}\0{tid}\0{text}".encode()).hexdigest()
                    if rid in seen:
                        continue
                    seen.add(rid)
                    tw[rid] = (hashtag, tid, text)
                    order_in_stream[rid] = npos
                    npos += 1
    texts = [f'CONTEST HASHTAG: #{tw[i][0]}\n\nTWEET: "{tw[i][2]}"' for i in ids]
    tweet_id = np.array([float(tw[i][1]) for i in ids])
    stream_ix = np.array([order_in_stream[i] for i in ids], dtype=float)
    return dict(tag="hashtagwars_verdict", ids=ids, y=y, groups=groups, A=A, V=V,
                a_names=[str(s) for s in meta["a_names"]],
                v_names=[str(s) for s in meta["v_names"]],
                texts=texts,
                _pos_raw={"tweet_id": tweet_id, "stream_ix": stream_ix})


# ------------------------------------------------------ Style Invitational --
def _load_si():
    meta, A, V = _bank("style_invitational")
    ids = [str(s) for s in meta["item_ids"]]
    groups = np.array([str(s) for s in meta["item_groups"]], dtype=object)
    y = np.array(meta["ys"]["top_tier"], dtype=int)

    rows = [json.loads(l) for l in open(SI_DIR / "style_invitational.jsonl") if l.strip()]
    by_id, entry_ix = {}, {}
    per_week = defaultdict(int)
    for i, r in enumerate(rows):
        rid = hashlib.sha1(f"{r['week_id']}\0{i}\0{r['entry_text']}".encode()).hexdigest()[:20]
        by_id[rid] = r
        entry_ix[rid] = per_week[str(r["week_id"])]
        per_week[str(r["week_id"])] += 1
    texts = [f'CONTEST PROMPT: {by_id[i]["contest_prompt"]}\n\n'
             f'ENTRY: "{by_id[i]["entry_text"]}"' for i in ids]
    e_ix = np.array([entry_ix[i] for i in ids], dtype=float)
    week_num = np.array([float(by_id[i]["week_id"]) for i in ids])
    return dict(tag="style_inv_toptier", ids=ids, y=y, groups=groups, A=A, V=V,
                a_names=[str(s) for s in meta["a_names"]],
                v_names=[str(s) for s in meta["v_names"]],
                texts=texts,
                _pos_raw={"entry_ix": e_ix, "week_num": week_num})


# ------------------------------------------------------------ dense join ----
def _attach_dense(d, cell):
    ids = d["ids"]
    n = len(ids)
    recs = _dense_records(cell)
    dsplit = np.array(["train"] * n, dtype=object)
    P = np.full((n, len(SEED_COLS)), np.nan)
    pos_of = {k: i for i, k in enumerate(ids)}
    if cell == "hashtagwars_verdict":
        for r in recs:
            i = pos_of[r["row_id"]]
            dsplit[i] = r["dense_split"]
            P[i] = [r[c] for c in SEED_COLS]
            assert int(r["judgement"]) == int(d["y"][i]), "HW label mismatch on join"
    else:
        # (group, text) join; 74 of 1,920 held-out rows are exact duplicate
        # (week, text) pairs -- identical text so identical dense probability,
        # assigned in a stable order, which leaves every readout unchanged.
        buckets = defaultdict(list)
        for i, k in enumerate(ids):
            buckets[(str(d["groups"][i]), d["texts"][i])].append(i)
        cursor = defaultdict(int)
        for r in recs:
            key = (str(r["group"]), r["text"])
            b = buckets[key]
            i = b[cursor[key]]
            cursor[key] += 1
            dsplit[i] = r["dense_split"]
            P[i] = [r[c] for c in SEED_COLS]
    d["dense_split"] = dsplit
    d["dense_seeds"] = P
    d["dense"] = np.nanmean(P, axis=1)          # seed ensemble; NaN on dense-train
    held = np.isin(dsplit, ["eval", "test"])
    assert np.isfinite(d["dense"][held]).all(), f"{cell}: missing dense prob on held-out row"
    assert not np.isfinite(d["dense"][~held]).any(), f"{cell}: dense prob on a train row"
    return d


# -------------------------------------------------------------- position ----
def _attach_position(d):
    """Position-in-container covariates (FREEZE ADDENDUM 4).  OBSERVED COVARIATE
    ONLY: never enters V, A, the bank, or any closure fit."""
    g = d["groups"]
    raw = d.pop("_pos_raw")
    size = pd.Series(np.ones(len(g))).groupby(pd.Series(g)).transform("sum").values
    out = {"container_size": size.astype(float)}
    if d["tag"] == "hashtagwars_verdict":
        key = raw["tweet_id"]
        out["submission_time_snowflake"] = key
        out["stream_ix"] = raw["stream_ix"]
    else:
        key = raw["entry_ix"]
        out["entry_ix"] = raw["entry_ix"]
        out["week_num"] = raw["week_num"]
    s = pd.Series(key)
    grp = s.groupby(pd.Series(g))
    out["within_container_rank"] = grp.rank(method="average").values.astype(float)
    out["within_container_pct"] = grp.rank(pct=True).values.astype(float)
    d["position"] = out
    return d


LOADERS = {"hashtagwars_verdict": _load_hw, "style_inv_toptier": _load_si}
CELLS = list(LOADERS)


def load(cell):
    d = LOADERS[cell]()
    d = _attach_dense(d, cell)
    d = _attach_position(d)
    d["meta"] = CELL_META[cell]
    led = json.loads((RESULTS / CELL_META[cell]["layer1"]).read_text())
    d["layer1"] = led
    assert len(d["y"]) == led["n"], f"{cell}: n={len(d['y'])} != layer1 n={led['n']}"
    assert abs(float(d["y"].mean()) - led["pos_rate"]) < 1e-9
    return d


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    for c in CELLS:
        d = load(c)
        held = np.isin(d["dense_split"], ["eval", "test"])
        per_seed = [float(roc_auc_score(d["y"][held], d["dense_seeds"][held, k]))
                    for k in range(3)]
        print(f"{c:22s} n={len(d['y']):6d} groups={len(set(d['groups'])):4d} "
              f"pos={d['y'].mean():.4f} A={d['A'].shape[1]} V={d['V'].shape[1]} "
              f"held={held.sum()} T_ens={roc_auc_score(d['y'][held], d['dense'][held]):.4f} "
              f"T_meanseedAUC={np.mean(per_seed):.4f} seeds={[round(v,4) for v in per_seed]}")
