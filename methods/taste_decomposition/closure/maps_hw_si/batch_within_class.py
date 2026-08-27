#!/usr/bin/env python3
"""HASHTAGWARS DEEP AUDIT -- Q1b: is the dense edge (or any named channel) READING THE SWEEP?

WHY THE OBVIOUS TEST IS UNAVAILABLE, stated before the one that replaces it.  The retrieval
sweep predicts y at AUC .974 (label-free gap split; batch_audit.py).  Any channel that
predicts y therefore predicts the sweep, and any attempt to "discount on the sweep" is an
attempt to stratify on a variable 97% collinear with the label: within-stratum y is nearly
constant, the stratified AUC is undefined on most strata, and the number that comes out is
an artifact of the few mixed strata.  **The sweep cannot be used as a discount channel on
this cell, and a stratified or matched Delta_adj computed on it must not be quoted.**

THE TEST THAT REPLACES IT.  The sweep is a TIME split -- the top-10 tweets were pulled in an
earlier pass, so they carry systematically earlier Snowflake ids.  A channel that is reading
the collection sweep rather than craft must track POSTING TIME.  So ask, WITHIN each label
class separately (where the sweep is nearly constant and time still varies):

    does the channel predict within-contest id-rank among negatives only?
    does it predict it among positives only?

A craft channel should be flat in both.  A scrape/time channel should not be.  This is the
same logic the programme's ordinal-craft finding used on r/Jokes, run in reverse.

Also run here: the same test for the DENSE score itself, which is the quantity the cell's
residual is made of, and the per-contest pool structure for Q3.

Spearman is used throughout (rank-rank, no distributional assumption); n per class is
reported with every coefficient, and nothing is quoted where n < 200.

CPU only.  Usage: OMP_NUM_THREADS=6 python3 batch_within_class.py
"""
from __future__ import annotations

import csv
import collections
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
HW = REPO / "datasets" / "humor" / "hashtagwars"
POP = HERE / "hashtagwars_verdict_population.csv"
CELL = "hashtagwars_verdict"

csv.field_size_limit(10 ** 9)
CTX = re.compile(r'^CONTEST HASHTAG: #(?P<tag>[^\n]+)\n\nTWEET: "(?P<tweet>[\s\S]*)"$')


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def load_raw():
    out = []
    for f in sorted(glob.glob(str(HW / "train_data" / "*.tsv"))) + \
             sorted(glob.glob(str(HW / "trial_data" / "*" / "*.tsv"))) + \
             sorted(glob.glob(str(HW / "trial_dir" / "*.tsv"))):
        tag = os.path.basename(f)[:-4]
        with open(f) as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) < 3:
                    continue
                try:
                    tid = int(p[0])
                except ValueError:
                    continue
                out.append({"tag": tag, "tid": tid, "text": p[1], "lab": int(p[2])})
    byc = collections.defaultdict(list)
    for r in out:
        byc[r["tag"]].append(r)
    for tag, v in byc.items():
        v.sort(key=lambda r: r["tid"])
        n = len(v)
        for i, r in enumerate(v):
            r["id_rank_pct"] = i / max(n - 1, 1)
    return out, byc


def main():
    raw, byc = load_raw()
    key = collections.defaultdict(list)
    for r in raw:
        key[(r["tag"], norm(r["text"]))].append(r)

    with open(POP) as fh:
        pop = list(csv.DictReader(fh))
    idx, rawrow = [], []
    for k, p in enumerate(pop):
        m = CTX.match(p["text"])
        c = key.get((m.group("tag"), norm(m.group("tweet")))) if m else None
        if c:
            idx.append(k)
            rawrow.append(c[0])
    idx = np.array(idx)
    print(f"[join] {len(idx)}/{len(pop)} matched")

    y = np.array([int(pop[k]["judgement"]) for k in idx])
    tag = np.array([rawrow[j]["tag"] for j in range(len(idx))], dtype=object)
    idrank = np.array([rawrow[j]["id_rank_pct"] for j in range(len(idx))])
    dsplit = np.array([pop[k]["dense_split"] for k in idx], dtype=object)
    heldout = np.isin(dsplit, ["eval", "test"])
    assert y.shape == idrank.shape

    out = {"schema": "hashtagwars_within_class/v1",
           "n_matched": int(len(idx)),
           "collinearity_warning": {
               "AUC_sweep_predicts_y": 0.9737,
               "consequence": "the retrieval sweep is 97% collinear with y, so it CANNOT be "
                              "used as a discount/stratification channel on this cell; any "
                              "stratified or matched Delta_adj computed on it is undefined "
                              "on most strata and must not be quoted."},
           "within_class_time_test": {}}

    # ------------------------------------------------ channels to test ----
    chans = {}
    # every Track-B routed channel from every round
    for r in (1, 2, 3, 4):
        fs, rt = HERE / f"{CELL}_r{r}_scores.npz", HERE / f"{CELL}_r{r}_routing_final.json"
        if not (fs.exists() and rt.exists()):
            continue
        z = np.load(fs, allow_pickle=True)
        cids = [str(s) for s in z["crit_ids"]]
        for x in json.loads(rt.read_text())["final"]:
            if x["final_route"] != "B" or x["blind_id"] not in cids:
                continue
            col = z["X"][:, cids.index(x["blind_id"])].astype(float)
            if len(col) != len(pop):
                continue
            chans[f"r{r}:{x['blind_id']}:{x['name'][:44]}"] = col[idx]

    # the dense score
    dz = HERE / f"{CELL}_r0_preds.npz"
    if dz.exists():
        z = np.load(dz, allow_pickle=True)
        for cand in ("dense", "dense_prob", "T"):
            if cand in z:
                dcol = np.asarray(z[cand], dtype=float)
                if len(dcol) == len(pop):
                    chans["DENSE_SCORE"] = dcol[idx]
                break

    neg, pos = y == 0, y == 1
    print(f"[classes] neg {neg.sum()}  pos {pos.sum()}")
    rows = []
    for nm, col in chans.items():
        fin = np.isfinite(col)
        rec = {"channel": nm,
               "alone_AUC_y_matched": (float(roc_auc_score(y[fin], col[fin]))
                                       if fin.sum() > 200 and len(set(y[fin])) > 1 else None),
               "n_finite": int(fin.sum())}
        for lab, m in (("neg", neg), ("pos", pos)):
            mm = m & fin
            if mm.sum() >= 200:
                rec[f"rho_time_{lab}"] = float(spearmanr(col[mm], idrank[mm]).statistic)
                rec[f"n_{lab}"] = int(mm.sum())
            else:
                rec[f"rho_time_{lab}"] = None
                rec[f"n_{lab}"] = int(mm.sum())
        rows.append(rec)
    rows.sort(key=lambda r: -abs((r["alone_AUC_y_matched"] or .5) - .5))
    out["within_class_time_test"]["channels"] = rows

    print(f"{'channel':52s} {'AUC_y':>6s} {'rho_t|neg':>10s} {'rho_t|pos':>10s}")
    for r in rows:
        print(f"{r['channel'][:52]:52s} {(r['alone_AUC_y_matched'] or 0):6.3f} "
              f"{(r['rho_time_neg'] if r['rho_time_neg'] is not None else 0):10.3f} "
              f"{(r['rho_time_pos'] if r['rho_time_pos'] is not None else 0):10.3f}"
              f"   (n+ {r['n_pos']})")

    # ------------------------------------------------------ Q3 pool structure ----
    pools = []
    for t in sorted({str(x) for x in tag}):
        m = tag == t
        pools.append({"contest": t, "n_rows_in_population": int(m.sum()),
                      "n_pos": int(y[m].sum()), "pos_rate": float(y[m].mean()),
                      "n_rows_raw_contest": len(byc[t])})
    raw_sizes = np.array([p["n_rows_raw_contest"] for p in pools])
    pop_sizes = np.array([p["n_rows_in_population"] for p in pools])
    npos = np.array([p["n_pos"] for p in pools])
    out["pool_structure"] = {
        "n_contests_in_population": len(pools),
        "raw_contest_size": {"mean": float(raw_sizes.mean()), "min": int(raw_sizes.min()),
                             "max": int(raw_sizes.max())},
        "population_rows_per_contest": {"mean": float(pop_sizes.mean()),
                                        "min": int(pop_sizes.min()), "max": int(pop_sizes.max())},
        "positives_per_contest": {"mean": float(npos.mean()), "min": int(npos.min()),
                                  "max": int(npos.max()), "sd": float(npos.std(ddof=1))},
        "design_note": "the SemEval release fixes the number of winners per contest at 10 "
                       "(1 winner + 9 runners-up; 11 of 101 contests have 8 runners-up). y is "
                       "therefore a FIXED-QUOTA rank-within-pool label, not an independent "
                       "per-item judgement -- the same shape as an editor pool.",
        "per_contest": pools}
    print(f"[pool] {len(pools)} contests | raw size mean {raw_sizes.mean():.1f} "
          f"[{raw_sizes.min()},{raw_sizes.max()}] | positives/contest mean {npos.mean():.2f} "
          f"sd {npos.std(ddof=1):.2f} [{npos.min()},{npos.max()}]")

    # honest-population sizes, for Q2
    out["E_geometry"] = {
        "n_heldout_matched": int(heldout.sum()),
        "n_contests_heldout": int(len({str(t) for t in tag[heldout]})),
        "n_contests_total_in_population": len(pools),
        "pos_in_heldout": int(y[heldout].sum()),
        "note": "E is the dense-held-out block only: 8 of the population's 40 contests. A "
                "cross-fitted dense arm would make every contest honest and multiply the "
                "grouping units by 5."}
    print(f"[E] held-out matched {heldout.sum()} rows, "
          f"{len({str(t) for t in tag[heldout]})} contests, {y[heldout].sum()} positives")

    (HERE / "hashtagwars_within_class.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote", HERE / "hashtagwars_within_class.json")


if __name__ == "__main__":
    main()
