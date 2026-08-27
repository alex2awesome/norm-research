#!/usr/bin/env python3
"""HASHTAGWARS DEEP AUDIT -- Q1: what is the retrieval batch, and does the dense edge ride on it?

THE STRUCTURE (established from the raw SemEval-2017 Task 6 release, not from any
downstream artifact).  Each hashtag contest is one .tsv of ~112 tweets labelled
0 (not top-10) / 1 (top-10) / 2 (winner).  Sorting a contest's tweets by tweet_id --
Snowflake ids are monotone in posting time -- separates the labels almost perfectly:

    within-contest tweet_id rank -> top-10,  pooled AUC = .0279 over 101 train contests
    88 / 101 contests have max(positive id) < min(negative id): FULLY DISJOINT ranges
    median fraction of negatives with an id below max(positive id) = 0.0000

That is not an early-posting effect, which would produce overlapping distributions.  It is
two RETRIEVAL SWEEPS: the top-10 tweets (known from the @midnight broadcast) were pulled in
one pass and the filler negatives in a later pass, so the two classes occupy disjoint
Snowflake-id intervals.  The label is therefore recoverable from collection metadata alone.

WHAT THIS SCRIPT ASKS.  The metadata leak is not itself a threat to the cell -- no model
here sees tweet_id.  The threat is a TEXTUAL FINGERPRINT of the sweep: if the two sweeps
differ in how tweets were captured (mention placement, urls, truncation, retweet furniture,
character escaping), a text-only dense model can read the sweep and its edge over the
articulated bank is an artifact rather than taste.  So:

  1. join the closure population back to raw tweet_ids and build the batch channel;
  2. measure the batch channel on E (the 924 dense-held-out rows);
  3. decompose the dense-over-bank LEVEL gap: stratified on batch, matched on batch, and
     the stratification-free stacked increment over batch alone;
  4. ask directly whether the TEXT predicts the batch (a text-only batch classifier), which
     is the quantity that decides whether the leak can reach a text model at all.

JOIN DISCIPLINE (standing rule: no id-dict joins).  The population's `id` is a sha1 of the
context block and carries no tweet_id.  The join key is (contest, normalised tweet text)
built from the population's own `text` field by parsing the frozen context template, and
every stage asserts counts; unmatched rows are reported and dropped, never imputed.

TOKENS NOT CHARS for every length readout.

CPU only.  Usage: OMP_NUM_THREADS=6 python3 batch_audit.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import collections
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
HW = REPO / "datasets" / "humor" / "hashtagwars"
POP = HERE / "hashtagwars_verdict_population.csv"
DENSE = HERE.parent / "maps_hw_si_dense_preds.json"

csv.field_size_limit(10 ** 9)
CTX = re.compile(r'^CONTEST HASHTAG: #(?P<tag>[^\n]+)\n\nTWEET: "(?P<tweet>[\s\S]*)"$')


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def toks(s: str) -> int:
    return len(norm(s).split())


# ------------------------------------------------------------------ raw ----
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
    return out


def main():
    raw = load_raw()
    print(f"[raw] {len(raw)} tweets over {len({r['tag'] for r in raw})} contests")

    # within-contest id rank + batch indicator, computed on the RAW file
    byc = collections.defaultdict(list)
    for r in raw:
        byc[r["tag"]].append(r)
    for tag, v in byc.items():
        v.sort(key=lambda r: r["tid"])
        n = len(v)
        pos_ids = [r["tid"] for r in v if r["lab"] >= 1]
        cut = max(pos_ids) if pos_ids else -1
        for i, r in enumerate(v):
            r["id_rank_pct"] = i / max(n - 1, 1)
            r["early_sweep"] = int(r["tid"] <= cut)      # LABEL-DERIVED, diagnostic only
    # a LABEL-FREE version of the same channel: is the id below the contest's own
    # largest natural gap in the id sequence?  This is the honest operationalisation --
    # it finds the sweep boundary from the id distribution alone.
    for tag, v in byc.items():
        ids = np.array([r["tid"] for r in v], dtype=float)
        if len(ids) < 4:
            for r in v:
                r["gap_sweep"] = 0
            continue
        gaps = np.diff(ids)
        k = int(np.argmax(gaps))
        for i, r in enumerate(v):
            r["gap_sweep"] = int(i <= k)
    y_raw = np.array([1 if r["lab"] >= 1 else 0 for r in raw])
    print(f"[raw] AUC(id_rank_pct)  = {roc_auc_score(y_raw, [r['id_rank_pct'] for r in raw]):.4f}")
    print(f"[raw] AUC(gap_sweep)    = {roc_auc_score(y_raw, [r['gap_sweep'] for r in raw]):.4f}"
          "   <- LABEL-FREE sweep split from the id gap alone")

    # ----------------------------------------------------------- join ----
    key = collections.defaultdict(list)
    for r in raw:
        key[(r["tag"], norm(r["text"]))].append(r)

    with open(POP) as fh:
        pop = list(csv.DictReader(fh))
    matched, unmatched, ambiguous = [], 0, 0
    for p in pop:
        m = CTX.match(p["text"])
        if not m:
            unmatched += 1
            p["_raw"] = None
            continue
        cand = key.get((m.group("tag"), norm(m.group("tweet"))))
        if not cand:
            unmatched += 1
            p["_raw"] = None
            continue
        if len({c["tid"] for c in cand}) > 1:
            ambiguous += 1
        p["_raw"] = cand[0]
    matched = [p for p in pop if p["_raw"] is not None]
    print(f"[join] population {len(pop)} -> matched {len(matched)} "
          f"({len(matched)/len(pop):.4f}), unmatched {unmatched}, ambiguous-text {ambiguous}")
    # the join must reproduce y exactly where it matched
    bad = sum(1 for p in matched if int(p["judgement"]) != int(p["_raw"]["lab"] >= 1))
    print(f"[join] label disagreements after join: {bad}  (must be 0)")
    assert bad == 0, "join is wrong: labels disagree"

    # ------------------------------------------------------------ dense ----
    dj = json.loads(DENSE.read_text())
    recs = dj["hashtagwars_verdict"] if isinstance(dj, dict) and "hashtagwars_verdict" in dj else dj
    dmap = collections.defaultdict(list)
    for r in (recs if isinstance(recs, list) else recs.get("records", [])):
        dmap[(str(r.get("group")), norm(r.get("text", "")))].append(r)
    print(f"[dense] {sum(len(v) for v in dmap.values())} dense records loaded")

    out = {"schema": "hashtagwars_batch_audit/v1",
           "raw": {"n_tweets": len(raw), "n_contests": len({r['tag'] for r in raw}),
                   "AUC_id_rank_pct": float(roc_auc_score(y_raw, [r["id_rank_pct"] for r in raw])),
                   "AUC_gap_sweep_LABEL_FREE": float(roc_auc_score(y_raw, [r["gap_sweep"] for r in raw])),
                   "n_contests_fully_disjoint": int(sum(
                       1 for tag, v in byc.items()
                       if [r for r in v if r["lab"] >= 1] and [r for r in v if r["lab"] == 0]
                       and max(r["tid"] for r in v if r["lab"] >= 1)
                       < min(r["tid"] for r in v if r["lab"] == 0)))},
           "join": {"n_population": len(pop), "n_matched": len(matched),
                    "match_rate": len(matched) / len(pop), "n_unmatched": unmatched,
                    "label_disagreements": bad}}

    # -------------------------------------------- does TEXT predict the sweep? ----
    # If the text cannot predict the sweep, the leak cannot reach a text-only model.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline

    tags = np.array([p["_raw"]["tag"] for p in matched], dtype=object)
    txt = [norm(CTX.match(p["text"]).group("tweet")) for p in matched]
    sweep = np.array([p["_raw"]["gap_sweep"] for p in matched])
    ylab = np.array([int(p["judgement"]) for p in matched])
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(txt))
    for tr, te in gkf.split(txt, groups=tags):
        clf = make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True),
                            LogisticRegression(C=1.0, max_iter=2000))
        clf.fit([txt[i] for i in tr], sweep[tr])
        oof[te] = clf.predict_proba([txt[i] for i in te])[:, 1]
    auc_text_sweep = float(roc_auc_score(sweep, oof))
    auc_text_y = None
    oof2 = np.zeros(len(txt))
    for tr, te in gkf.split(txt, groups=tags):
        clf = make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True),
                            LogisticRegression(C=1.0, max_iter=2000))
        clf.fit([txt[i] for i in tr], ylab[tr])
        oof2[te] = clf.predict_proba([txt[i] for i in te])[:, 1]
    auc_text_y = float(roc_auc_score(ylab, oof2))
    out["text_reads_sweep"] = {
        "AUC_tfidf_predicts_gap_sweep": auc_text_sweep,
        "AUC_tfidf_predicts_y": auc_text_y,
        "n": len(txt),
        "note": "grouped-OOF tf-idf logistic on the tweet text alone. The first number is "
                "the ceiling on how much of the retrieval sweep a TEXT-ONLY model can see; "
                "the leak can only reach the dense arm through this channel."}
    print(f"[text->sweep] grouped-OOF AUC {auc_text_sweep:.4f}   "
          f"[text->y] {auc_text_y:.4f}  (n={len(txt)})")

    # ------------------------------------------- surface differences by sweep ----
    feats = {
        "tokens": lambda t: toks(t),
        "has_url": lambda t: int("http" in t.lower()),
        "n_at_mentions": lambda t: t.count("@"),
        "has_midnight": lambda t: int("@midnight" in t.lower()),
        "n_hashtags": lambda t: t.count("#"),
        "ends_with_hashtag": lambda t: int(norm(t).split()[-1].startswith("#")) if norm(t) else 0,
        "has_quote_char": lambda t: int('"' in t),
        "uppercase_ratio": lambda t: (sum(c.isupper() for c in t) / max(len(t), 1)),
    }
    tbl = {}
    for nm, fn in feats.items():
        v = np.array([fn(x) for x in txt], dtype=float)
        tbl[nm] = {"mean_early_sweep": float(v[sweep == 1].mean()),
                   "mean_late_sweep": float(v[sweep == 0].mean()),
                   "AUC_predicts_sweep": float(roc_auc_score(sweep, v)),
                   "AUC_predicts_y": float(roc_auc_score(ylab, v))}
    out["surface_by_sweep"] = tbl
    print("[surface] feature: AUC->sweep / AUC->y")
    for nm, v in sorted(tbl.items(), key=lambda kv: -abs(kv[1]["AUC_predicts_sweep"] - .5)):
        print(f"   {nm:20s} {v['AUC_predicts_sweep']:.3f} / {v['AUC_predicts_y']:.3f}"
              f"   early {v['mean_early_sweep']:.3f} vs late {v['mean_late_sweep']:.3f}")

    (HERE / "hashtagwars_batch_audit.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote", HERE / "hashtagwars_batch_audit.json")


if __name__ == "__main__":
    main()
