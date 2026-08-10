#!/usr/bin/env python3
"""Build fixed train/eval/test splits for the Llama-8B dense runs (three claim-matcher
domains). Consumed by methods/dense/train_reward_model.py (which requires ~80/10/10 and
reads split_dir/{train,eval,test}.csv with 'text' + 'judgement' columns).

  peer/           test = the CANONICAL ICLR 2,400 (npz row order) -> AUC comparable to the
                  linear dense .690 protocol (fit on pool, evaluate once on the 2,400).
                  train/eval = big-train pool (other ICLR train abstracts, text-reuse guard).
  news_pooled/    9,919 EN docs, honest y; split by outlet|day CELL (hash) 80/10/10.
  news_loo_latimes/, news_loo_guardian/
                  train/eval = other 3 outlets (hash 8:1); test = held-out outlet SUBSAMPLED
                  to ~10% of dir total (harness enforces ratios) -> outlet-transport readout.
  patents/        pair testbed 71,714; text = element + reference passage; APP-grouped
                  hash 80/10/10 (uid kept for within-claim paired readout post-eval).

All splits stable-hash (sha1), no seeded shuffles. Run on sk3 (CPU):
  $HOME/envs/ai_usage/bin/python methods/dense/prep_dense8b_splits.py
"""
import csv, glob, gzip, hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
OUT = ROOT / "outputs" / "dense8b"
NH = ROOT / "datasets" / "news-homepages"

def h(s):
    return hashlib.sha1(str(s).encode()).hexdigest()

def write_split(d, train, ev, test):
    d.mkdir(parents=True, exist_ok=True)
    (d / "splits").mkdir(exist_ok=True)
    for name, df in [("train", train), ("eval", ev), ("test", test)]:
        df.to_csv(d / "splits" / f"{name}.csv", index=False)
    pd.concat([train, ev, test]).to_csv(d / "all.csv", index=False)
    tot = len(train) + len(ev) + len(test)
    print(f"[{d.name}] train {len(train)} ({len(train)/tot:.3f}) | eval {len(ev)} "
          f"({len(ev)/tot:.3f}) | test {len(test)} ({len(test)/tot:.3f}) | "
          f"y-rate test {test.judgement.mean():.3f}", flush=True)

def norm_prefix(t):
    return re.sub(r"[^a-z0-9]+", "", str(t).lower())[:150]

# ---------------- peer ----------------
d = np.load(ROOT / "datasets/peer-review/peer_review_scores_iclr.npz", allow_pickle=True)
ids = [str(i) for i in d["ids"]]; y = d["y"].astype(int)
t = pd.read_csv(ROOT / "datasets/peer-review/splits/train.csv.gz")
by_id = t.drop_duplicates("id").set_index("id")
test = pd.DataFrame({"id": ids,
                     "text": [str(by_id.loc[i, "text"]) for i in ids],
                     "judgement": y})
pref2400 = set(test.text.map(norm_prefix))
pool = t[t.venue.astype(str).str.lower().str.contains("iclr") & ~t.id.isin(set(ids))].copy()
pool = pool.dropna(subset=["text", "judgement"])
pool = pool[~pool.text.map(norm_prefix).isin(pref2400)]
pool["judgement"] = pool.judgement.astype(int)
pool = pool.assign(hh=pool.id.map(h)).sort_values("hh")
n_eval = 2260
ev = pool.iloc[:n_eval][["id", "text", "judgement"]]
tr = pool.iloc[n_eval:][["id", "text", "judgement"]]
write_split(OUT / "peer", tr, ev, test)

# ---------------- news ----------------
M = pd.read_csv(ROOT / "outputs/multi_y_news/doc_metrics_v2.csv")
urls = set(M.url); text = {}
for p in glob.glob(str(NH / "fulltext" / "fulltext_v2_shard*.jsonl")):
    for ln in open(p):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("url") in urls and len(r.get("text") or "") > 400:
            text[r["url"]] = r["text"]
M = M[M.url.isin(text)].reset_index(drop=True)
N = pd.DataFrame({"url": M.url, "outlet": M.outlet, "day": M.day,
                  "text": M.url.map(text), "judgement": M.y.astype(int)})
print(f"[news] {len(N)} docs", flush=True)

# pooled: split by outlet|day cell
N["cell"] = N.outlet.astype(str) + "|" + N.day.astype(str)
cells = sorted(N.cell.unique(), key=h)
sizes = N.cell.value_counts()
tr_c, ev_c, te_c, cum = [], [], [], 0
tot = len(N)
for c in cells:
    frac = cum / tot
    (tr_c if frac < .80 else ev_c if frac < .90 else te_c).append(c)
    cum += sizes[c]
write_split(OUT / "news_pooled",
            N[N.cell.isin(tr_c)].drop(columns="cell"),
            N[N.cell.isin(ev_c)].drop(columns="cell"),
            N[N.cell.isin(te_c)].drop(columns="cell"))

# LOO outlets
for held in ["latimes", "guardian"]:
    rest = N[N.outlet != held].assign(hh=lambda x: x.url.map(h)).sort_values("hh")
    n_test = int(round(len(rest) / 0.9 * 0.1))
    ho = N[N.outlet == held].assign(hh=lambda x: x.url.map(h)).sort_values("hh").iloc[:n_test]
    n_ev = n_test
    write_split(OUT / f"news_loo_{held}",
                rest.iloc[n_ev:].drop(columns=["cell", "hh"]),
                rest.iloc[:n_ev].drop(columns=["cell", "hh"]),
                ho.drop(columns=["cell", "hh"]))

# ---------------- patents ----------------
rows = []
for ln in open(ROOT / "datasets/claim-matching/testbed/pair_testbed.jsonl"):
    r = json.loads(ln)
    rows.append({"uid": r["uid"], "app_id": r["app_id"], "rejection_type": r.get("rejection_type"),
                 "text": f"CLAIM ELEMENT:\n{r['element']}\n\nREFERENCE PASSAGE:\n{r['span']}",
                 "judgement": int(r["y"])})
P = pd.DataFrame(rows)
print(f"[patents] {len(P)} pairs / {P.uid.nunique()} claims / {P.app_id.nunique()} apps", flush=True)
apps = sorted(P.app_id.unique(), key=h)
sizes = P.app_id.value_counts()
tr_a, ev_a, te_a, cum = [], [], [], 0
tot = len(P)
for a in apps:
    frac = cum / tot
    (tr_a if frac < .80 else ev_a if frac < .90 else te_a).append(a)
    cum += sizes[a]
write_split(OUT / "patents", P[P.app_id.isin(tr_a)], P[P.app_id.isin(ev_a)], P[P.app_id.isin(te_a)])
print("PREP_DONE", flush=True)
