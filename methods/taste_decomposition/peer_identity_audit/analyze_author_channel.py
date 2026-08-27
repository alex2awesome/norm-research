#!/usr/bin/env python3
"""peer_revealed identity-leak audit, step 2 (user request 2026-08-12: "is it fitting
on author identity? or school/company/institution? were these in the spurious
variables?" — they were NOT: the 57 F2 nuisance channels are all judged text
criteria, nuisance_struct=0).

Three readouts, all label-only + metadata (no model, no text):
  1. AUTHOR/INSTITUTION FOLD OVERLAP — the dense splits group by ntitle (paper), so
     the same author's other papers CAN straddle train/held-out. Quantify it.
  2. AUTHOR-IDENTITY-ALONE AUC — score each paper by the mean y of the same
     authors' OTHER papers in the corpus (self and same-ntitle excluded). If this
     is well above .5, author identity is a live confound channel the campaign
     never named.
  3. INSTITUTION-ALONE AUC — same estimator over institutions.

CPU, seconds. Writes author_channel_report.json.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2] / "datasets/peer-review/vat_3y/revealed.jsonl"

rows = []
for line in open(SRC):
    r = json.loads(line)
    yv = r.get("judgement")
    try:
        yv = float(yv)
    except (TypeError, ValueError):
        continue
    if yv not in (0.0, 1.0):
        continue
    rows.append({"wid": r["id"].rsplit("/", 1)[-1], "ntitle": r["ntitle"],
                 "y": int(yv), "split": r.get("split"), "year": r.get("year")})

meta = {}
for line in open(HERE / "openalex_authorships.jsonl"):
    w = json.loads(line)
    meta[w["work_id"]] = w

for r in rows:
    m = meta.get(r["wid"], {})
    r["authors"] = [a["author_id"] for a in m.get("authors", []) if a.get("author_id")]
    r["insts"] = sorted({i["id"] for a in m.get("authors", [])
                         for i in a.get("institutions", []) if i.get("id")})
    r["cites"] = m.get("cited_by_count")

n = len(rows)
have_auth = sum(1 for r in rows if r["authors"])
rep = {"n_valid_y_rows": n, "rows_with_authorship": have_auth,
       "split_counts": {}, "pos_rate": float(np.mean([r["y"] for r in rows]))}
for r in rows:
    rep["split_counts"][r["split"]] = rep["split_counts"].get(r["split"], 0) + 1

# ---- 1. overlap: held-out rows sharing an author/institution with ANY train row
train_auth, train_inst = set(), set()
for r in rows:
    if r["split"] == "train":
        train_auth.update(r["authors"])
        train_inst.update(r["insts"])
held = [r for r in rows if r["split"] != "train"]
ov_a = [bool(set(r["authors"]) & train_auth) for r in held if r["authors"]]
ov_i = [bool(set(r["insts"]) & train_inst) for r in held if r["insts"]]
rep["heldout_rows"] = len(held)
rep["heldout_share_author_with_train"] = float(np.mean(ov_a)) if ov_a else None
rep["heldout_share_institution_with_train"] = float(np.mean(ov_i)) if ov_i else None

# ---- 2/3. identity-alone AUC (leave-own-paper-and-group-out mean-y encoder)
def identity_auc(key):
    by_id = defaultdict(list)          # entity -> list of (ntitle, y)
    for r in rows:
        for e in r[key]:
            by_id[e].append((r["ntitle"], r["y"]))
    scores, ys, covered = [], [], 0
    for r in rows:
        vals = []
        for e in r[key]:
            vals += [y for nt, y in by_id[e] if nt != r["ntitle"]]
        if vals:
            covered += 1
            scores.append(np.mean(vals))
            ys.append(r["y"])
    if len(set(ys)) < 2:
        return None
    return {"auc": float(roc_auc_score(ys, scores)), "n_covered": covered,
            "coverage": covered / n}

def identity_auc_split(key, split):
    by_id = defaultdict(list)
    for r in rows:
        if r["split"] == "train":      # encoder built from TRAIN ONLY = the leak path
            for e in r[key]:
                by_id[e].append((r["ntitle"], r["y"]))
    scores, ys = [], []
    for r in rows:
        if r["split"] == "train" or (split and r["split"] != split):
            continue
        vals = [y for e in r[key] for nt, y in by_id[e] if nt != r["ntitle"]]
        if vals:
            scores.append(np.mean(vals))
            ys.append(r["y"])
    if len(set(ys)) < 2:
        return None
    return {"auc": float(roc_auc_score(ys, scores)), "n": len(ys)}

rep["author_identity_alone_full_loo"] = identity_auc("authors")
rep["institution_identity_alone_full_loo"] = identity_auc("insts")
rep["author_train_encoder_on_heldout"] = identity_auc_split("authors", None)
rep["institution_train_encoder_on_heldout"] = identity_auc_split("insts", None)

# fame proxy: does the authors' mean citation mass predict y (bibliometric, honest-ish
# but text-invisible)?  Uses each author's OTHER papers' cited_by_count mean.
by_auth_c = defaultdict(list)
for r in rows:
    if r["cites"] is not None:
        for e in r["authors"]:
            by_auth_c[e].append((r["ntitle"], r["cites"]))
sc, ys = [], []
for r in rows:
    vals = [np.log1p(c) for e in r["authors"] for nt, c in by_auth_c[e] if nt != r["ntitle"]]
    if vals:
        sc.append(np.mean(vals))
        ys.append(r["y"])
rep["author_fame_logcites_alone"] = {"auc": float(roc_auc_score(ys, sc)), "n": len(ys)}

json.dump(rep, open(HERE / "author_channel_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
