"""m_omega experiment S1: universe + candidate families (runs on sk3, CPU).

Universe: e-cert slice banks (fresh 4-family Omega pools, 8B-mined) INTERSECT mention-y
evaluable metrics, tasks {humor, cw}: slice gi == hierarchy group == mention aid directly.
(peer needs a name-join to a different bank space — deferred to v1.1 if n is thin.)

Candidate family per metric (structured assemblies; label-free construction):
  C0 = definition (merged_name: merged_description)             <- m_desc
  C1 = definition + top-2 units   C2 = definition + top-4 units
  C3 = definition + top-8 units   C4 = top-4 units alone
  C5 = best single unit
Unit pre-ranking INSIDE the bank (no external data): i_binary(unit sig row, M_i) from the
npz — the bank's own signals. Units drawn from freegen+children prompts.
Output: momega_candidates.json + universe report.
"""
import glob
import json
import math
import os
import re
from pathlib import Path

import numpy as np

BANKS = "/lfs/skampere3/0/alexspan/outputs/ecert_slice_v1"
HIER = Path("/lfs/skampere3/0/alexspan/norm-research/outputs/hierarchy")
MD = Path("/lfs/skampere3/0/alexspan/mention_auc")
HIERFILE = {"humor": "humor", "creative-writing": "creative-writing",
            "peer-review": "peer-review"}
MENTASK = {"humor": "humor", "creative-writing": "cw", "peer-review": "peer"}
YFILE = {"humor": "humor_ypos.json", "cw": "variant_ypos_cw.json",
         "peer": "peer_y_pos.json"}
PEERBANK = "/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful/banks/peer-review.json"
_norm = lambda s_: re.sub(r"[^a-z0-9]+", " ", str(s_).lower()).strip()


def ibin(a, b):
    ab = (a >= np.median(a)).astype(int)
    bb = (b >= np.median(b)).astype(int)
    if ab.std() == 0 or bb.std() == 0:
        return -1.0
    P = np.zeros((2, 2))
    for x, y in zip(ab, bb):
        P[x, y] += 1
    P /= P.sum()
    mi = 0.0
    for x in (0, 1):
        for y in (0, 1):
            if P[x, y] > 0:
                mi += P[x, y] * math.log2(P[x, y] / (P[x].sum() * P[:, y].sum()))
    return mi


ycov = {}
for t, yf in YFILE.items():
    raw = json.load(open(MD / yf))
    k = next(iter(raw))
    if re.fullmatch(r"a\d+", k):
        ycov[t] = {m: len(v) for m, v in raw.items()}
    else:
        cnt = {}
        for d, ms in raw.items():
            for m in ms:
                cnt[m] = cnt.get(m, 0) + 1
        ycov[t] = cnt

out = []
for f in sorted(glob.glob(f"{BANKS}/*_R2_metric*_sigs.npz")):
    base = os.path.basename(f)
    task = base.split("_R2_")[0]
    if task not in HIERFILE:
        continue
    gi = int(re.search(r"metric(\d+)_", base).group(1))
    mt = MENTASK[task]
    if task == "peer-review":
        # slice gi lives in hierarchy space; mention aid lives in silver-bank space -> name-join
        gh = json.load(open(HIER / "peer-review_general_r2_expanded.json"))["merged_groups"]
        if gi >= len(gh):
            continue
        target = _norm(gh[gi].get("merged_name", ""))
        pb = json.load(open(PEERBANK))["metrics"]
        mid = None
        for i, m in enumerate(pb):
            if _norm(m["name"]) == target:
                mid = f"a{i}"
                break
        if mid is None:
            continue
    else:
        mid = f"a{gi}"
    if ycov[mt].get(mid, 0) < 10:
        continue
    z = np.load(f, allow_pickle=True)
    sigs = np.asarray(z["sigs"], float)
    Mi = np.asarray(z["M_i"], float)
    prompts = [str(p) for p in z["prompts"]]
    med = np.median(Mi)
    if (Mi >= med).std() == 0:
        continue
    scores = []
    for i, row in enumerate(sigs):
        ok = np.isfinite(row)
        if ok.sum() < 100:
            continue
        scores.append((ibin(row[ok], Mi[ok]), i))
    scores.sort(reverse=True)
    top = [prompts[i] for _, i in scores[:8]]
    if len(top) < 4:
        continue
    g = json.load(open(HIER / f"{HIERFILE[task]}_general_r2_expanded.json"))["merged_groups"]
    nm = g[gi].get("merged_name", mid)
    desc = g[gi].get("merged_description", "")[:600]
    definition = f"{nm}: {desc}" if desc else nm
    cands = {
        "C0": definition,
        "C1": definition + " Key checks: " + " ".join(f"({j+1}) {u}" for j, u in enumerate(top[:2])),
        "C2": definition + " Key checks: " + " ".join(f"({j+1}) {u}" for j, u in enumerate(top[:4])),
        "C3": definition + " Key checks: " + " ".join(f"({j+1}) {u}" for j, u in enumerate(top[:8])),
        "C4": "Evaluate: " + " ".join(f"({j+1}) {u}" for j, u in enumerate(top[:4])),
        "C5": top[0],
    }
    out.append({"task": mt, "metric": mid, "bank": base, "name": nm,
                "n_pos": ycov[mt].get(mid, 0), "candidates": cands})

json.dump(out, open(f"{BANKS}/momega_candidates.json", "w"), indent=1)
import collections
print("universe:", len(out), dict(collections.Counter(r["task"] for r in out)))
print("median n_pos:", int(np.median([r["n_pos"] for r in out])))
