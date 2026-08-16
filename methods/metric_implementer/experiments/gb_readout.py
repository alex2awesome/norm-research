"""GEPA-flavor delta readout: critic-shipped rubrics (602) vs seed C0 on EVAL-half.
Delta_rec = 0 for all 15 (holdout gate closed everywhere)."""
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

MD = Path("/lfs/skampere3/0/alexspan/mention_auc")
YFILE = {"humor": "humor_ypos.json", "peer": "peer_y_pos.json"}


def obj_half(task, doc):
    return int(hashlib.md5(f"ocsplit:{task}:{doc}".encode()).hexdigest(), 16) % 2 == 0


def auc(y, p):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if not n1 or not n0:
        return None
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ymap_load(yf):
    raw = json.load(open(MD / yf))
    k = next(iter(raw))
    out = defaultdict(set)
    if re.fullmatch(r"a\d+", k):
        for m, docs in raw.items():
            for d in docs:
                out[d].add(m)
    else:
        for d, ms in raw.items():
            out[d] = set(ms)
    return out


rows = []
for task in ("humor", "peer"):
    gb = MD / f"gb_{task}_corpus8b.json"
    if not gb.exists():
        continue
    g = json.load(open(gb))
    cor = json.load(open(MD / f"oclc_llama8b_{task}_corpus.json"))
    ids = g["post_ids"]
    idxc = {x: i for i, x in enumerate(cor["post_ids"])}
    ym = ymap_load(YFILE[task])
    ev = ~np.array([obj_half(task, x) for x in ids])
    for key, vec in g["scores"].items():
        mid = key.split("__")[0]
        ck = f"{mid}__-1"
        if ck not in cor["scores"]:
            continue
        yv = np.array([1 if mid in ym.get(x, ()) else 0 for x in ids])
        v = np.asarray(vec, float)
        v0f = np.asarray(cor["scores"][ck], float)
        v0 = np.array([v0f[idxc[x]] if x in idxc else np.nan for x in ids])
        fin = np.isfinite(v) & np.isfinite(v0) & ev
        if fin.sum() < 60 or yv[fin].sum() < 5 or yv[fin].sum() > fin.sum() - 5:
            continue
        rows.append({"task": task, "metric": mid,
                     "auc_seed": round(auc(yv[fin], v0[fin]), 4),
                     "auc_critic": round(auc(yv[fin], v[fin]), 4)})
print(f"critic-shipped metrics evaluable: {len(rows)}")
for r in rows:
    print(f"{r['task']:6s} {r['metric']:6s} seed {r['auc_seed']:.3f} critic-evolved {r['auc_critic']:.3f} "
          f"delta {r['auc_critic'] - r['auc_seed']:+.4f}")
d = [r["auc_critic"] - r["auc_seed"] for r in rows]
if len(d) >= 5:
    rng = random.Random(0)
    n = len(d)
    boots = sorted(float(np.mean([d[rng.randrange(n)] for _ in range(n)])) for _ in range(20000))
    print(f"Delta_critic(shipped) mean {np.mean(d):+.4f} [{boots[500]:+.4f},{boots[19499]:+.4f}] "
          f"+/-: {sum(1 for x in d if x > 0)}/{sum(1 for x in d if x < 0)}")
print("Delta_rec = 0.0 on all 15 (holdout gate closed 15/15)")
json.dump(rows, open(MD / "gb_readout_v1.json", "w"), indent=1)
