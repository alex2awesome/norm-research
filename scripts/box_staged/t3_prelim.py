import glob
import json
from collections import defaultdict

import numpy as np

MD = "/lfs/skampere3/0/alexspan/mention_auc"
items = {it["item_id"]: it for it in json.load(open(f"{MD}/t3_items_v2.json"))}
V = {}
for f in glob.glob(f"{MD}/verdicts/*.json"):
    for v in json.load(open(f)):
        V[v["item_id"]] = v
print("verdicts joined:", len(V), "of", len(items))

key = {(k.get("task", "peer"), k["metric"], k["doc"]): k["truth"]
       for k in json.load(open(f"{MD}/t3_anchor_key_SEALED.json"))}
ok = tot = 0
for iid, v in V.items():
    it = items.get(iid)
    if not it or it["stratum"] != "anchor":
        continue
    t = key.get((it.get("task", "peer"), it["metric"], it["doc"]))
    if t is None:
        continue
    tot += 1
    ok += int(bool(v["applies"]) == bool(t))
print(f"ANCHOR PREVIEW: {ok}/{tot} correct")


def load_scores(task):
    if task in ("peer", "crx"):
        f = "peer_p_scores.json" if task == "peer" else "crx_p_scores.json"
        d = json.load(open(f"{MD}/{f}"))
        return d["post_ids"], d["scores"]
    if task == "math":
        d = json.load(open(f"{MD}/math_canon_probes_g4.json"))
        return d["post_ids"], {k.split("__")[0]: v for k, v in d["scores"].items()}
    man = json.load(open(f"{MD}/{task}_forms_manifest.json"))
    d = json.load(open(f"{MD}/{task}_scores_g4.json"))
    best = {}
    for e in man:
        m = e["metric_id"]
        if m not in best or e.get("mi_form", 0) > best[m][1]:
            best[m] = (e["form_idx"], e.get("mi_form", 0))
    return d["post_ids"], {m: d["scores"][f"{m}__{fi}"]
                           for m, (fi, _) in best.items() if f"{m}__{fi}" in d["scores"]}


def auc(y, p):
    o = np.argsort(p)
    r = np.empty(len(p))
    r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    if not n1 or not n0:
        return None
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


cache = {}
rows = defaultdict(list)
for iid, v in V.items():
    it = items.get(iid)
    if not it or it["stratum"] == "anchor":
        continue
    task = it.get("task", "peer")
    if task not in cache:
        cache[task] = load_scores(task)
    ids, S = cache[task]
    if it["metric"] not in S:
        continue
    try:
        i = ids.index(it["doc"])
    except ValueError:
        continue
    sc = float(np.asarray(S[it["metric"]], float)[i])
    if not np.isfinite(sc):
        continue
    ym = 1 if it["stratum"].startswith("y1") else 0
    rows[task].append((sc, ym, 1 if v["applies"] else 0))

print("task    n   AUC_vs_mention  AUC_vs_arbiter  arb+rate  pure_neg  y0_to_arb+")
res = {}
for task, rr in sorted(rows.items()):
    sc = np.array([r[0] for r in rr])
    ym = np.array([r[1] for r in rr])
    ya = np.array([r[2] for r in rr])
    y0 = ym == 0
    flip = float(ya[y0].mean()) if y0.sum() else float("nan")
    a1, a2 = auc(ym, sc), auc(ya, sc)
    res[task] = {"n": len(rr), "auc_mention": a1, "auc_arbiter": a2,
                 "arb_pos_rate": float(ya.mean()), "pure_neg": int((ya == 0).sum()),
                 "y0_flip_rate": flip}
    print(f"{task:7s}{len(rr):4d}   {a1:.3f}          {a2!s:>6.4s}          "
          f"{ya.mean():.2f}      {int((ya == 0).sum()):4d}      {flip:.2f}")
json.dump(res, open(f"{MD}/t3_prelim_readout.json", "w"), indent=1)
