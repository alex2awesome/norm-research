"""Compare free-gen vs MCQ recovery channels + the graded distractor-difficulty sweep.

Reads recon_free / recon_mcq_{random,hard,graded}.json from tmp_vinfo and prints:
  (1) per-metric free vs MCQ transmission + articulation tax (T_hard - T_free),
  (2) per-task means, (3) the graded transmission-&-identification vs difficulty-S curve.
"""
import json
import collections
import numpy as np

OUT = "/lfs/skampere3/0/alexspan/tmp_vinfo"


def load(n):
    try:
        return json.load(open(f"{OUT}/{n}"))
    except Exception as e:
        print("MISS", n, e)
        return []


def nz(x):
    return float("nan") if x is None else x


free = load("recon_free.json")
rnd = load("recon_mcq_random.json")
hard = load("recon_mcq_hard.json")
grad = load("recon_mcq_graded.json")


def key(r):
    return (r["task"], r["metric_id"])


fmap = {key(r): r for r in free}
rmap = {key(r): r for r in rnd}
hmap = {key(r): r for r in hard}

print("\n=== FREE vs MCQ (pinned) -- transmission I(m;m^) bits; tax = T_hard - T_free ===")
hdr = "{:11s} {:28s} {:>7s} {:>16s} {:>16s} {:>7s}".format(
    "task", "metric", "T_free", "T_rand(S/id)", "T_hard(S/id)", "tax")
print(hdr)
for k in sorted(fmap):
    f = fmap[k]
    r = rmap.get(k, {})
    h = hmap.get(k, {})
    tf, tr, th = nz(f.get("iv_transmission")), nz(r.get("iv_transmission")), nz(h.get("iv_transmission"))
    rcell = "{:.3f}({:.2f}/{:.2f})".format(tr, nz(r.get("option_set_S")), nz(r.get("identification_acc")))
    hcell = "{:.3f}({:.2f}/{:.2f})".format(th, nz(h.get("option_set_S")), nz(h.get("identification_acc")))
    print("{:11s} {:28s} {:7.3f} {:>16s} {:>16s} {:+7.3f}".format(
        k[0], f.get("name", "")[:28], tf, rcell, hcell, th - tf))

print("\n=== per-task means ===")
for task in sorted(set(k[0] for k in fmap)):
    ks = [k for k in fmap if k[0] == task]
    mf = np.nanmean([nz(fmap[k].get("iv_transmission")) for k in ks])
    mr = np.nanmean([nz(rmap[k].get("iv_transmission")) for k in ks if k in rmap]) if any(k in rmap for k in ks) else float("nan")
    mh = np.nanmean([nz(hmap[k].get("iv_transmission")) for k in ks if k in hmap]) if any(k in hmap for k in ks) else float("nan")
    idr = np.nanmean([nz(rmap[k].get("identification_acc")) for k in ks if k in rmap]) if any(k in rmap for k in ks) else float("nan")
    idh = np.nanmean([nz(hmap[k].get("identification_acc")) for k in ks if k in hmap]) if any(k in hmap for k in ks) else float("nan")
    print("  {:11s} T_free={:.3f}  T_rand={:.3f}(id {:.2f})  T_hard={:.3f}(id {:.2f})  n={}".format(
        task, mf, mr, idr, mh, idh, len(ks)))

print("\n=== GRADED sweep: transmission & identification vs difficulty S (mean over metrics) ===")
g = collections.defaultdict(lambda: collections.defaultdict(lambda: {"T": [], "S": [], "id": []}))
for r in grad:
    band = r.get("distractor", "?")
    g[r["task"]][band]["T"].append(nz(r.get("iv_transmission")))
    g[r["task"]][band]["S"].append(nz(r.get("option_set_S")))
    g[r["task"]][band]["id"].append(nz(r.get("identification_acc")))
for task in sorted(g):
    print("  {}:".format(task))
    for band in sorted(g[task]):
        d = g[task][band]
        print("    {:10s} S={:.2f}  T={:.3f}  id={:.2f}".format(
            band, np.nanmean(d["S"]), np.nanmean(d["T"]), np.nanmean(d["id"])))
