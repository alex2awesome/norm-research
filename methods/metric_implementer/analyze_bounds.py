"""Articulability BOUNDS: free-gen (lower) vs realistic-MCQ identification (upper), same [0,1] scale.

  lower bound  = free-gen transmission_norm = I(m;m^)/H(m): fraction of the metric's verdict-entropy
                 recovered when the model must AUTHOR the rule and have it re-execute.
  upper bound  = MCQ chance-corrected identification = (id - 1/M)/(1 - 1/M): can the model RECOGNIZE
                 the metric among behaviorally-confusable alternatives. Recognition >= recall, so it
                 upper-bounds articulability. Taken as the MAX over realistic difficulty bands.

Normalized (uniform-register option descriptions) is the realistic upper bound; raw is shown to
quantify the phrasing-distinctiveness confound.
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


free = load("recon_free_bounds.json")
raw = load("recon_mcq_graded_raw.json")
norm = load("recon_mcq_graded_norm.json")
tasks = sorted(set(r["task"] for r in free))


def task_lower(task):
    vs = [nz(r.get("transmission_norm")) for r in free if r["task"] == task]
    return np.nanmean(vs) if vs else float("nan")


def graded_by_band(rows, task):
    g = collections.defaultdict(lambda: {"S": [], "id": [], "idcc": []})
    for r in rows:
        if r["task"] != task:
            continue
        b = r.get("distractor", "?")
        g[b]["S"].append(nz(r.get("option_set_S")))
        g[b]["id"].append(nz(r.get("identification_acc")))
        g[b]["idcc"].append(nz(r.get("identification_cc")))
    return g


def task_upper(rows, task):
    g = graded_by_band(rows, task)
    band_means = [np.nanmean(d["idcc"]) for d in g.values() if np.isfinite(np.nanmean(d["idcc"]))]
    return max(band_means) if band_means else float("nan")


print("\n=== ARTICULABILITY BOUNDS (both in [0,1]) ===")
print("{:16s} {:>11s} {:>12s} {:>12s}".format("task", "LOWER(free)", "UPPER(raw)", "UPPER(norm)"))
for t in tasks:
    print("{:16s} {:11.3f} {:12.3f} {:12.3f}".format(
        t, task_lower(t), task_upper(raw, t), task_upper(norm, t)))

print("\n=== graded detail: chance-corrected identification per band (raw -> norm) ===")
for t in tasks:
    gr, gn = graded_by_band(raw, t), graded_by_band(norm, t)
    print("  {}:".format(t))
    for b in sorted(set(gr) | set(gn)):
        sr = np.nanmean(gr[b]["S"]) if b in gr else float("nan")
        idr = np.nanmean(gr[b]["idcc"]) if b in gr else float("nan")
        idn = np.nanmean(gn[b]["idcc"]) if b in gn else float("nan")
        rawid = np.nanmean(gr[b]["id"]) if b in gr else float("nan")
        nrmid = np.nanmean(gn[b]["id"]) if b in gn else float("nan")
        print("    {:10s} S={:.2f}  raw id={:.2f}(cc {:+.2f})  norm id={:.2f}(cc {:+.2f})".format(
            b, sr, rawid, idr, nrmid, idn))
