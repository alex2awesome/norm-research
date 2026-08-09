"""Decoder-channel MI upper-bound census (user request 2026-08-08).

For each decoder-output row (base||arm scored by one receiver over 300 probes):
  - The reference labeling maximizing MI with the row is the row itself, so the
    channel's upper bound is H(pred), the entropy of the decoder's own verdicts.
  - Achieved MI is computed vs the frontier dossier reference (4-voter majority,
    ties dropped), efficiency = MI / min(H(pred), H(ref)) on the same items.
  - Degenerate = near-constant rows (frac_yes <= .02 or >= .98): upper bound ~0;
    raw agreement with a skewed reference can still look high, balanced accuracy
    is already immune (constant rows -> .5), MI makes the collapse explicit.
News task is scored PANEL-INTERNAL only (its probe universe is news_probes.jsonl —
see reference_news_zxa_probe_universe landmine); ref and rows share that universe.
CPU-only; no new scoring. Output: mi_upper_census_v1.json + two lean tables.
"""
import json
import math
import os

import numpy as np

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"
TASKS = ["humor", "creative_writing", "math", "peer_review", "news_homepages"]
EXECS = ["llama1b", "llama3b", "qwen25-3b", "qwen25-7b", "llama8b", "qwen25-14b",
         "gemma2-27b", "qwen25-32b", "llama70b", "qwen25-72b"]
GLM = ["glm-47", "glm-52"]
V1_ARMS = ["name", "definition", "explanation", "dossier", "dossier_mismatched",
           "definition_padded"]
DEG_LO, DEG_HI = 0.02, 0.98


def load(p):
    if not os.path.exists(p):
        return {}
    z = np.load(p, allow_pickle=True)
    return {str(n): z["m_bar"][i] for i, n in enumerate([str(x) for x in z["names"]])}


def h_bin(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def mi_bits(x, y):
    # binary-binary plug-in MI
    n = len(x)
    if n == 0:
        return 0.0, 0.0, 0.0
    mi = 0.0
    px1 = float(np.mean(x))
    py1 = float(np.mean(y))
    for a in (0, 1):
        for b_ in (0, 1):
            pab = float(np.mean((x == a) & (y == b_)))
            pa = px1 if a else 1 - px1
            pb = py1 if b_ else 1 - py1
            if pab > 0 and pa > 0 and pb > 0:
                mi += pab * math.log2(pab / (pa * pb))
    return max(mi, 0.0), h_bin(px1), h_bin(py1)


rows_out = []
for TASK in TASKS:
    pan = {ex: load(f"{OM}/mbar_zxa_{TASK}_{ex}.npz") for ex in EXECS}
    glm = {ex: load(f"{OM}/mbar_zxaglm_{TASK}_{ex}.npz") for ex in GLM}
    all_src = {**pan, **glm}
    # frontier ref per base (4-voter majority; news = panel-internal, same universe)
    bases = sorted({k.split("||")[0] for d in all_src.values() for k in d})
    ref = {}
    for b in bases:
        votes = []
        for src_name in ("llama70b", "qwen25-72b", "glm-47", "glm-52"):
            r = all_src.get(src_name, {}).get(f"{b}||dossier")
            if r is not None:
                votes.append((np.asarray(r, float) > .5).astype(float))
        if len(votes) < 2:
            continue
        mean = np.stack(votes).mean(0)
        ref[b] = np.where(mean > .5, 1, np.where(mean < .5, 0, -1))
    for ex, d in all_src.items():
        for key, row in d.items():
            b, arm = key.rsplit("||", 1)
            if arm not in V1_ARMS:
                continue
            r = np.asarray(row, float)
            fin = np.isfinite(r)
            if fin.sum() < 30:
                continue
            pred = (r[fin] > .5).astype(int)
            fy = float(pred.mean())
            H = h_bin(fy)
            rec = {"task": TASK, "ex": ex, "arm": arm, "base": b,
                   "frac_yes": round(fy, 4), "H_pred": round(H, 4),
                   "degenerate": bool(fy <= DEG_LO or fy >= DEG_HI)}
            lab = ref.get(b)
            if lab is not None:
                ok = fin & (np.asarray(lab) >= 0)
                if ok.sum() >= 30:
                    x = (r[ok] > .5).astype(int)
                    y = np.asarray(lab)[ok].astype(int)
                    mi, hx, hy = mi_bits(x, y)
                    cap = min(hx, hy)
                    rec.update({"MI": round(mi, 4), "H_ref": round(hy, 4),
                                "eff": round(mi / cap, 4) if cap > 1e-9 else None})
            rows_out.append(rec)

json.dump({"deg_lo": DEG_LO, "deg_hi": DEG_HI, "rows": rows_out},
          open(f"{OM}/mi_upper_census_v1.json", "w"))
print(f"rows={len(rows_out)}")

# Table 1: per (task, arm) pooled over receivers+bases
print("\n=== per task x channel (pooled receivers x bases) ===")
print(f"{'task':16s} {'arm':20s} {'n':>5s} {'%deg':>5s} {'medH':>6s} "
      f"{'medMI':>6s} {'maxMI':>6s} {'medEff':>6s}")
for TASK in TASKS:
    for arm in V1_ARMS:
        sub = [r for r in rows_out if r["task"] == TASK and r["arm"] == arm]
        if not sub:
            continue
        mis = [r["MI"] for r in sub if "MI" in r]
        effs = [r["eff"] for r in sub if r.get("eff") is not None]
        print(f"{TASK:16s} {arm:20s} {len(sub):5d} "
              f"{100 * np.mean([r['degenerate'] for r in sub]):5.1f} "
              f"{np.median([r['H_pred'] for r in sub]):6.3f} "
              f"{(np.median(mis) if mis else float('nan')):6.3f} "
              f"{(np.max(mis) if mis else float('nan')):6.3f} "
              f"{(np.median(effs) if effs else float('nan')):6.3f}")

# Table 2: degeneracy rate per receiver (pooled tasks/arms) — capability dependence
print("\n=== degeneracy rate per receiver (pooled tasks x arms) ===")
for ex in EXECS + GLM:
    sub = [r for r in rows_out if r["ex"] == ex]
    if not sub:
        continue
    deg = 100 * np.mean([r["degenerate"] for r in sub])
    mh = np.median([r["H_pred"] for r in sub])
    mis = [r["MI"] for r in sub if "MI" in r]
    print(f"{ex:12s} n={len(sub):5d} deg={deg:5.1f}%  medH={mh:.3f}  "
          f"medMI={(np.median(mis) if mis else float('nan')):.3f}")

print("\nDONE ->", f"{OM}/mi_upper_census_v1.json")
