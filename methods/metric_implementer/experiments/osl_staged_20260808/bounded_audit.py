"""STEP 1 — BOUNDED audit (user 2026-07-08: 'extremely important').
Every BOUNDED verdict is a tacitness claim with two known impostors. Per BOUNDED metric:
  (a) WITHIN-FAMILY bend check: does each family ladder (llama 1/3/8/70, qwen 3/7/14/72),
      alone, saturate? Families disagree -> DIALECT-SUSPECT (pooled bend = family composition
      artifact, not a plateau).
  (b) FLOOR-CONTACT: per-metric frontier floor = mean cross-family agreement among top-z
      executors (Spearman of z-scored m_bar vectors). Fitted L <= floor + margin -> AT-FLOOR:
      the plateau is criterion underdetermination, NOT executor tacitness.
  (c) survivors = TACIT-CANDIDATE: both families saturate, plateau ABOVE the floor and BELOW
      the ceiling. These are the metrics licensed to carry the negative certificate.
Also audits REACHES for the converse error (dialect-inflated ceilings). Descriptions attached
for spot-reading. Output: outputs/osl_multi/bounded_audit.json + printed tables.
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.stats import spearmanr

B = "/lfs/skampere3/0/alexspan"
O = f"{B}/outputs/osl_multi"
FAM = {"llama1b": "llama", "llama3b": "llama", "llama8b": "llama", "llama70b": "llama",
       "llama405b": "llama", "qwen25-3b": "qwen", "qwen25-7b": "qwen", "qwen25-14b": "qwen",
       "qwen25-32b": "qwen", "qwen25-72b": "qwen", "gemma2-9b": "gemma2", "gemma2-27b": "gemma2",
       "mistral7b": "mistral", "mistral-24b": "mistral", "phi4": "phi", "qwen35-122b": "qwen35"}
HIER = {"creative_writing": "creative-writing", "press_releases": "press-releases",
        "math": "math", "news_homepages": "news-homepages", "peer_review": "peer-review",
        "notice_and_comment": "notice-and-comment", "patents": "patents", "humor": "humor"}
MARGIN = 0.10          # floor / ceiling contact margin
LATE = 0.05            # family late-gain below this = saturating


def load_panels(task):
    panels = {}
    def add(path, ex):
        z = np.load(path, allow_pickle=True)
        d = panels.setdefault(ex, {})
        names = [str(x) for x in z["names"]]
        for i, n in enumerate(names):
            d[n] = z["m_bar"][i]
    if task == "humor":
        for f in sorted(glob.glob(f"{B}/outputs/osl/mbar285_*.npz")) + \
                 sorted(glob.glob(f"{B}/outputs/osl/mbar285c_*.npz")):
            add(f, re.sub(r"^mbar285c?_|\.npz$", "", os.path.basename(f)))
        for f in sorted(glob.glob(f"{O}/mbar2_humor_sup_*.npz")) + \
                 sorted(glob.glob(f"{O}/mbar2_humor_[!s]*.npz")):
            add(f, re.sub(r"^mbar2_humor(_sup)?_|\.npz$", "", os.path.basename(f)))
    else:
        for f in sorted(glob.glob(f"{O}/mbar2_{task}_*.npz")):
            add(f, re.sub(rf"^mbar2_{task}_|\.npz$", "", os.path.basename(f)))
    return {e: d for e, d in panels.items() if e in FAM}


def zsv(v):
    s = np.nanstd(v)
    return (v - np.nanmean(v)) / s if s > 1e-9 else np.full_like(v, np.nan)


def fam_verdict(c, fam):
    """within-family shape from the curve points of one family (z-ordered)."""
    pts = [(z, y) for z, y, e in zip(c["z"], c["y"], c["execs"]) if FAM.get(e) == fam]
    if len(pts) < 3:
        return "UNDERPOWERED"
    pts.sort()
    ys = [y for _, y in pts]
    late = ys[-1] - ys[-2]
    early = max(ys[:-1]) - ys[0]
    if early <= 0.05 and abs(late) <= LATE:
        return "FLAT"
    if late >= LATE:
        return "RISING"
    if late <= -2 * LATE and ys[-1] < max(ys) - 2 * LATE:
        return "DECLINING"           # non-monotone fall at top = dialect signature
    return "SATURATING"


def metric_floor(panels, zmap, name):
    """cross-family agreement among the 3 top-z executors that scored this metric."""
    es = sorted((e for e in panels if name in panels[e] and e in zmap),
                key=lambda e: -zmap[e])
    top = []
    for e in es:                      # top-z, one per family, up to 3 families
        if FAM[e] not in {FAM[t] for t in top}:
            top.append(e)
        if len(top) == 3:
            break
    if len(top) < 2:
        return np.nan
    rs = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = zsv(panels[top[i]][name]), zsv(panels[top[j]][name])
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 20:
                rs.append(spearmanr(a[m], b[m]).correlation)
    return float(np.mean(rs)) if rs else np.nan


zmap = {}
for f in glob.glob(f"{B}/outputs/osl/*.json"):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    if isinstance(r, dict) and r.get("executor") and r.get("battery"):
        zmap[r["executor"]] = r["battery"]["z"]

audit = {}
for task in HIER:
    lp = f"{O}/laws_{task}.json"
    cp = f"{O}/curves_{task}.json"
    if not (os.path.exists(lp) and os.path.exists(cp)):
        continue
    laws = json.load(open(lp))
    curves = json.load(open(cp))
    ceil = laws.get("ceiling") or np.nan
    mg = json.load(open(f"{B}/norm-research/outputs/hierarchy/{HIER[task]}_general_r2_expanded.json"))["merged_groups"]
    descs = {g["merged_name"]: (g.get("merged_description") or "")[:140] for g in mg
             if g.get("merged_name")}
    panels = load_panels(task)
    rows = []
    for r in laws["rows"]:
        if r.get("verdict") not in ("BOUNDED", "REACHES") or r["name"] not in curves:
            continue
        c = curves[r["name"]]
        fl = fam_verdict(c, "llama")
        fq = fam_verdict(c, "qwen")
        floor = metric_floor(panels, zmap, r["name"])
        L = r.get("L")
        klass = "UNAUDITABLE"
        if r["verdict"] == "BOUNDED" and L is not None:
            fams = {fl, fq}
            if "DECLINING" in fams or ("RISING" in fams and
                                       ("SATURATING" in fams or "FLAT" in fams)):
                klass = "DIALECT-SUSPECT"
            elif np.isfinite(floor) and L <= floor + MARGIN:
                klass = "AT-FLOOR"
            elif np.isfinite(ceil) and L >= ceil - MARGIN:
                klass = "CEILING-ADJACENT"
            elif fams <= {"SATURATING", "FLAT", "UNDERPOWERED"} and "UNDERPOWERED" in fams \
                    and fams != {"UNDERPOWERED"}:
                klass = "TACIT-CANDIDATE (weak)"
            elif fams == {"UNDERPOWERED"}:
                klass = "UNDERPOWERED"
            elif fams <= {"SATURATING", "FLAT"}:
                klass = "TACIT-CANDIDATE"
        elif r["verdict"] == "REACHES":
            klass = "REACHES-DIALECT" if ("DECLINING" in {fl, fq}) else "REACHES-OK"
        rows.append(dict(name=r["name"], verdict=r["verdict"], L=L, L_hi=r.get("L_hi"),
                         floor=None if not np.isfinite(floor) else round(floor, 3),
                         ceiling=None if not np.isfinite(ceil) else round(ceil, 3),
                         fam_llama=fl, fam_qwen=fq, klass=klass,
                         desc=descs.get(r["name"], "")))
    audit[task] = rows
    from collections import Counter
    b = Counter(x["klass"] for x in rows if x["verdict"] == "BOUNDED")
    rc = Counter(x["klass"] for x in rows if x["verdict"] == "REACHES")
    if b or rc:
        print(f"[{task}] BOUNDED audit: {dict(b)}   REACHES audit: {dict(rc)}")

print("\n== TACIT-CANDIDATES (both families saturate, above floor, below ceiling) ==")
for task, rows in audit.items():
    for x in rows:
        if x["klass"].startswith("TACIT-CANDIDATE"):
            print(f"  [{task}] L={x['L']:.2f} floor={x['floor']} ceil={x['ceiling']} "
                  f"({x['fam_llama']}/{x['fam_qwen']})  {x['name'][:48]}")
            print(f"       {x['desc'][:120]}")
print("\n== AT-FLOOR (plateau = criterion underdetermination, not executor tacitness) ==")
for task, rows in audit.items():
    for x in rows:
        if x["klass"] == "AT-FLOOR":
            print(f"  [{task}] L={x['L']:.2f} floor={x['floor']}  {x['name'][:60]}")
json.dump(audit, open(f"{O}/bounded_audit.json", "w"), indent=1, default=float)
print(f"\n-> {O}/bounded_audit.json")
