"""E7 provenance grid — distillability x transport, crossed with CODIF + E1 (§6.3).

Joins: e7_sel_pilot.json (TF-IDF) + e7_sel_pilot_bge.json (BGE) per field;
inventory.json ratio_llama per criterion; codif_merged.jsonl thick-predicate tags;
key_eval_<task>.json frac_nonce per criterion (E1-selected subset).

distill = max(frac_distilled_tfidf, frac_distilled_bge)  — best surface attempt
transport ratio: LOW = transports well (td/fm; .30 median fleet-wide)

Grid (thresholds distill .5 / ratio .5):
  distill>=.5, ratio<=.5  CODIFIED-SURFACE (MECH-adjacent; construct lives on the surface)
  distill<.5,  ratio<=.5  ENCULTURATED (TASTE-adjacent; shared, not surface-reducible)
  distill<.5,  ratio>.5   IDIOSYNCRATIC RESIDUE
  distill>=.5, ratio>.5   OVERFIT-SURFACE (family-specific surface reading)

Usage: python3 e7_provenance_grid.py
-> outputs/metric_seam_pilot/battery/e7_provenance_grid.json
"""
import json, pathlib, sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import BASE  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from certificates import spearman  # noqa: E402


def med(xs):
    xs = sorted(x for x in xs if x is not None and x == x)
    return round(xs[len(xs) // 2], 3) if xs else None


def main():
    tfidf = json.load(open(BASE / "battery/e7_sel_pilot.json"))
    bge = json.load(open(BASE / "battery/e7_sel_pilot_bge.json"))
    inv = json.load(open(BASE / "battery/inventory.json"))
    codif = {}
    for line in open(BASE / "battery/codif/codif_merged.jsonl"):
        r = json.loads(line)
        codif[(r["task"], r["aid"])] = r
    keyev = {}
    for t in tfidf:
        p = BASE / f"battery/key_eval_{t}.json"
        if p.exists():
            keyev[t] = json.load(open(p))

    rows = []
    for task, tv in tfidf.items():
        if "fields" not in tv:
            continue
        for fk, r in tv["fields"].items():
            aid, field = fk.split("__", 1)
            b = bge.get(task, {}).get("fields", {}).get(fk, {})
            ds = [x for x in (r.get("frac_distilled"), b.get("frac_distilled"))
                  if x is not None]
            ratio = r.get("ratio_llama")
            if not ds or ratio is None:
                continue
            cd = codif.get((task, aid), {})
            ke = keyev.get(task, {})
            ker = ke.get(aid) or (ke.get("aspects", {}) or {}).get(aid) or {}
            rows.append({
                "task": task, "aid": aid, "field": field,
                "distill": max(ds), "ratio": ratio,
                "thick": (cd.get("llm_fields") or {}).get(field),
                "c8_share": cd.get("c8_share"),
                "frac_nonce": ker.get("frac_nonce") if isinstance(ker, dict) else None})

    grid = defaultdict(list)
    for r in rows:
        cell = (("CODIFIED-SURFACE" if r["ratio"] <= .5 else "OVERFIT-SURFACE")
                if r["distill"] >= .5 else
                ("ENCULTURATED" if r["ratio"] <= .5 else "IDIOSYNCRATIC"))
        r["cell"] = cell
        grid[cell].append(r)

    print(f"{len(rows)} fields with both coordinates")
    for cell in ("CODIFIED-SURFACE", "ENCULTURATED", "IDIOSYNCRATIC",
                 "OVERFIT-SURFACE"):
        g = grid.get(cell, [])
        bytask = defaultdict(int)
        for r in g:
            bytask[r["task"]] += 1
        ex = ", ".join(f"{r['task'][:2]}/{r['aid']}.{r['field']}" for r in g[:4])
        print(f"{cell:18s} n={len(g):3d}  {dict(bytask)}  e.g. {ex}")

    d = [r["distill"] for r in rows]
    t = [r["ratio"] for r in rows]
    rho_dt = spearman(d, t)
    print(f"Spearman(distill, transport_ratio) = {rho_dt:.3f} (n={len(rows)})")
    fn = [(r["distill"], r["frac_nonce"]) for r in rows if r["frac_nonce"] is not None]
    rho_dn = spearman([x for x, _ in fn], [y for _, y in fn]) if len(fn) >= 8 else None
    if rho_dn is not None:
        print(f"Spearman(distill, frac_nonce)      = {rho_dn:.3f} (n={len(fn)})")

    bythick = defaultdict(list)
    for r in rows:
        if r["thick"]:
            bythick[r["thick"].split(":")[0]].append(r["distill"])
    print("median distill by thick-predicate:")
    for k, v in sorted(bythick.items(), key=lambda kv: -len(kv[1])):
        if len(v) >= 4:
            print(f"  {k:20s} {med(v)}  (n={len(v)})")

    out = {"rows": rows,
           "cells": {c: len(g) for c, g in grid.items()},
           "rho_distill_transport": round(rho_dt, 3) if rho_dt == rho_dt else None,
           "rho_distill_nonce": (round(rho_dn, 3)
                                 if rho_dn is not None and rho_dn == rho_dn else None),
           "distill_by_thick": {k: {"median": med(v), "n": len(v)}
                                for k, v in bythick.items() if len(v) >= 4}}
    path = BASE / "battery/e7_provenance_grid.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
