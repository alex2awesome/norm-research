#!/usr/bin/env python
"""
Iso-morphism census — how much does ARTICULATING a metric change its behavior?
(2026-07-04, tacit-knowledge scale-out.)

USER GOAL: "find significant amounts of iso-morphism between pairs." Two forms of a
metric are ISO-MORPHIC if they produce interchangeable behavior — a fixed reader
reproduces the metric's own canonical (full-rubric) verdict about equally well from
either form. Iso-morphism between (compact, rich) form pairs = adding the articulation
made no difference (the knowledge was already indexed by the shorter form).

MEASURE (executor-consistent self-readout, the PRIMARY readout). For reader size s and
metric m, the target is the reader's OWN full_rubric verdict M̄_s(x) (reconstruction-only:
never a task label). Each rung form (name / definition / explanation / exemplars / dossier)
is judged and scored for how many bits it transmits about that target. We normalize by
H_self(m,s) = entropy of the own verdict, giving rec(m,s,rung) ∈ [0,1] = fraction of the
metric's own decision recovered by that rung.

WITHIN-size rung SHAPE is the valid comparison (cross-size levels are confounded — each
size targets its own verdict). Per-metric iso-morphism verdicts use the small-reader
average (1B/3B/8B).

Iso-morphism verdicts per metric (target pair defaults to name<->definition, plus the
compact<->rich contrast name<->dossier):
  iso_name_def   : |rec(def) - rec(name)| <= tau         (definition adds ~nothing)
  regressive     : rec(dossier) < rec(name) - tau        (rich articulation HURTS)
  best_rung      : argmax_rung rec                        (where the metric actually lives)

RUNG-CONSTRUCTION CAVEAT (audit 2026-07-04): in the v1 CW/humor grids the `exemplars` rung
is ostension-BY-DESIGN ("Judge by these examples ONLY." + k=2/2 400-ch snippets) and every
`dossier` EMBEDS that block including the ONLY instruction — so `regressive_dossier` on v1
grids measures a self-contradicting prompt, NOT full articulation. Treat exemplars' collapse
as the ostension-fails-census result; do NOT quote regressive_dossier as "articulation hurts"
until grids carry a dossier_v2 rung (examples reframed, no ONLY line). The clean, quotable
shape is name -> definition -> explanation.

Scale: pass --self GRID to point at any task's grid_bits_self-shaped JSON; the census
auto-runs over every domain present. Concept tags (--tags) are optional per domain.

Run:  python -m methods.metric_implementer.experiments.isomorphism_census \
        --self notebooks/data/two_faces_20260702/grid_bits_self.json \
        --mi   notebooks/data/two_faces_20260702/grid_bits.json \
        --tags notebooks/data/two_faces_20260702/concept_tags.json \
        --out  notebooks/data/two_faces_20260702/isomorphism_census.json
"""
import argparse
import json
import numpy as np
from scipy.stats import mannwhitneyu

RUNGS = ["name", "definition", "explanation", "exemplars", "dossier", "dossier_v2"]
# dossier_v2 exists only in post-2026-07-04 grids (the ONLY-line fix); older self-grids
# (CW/humor v1) simply lack the key and every consumer here treats missing rungs as None.
SMALL = ["1B", "3B", "8B"]
TAU = 0.05


def rec(cell, rung):
    h = cell.get("H_self", 0.0)
    if h is None or h <= 0 or rung not in cell:
        return None
    return cell[rung] / h


def summarize(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, float)
    return {"n": len(a), "mean": round(float(a.mean()), 4),
            "median": round(float(np.median(a)), 4),
            "sd": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0}


def mw(a, b):
    a = [x for x in a if x is not None and np.isfinite(x)]
    b = [x for x in b if x is not None and np.isfinite(x)]
    if len(a) < 3 or len(b) < 3:
        return {"n_a": len(a), "n_b": len(b), "note": "too few"}
    try:
        U, p = mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return {"n_a": len(a), "n_b": len(b), "note": "degenerate"}
    return {"n_a": len(a), "n_b": len(b), "p": round(float(p), 4),
            "rank_biserial": round(float(1 - 2 * U / (len(a) * len(b))), 3)}


def small_avg(sg, m, rung, sizes):
    vals = [rec(sg[s][m], rung) for s in sizes
            if s in sg and m in sg[s] and rec(sg[s][m], rung) is not None]
    return float(np.mean(vals)) if vals else None


def census_domain(sg, tags_for_dom):
    sizes_present = [s for s in ["1B", "3B", "8B", "70B"] if s in sg]
    small = [s for s in SMALL if s in sg]
    ref_size = small[0] if small else sizes_present[0]
    metrics = sorted(sg[ref_size].keys(), key=int)

    # per-size curve (within-size shape is the valid comparison)
    per_size = {}
    for s in sizes_present:
        row = {}
        for rung in RUNGS:
            vals = [rec(sg[s][m], rung) for m in metrics
                    if m in sg[s] and rec(sg[s][m], rung) is not None]
            row[rung] = round(float(np.mean(vals)), 4) if vals else None
        row["n"] = len([m for m in metrics if m in sg[s] and "name" in sg[s][m]])
        per_size[s] = row

    # small-reader-averaged curve
    curve = {rung: round(float(np.mean([v for v in
             (small_avg(sg, m, rung, small) for m in metrics) if v is not None])), 4)
             for rung in RUNGS}

    # per-metric verdicts on the small-reader average
    per_metric = {}
    for m in metrics:
        r = {rung: small_avg(sg, m, rung, small) for rung in RUNGS}
        if r["name"] is None or r["definition"] is None:
            per_metric[m] = {"skip": True, **r}
            continue
        best = max((rung for rung in RUNGS if r[rung] is not None),
                   key=lambda k: r[k])
        gap_def = r["definition"] - r["name"]
        gap_doss = (r["dossier"] - r["name"]) if r["dossier"] is not None else None
        per_metric[m] = {
            "name": round(r["name"], 4), "definition": round(r["definition"], 4),
            "dossier": round(r["dossier"], 4) if r["dossier"] is not None else None,
            "best_rung": best, "gap_def_name": round(gap_def, 4),
            "gap_dossier_name": round(gap_doss, 4) if gap_doss is not None else None,
            "iso_name_def": bool(abs(gap_def) <= TAU),
            "regressive_dossier": bool(gap_doss is not None and gap_doss < -TAU),
        }
        if tags_for_dom and int(m) in tags_for_dom:
            per_metric[m]["label"] = tags_for_dom[int(m)]["label"]

    live = [m for m in metrics if not per_metric[m].get("skip")]
    n = len(live)
    agg = {
        "n_metrics": n,
        "iso_name_def_fraction": round(sum(per_metric[m]["iso_name_def"] for m in live) / n, 3),
        "regressive_dossier_fraction": round(
            sum(per_metric[m]["regressive_dossier"] for m in live) / n, 3),
        "best_rung_dist": {rung: sum(per_metric[m]["best_rung"] == rung for m in live)
                           for rung in RUNGS},
        "gap_def_name": summarize([per_metric[m]["gap_def_name"] for m in live]),
        "gap_dossier_name": summarize([per_metric[m]["gap_dossier_name"] for m in live]),
        "name_frac": summarize([per_metric[m]["name"] for m in live]),
    }

    # by concept type
    by_type = None
    if tags_for_dom:
        buckets = {"TASTE": [], "STRUCTURAL_CRAFT": []}
        name_by = {"TASTE": [], "STRUCTURAL_CRAFT": []}
        for m in live:
            lab = per_metric[m].get("label")
            if lab in buckets:
                buckets[lab].append(per_metric[m]["gap_def_name"])
                name_by[lab].append(per_metric[m]["name"])
        by_type = {
            "taste_gap_def_name": summarize(buckets["TASTE"]),
            "craft_gap_def_name": summarize(buckets["STRUCTURAL_CRAFT"]),
            "taste_name_frac": summarize(name_by["TASTE"]),
            "craft_name_frac": summarize(name_by["STRUCTURAL_CRAFT"]),
            "mw_gap_taste_vs_craft": mw(buckets["TASTE"], buckets["STRUCTURAL_CRAFT"]),
            "mw_namefrac_taste_vs_craft": mw(name_by["TASTE"], name_by["STRUCTURAL_CRAFT"]),
        }

    return {"sizes": sizes_present, "per_size_curve": per_size,
            "small_reader_curve": curve, "aggregate": agg,
            "by_concept_type": by_type, "per_metric": per_metric}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self", dest="self_grid", required=True)
    ap.add_argument("--mi", dest="mi_grid", default=None)
    ap.add_argument("--tags", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    self_grid = json.load(open(a.self_grid))
    tags = None
    if a.tags:
        raw = json.load(open(a.tags))["tags"]
        tags = {}
        for t in raw:
            tags.setdefault(t["domain"], {})[t["gi"]] = t

    report = {"tau": TAU, "primary": "executor-consistent self-readout (name..dossier)",
              "note": "rec = rung_bits / H_self (fraction of own full-rubric verdict "
                      "recovered). WITHIN-size shape is valid; per-metric uses small-reader avg.",
              "domains": {}}
    for dom, sg in self_grid.items():
        report["domains"][dom] = census_domain(sg, tags.get(dom) if tags else None)

    # pooled across domains that have concept tags
    pooled = {"taste_gap": [], "craft_gap": [], "taste_name": [], "craft_name": []}
    n_iso = [0, 0]
    n_reg = [0, 0]
    for dom, d in report["domains"].items():
        for m, v in d["per_metric"].items():
            if v.get("skip"):
                continue
            n_iso[1] += 1
            n_iso[0] += v["iso_name_def"]
            n_reg[1] += 1
            n_reg[0] += v["regressive_dossier"]
            lab = v.get("label")
            if lab == "TASTE":
                pooled["taste_gap"].append(v["gap_def_name"]); pooled["taste_name"].append(v["name"])
            elif lab == "STRUCTURAL_CRAFT":
                pooled["craft_gap"].append(v["gap_def_name"]); pooled["craft_name"].append(v["name"])
    report["pooled"] = {
        "n_total": n_iso[1],
        "iso_name_def_fraction": round(n_iso[0] / n_iso[1], 3) if n_iso[1] else None,
        "regressive_dossier_fraction": round(n_reg[0] / n_reg[1], 3) if n_reg[1] else None,
        "taste_gap_def_name": summarize(pooled["taste_gap"]),
        "craft_gap_def_name": summarize(pooled["craft_gap"]),
        "mw_gap_taste_vs_craft": mw(pooled["taste_gap"], pooled["craft_gap"]),
        "mw_namefrac_taste_vs_craft": mw(pooled["taste_name"], pooled["craft_name"]),
    }

    json.dump(report, open(a.out, "w"), indent=1)

    # console
    print(f"ISO-MORPHISM CENSUS (tau={TAU})  ->  {a.out}\n")
    for dom, d in report["domains"].items():
        c = d["small_reader_curve"]; ag = d["aggregate"]
        print(f"[{dom}] n={ag['n_metrics']}  curve(small-avg): " +
              "  ".join(f"{r[:4]} {c[r]:.3f}" for r in RUNGS))
        print(f"   iso(name~def)={ag['iso_name_def_fraction']}  "
              f"regressive(dossier<name)={ag['regressive_dossier_fraction']}  "
              f"best_rung={ag['best_rung_dist']}")
        if d["by_concept_type"]:
            bt = d["by_concept_type"]
            print(f"   name_frac TASTE={bt['taste_name_frac'].get('mean')} "
                  f"CRAFT={bt['craft_name_frac'].get('mean')}  "
                  f"MW(gap)={bt['mw_gap_taste_vs_craft']}")
    pl = report["pooled"]
    print(f"\n[POOLED] n={pl['n_total']}  iso(name~def)={pl['iso_name_def_fraction']}  "
          f"regressive={pl['regressive_dossier_fraction']}")
    print(f"   MW gap taste-vs-craft: {pl['mw_gap_taste_vs_craft']}")


if __name__ == "__main__":
    main()
