"""Certified seam-survey table for a generalized task (survey level: all items, no split).
Usage: python3 analyze_task.py <task>
"""
import json, pathlib, statistics as st, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman, attenuation_ceiling, ceiling_normalized

FLAVORS = ["v0_keyword", "v1_structure", "v2_holistic"]


def main():
    task = sys.argv[1]
    OUT = ROOT / "outputs/metric_seam_pilot/tasks" / task
    aspects = json.load(open(OUT / "aspects_used.json"))
    names = {x["aspect_id"]: x["name"]
             for x in json.load(open(ROOT / "runs/validity_full/v2" / task / "aspects.json"))}
    code = json.load(open(OUT / "code_scores.json"))

    p1, p2, scope = {}, {}, {}
    for line in open(OUT / "results.jsonl"):
        r = json.loads(line)
        if not isinstance(r["score"], int):
            continue
        if r["channel"] == "pass1":
            p1.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
        elif r["channel"] == "pass2":
            p2.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
        elif r["channel"] == "scope":
            scope[r["datapoint_id"]] = r["score"]
    in_scope = {d for d, s in scope.items() if s >= 7}
    n_items = len(json.load(open(OUT / "items.json")))   # not hardcoded (corpora vary: 250/300)
    judge = {}
    for aid in set(p1) | set(p2):
        for d in set(p1.get(aid, {})) | set(p2.get(aid, {})):
            vals = [m[aid][d] for m in (p1, p2) if d in m.get(aid, {})]
            judge.setdefault(aid, {})[d] = sum(vals) / len(vals) / 10.0

    table = []
    print(f"{task}: scope {len(in_scope)}/{n_items} in-scope (mean {st.mean(scope.values()):.1f})"
          if scope else f"{task}: NO scope channel")
    print(f"{'aspect':7} {'rel1':>5} {'ceil':>5} {'rho':>6} {'ρ/ceil':>7} "
          f"{'ρ_scoped':>8} {'NA':>4}  name")
    for aid in aspects:
        both = [d for d in p1.get(aid, {}) if d in p2.get(aid, {})]
        if len(both) < 30:
            table.append({"aspect": aid, "verdict": "degenerate", "n": len(both)})
            print(f"{aid:7} DEGENERATE n={len(both)}  {names.get(aid,'')[:40]}")
            continue
        rel1 = spearman([p1[aid][d] for d in both], [p2[aid][d] for d in both])
        if rel1 != rel1:   # constant pass -> undefined reliability
            table.append({"aspect": aid, "verdict": "degenerate-reliability"})
            print(f"{aid:7} DEGENERATE-REL  {names.get(aid,'')[:40]}")
            continue
        if rel1 <= 0.05:   # unreliable channel: ceiling ~0, normalization meaningless
            table.append({"aspect": aid, "name": names.get(aid, ""),
                          "rel1": round(rel1, 3), "verdict": "unreliable-channel"})
            print(f"{aid:7} {rel1:5.2f} UNRELIABLE-CHANNEL  {names.get(aid,'')[:40]}")
            continue
        ceil = attenuation_ceiling(min(max(rel1, 0.0), 1.0), 2)
        na = n_items - len(judge.get(aid, {}))
        best = (None, float("-inf"))
        for fl in FLAVORS:
            col = code.get(f"{aid}_{fl}")
            if col is None:
                continue
            sel = [d for d in judge.get(aid, {}) if col.get(d) is not None]
            if len(sel) < 30:
                continue
            r = spearman([col[d] for d in sel], [judge[aid][d] for d in sel])
            if r == r and r > best[1]:
                best = (fl, r)
        fl, rho = best
        rho_sc = float("nan")
        if fl:
            col = code[f"{aid}_{fl}"]
            ssel = [d for d in judge.get(aid, {})
                    if col.get(d) is not None and d in in_scope]
            if len(ssel) >= 20:
                rho_sc = spearman([col[d] for d in ssel], [judge[aid][d] for d in ssel])
        norm = ceiling_normalized(rho, max(rel1, 1e-6), 2) if fl else float("nan")
        table.append({"aspect": aid, "name": names.get(aid, ""), "rel1": round(rel1, 3),
                      "ceiling": round(ceil, 3), "best_flavor": fl,
                      "rho": round(rho, 3) if fl else None,
                      "rho_over_ceiling": round(norm, 3) if norm == norm else None,
                      "rho_scoped": round(rho_sc, 3) if rho_sc == rho_sc else None,
                      "na": na})
        print(f"{aid:7} {rel1:5.2f} {ceil:5.2f} "
              f"{(rho if fl else float('nan')):6.2f} {norm:7.2f} {rho_sc:8.2f} {na:4}  "
              f"{names.get(aid, '')[:40]}")
    json.dump({"scope_n": len(in_scope), "table": table},
              open(OUT / "seam_table.json", "w"), indent=1)


if __name__ == "__main__":
    main()
