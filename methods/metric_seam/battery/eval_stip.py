"""E2 STIPULATION-OVERRIDE eval (pre-reg note §2.3) — field-level, no program re-run.

For each E2 field: the extractor was given the construct NAME plus a DEVIANT stipulated
definition (mechanically checkable). Readout on the CONFLICT set (items where the
stipulated answer differs from the certified Gemma answer under the original prompt):
  compliance = frac(answer == stipulated truth)   [followed the stipulation]
  snapback   = frac(answer == original answer)    [retrieved the community meaning]
T-RET predicts snapback mass is substantial; H_spec (prompt-as-program) predicts
compliance ≈ 1. "other" = neither (format drift, refusals).

Usage: python3 eval_stip.py <task>
-> outputs/metric_seam_pilot/battery/stip_eval_<task>.json
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, BASE  # noqa: E402

SAFE_BUILTINS = {"len": len, "any": any, "sum": sum, "min": min, "max": max, "all": all, "abs": abs}


def norm(x):
    return (x or "").strip().strip('."’”').lower()


def main():
    task = sys.argv[1]
    ctx = load_ctx(task)
    inv = json.load(open(BASE / "battery/inventory.json"))[task]
    variants = json.load(open(BASE / "battery" / f"variants_{task}.json"))

    stip_raw = {}
    for line in open(ctx["outdir"] / "battery_results.jsonl"):
        r = json.loads(line)
        a = r["aspect_id"]
        if ".stip__" not in a:
            continue
        head, field = a.split("__", 1)
        aid = head.split(".")[0]
        stip_raw.setdefault((aid, field), {})[r["datapoint_id"]] = norm(r.get("raw"))

    orig_raw = {}
    for aid, per_dp in ctx["f_orig"].items():
        for d, fm in per_dp.items():
            for field, val in fm.items():
                orig_raw.setdefault((aid, field), {})[d] = norm(val)

    report = {}
    for key, v in variants.items():
        if not v.get("e2"):
            continue
        aid, field = v["aid"], v["field"]
        answers = stip_raw.get((aid, field))
        if not answers:
            report[key] = {"error": "no stip extraction rows"}
            continue
        expr = v["e2"]["checker_expr"]
        orig = orig_raw.get((aid, field), {})
        n = comp = snap = other = conflict = agree_cells = 0
        for d, ans in answers.items():
            text = ctx["items"].get(d)
            if text is None:
                continue
            try:
                truth = norm(str(eval(expr, {"__builtins__": SAFE_BUILTINS, "text": text}, {})))
            except Exception:
                continue
            o = orig.get(d)
            n += 1
            if o is None:
                continue
            if truth == o:
                agree_cells += 1
                continue          # uninformative cell: stipulation agrees with community answer
            conflict += 1
            if ans == truth:
                comp += 1
            elif ans == o:
                snap += 1
            else:
                other += 1
        if conflict < 20:
            report[key] = {"error": f"conflict_n={conflict} (rule too aligned or NA-heavy)",
                           "n": n, "agree_cells": agree_cells}
            continue
        report[key] = {
            "construct": v["construct_name"], "rule": v["e2"]["rule_gloss"],
            "n": n, "n_conflict": conflict,
            "compliance": round(comp / conflict, 3),
            "snapback": round(snap / conflict, 3),
            "other": round(other / conflict, 3),
        }
        r = report[key]
        print(f"{key}: conflict={conflict} comply={r['compliance']} "
              f"snapback={r['snapback']} other={r['other']}  [{r['rule'][:50]}]")

    ok = [v for v in report.values() if "error" not in v]
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
    summ = {"n_fields": len(ok),
            "median_compliance": med([v["compliance"] for v in ok]),
            "median_snapback": med([v["snapback"] for v in ok]),
            "n_snapback_gt_compliance": sum(1 for v in ok
                                            if v["snapback"] > v["compliance"]),
            "errors": [k for k, v in report.items() if "error" in v]}
    out = BASE / "battery" / f"stip_eval_{task}.json"
    json.dump({"fields": report, "summary": summ}, open(out, "w"), indent=1)
    print("summary:", json.dumps(summ))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
