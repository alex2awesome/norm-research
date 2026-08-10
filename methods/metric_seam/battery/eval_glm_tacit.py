"""GLM-family tacitity eval (2026-07-08) — semantic gravity across GLM endpoints.

For each GLM version's results_<model>.jsonl (comm + stip conditions): on the CONFLICT
set (items where the deviant-stipulated truth != that version's OWN community answer),
  compliance = frac(stip answer == stipulated truth)   [followed the rule]
  snapback   = frac(stip answer == own community answer) [retrieved community meaning]
Snapback = the tacitness signature. Reported per version and split thick(humor)/thin(math).
Comparable in SHAPE to the Llama-ladder / Qwen-toggle e2 numbers.

-> outputs/metric_seam_pilot/battery/glm_tacit_eval.json
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx  # noqa: E402
BASE = pathlib.Path(__file__).resolve().parents[3] / "outputs/metric_seam_pilot"
GT = BASE / "battery/glm_tacit"
SAFE = {"len": len, "any": any, "sum": sum, "min": min, "max": max, "all": all, "abs": abs}
MODELS = ["glm-4.5", "glm-4.6", "glm-4.7", "glm-5.2"]


def norm(x):
    return (x or "").strip().strip('."’”').lower()


def ev(expr, text):
    try:
        return norm(str(eval(expr, {"__builtins__": SAFE, "text": text}, {})))
    except Exception:
        return None


def load(model):
    """results_<model>.jsonl -> {(task,aid,field,cond): {dpid: norm(raw)}}"""
    p = GT / f"results_{model}.jsonl"
    out = {}
    if not p.exists():
        return out
    for line in open(p):
        r = json.loads(line)
        head, field = r["aspect_id"].split("__", 1)
        task, rest = head.split("::")
        aid, cond = rest.rsplit(".", 1)
        out.setdefault((task, aid, field, cond), {})[r["datapoint_id"]] = norm(r.get("raw"))
    return out


def main():
    checkers = json.load(open(GT / "checkers.json"))
    items = {t: load_ctx(t)["items"] for t in ("humor", "math")}
    report, summary = {}, {}
    for model in MODELS:
        res = load(model)
        if not res:
            continue
        rows = {}
        for ck, meta in checkers.items():
            task, aid, field = meta["task"], meta["aid"], meta["field"]
            expr = meta["checker_expr"]
            comm = res.get((task, aid, field, "comm"), {})
            stip = res.get((task, aid, field, "stip"), {})
            comp = snap = other = conflict = agree = 0
            for d, ans in stip.items():
                text = items[task].get(d)
                o = comm.get(d)
                if text is None or not o or not ans:
                    continue
                truth = ev(expr, text)
                if truth is None:
                    continue
                if truth == o:
                    agree += 1
                    continue
                conflict += 1
                if ans == truth or (truth in ans and o not in ans):
                    comp += 1
                elif ans == o or (o in ans):
                    snap += 1
                else:
                    other += 1
            if conflict >= 10:
                rows[ck] = {"task": task, "n_conflict": conflict,
                            "compliance": round(comp / conflict, 3),
                            "snapback": round(snap / conflict, 3),
                            "other": round(other / conflict, 3)}
        report[model] = rows

        def med(xs):
            return round(sorted(xs)[len(xs) // 2], 3) if xs else None
        s = {}
        for tk in ("humor", "math", "all"):
            rs = [r for r in rows.values() if tk == "all" or r["task"] == tk]
            s[tk] = {"n_fields": len(rs),
                     "median_compliance": med([r["compliance"] for r in rs]),
                     "median_snapback": med([r["snapback"] for r in rs]),
                     "n_snap_gt_comply": sum(1 for r in rs
                                             if r["snapback"] > r["compliance"])}
        summary[model] = s

    json.dump({"summary": summary, "per_field": report,
               "note": "snapback vs each GLM version's OWN community answer; "
                       "conflict set only; humor=thick, math=thin"},
              open(BASE / "battery/glm_tacit_eval.json", "w"), indent=1)
    print(f"{'model':9s} | {'humor comply/snap':>20s} | {'math comply/snap':>18s} | "
          f"{'snap>comply (h/m)':>17s}")
    for model in MODELS:
        if model not in summary:
            continue
        s = summary[model]
        h, m = s["humor"], s["math"]
        print(f"{model:9s} | {str(h['median_compliance'])+'/'+str(h['median_snapback']):>20s} | "
              f"{str(m['median_compliance'])+'/'+str(m['median_snapback']):>18s} | "
              f"{str(h['n_snap_gt_comply'])+'/'+str(h['n_fields'])+' , '+str(m['n_snap_gt_comply'])+'/'+str(m['n_fields']):>17s}")
    print(f"-> {BASE / 'battery/glm_tacit_eval.json'}")


if __name__ == "__main__":
    main()
