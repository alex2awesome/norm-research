"""GLM E2-KIND eval (2026-07-08) — deficit-corrected gravity across GLM endpoints.

Mirrors eval_e2kind_task's cell logic but for the 4 GLM versions on the 50-item subsample,
using each version's OWN comm answers as the community baseline. Per field:
  cell4 (nonce+deviant): comply / phantom_snap on the conflict set
  cell6 (nonce+neutral): execution accuracy = pure rule-following CAPACITY
  gravity_effect = cell6.acc - cell4.comply  (capacity minus deviant-compliance)
High gravity_effect = CAN follow a neutral rule but RESISTS the deviant one = genuine
gravity. If gravity_effect ~ 0 at the top endpoint, its low raw snap-back was capacity.

-> outputs/metric_seam_pilot/battery/glm_e2kind_eval.json
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, BASE  # noqa: E402
GE = BASE / "battery/glm_e2kind"
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
    """-> {(task, aid.cell, field): {dpid: norm(raw)}}"""
    p = GE / f"results_{model}.jsonl"
    out = {}
    if not p.exists():
        return out
    for line in open(p):
        r = json.loads(line)
        task, rest = r["aspect_id"].split("::")
        head, field = rest.split("__", 1)  # head = aid.cellX  (aid.comm for comm)
        out.setdefault((task, head, field), {})[r["datapoint_id"]] = norm(r.get("raw"))
    return out


def main():
    kv = {t: {k: v for k, v in json.load(open(
              BASE / f"battery/variants_e2kind_{t}.json")).items()
              if not k.startswith("_")} for t in ("humor", "math")}
    comm_ck = {t: json.load(open(BASE / f"battery/variants_{t}.json"))
               for t in ("humor", "math")}
    items = {t: load_ctx(t)["items"] for t in ("humor", "math")}

    report, summary = {}, {}
    for model in MODELS:
        res = load(model)
        if not res:
            continue
        rows = {}
        for t in ("humor", "math"):
            for key, spec in kv[t].items():
                aid, field = key.split("__", 1)
                comm_expr = comm_ck[t].get(key, {}).get("e2", {}).get("checker_expr")
                if not comm_expr:
                    continue
                own = res.get((t, f"{aid}.comm", field), {})
                if not own:
                    continue
                # community -> cell4-nonce label map (checkers on text)
                lmap = {}
                for d, text in list(items[t].items())[:500]:
                    a = ev(comm_expr, text)
                    b = ev(spec["cell4_checker_expr"], text)
                    if a and b:
                        lmap.setdefault(a, b)
                # cell4 conflict readout
                c4 = res.get((t, f"{aid}.cell4", field), {})
                comp = phant = other = conflict = 0
                for d, ans in c4.items():
                    text = items[t].get(d)
                    o = own.get(d)
                    if not text or not o or not ans:
                        continue
                    t4 = ev(spec["cell4_checker_expr"], text)
                    if t4 is None or lmap.get(o) is None or t4 == lmap[o]:
                        continue
                    conflict += 1
                    if ans == t4 or (t4 in ans and lmap[o] not in ans):
                        comp += 1
                    elif ans == o or ans == lmap.get(o):
                        phant += 1
                    else:
                        other += 1
                row = {"task": t, "nonce_locus": spec.get("nonce_locus")}
                if conflict >= 8:
                    row["cell4"] = {"n": conflict, "comply": round(comp / conflict, 3),
                                    "phantom_snap": round(phant / conflict, 3)}
                for cell in ("cell5", "cell6"):
                    cm = res.get((t, f"{aid}.{cell}", field), {})
                    ok = tot = 0
                    for d, ans in cm.items():
                        text = items[t].get(d)
                        if not text or not ans:
                            continue
                        tr = ev(spec[f"{cell}_checker_expr"], text)
                        if tr is None:
                            continue
                        tot += 1
                        ok += (ans == tr) or (tr in ans)
                    if tot >= 15:
                        row[cell] = {"n": tot, "acc": round(ok / tot, 3)}
                if row.get("cell4") and row.get("cell6"):
                    row["gravity_effect"] = round(
                        row["cell6"]["acc"] - row["cell4"]["comply"], 3)
                rows[key + "@" + t] = row
        report[model] = rows

        def med(xs):
            return round(sorted(xs)[len(xs) // 2], 3) if xs else None
        s = {}
        for tk in ("humor", "math"):
            rs = [r for r in rows.values() if r["task"] == tk]
            s[tk] = {
                "median_comply4": med([r["cell4"]["comply"] for r in rs if r.get("cell4")]),
                "median_acc6_capacity": med([r["cell6"]["acc"] for r in rs if r.get("cell6")]),
                "median_gravity_effect": med([r["gravity_effect"] for r in rs
                                              if r.get("gravity_effect") is not None]),
                "n": sum(1 for r in rs if r.get("gravity_effect") is not None)}
        summary[model] = s

    json.dump({"summary": summary, "per_field": report,
               "note": "gravity_effect = cell6(nonce+neutral capacity).acc - "
                       "cell4(nonce+deviant).comply; own comm baseline per version"},
              open(BASE / "battery/glm_e2kind_eval.json", "w"), indent=1)
    print(f"{'model':9s} | {'HUMOR cap6/comply4/GRAV':>26s} | {'MATH cap6/comply4/GRAV':>24s}")
    for model in MODELS:
        if model not in summary:
            continue
        h, m = summary[model]["humor"], summary[model]["math"]
        def fmt(x):
            return f"{x['median_acc6_capacity']}/{x['median_comply4']}/{x['median_gravity_effect']}"
        print(f"{model:9s} | {fmt(h):>26s} | {fmt(m):>24s}")
    print(f"-> {BASE / 'battery/glm_e2kind_eval.json'}")


if __name__ == "__main__":
    main()
