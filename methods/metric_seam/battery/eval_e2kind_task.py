"""E2-KIND grid eval, task-generalized (scale-up of eval_e2kind.py; PR keeps the
original script).

Per extractor with tasks/<task>/e2kind_results_<ext>.jsonl: cell4 comply/phantom on
the conflict set, cell5/6 execution accuracy, gravity = exec6 - comply4. Empirical
community->nonce label map from evaluating both checkers on the same items.

Own community answers per extractor: gemma f_orig, llama3b/8b/70 field_results_*,
qwen_toff field_results_qwen. qwen_ton has NO task-level full-condition extraction:
own answers PROXIED from qwen_toff (same weights, different mode) and marked as such.

Usage: python3 eval_e2kind_task.py <task>
-> outputs/metric_seam_pilot/battery/e2kind_eval_<task>.json
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, load_fields, BASE  # noqa: E402

SAFE_BUILTINS = {"len": len, "any": any, "sum": sum, "min": min, "max": max,
                 "all": all, "abs": abs}


def norm(x):
    return (x or "").strip().strip('."’”').lower()


def ev(expr, text):
    try:
        return norm(str(eval(expr, {"__builtins__": SAFE_BUILTINS, "text": text}, {})))
    except Exception:
        return None


def load_closed(path):
    out = {}
    if not path.exists():
        return out
    for line in open(path):
        r = json.loads(line)
        if r.get("channel") != "field" or r.get("think_unclosed"):
            continue
        aid, field = r["aspect_id"].split("__", 1)
        out.setdefault(aid, {}).setdefault(r["datapoint_id"], {})[field] = \
            (r.get("raw") or "").strip()
    return out


def main():
    task = sys.argv[1]
    ctx = load_ctx(task)
    kv = {k: v for k, v in json.load(open(
        BASE / f"battery/variants_e2kind_{task}.json")).items()
        if not k.startswith("_")}
    variants = json.load(open(BASE / f"battery/variants_{task}.json"))
    td = ctx["outdir"]

    own = {"gemma": ctx["f_orig"],
           "llama3b": load_fields(td / "field_results_llama3b.jsonl"),
           "llama8b": load_fields(td / "field_results_llama8b.jsonl"),
           "llama70": load_fields(td / "field_results_llama.jsonl"),
           "qwen_toff": load_fields(td / "field_results_qwen.jsonl"),
           "qwen_ton": load_fields(td / "field_results_qwen.jsonl")}  # PROXY (toff)

    report = {}
    for ext in ("gemma", "llama3b", "llama8b", "llama70", "qwen_toff", "qwen_ton"):
        p = td / f"e2kind_results_{ext}.jsonl"
        if not p.exists():
            continue
        res = load_closed(p) if ext == "qwen_ton" else load_fields(p)
        # extension runs (e.g. re-authored fields) live in *_ext files; disjoint aids
        pe = td / f"e2kind_results_{ext}_ext.jsonl"
        if pe.exists():
            res.update(load_closed(pe) if ext == "qwen_ton" else load_fields(pe))
        rows = {}
        for key, v in kv.items():
            aid, field = key.split("__", 1)
            # community-side reference: re-authored fields carry their own checker
            # (the original e2 checker is corpus-degenerate for exactly those)
            comm_expr = kv[key].get("community_checker_expr") or \
                variants.get(key, {}).get("e2", {}).get("checker_expr", "None")
            own_map = {d: norm(x.get(field))
                       for d, x in own[ext].get(aid, {}).items() if x.get(field)}
            lmap = {}
            for d, text in list(ctx["items"].items())[:500]:
                a = ev(comm_expr, text)
                b = ev(kv[key]["cell4_checker_expr"], text)
                if a and b:
                    lmap.setdefault(a, b)
            row = {}
            c4 = res.get(f"{aid}.cell4", {})
            comp = phant = other = conflict = 0
            for d, x in c4.items():
                ans = norm(x.get(field))
                text = ctx["items"].get(d)
                o = own_map.get(d)
                if not text or not o or not ans:
                    continue
                t4 = ev(kv[key]["cell4_checker_expr"], text)
                if t4 is None or lmap.get(o) is None or t4 == lmap[o]:
                    continue
                conflict += 1
                if ans == t4 or (t4 in ans and lmap[o] not in ans):
                    comp += 1
                elif ans == o or ans == lmap.get(o):
                    phant += 1
                else:
                    other += 1
            if conflict >= 15:
                row["cell4"] = {"n_conflict": conflict,
                                "comply": round(comp / conflict, 3),
                                "phantom_snap": round(phant / conflict, 3),
                                "other": round(other / conflict, 3)}
            for cell in ("cell5", "cell6"):
                cm = res.get(f"{aid}.{cell}", {})
                ok = tot = 0
                for d, x in cm.items():
                    ans = norm(x.get(field))
                    text = ctx["items"].get(d)
                    if not text or not ans:
                        continue
                    t = ev(kv[key][f"{cell}_checker_expr"], text)
                    if t is None:
                        continue
                    tot += 1
                    ok += (ans == t) or (t in ans)
                if tot >= 30:
                    row[cell] = {"n": tot, "acc": round(ok / tot, 3)}
            if row.get("cell4") and row.get("cell6"):
                row["gravity_effect"] = round(
                    row["cell6"]["acc"] - row["cell4"]["comply"], 3)
            row["nonce_locus"] = kv[key].get("nonce_locus")
            rows[key] = row
            c4s = row.get("cell4", {})
            print(f"[{ext}] {key} ({row['nonce_locus']}): "
                  f"c4 comply={c4s.get('comply')} phantom={c4s.get('phantom_snap')} "
                  f"(n={c4s.get('n_conflict')}) | exec5={row.get('cell5', {}).get('acc')} "
                  f"exec6={row.get('cell6', {}).get('acc')} "
                  f"gravity={row.get('gravity_effect')}")
        report[ext] = rows

    med = lambda xs: (sorted(xs)[len(xs) // 2] if xs else None)
    summ = {}
    for ext, rows in report.items():
        summ[ext] = {
            "median_c4_comply": med([r["cell4"]["comply"] for r in rows.values()
                                     if r.get("cell4")]),
            "median_c4_phantom": med([r["cell4"]["phantom_snap"] for r in rows.values()
                                      if r.get("cell4")]),
            "median_exec5": med([r["cell5"]["acc"] for r in rows.values()
                                 if r.get("cell5")]),
            "median_exec6": med([r["cell6"]["acc"] for r in rows.values()
                                 if r.get("cell6")]),
            "median_gravity": med([r["gravity_effect"] for r in rows.values()
                                   if r.get("gravity_effect") is not None]),
            "n_gradable_c4": sum(1 for r in rows.values() if r.get("cell4"))}
        print(f"== {ext}: {json.dumps(summ[ext])}")
    out = BASE / f"battery/e2kind_eval_{task}.json"
    json.dump({"per_extractor": report, "summary": summ,
               "note": "qwen_ton own-answers proxied from qwen_toff"},
              open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
