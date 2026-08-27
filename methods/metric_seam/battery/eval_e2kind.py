"""E2-KIND grid eval (seam note §7) — cells 4/5/6 across extractors.

Per extractor with e2kind_results_<ext>.jsonl:
  cell4 nonce+deviant: on the conflict set (deviant truth != nonce-image of the
    extractor's own community answer): comply (matches deviant truth, nonce vocab),
    phantom_snap (outputs its community answer anyway), other.
  cell5 name+neutral / cell6 nonce+neutral: accuracy vs fresh-label checker truth on
    all items = pure rule-execution capacity (deficit control).
  gravity_effect := acc(cell6) - comply(cell4)   [same nonce framing, neutral vs deviant]

Label mapping community->nonce derived EMPIRICALLY per field by evaluating both the
original e2 checker and the cell4 checker on the same items (rules are semantically
identical, so pairs define the bijection).

Own community answers per extractor: gemma f_orig, llama3b/8b/70 field_results_llamaX,
qwen_toff field_results_qwen, qwen_ton qwen_e2_full5_ton(+ton2, closed rows only).

Usage: python3 eval_e2kind.py
-> outputs/metric_seam_pilot/battery/e2kind_eval.json
"""
import json, pathlib, sys
from collections import defaultdict

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


def load_closed(path, overlay=None):
    raw = {}
    for p in ([path] + ([overlay] if overlay else [])):
        if not p or not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            if r.get("channel") != "field":
                continue
            raw[(r["aspect_id"], r["datapoint_id"])] = r
    out = {}
    for (aspect, d), r in raw.items():
        if r.get("think_unclosed"):
            continue
        aid, field = aspect.split("__", 1)
        ans = (r.get("raw") or "").strip()
        out.setdefault(aid, {}).setdefault(d, {})[field] = ans
    return out


def main():
    ctx = load_ctx("press_releases")
    kv = json.load(open(BASE / "battery/variants_e2kind_pr.json"))
    variants = json.load(open(BASE / "battery/variants_press_releases.json"))
    v2 = ctx["outdir"]

    own = {"gemma": ctx["f_orig"],
           "llama3b": load_fields(v2 / "field_results_llama3b.jsonl"),
           "llama8b": load_fields(v2 / "field_results_llama8b.jsonl"),
           "llama70": load_fields(v2 / "field_results_llama.jsonl"),
           "qwen_toff": load_fields(v2 / "field_results_qwen.jsonl"),
           "qwen_ton": load_closed(v2 / "qwen_e2_full5_ton.jsonl",
                                   v2 / "qwen_e2_full5_ton2.jsonl")}

    report = {}
    for ext in ("gemma", "llama3b", "llama8b", "llama70", "qwen_toff", "qwen_ton"):
        p = v2 / f"e2kind_results_{ext}.jsonl"
        if not p.exists():
            continue
        res = load_closed(p) if ext == "qwen_ton" else load_fields(p)
        # load_fields keys "{aid}.{cell}" -> {d: {field: ans}}; load_closed likewise
        rows = {}
        for key, v in kv.items():
            aid, field = key.split("__", 1)
            e2 = variants.get(key, {}).get("e2", {})
            own_map = {d: norm(x.get(field))
                       for d, x in own[ext].get(aid, {}).items()
                       if x.get(field)}
            # empirical community->nonce label map
            lmap = {}
            for d, text in list(ctx["items"].items())[:500]:
                a = ev(e2.get("checker_expr", "None"), text)
                b = ev(kv[key]["cell4_checker_expr"], text)
                if a and b:
                    lmap.setdefault(a, b)
            row = {}
            # cell4
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
                if ans == t4:
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
            # cells 5/6: accuracy vs fresh-label truth
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
                  f"(n={c4s.get('n_conflict')}) | "
                  f"exec5={row.get('cell5', {}).get('acc')} "
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
                                   if r.get("gravity_effect") is not None])}
        print(f"== {ext}: {json.dumps(summ[ext])}")
    out = BASE / "battery/e2kind_eval.json"
    json.dump({"per_extractor": report, "summary": summ}, open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
