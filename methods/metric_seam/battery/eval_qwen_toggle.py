"""E2 reasoning-toggle eval — Qwen3.5-122B, thinking on vs off, same weights.

Isolates the candidate driver of stipulation snap-back (results note §E2-3FAM):
if snap-back comes from reasoning-style generation (model re-derives the community
concept mid-chain and overrides the deviant stipulation), thinking=on should snap back
more than thinking=off WITH IDENTICAL WEIGHTS.

Conflict rule identical to probe_eval/eval_stip: checker-truth != model's own
full-condition answer (own answers taken from the SAME mode: toff full = the certified
transport extraction field_results_qwen.jsonl [thinking off]; ton full = qwen_e2_full5_ton).

Usage: python3 eval_qwen_toggle.py
-> outputs/metric_seam_pilot/battery/qwen_toggle_eval.json
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, load_fields, BASE  # noqa: E402

SAFE_BUILTINS = {"len": len, "any": any, "sum": sum, "min": min, "max": max, "all": all, "abs": abs}

E2FIELDS = ["a25__content_type", "a42__doc_type", "a87__doc_type",
            "a104__doc_kind", "a112__page_kind"]


def norm(x):
    return (x or "").strip().strip('."’”').lower()


def stip_stats(stip_map, full_map, items, expr):
    comp = snap = other = conflict = 0
    for d, ans in stip_map.items():
        text = items.get(d)
        o = full_map.get(d)
        if text is None or o is None or not o:
            continue
        try:
            truth = norm(str(eval(expr, {"__builtins__": SAFE_BUILTINS, "text": text}, {})))
        except Exception:
            continue
        if truth == o:
            continue
        conflict += 1
        if ans == truth:
            comp += 1
        elif ans == o:
            snap += 1
        else:  # fuzzy fallback for long-form answers (containment, unambiguous only)
            c_t = bool(truth) and truth in ans
            c_o = bool(o) and o in ans
            if c_t and not c_o:
                comp += 1
            elif c_o and not c_t:
                snap += 1
            else:
                other += 1
    if conflict < 15:
        return None
    return {"n_conflict": conflict, "compliance": round(comp / conflict, 3),
            "snapback": round(snap / conflict, 3), "other": round(other / conflict, 3)}


def main():
    ctx = load_ctx("press_releases")
    variants = json.load(open(BASE / "battery/variants_press_releases.json"))
    v2 = ctx["outdir"]

    def load_closed(path, overlay=None):
        """Like load_fields but DROPS rows whose thinking never closed —
        an empty answer from truncation/stall is missing data, not a verdict.
        overlay: a second file (4k-budget re-run) whose rows take precedence."""
        raw = {}
        for p in ([path] + ([overlay] if overlay else [])):
            if not p or not p.exists():
                continue
            for line in open(p):
                r = json.loads(line)
                if r.get("channel") != "field":
                    continue
                raw[(r["aspect_id"], r["datapoint_id"])] = r  # later file wins
        out = {}
        for (aspect, d), r in raw.items():
            if r.get("think_unclosed"):
                continue
            aid, field = aspect.split("__", 1)
            ans = (r.get("raw") or "").strip()
            if ans.upper() == "NONE":
                ans = ""
            out.setdefault(aid, {}).setdefault(d, {})[field] = ans
        return out

    stip_toff = load_closed(v2 / "qwen_e2_stip_toff.jsonl")
    stip_ton = load_closed(v2 / "qwen_e2_stip_ton.jsonl",
                           v2 / "qwen_e2_stip_ton2.jsonl")
    full_toff = load_fields(v2 / "field_results_qwen.jsonl")
    full_ton = load_closed(v2 / "qwen_e2_full5_ton.jsonl",
                           v2 / "qwen_e2_full5_ton2.jsonl")

    # thinking-on audit: unclosed </think> rate
    unclosed = total_ton = 0
    for fn in ("qwen_e2_stip_ton.jsonl", "qwen_e2_full5_ton.jsonl"):
        p = v2 / fn
        if p.exists():
            for line in open(p):
                r = json.loads(line)
                total_ton += 1
                unclosed += bool(r.get("think_unclosed"))
    print(f"thinking-on rows {total_ton}, unclosed </think>: {unclosed}")

    report = {"unclosed_rate": round(unclosed / total_ton, 4) if total_ton else None}
    for key in E2FIELDS:
        v = variants.get(key)
        if not v or not v.get("e2"):
            continue
        aid, field = key.split("__", 1)
        expr = v["e2"]["checker_expr"]
        row = {}
        for mode, stip_f, full_f in [("toff", stip_toff, full_toff),
                                     ("ton", stip_ton, full_ton)]:
            sm = {d: norm(fm.get(field))
                  for d, fm in stip_f.get(f"{aid}.stip", {}).items()}
            fm_norm = {d: norm(x.get(field))
                       for d, x in full_f.get(aid, {}).items()}
            st = stip_stats(sm, fm_norm, ctx["items"], expr)
            if st:
                row[mode] = st
        if row:
            report[key] = row
            for m, s in row.items():
                print(f"{key} [{m}]: comply={s['compliance']} "
                      f"snapback={s['snapback']} other={s['other']} "
                      f"(n={s['n_conflict']})")

    out = BASE / "battery/qwen_toggle_eval.json"
    json.dump(report, open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
