"""WS6.1 readout: a42 (math) h1-vs-h0 on the expanded held-out set (see
build_a42_expansion.py header for the FROZEN resolution rule).

Combined held-out = 100 original test items (judge/fields from the frozen fleet run) +
400 fresh items (judge/fields from a42_expansion_results.jsonl). Judges intersection-only
2-pass mean /10. Paired bootstrap B=2000 over the combined set:
  P(h1>h0), P(rho_h1 >= .60), plus per-pool rhos for transparency.
Usage: python3 eval_a42_expansion.py   (after the sk3 Gemma pass is rsynced back)
"""
import importlib.util
import json
import pathlib
import random
import re
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from battery_common import load_ctx, run_prog, ROOT  # noqa: E402
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman  # noqa: E402

OUT = ROOT / "outputs/metric_seam_pilot/tasks/math"
HYB = ROOT / "methods/metric_seam/hybrids/programs_math"
B = 2000


def clean(raw):
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    line = re.sub(r"^(answer|reply)\s*[:\-]\s*", "", line, flags=re.I).strip()
    return "" if line.upper().startswith("NONE") else line[:200]


def load_mod(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p1, p2, fields = {}, {}, {}
    for line in open(OUT / "a42_expansion_results.jsonl"):
        r = json.loads(line)
        if r["channel"] == "field":
            fields.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = clean(r.get("raw", ""))
        elif r["channel"] in ("pass1", "pass2") and isinstance(r["score"], int):
            (p1 if r["channel"] == "pass1" else p2)[r["datapoint_id"]] = r["score"]
    both = sorted(set(p1) & set(p2))
    fresh_judge = {d: (p1[d] + p2[d]) / 2 / 10.0 for d in both}
    rel1 = spearman([p1[d] for d in both], [p2[d] for d in both])
    print(f"fresh: {len(both)} 2-pass judged (rel1={rel1:.3f})")

    ctx = load_ctx("math")
    old_test = [d for d in ctx["test"] if d in ctx["judge"].get("a42", {})]
    fresh_items = {it["datapoint_id"]: it["ctext"]
                   for it in json.load(open(OUT / "a42_expansion_items.json"))}

    judge = {**{d: ctx["judge"]["a42"][d] for d in old_test}, **fresh_judge}
    fresh_fmap = {d: {a.split("__", 1)[1]: fields[a].get(d, "")
                      for a in fields} for d in fresh_items}
    # old-item field maps are PER-PROGRAM: h0's fields were extracted in the original
    # fleet pass (f_orig, 'a42__<f>'), h1's NEW fields in field_results_h1.jsonl
    # ('a42.h1__<f>') — feeding h1 the h0 map collapses it (caught 2026-07-10).
    h1f = {}
    for line in open(OUT / "field_results_h1.jsonl"):
        r = json.loads(line)
        if r["aspect_id"].startswith("a42.h1__"):
            h1f.setdefault(r["datapoint_id"], {})[
                r["aspect_id"].split("__", 1)[1]] = clean(r.get("raw", ""))
    old_fmap = {"a42_h0": ctx["f_orig"].get("a42", {}), "a42_h1": h1f}

    cols = {}
    for name in ("a42_h0", "a42_h1"):
        mod = load_mod(HYB / f"{name}.py")
        c_old = run_prog(mod.score, {d: ctx["items"][d] for d in old_test},
                         old_fmap[name], ctx["ops"])
        c_new = run_prog(mod.score, fresh_items, fresh_fmap, ctx["ops"])
        cols[name] = {**c_old, **c_new}

    sel = [d for d in judge if cols["a42_h0"].get(d) is not None
           and cols["a42_h1"].get(d) is not None]
    sel_old = [d for d in sel if d in set(old_test)]
    sel_new = [d for d in sel if d not in set(old_test)]
    for tag, ss in (("combined", sel), ("old-100", sel_old), ("fresh", sel_new)):
        r0 = spearman([cols["a42_h0"][d] for d in ss], [judge[d] for d in ss])
        r1 = spearman([cols["a42_h1"][d] for d in ss], [judge[d] for d in ss])
        print(f"{tag:9} n={len(ss):3}  rho h0={r0:.4f}  h1={r1:.4f}  d={r1 - r0:+.4f}")

    rng = random.Random(0)
    wins = floor_ok = 0
    for _ in range(B):
        s = [sel[rng.randrange(len(sel))] for _ in sel]
        j = [judge[d] for d in s]
        r0 = spearman([cols["a42_h0"][d] for d in s], j)
        r1 = spearman([cols["a42_h1"][d] for d in s], j)
        wins += r1 > r0
        floor_ok += r1 >= 0.60
    p_beat, p_floor = wins / B, floor_ok / B
    promote = p_beat >= 0.80 and p_floor >= 0.95
    print(f"P(h1>h0)={p_beat:.3f} (need >=.80)   P(rho_h1>=.60)={p_floor:.3f} (need >=.95)")
    print("VERDICT: " + ("PROMOTE h1" if promote else "h0 STAYS HEAD"))
    json.dump({"rel1_fresh": round(rel1, 3), "n_combined": len(sel),
               "P_h1_beats_h0": p_beat, "P_floor": p_floor,
               "verdict": "promote_h1" if promote else "h0_stays"},
              open(OUT / "a42_expansion_verdict.json", "w"), indent=1)


if __name__ == "__main__":
    main()
