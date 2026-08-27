"""UNIT->CODE analysis (2026-07-08), run AFTER GPU judging + field extraction.

For each of the 30 humor census units: judge reliability, code floor (best of 3
flavors, ceiling-normalized), hybrid r-tilde, field marginal fm. Grouped by census
type (MECHANICAL / CRAFT / TASTE) to test the two-faces prediction:
  MECHANICAL -> high code floor; CRAFT -> code floor via structure, fm moderate;
  TASTE      -> low code floor, name-indexed (fm carries what code can't).

-> outputs/metric_seam_pilot/battery/units_eval.json
"""
import json, math, pathlib, sys, collections

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_mod, run_prog, BASE, ROOT  # noqa: E402
from eval_hybrids_task import (load_judge, load_fields, split_task_ids,  # noqa: E402
                               FLAVORS)
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman  # noqa: E402
from ops import Ops  # noqa: E402

TD = BASE / "tasks/humor_units"
PU = ROOT / "methods/metric_seam/hybrids/programs_units"


def ceiling(rel, k=2):
    if rel is None or rel != rel or rel <= 0:
        return float("nan")
    return math.sqrt(2 * rel / (1 + rel))  # Spearman-Brown K=2 attenuation ceiling


def rt(ids, col, judge, c):
    s = [d for d in ids if col.get(d) is not None and d in judge]
    if len(s) < 20 or c != c or c <= 0:
        return float("nan")
    r = spearman([col[d] for d in s], [judge[d] for d in s])
    return max(0.0, min(1.0, r / c)) if r == r else float("nan")


def main():
    units = {u["aspect_id"]: u for u in json.load(open(TD / "units_selected.json"))}
    items_l = json.load(open(TD / "items.json"))
    items = {x["datapoint_id"]: x["ctext"] for x in items_l}
    train, test = split_task_ids(items_l)
    judge, rel = load_judge(TD / "results.jsonl")
    fields = load_fields(TD / "field_results.jsonl")
    code = json.load(open(TD / "code_scores.json"))
    ops = Ops(corpus_path=str(TD / "items.json"))

    rows = []
    for uid, u in units.items():
        if uid not in judge:
            continue
        c = ceiling(rel.get(uid))
        # best code flavor by train
        best_tr, base_col = -2, None
        for fl in FLAVORS:
            col = code.get(f"{uid}_{fl}") or {}
            s = [d for d in train if d in judge[uid] and col.get(d) is not None]
            if len(s) < 30:
                continue
            r = spearman([col[d] for d in s], [judge[uid][d] for d in s])
            if r == r and r > best_tr:
                best_tr, base_col = r, col
        if base_col is None:
            continue
        code_rt = rt(test, base_col, judge[uid], c)
        # hybrid
        prog = PU / f"{uid}_h0.py"
        hyb_rt, fm = float("nan"), float("nan")
        if prog.exists():
            fmap = fields.get(uid, {})
            hyb_col = run_prog(load_mod(prog).score, items, fmap, ops)
            blank_col = run_prog(load_mod(prog).score, items, {}, ops)
            hyb_rt = rt(test, hyb_col, judge[uid], c)
            s = [d for d in test if d in judge[uid] and hyb_col.get(d) is not None
                 and blank_col.get(d) is not None]
            if len(s) >= 20:
                fm = (spearman([hyb_col[d] for d in s], [judge[uid][d] for d in s])
                      - spearman([blank_col[d] for d in s], [judge[uid][d] for d in s]))
        rows.append({"uid": uid, "type": u["type"], "n_sources": u["n_sources"],
                     "name": u["name"], "rel1": round(rel.get(uid, float('nan')), 3),
                     "ceiling": round(c, 3) if c == c else None,
                     "code_rt": round(code_rt, 3) if code_rt == code_rt else None,
                     "hyb_rt": round(hyb_rt, 3) if hyb_rt == hyb_rt else None,
                     "fm": round(fm, 3) if fm == fm else None})

    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for m in ("code_rt", "hyb_rt", "fm", "rel1"):
            if r[m] is not None:
                by[r["type"]][m].append(r[m])

    def med(xs):
        return round(sorted(xs)[len(xs) // 2], 3) if xs else None
    summary = {t: {m: med(v) for m, v in d.items()} | {"n": len(d.get("code_rt", []))}
               for t, d in by.items()}
    out = {"summary_by_type": summary, "n_units": len(rows), "per_unit": rows}
    json.dump(out, open(BASE / "battery/units_eval.json", "w"), indent=1)
    print("UNIT->CODE by census type (median):")
    print(f"{'type':12s} {'n':>3s} {'code_rt':>8s} {'hyb_rt':>8s} {'fm':>6s} {'rel1':>6s}")
    for t in ("MECHANICAL", "CRAFT", "TASTE"):
        if t in summary:
            s = summary[t]
            print(f"{t:12s} {s['n']:>3d} {str(s.get('code_rt')):>8s} "
                  f"{str(s.get('hyb_rt')):>8s} {str(s.get('fm')):>6s} {str(s.get('rel1')):>6s}")
    print(f"-> {BASE / 'battery/units_eval.json'}")


if __name__ == "__main__":
    main()
