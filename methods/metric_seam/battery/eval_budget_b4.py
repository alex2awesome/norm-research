"""Budget-4 arm of the field-budget ladder (2026-07-08), run AFTER GPU extraction.

Merges the new b4 fields (battery/b4_field_results.jsonl, keyed <task>::<aid>__<field>)
with each criterion's existing f_orig, runs programs_b4/<task>__<aid>_h4.py, computes the
TEST rho, and writes it into battery/budget_ladder.json as b4_test. Also folds the a108
r2 technique_novelty extraction into agentic_cert if present.

-> updates outputs/metric_seam_pilot/battery/budget_ladder.json (b4_test filled)
"""
import json, pathlib, sys, collections

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, load_mod, run_prog, BASE, ROOT  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import certificates  # noqa: E402
spearman = certificates.spearman
B4 = ROOT / "methods/metric_seam/hybrids/programs_b4"


def load_b4_fields():
    """battery/b4_field_results.jsonl -> {(task,aid): {dpid: {field: val}}}"""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    p = BASE / "battery/b4_field_results.jsonl"
    if not p.exists():
        return out
    for line in open(p):
        r = json.loads(line)
        if r.get("channel") != "field":
            continue
        key, field = r["aspect_id"].split("__", 1)
        task, aid = key.split("::")
        ans = (r.get("raw") or "").strip()
        if ans.upper() == "NONE":
            ans = ""
        out[(task, aid)][r["datapoint_id"]][field] = ans
    return out


def rho_on(ids, col, judge):
    s = [d for d in ids if col.get(d) is not None and d in judge]
    return spearman([col[d] for d in s], [judge[d] for d in s]) if len(s) >= 20 else float("nan")


def main():
    ladder = json.load(open(BASE / "battery/budget_ladder.json"))
    newf = load_b4_fields()
    ctxs = {}
    for key, row in ladder.items():
        task, aid = key.split(".")
        if task not in ctxs:
            ctxs[task] = load_ctx(task)
        ctx = ctxs[task]
        prog = B4 / f"{task}__{aid}_h4.py"
        if not prog.exists():
            continue
        judge = ctx["judge"].get(aid, {})
        # merge existing f_orig with new b4 fields per dpid
        base = ctx["f_orig"].get(aid, {})
        extra = newf.get((task, aid), {})
        merged = {}
        for dpid in set(base) | set(extra):
            merged[dpid] = {**base.get(dpid, {}), **extra.get(dpid, {})}
        col = run_prog(load_mod(prog).score, ctx["items"], merged, ctx["ops"])
        r = rho_on(sorted(ctx["test"]), col, judge)
        row["b4_test"] = round(r, 3) if r == r else None
        row["b4_new_fields"] = sorted({f for d in extra.values() for f in d})
        print(f"{key}: b0={row['b0_test']} b1={row['b1_test']} b2={row['b2_test']} "
              f"b4={row['b4_test']}  (+{row.get('b4_new_fields')})")
    json.dump(ladder, open(BASE / "battery/budget_ladder.json", "w"), indent=1)

    # seam-depth summary: minimal budget to reach 95% of the max-budget rho, per criterion
    def depth(row):
        pts = [(b, row.get(f"b{b}_test")) for b in (0, 1, 2, 4)]
        pts = [(b, v) for b, v in pts if v is not None]
        if not pts:
            return None
        top = max(v for _, v in pts)
        if top <= 0:
            return None
        for b, v in pts:
            if v >= 0.95 * top:
                return b
        return pts[-1][0]
    depths = {k: depth(v) for k, v in ladder.items()}
    by_share = collections.defaultdict(list)
    for k, v in ladder.items():
        d = depths[k]
        if d is not None:
            by_share[v["c8_share"]].append(d)
    print("\nseam depth (min budget for 95% of max rho) by c8_share:")
    for sh, ds in sorted(by_share.items()):
        print(f"  {sh}: median depth {sorted(ds)[len(ds)//2]}  (n={len(ds)}, {sorted(ds)})")
    print(f"-> {BASE / 'battery/budget_ladder.json'}")


if __name__ == "__main__":
    main()
