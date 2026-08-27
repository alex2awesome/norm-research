"""Field-budget ladder (seam expansion, 2026-07-08) — CPU arms: budgets 0/1/2.

For each sampled criterion (battery/ladder_sample.json): run its h0 hybrid with
b0 = all fields blanked, b1 = exactly one field served (each field tried; selected
on TRAIN), b2 = full extraction. TEST Spearman per arm. Budget-4 arm (h4 programs +
2 new fields) is filled in later by eval_budget_ladder_b4.py after GPU extraction.

-> outputs/metric_seam_pilot/battery/budget_ladder.json
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, load_mod, run_prog, BASE  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import certificates  # noqa: E402
spearman = certificates.spearman


def rho_on(ids, col, judge):
    s = [d for d in ids if col.get(d) is not None and d in judge]
    return (spearman([col[d] for d in s], [judge[d] for d in s]), len(s)) \
        if len(s) >= 20 else (float("nan"), len(s))


def sub_fields(fmap, keep):
    return {d: {k: v for k, v in row.items() if k in keep}
            for d, row in fmap.items()}


def main():
    sample = json.load(open(BASE / "battery/ladder_sample.json"))
    ctxs = {}
    out = {}
    for s in sample:
        t, aid = s["task"], s["aid"]
        if t not in ctxs:
            ctxs[t] = load_ctx(t)
        ctx = ctxs[t]
        judge = ctx["judge"].get(aid, {})
        fmap = ctx["f_orig"].get(aid, {})
        pm = load_mod(ctx["hyb"] / f"{aid}_h0.py")
        fields = sorted(getattr(pm, "LLM_FIELDS", {}) or {})
        train, test = sorted(ctx["train"]), sorted(ctx["test"])

        col_b0 = run_prog(pm.score, ctx["items"], {}, ctx["ops"])
        col_b2 = run_prog(pm.score, ctx["items"], fmap, ctx["ops"])
        r0_te, n0 = rho_on(test, col_b0, judge)
        r2_te, n2 = rho_on(test, col_b2, judge)

        b1 = {}
        best_f, best_tr = None, -2
        for f in fields:
            col = run_prog(pm.score, ctx["items"], sub_fields(fmap, {f}), ctx["ops"])
            r_tr, _ = rho_on(train, col, judge)
            r_te, _ = rho_on(test, col, judge)
            b1[f] = {"train": round(r_tr, 3) if r_tr == r_tr else None,
                     "test": round(r_te, 3) if r_te == r_te else None}
            if r_tr == r_tr and r_tr > best_tr:
                best_f, best_tr = f, r_tr
        r1_te = b1.get(best_f, {}).get("test")

        out[f"{t}.{aid}"] = {
            "c8_share": s["c8_share"], "fm_ref": s["fm"], "fields": fields,
            "b0_test": round(r0_te, 3) if r0_te == r0_te else None,
            "b1_test": r1_te, "b1_field": best_f, "b1_all": b1,
            "b2_test": round(r2_te, 3) if r2_te == r2_te else None,
            "n_test": n2, "b4_test": None}
        print(f"{t}.{aid} [{s['c8_share']}] b0={out[f'{t}.{aid}']['b0_test']} "
              f"b1={r1_te} ({best_f}) b2={out[f'{t}.{aid}']['b2_test']}")
    json.dump(out, open(BASE / "battery/budget_ladder.json", "w"), indent=1)
    print(f"-> {BASE / 'battery/budget_ladder.json'}")


if __name__ == "__main__":
    main()
