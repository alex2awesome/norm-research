"""AGENTIC-COMPILE round-2 held-out certification (2026-07-08).

Same machinery as cert_agentic.py, over programs_agentic/*_agentic_r2.py. Compares each
r2 candidate to its h0 on the frozen test split (existing f_orig fields; a108's
technique_novelty served from f_orig if present, else None — the program degrades).

-> outputs/metric_seam_pilot/battery/agentic_cert_r2.json
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, load_mod, run_prog, BASE, ROOT  # noqa: E402
from eval_hybrids_task import paired_boot, FLAVORS  # noqa: E402
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman  # noqa: E402

AG = ROOT / "methods/metric_seam/hybrids/programs_agentic"
SUFFIX = {"legal": "legal_title_vii", "cw": "creative_writing", "math": "math",
          "humor": "humor"}


def parse_name(fname):
    stem = fname[:-len("_agentic_r2.py")]
    for suf, task in SUFFIX.items():
        if stem.endswith(suf):
            return task, stem[:-len(suf)]
    raise ValueError(fname)


def rho_on(sel, col, judge):
    s = [d for d in sel if col.get(d) is not None]
    if len(s) < 20:
        return float("nan"), 0
    return spearman([col[d] for d in s], [judge[d] for d in s]), len(s)


def main():
    cands = sorted(p.name for p in AG.glob("*_agentic_r2.py"))
    report, ctxs = {}, {}
    for fname in cands:
        task, aid = parse_name(fname)
        if task not in ctxs:
            ctxs[task] = load_ctx(task)
        ctx = ctxs[task]
        judge = ctx["judge"].get(aid, {})
        fmap = ctx["f_orig"].get(aid, {})
        cand = load_mod(AG / fname)
        h0 = load_mod(ctx["hyb"] / f"{aid}_h0.py")
        col_c = run_prog(cand.score, ctx["items"], fmap, ctx["ops"])
        col_0 = run_prog(h0.score, ctx["items"], fmap, ctx["ops"])

        cs_path = (ctx["outdir"] / ("code_scores_v2.json" if task == "press_releases"
                                    else "code_scores.json"))
        code = json.load(open(cs_path)) if cs_path.exists() else {}
        train = sorted(ctx["train"])
        best_fl, best_tr, base_col = None, -2, None
        for fl in FLAVORS:
            col = code.get(f"{aid}_{fl}") or {}
            sel = [d for d in train if d in judge and col.get(d) is not None]
            if len(sel) < 30:
                continue
            r = spearman([col[d] for d in sel], [judge[d] for d in sel])
            if r == r and r > best_tr:
                best_fl, best_tr, base_col = fl, r, col

        tr_sel = [d for d in train if d in judge]
        te_sel = [d for d in sorted(ctx["test"]) if d in judge]
        r_tr_c, _ = rho_on(tr_sel, col_c, judge)
        r_tr_0, _ = rho_on(tr_sel, col_0, judge)
        r_te_c, n_te = rho_on(te_sel, col_c, judge)
        r_te_0, _ = rho_on(te_sel, col_0, judge)
        row = {"n_test": n_te,
               "rho_train": {"cand": round(r_tr_c, 4), "h0": round(r_tr_0, 4)},
               "rho_test": {"cand": round(r_te_c, 4), "h0": round(r_te_0, 4)},
               "gap": {"cand": round(r_tr_c - r_te_c, 4), "h0": round(r_tr_0 - r_te_0, 4)},
               "delta_test": round(r_te_c - r_te_0, 4)}
        sel_b = [d for d in te_sel if col_c.get(d) is not None and col_0.get(d) is not None]
        _, p_beat0, _ = paired_boot(sel_b, col_c, col_0, judge, gate_floor=-2, margin=-2)
        row["P_cand_gt_h0"] = p_beat0
        if base_col is not None:
            sel_g = [d for d in sel_b if base_col.get(d) is not None]
            pg_c, _, _ = paired_boot(sel_g, col_c, base_col, judge)
            pg_0, _, _ = paired_boot(sel_g, col_0, base_col, judge)
            r_te_b, _ = rho_on(sel_g, base_col, judge)
            row["gate"] = {"baseline_flavor": best_fl,
                           "rho_test_baseline": round(r_te_b, 3),
                           "P_gate_cand": pg_c, "P_gate_h0": pg_0}
        report[f"{task}.{aid}"] = row
        g = row.get("gate", {})
        print(f"{task}.{aid}: test cand {r_te_c:+.3f} vs h0 {r_te_0:+.3f} "
              f"(train {r_tr_c:+.3f}/{r_tr_0:+.3f}, gap {row['gap']['cand']:+.3f}) "
              f"P(c>h0)={p_beat0} P_gate c/h0={g.get('P_gate_cand')}/{g.get('P_gate_h0')}")

    json.dump(report, open(BASE / "battery/agentic_cert_r2.json", "w"), indent=1)
    # summary
    n_gate_c = sum(1 for r in report.values()
                   if (r.get("gate", {}).get("P_gate_cand") or 0) >= .95)
    n_gate_0 = sum(1 for r in report.values()
                   if (r.get("gate", {}).get("P_gate_h0") or 0) >= .95)
    n_beat = sum(1 for r in report.values() if (r.get("P_cand_gt_h0") or 0) >= .95)
    ds = sorted(r["delta_test"] for r in report.values())
    print(f"\nSUMMARY n={len(report)}  median delta_test={ds[len(ds)//2]:+.3f}  "
          f"cand beats h0 P>=.95: {n_beat}  gate-cert cand/h0: {n_gate_c}/{n_gate_0}")
    print(f"-> {BASE / 'battery/agentic_cert_r2.json'}")


if __name__ == "__main__":
    main()
