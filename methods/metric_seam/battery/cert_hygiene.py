"""Recertify hygiene-patched programs on the held-out test split.

For every entry in battery/hygiene_manifest.json (hygiene file -> (task, source
program)): score patched vs original on TEST, paired bootstrap P(patched>orig), and
G1 gate P_gate for both where a codegen baseline exists. Bug fixes are kept on
correctness grounds — rho deltas are REPORTED, not used to accept/reject.

Usage: python3 cert_hygiene.py
-> outputs/metric_seam_pilot/battery/hygiene_cert.json
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, load_mod, run_prog, BASE, ROOT  # noqa: E402
from eval_hybrids_task import paired_boot, FLAVORS  # noqa: E402
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman  # noqa: E402

HYB = ROOT / "methods/metric_seam/hybrids"
HYG = HYB / "programs_hygiene"


def rho_on(sel, col, judge):
    s = [d for d in sel if col.get(d) is not None]
    if len(s) < 20:
        return float("nan"), 0
    return spearman([col[d] for d in s], [judge[d] for d in s]), len(s)


def main():
    manifest = json.load(open(BASE / "battery/hygiene_manifest.json"))
    ctxs, codes = {}, {}
    report = {}
    for hyg_name, (task, src) in sorted(manifest.items()):
        if task not in ctxs:
            ctxs[task] = load_ctx(task)
            cs = (ctxs[task]["outdir"] /
                  ("code_scores_v2.json" if task == "press_releases" else "code_scores.json"))
            codes[task] = json.load(open(cs)) if cs.exists() else {}
        ctx = ctxs[task]
        aid = pathlib.Path(src).name.split("_")[0].replace("cw", "")
        aid = pathlib.Path(src).name.split("_")[0]
        if aid.endswith("cw"):
            aid = aid[:-2]
        judge = ctx["judge"].get(aid, {})
        if not judge:
            report[hyg_name] = {"error": "no judge"}
            continue
        fmap = ctx["f_orig"].get(aid, {})
        try:
            pm = load_mod(HYG / hyg_name)
            om = load_mod(HYB / src)
        except Exception as e:
            report[hyg_name] = {"error": f"load: {e}"}
            continue
        col_p = run_prog(pm.score, ctx["items"], fmap, ctx["ops"])
        col_o = run_prog(om.score, ctx["items"], fmap, ctx["ops"])
        te = [d for d in sorted(ctx["test"]) if d in judge]
        r_p, n = rho_on(te, col_p, judge)
        r_o, _ = rho_on(te, col_o, judge)
        sel = [d for d in te if col_p.get(d) is not None and col_o.get(d) is not None]
        _, p_beat, _ = paired_boot(sel, col_p, col_o, judge, gate_floor=-2, margin=-2)
        row = {"task": task, "src": src, "n_test": n,
               "rho_test": {"patched": round(r_p, 4) if r_p == r_p else None,
                            "orig": round(r_o, 4) if r_o == r_o else None},
               "delta": round(r_p - r_o, 4) if r_p == r_p and r_o == r_o else None,
               "P_patched_gt_orig": p_beat}
        # gate for h0 sources only (baseline flavor frozen on train, as elsewhere)
        if src.endswith("_h0.py"):
            train = sorted(ctx["train"])
            best_fl, best_tr, base_col = None, -2, None
            for fl in FLAVORS:
                col = codes[task].get(f"{aid}_{fl}") or {}
                s = [d for d in train if d in judge and col.get(d) is not None]
                if len(s) < 30:
                    continue
                r = spearman([col[d] for d in s], [judge[d] for d in s])
                if r == r and r > best_tr:
                    best_fl, best_tr, base_col = fl, r, col
            if base_col is not None:
                sg = [d for d in sel if base_col.get(d) is not None]
                pg_p, _, _ = paired_boot(sg, col_p, base_col, judge)
                pg_o, _, _ = paired_boot(sg, col_o, base_col, judge)
                row["gate"] = {"flavor": best_fl, "P_gate_patched": pg_p,
                               "P_gate_orig": pg_o}
        report[hyg_name] = row
        g = row.get("gate", {})
        print(f"{task}.{hyg_name}: patched {row['rho_test']['patched']} vs orig "
              f"{row['rho_test']['orig']} (d={row['delta']}) P={p_beat} "
              f"gate {g.get('P_gate_orig')}→{g.get('P_gate_patched')}")

    out = BASE / "battery/hygiene_cert.json"
    json.dump(report, open(out, "w"), indent=1)
    ds = [r["delta"] for r in report.values() if isinstance(r.get("delta"), float)]
    ds.sort()
    print(f"-> {out}; n={len(ds)} median delta={ds[len(ds)//2] if ds else None} "
          f"min={ds[0] if ds else None} max={ds[-1] if ds else None}")


if __name__ == "__main__":
    main()
