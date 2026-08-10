"""Legal transport ratios (R19) — folds legal into the E6 three-family panel.

For every gradable legal aspect: run the h0 hybrid with gemma / llama / qwen field
extractions and a blank condition, compute fm = rho_gemma - rho_blank, td_fam =
rho_gemma - rho_fam, ratio_fam = td/fm (fraction of field signal lost under swap;
low=portable, high=bound), P_degrade_fam via paired bootstrap. Emits a file in the
transport_eval_3fam.json schema so eval_artic.py can consume legal.

Usage: python3 legal_transport.py
-> outputs/metric_seam_pilot/tasks/legal_title_vii/transport_eval_3fam.json
"""
import json, pathlib, random, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, load_mod, load_fields, run_prog  # noqa: E402
import certificates  # noqa: E402
spearman = certificates.spearman
B = 2000


def rho_on(ids, col, judge):
    s = [d for d in ids if col.get(d) is not None and d in judge]
    return spearman([col[d] for d in s], [judge[d] for d in s]) if len(s) >= 20 \
        else float("nan")


def main():
    ctx = load_ctx("legal_title_vii")
    td_ = ctx["outdir"]
    _, rel = None, {}
    from eval_hybrids_task import load_judge
    judge, rel = load_judge(td_ / "results.jsonl")
    fam = {"gemma": ctx["f_orig"],
           "llama": load_fields(td_ / "field_results_llama.jsonl"),
           "qwen": load_fields(td_ / "field_results_qwen.jsonl")}
    test = sorted(ctx["test"])
    aspects = {}
    for aid in sorted(judge):
        prog = ctx["hyb"] / f"{aid}_h0.py"
        if not prog.exists():
            continue
        try:
            mod = load_mod(prog)
        except Exception:
            continue
        cols = {c: run_prog(mod.score, ctx["items"],
                            fam[c].get(aid, {}) if c != "blank" else {}, ctx["ops"])
                for c in ("gemma", "llama", "qwen", "blank")}
        tsel = [d for d in test if d in judge[aid]
                and all(cols[c].get(d) is not None for c in cols)]
        if len(tsel) < 30:
            continue
        rho = {c: rho_on(tsel, cols[c], judge[aid]) for c in cols}
        fm = rho["gemma"] - rho["blank"]
        row = {"n_test": len(tsel), "rel1": round(rel.get(aid, float("nan")), 3),
               "rho": {c: round(v, 3) if v == v else None for c, v in rho.items()},
               "field_marginal": round(fm, 3) if fm == fm else None}
        rng = random.Random(11)
        for f in ("llama", "qwen"):
            tdv = rho["gemma"] - rho[f]
            deg = used = 0
            for _ in range(B):
                s = [tsel[rng.randrange(len(tsel))] for _ in tsel]
                rg = spearman([cols["gemma"][d] for d in s], [judge[aid][d] for d in s])
                rf = spearman([cols[f][d] for d in s], [judge[aid][d] for d in s])
                if rg == rg and rf == rf:
                    used += 1
                    deg += rg > rf
            row[f"td_{f}"] = round(tdv, 3) if tdv == tdv else None
            row[f"ratio_{f}"] = (round(tdv / fm, 3)
                                 if fm == fm and abs(fm) > 0.05 and tdv == tdv else None)
            row[f"P_degrade_{f}"] = round(deg / used, 4) if used else None
        aspects[aid] = row
        print(f"{aid}: fm={row['field_marginal']} r_l={row['ratio_llama']} "
              f"r_q={row['ratio_qwen']} Pdeg_l={row['P_degrade_llama']} "
              f"Pdeg_q={row['P_degrade_qwen']}")

    def med(xs):
        xs = sorted(x for x in xs if x is not None)
        return round(xs[len(xs) // 2], 3) if xs else None
    pairs_l = [(v["field_marginal"], v["td_llama"]) for v in aspects.values()
               if v["field_marginal"] is not None and v.get("td_llama") is not None]
    fmtd = spearman([p[0] for p in pairs_l], [p[1] for p in pairs_l]) \
        if len(pairs_l) >= 8 else float("nan")
    rl = [v["ratio_llama"] for v in aspects.values()]
    rq = [v["ratio_qwen"] for v in aspects.values()]
    both = [(v["ratio_llama"], v["ratio_qwen"]) for v in aspects.values()
            if v.get("ratio_llama") is not None and v.get("ratio_qwen") is not None]
    e6 = spearman([b[0] for b in both], [b[1] for b in both]) \
        if len(both) >= 8 else float("nan")
    summ = {"n": len(aspects), "median_ratio_llama": med(rl),
            "median_ratio_qwen": med(rq),
            "spearman_fm_td_llama": round(fmtd, 3) if fmtd == fmtd else None,
            "E6_spearman_ratio_l_q": round(e6, 3) if e6 == e6 else None, "E6_n": len(both)}
    out = td_ / "transport_eval_3fam.json"
    json.dump({"aspects": aspects, "summary": summ}, open(out, "w"), indent=1)
    print("SUMMARY:", json.dumps(summ))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
