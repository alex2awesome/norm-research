"""THREE-FAMILY transport test (E6 in the 2026-07-05 seam-position/retrieval note §2.3).

Extends transport_eval_task.py with the Qwen-3.5-122B third-family re-extraction:
  gemma (certified) / llama (swap 1) / qwen (swap 2) / blank (ablation).
E6 question: is extractor-boundness a property of the CRITERION (its cultural
specificity) rather than of the family pair? Prediction: per-criterion transport
ratios correlate across pairs, Spearman(ratio_g->l, ratio_g->q) > 0 on aspects with
a real field marginal (|fm| > 0.05).

Usage: python3 transport_eval_3fam.py <task> <progdir> [math]
-> outputs/metric_seam_pilot/tasks/<task>/transport_eval_3fam.json
"""
import json, pathlib, random, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from certificates import spearman  # noqa: E402
from ops import Ops                # noqa: E402
from eval_hybrids_task import split_task_ids, load_mod, load_judge, load_fields, run_prog

B = 2000
FAMS = ("llama", "qwen")


def rho_on(sel, col, judge):
    s = [d for d in sel if col.get(d) is not None]
    return spearman([col[d] for d in s], [judge[d] for d in s]) if len(s) >= 20 else float("nan")


def main():
    task, progdir = sys.argv[1], sys.argv[2]
    use_math = len(sys.argv) > 3 and sys.argv[3] == "math"
    OUT = ROOT / "outputs/metric_seam_pilot/tasks" / task
    HYB = pathlib.Path(__file__).parent / progdir

    items_l = json.load(open(OUT / "items.json"))
    items = {x["datapoint_id"]: x["ctext"] for x in items_l}
    _, test = split_task_ids(items_l)
    judge, rel = load_judge(OUT / "results.jsonl")
    fields = {"gemma": load_fields(OUT / "field_results.jsonl"),
              "llama": load_fields(OUT / "field_results_llama.jsonl"),
              "qwen": load_fields(OUT / "field_results_qwen.jsonl")}

    if use_math:
        from ops_math import MathOps
        ops = MathOps(corpus_path=str(OUT / "items.json"))
    else:
        ops = Ops(corpus_path=str(OUT / "items.json"))

    report = {}
    for prog in sorted(HYB.glob("*_h0.py")):
        aid = prog.stem.split("_")[0]
        if aid not in judge:
            continue
        try:
            mod = load_mod(prog)
        except Exception as e:
            report[aid] = {"error": str(e)}
            continue
        cols = {c: run_prog(mod.score, items, fields[c].get(aid, {}) if c != "blank" else {},
                            ops)
                for c in ("gemma", "llama", "qwen", "blank")}
        tsel = [d for d in test if d in judge[aid]
                and all(cols[c].get(d) is not None for c in cols)]
        if len(tsel) < 30:
            report[aid] = {"error": f"n_test={len(tsel)}"}
            continue
        rho = {c: rho_on(tsel, cols[c], judge[aid]) for c in cols}
        fm = rho["gemma"] - rho["blank"]
        row = {"n_test": len(tsel), "rel1": round(rel.get(aid, float("nan")), 3),
               "rho": {c: round(v, 3) if v == v else None for c, v in rho.items()},
               "field_marginal": round(fm, 3) if fm == fm else None}
        rng = random.Random(11)
        for fam in FAMS:
            td = rho["gemma"] - rho[fam]
            deg = used = 0
            for _ in range(B):
                s = [tsel[rng.randrange(len(tsel))] for _ in tsel]
                rg = spearman([cols["gemma"][d] for d in s], [judge[aid][d] for d in s])
                rf = spearman([cols[fam][d] for d in s], [judge[aid][d] for d in s])
                if rg == rg and rf == rf:
                    used += 1
                    deg += rg > rf
            row[f"td_{fam}"] = round(td, 3) if td == td else None
            row[f"ratio_{fam}"] = (round(td / fm, 3)
                                   if fm == fm and abs(fm) > 0.05 and td == td else None)
            row[f"P_degrade_{fam}"] = round(deg / used, 4) if used else None
        report[aid] = row
        print(f"{aid}: g={row['rho']['gemma']} l={row['rho']['llama']} "
              f"q={row['rho']['qwen']} b={row['rho']['blank']} fm={row['field_marginal']} "
              f"r_l={row['ratio_llama']} r_q={row['ratio_qwen']}")

    ok = [v for v in report.values() if isinstance(v, dict) and "error" not in v]
    summ = {"n": len(ok)}
    for fam in FAMS:
        pairs = [(v["field_marginal"], v[f"td_{fam}"]) for v in ok
                 if v["field_marginal"] is not None and v[f"td_{fam}"] is not None]
        if len(pairs) >= 5:
            summ[f"spearman_fm_td_{fam}"] = round(
                spearman([p[0] for p in pairs], [p[1] for p in pairs]), 3)
            tds = sorted(p[1] for p in pairs)
            summ[f"median_td_{fam}"] = round(tds[len(tds) // 2], 3)
        rats = sorted(v[f"ratio_{fam}"] for v in ok if v[f"ratio_{fam}"] is not None)
        if rats:
            summ[f"median_ratio_{fam}"] = round(rats[len(rats) // 2], 3)
    # E6: criterion-level boundness across pairs
    both = [(v["ratio_llama"], v["ratio_qwen"]) for v in ok
            if v.get("ratio_llama") is not None and v.get("ratio_qwen") is not None]
    summ["n_ratio_pairs"] = len(both)
    if len(both) >= 5:
        summ["spearman_ratio_l_q"] = round(
            spearman([p[0] for p in both], [p[1] for p in both]), 3)
    summ["degrade95_both"] = sum(
        1 for v in ok if (v.get("P_degrade_llama") or 0) >= .95
        and (v.get("P_degrade_qwen") or 0) >= .95)
    summ["degrade95_llama_only"] = sum(
        1 for v in ok if (v.get("P_degrade_llama") or 0) >= .95
        and (v.get("P_degrade_qwen") or 0) < .95)
    summ["degrade95_qwen_only"] = sum(
        1 for v in ok if (v.get("P_degrade_qwen") or 0) >= .95
        and (v.get("P_degrade_llama") or 0) < .95)
    json.dump({"aspects": report, "summary": summ},
              open(OUT / "transport_eval_3fam.json", "w"), indent=1)
    print("summary:", json.dumps(summ))
    print(f"-> {OUT / 'transport_eval_3fam.json'}")


if __name__ == "__main__":
    main()
