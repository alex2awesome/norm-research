"""THREE-FAMILY transport test, v2 (press releases) variant — see transport_eval_3fam.py.

gemma (certified) / llama / qwen (Qwen3.5-122B, thinking disabled) / blank on the frozen
programs_v2 hybrids, harness.split_ids held-out 100. E6: Spearman(ratio_g->l, ratio_g->q).

Usage: python3 transport_eval_v2_3fam.py
-> outputs/metric_seam_pilot/v2/transport_eval_3fam.json
"""
import json, pathlib, random, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(ROOT / "methods/metric_seam/pilot"))
from certificates import spearman              # noqa: E402
from ops import Ops                            # noqa: E402
from harness import split_ids                  # noqa: E402
from analyze_v2 import load_judge_v2, OUT, V1  # noqa: E402
from eval_hybrids_task import load_mod, load_fields, run_prog  # noqa: E402

B = 2000
FAMS = ("llama", "qwen")


def rho_on(sel, col, judge):
    s = [d for d in sel if col.get(d) is not None]
    return spearman([col[d] for d in s], [judge[d] for d in s]) if len(s) >= 20 else float("nan")


def main():
    HYB = pathlib.Path(__file__).parent / "programs_v2"
    items = {x["datapoint_id"]: x["ctext"] for x in json.load(open(V1 / "items_v1.json"))}
    _, test = split_ids()
    judge, p1, p2 = load_judge_v2()
    fields = {"gemma": load_fields(OUT / "field_results_v2.jsonl"),
              "llama": load_fields(OUT / "field_results_llama.jsonl"),
              "qwen": load_fields(OUT / "field_results_qwen.jsonl")}
    ops = Ops(corpus_path=str(ROOT / "runs/validity_full/v2/press_releases/datapoints.json"))

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
        cols = {c: run_prog(mod.score, items,
                            fields[c].get(aid, {}) if c != "blank" else {}, ops)
                for c in ("gemma", "llama", "qwen", "blank")}
        tsel = [d for d in test if d in judge[aid]
                and all(cols[c].get(d) is not None for c in cols)]
        if len(tsel) < 30:
            report[aid] = {"error": f"n_test={len(tsel)}"}
            continue
        rho = {c: rho_on(tsel, cols[c], judge[aid]) for c in cols}
        fm = rho["gemma"] - rho["blank"]
        row = {"n_test": len(tsel),
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
