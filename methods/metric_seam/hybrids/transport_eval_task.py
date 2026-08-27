"""Interpreter-swap TRANSPORT TEST (theory note §7, binding/provenance 2026-07-04).

For every hybrid program of a task, score the SAME frozen program on the SAME held-out split
under three field conditions:
  gemma  — original extractions (field_results.jsonl)         [the certified condition]
  llama  — Llama-3.3-70B re-extractions (field_results_llama.jsonl)  [family swap]
  blank  — all fields ""                                       [borrowed-meaning ablation]

Per aspect:
  field_marginal   = rho_gemma - rho_blank    (weight of the borrowed enculturated payload)
  transport_delta  = rho_gemma - rho_llama    (certificate loss under interpreter swap)
  P_degrade        = paired-bootstrap P(rho_gemma > rho_llama), B=2000
Retrieval-theory prediction: transport_delta grows with field_marginal (slope > 0) but is
SMALLER than field_marginal where the two families share the enculturated competence
(transport_ratio = transport_delta / field_marginal < 1 on shared-culture constructs).

Usage: python3 transport_eval_task.py <task> <progdir> [math]
-> outputs/metric_seam_pilot/tasks/<task>/transport_eval.json
"""
import importlib.util, json, pathlib, random, signal, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from certificates import spearman  # noqa: E402
from ops import Ops                # noqa: E402
from eval_hybrids_task import split_task_ids, load_mod, load_judge, load_fields, run_prog

B = 2000


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
    f_gemma = load_fields(OUT / "field_results.jsonl")
    f_llama = load_fields(OUT / "field_results_llama.jsonl")

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
        cols = {}
        for cond, fmap in (("gemma", f_gemma.get(aid, {})),
                           ("llama", f_llama.get(aid, {})),
                           ("blank", {})):
            cols[cond] = run_prog(mod.score, items, fmap, ops)
        tsel = [d for d in test if d in judge[aid]
                and all(cols[c].get(d) is not None for c in cols)]
        if len(tsel) < 30:
            report[aid] = {"error": f"n_test={len(tsel)}"}
            continue
        rho = {c: rho_on(tsel, cols[c], judge[aid]) for c in cols}
        # paired bootstrap P(gemma > llama)
        rng = random.Random(11)
        deg = used = 0
        for _ in range(B):
            s = [tsel[rng.randrange(len(tsel))] for _ in tsel]
            rg = spearman([cols["gemma"][d] for d in s], [judge[aid][d] for d in s])
            rl = spearman([cols["llama"][d] for d in s], [judge[aid][d] for d in s])
            if rg == rg and rl == rl:
                used += 1
                deg += rg > rl
        fm = rho["gemma"] - rho["blank"]
        td = rho["gemma"] - rho["llama"]
        report[aid] = {
            "n_test": len(tsel), "rel1": round(rel.get(aid, float("nan")), 3),
            "rho": {c: round(v, 3) if v == v else None for c, v in rho.items()},
            "field_marginal": round(fm, 3) if fm == fm else None,
            "transport_delta": round(td, 3) if td == td else None,
            "transport_ratio": (round(td / fm, 3) if fm == fm and abs(fm) > 0.05
                                and td == td else None),
            "P_degrade": round(deg / used, 4) if used else None,
        }
        r = report[aid]
        print(f"{aid}: g={r['rho']['gemma']} l={r['rho']['llama']} b={r['rho']['blank']}  "
              f"fm={r['field_marginal']} td={r['transport_delta']} "
              f"ratio={r['transport_ratio']} P_deg={r['P_degrade']}")

    # cross-aspect summary: does transport_delta track field_marginal?
    pairs = [(v["field_marginal"], v["transport_delta"]) for v in report.values()
             if isinstance(v, dict) and v.get("field_marginal") is not None
             and v.get("transport_delta") is not None]
    summ = {"n": len(pairs)}
    if len(pairs) >= 5:
        summ["spearman_fm_td"] = round(spearman([p[0] for p in pairs],
                                                [p[1] for p in pairs]), 3)
        tds = sorted(p[1] for p in pairs)
        summ["median_transport_delta"] = round(tds[len(tds) // 2], 3)
        fms = sorted(p[0] for p in pairs)
        summ["median_field_marginal"] = round(fms[len(fms) // 2], 3)
    json.dump({"aspects": report, "summary": summ},
              open(OUT / "transport_eval.json", "w"), indent=1)
    print("summary:", summ)
    print(f"-> {OUT / 'transport_eval.json'}")


if __name__ == "__main__":
    main()
