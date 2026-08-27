"""Interpreter-swap TRANSPORT TEST for wave-2 hybrids (theory note §7, binding/provenance
2026-07-04), v2-machinery variant of transport_eval_task.py.

For every hybrid program in hybrids/programs_v2/<aid>_h0.py, score the SAME frozen program
on the SAME held-out split (harness.split_ids: 150 train / 100 test, seed 7, over
outputs/metric_seam_pilot/v1/items_v1.json — v2 reuses the v1 item universe) under three
field conditions:
  gemma  — original extractions (outputs/metric_seam_pilot/v2/field_results_v2.jsonl)
  llama  — Llama re-extractions (outputs/metric_seam_pilot/v2/field_results_llama.jsonl)
  blank  — all fields ""                                       [borrowed-meaning ablation]

Judge = v2 pass1/pass2 Gemma scores (outputs/metric_seam_pilot/v2/results_v2.jsonl, via
analyze_v2.load_judge_v2 — SAME judge for all three conditions).

Per aspect:
  field_marginal   = rho_gemma - rho_blank    (weight of the borrowed enculturated payload)
  transport_delta  = rho_gemma - rho_llama    (certificate loss under interpreter swap)
  transport_ratio  = transport_delta / field_marginal  (only where |field_marginal| > 0.05)
  P_degrade        = paired-bootstrap P(rho_gemma > rho_llama), B=2000, seed=11

Sanity check: rho_gemma (this script's point estimate on `test`) is compared to the
certified outputs/metric_seam_pilot/v2/hybrid_eval_v2.json[aid]["gate"]["rho_mean"]
(bootstrap-mean rho on the same test set, same gemma fields). Aspects differing by
>0.08 are flagged rather than silently accepted.

Usage: python3 transport_eval_v2.py
-> outputs/metric_seam_pilot/v2/transport_eval.json
"""
import json, pathlib, random, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(ROOT / "methods/metric_seam/pilot"))
from certificates import spearman              # noqa: E402
from ops import Ops                            # noqa: E402
from harness import split_ids                   # noqa: E402  (v1/v2-shared 150/100 seed-7 split)
from analyze_v2 import load_judge_v2, OUT, V1   # noqa: E402
from eval_hybrids_task import load_mod, load_fields, run_prog  # noqa: E402

B = 2000
FLAG_TOL = 0.08


def rho_on(sel, col, judge):
    s = [d for d in sel if col.get(d) is not None]
    return spearman([col[d] for d in s], [judge[d] for d in s]) if len(s) >= 20 else float("nan")


def compute_rel1(p1, p2, aid):
    both = sorted(set(p1.get(aid, {})) & set(p2.get(aid, {})))
    if len(both) < 30:
        return float("nan")
    return spearman([p1[aid][d] for d in both], [p2[aid][d] for d in both])


def main():
    HYB = pathlib.Path(__file__).parent / "programs_v2"

    items = {x["datapoint_id"]: x["ctext"] for x in json.load(open(V1 / "items_v1.json"))}
    train, test = split_ids()
    judge, p1, p2 = load_judge_v2()
    f_gemma = load_fields(OUT / "field_results_v2.jsonl")
    f_llama = load_fields(OUT / "field_results_llama.jsonl")

    ops = Ops(corpus_path=str(ROOT / "runs/validity_full/v2/press_releases/datapoints.json"))

    certified = {}
    cert_path = OUT / "hybrid_eval_v2.json"
    if cert_path.exists():
        certified = json.load(open(cert_path))

    report = {}
    flags = []
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
        rel1 = compute_rel1(p1, p2, aid)
        row = {
            "n_test": len(tsel), "rel1": round(rel1, 3) if rel1 == rel1 else None,
            "rho": {c: round(v, 3) if v == v else None for c, v in rho.items()},
            "field_marginal": round(fm, 3) if fm == fm else None,
            "transport_delta": round(td, 3) if td == td else None,
            "transport_ratio": (round(td / fm, 3) if fm == fm and abs(fm) > 0.05
                                and td == td else None),
            "P_degrade": round(deg / used, 4) if used else None,
        }
        # sanity check vs certified rho_mean (gate)
        cert_rho = certified.get(aid, {}).get("gate", {}).get("rho_mean")
        if cert_rho is not None and rho["gemma"] == rho["gemma"]:
            diff = rho["gemma"] - cert_rho
            row["certified_rho_mean"] = cert_rho
            row["certified_diff"] = round(diff, 3)
            if abs(diff) > FLAG_TOL:
                row["FLAG"] = f"rho_gemma diverges from certified rho_mean by {diff:+.3f}"
                flags.append((aid, row["FLAG"]))
        report[aid] = row
        r = report[aid]
        print(f"{aid}: g={r['rho']['gemma']} l={r['rho']['llama']} b={r['rho']['blank']}  "
              f"fm={r['field_marginal']} td={r['transport_delta']} "
              f"ratio={r['transport_ratio']} P_deg={r['P_degrade']}"
              + (f"  [{r['FLAG']}]" if "FLAG" in r else ""))

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
    summ["flags"] = flags
    json.dump({"aspects": report, "summary": summ},
              open(OUT / "transport_eval.json", "w"), indent=1)
    print("summary:", summ)
    print(f"-> {OUT / 'transport_eval.json'}")


if __name__ == "__main__":
    main()
