"""Evaluate hybrid programs (round N) against the judge; apply the pilot gate.

Usage: python3 eval_hybrids.py [round_suffix]   (default h0)
Gate:
  G1 rho_test >= max(baseline_rho_test + 0.10, 0.60)   [faithfulness, same held-out items]
  G3 (a86) CF delta_hybrid <= delta_judge + 0.05        [construct, not presence]
Also reports scoped-subset rho (true press releases only) and train/test split honestly.
"""
import json, pathlib, statistics as st, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from harness import (OUT, ROOT, load_judge, load_scope, split_ids, spearman,
                     load_hybrid, load_fields, run_hybrid)
from ops import Ops

FLAVORS = ["v0_keyword", "v1_structure", "v2_holistic"]


def main():
    suffix = sys.argv[1] if len(sys.argv) > 1 else "h0"
    judge, p1, _ = load_judge()
    in_scope, _ = load_scope()
    train, test = split_ids()
    items = {x["datapoint_id"]: x["ctext"] for x in json.load(open(OUT / "items_v1.json"))}
    code = json.load(open(OUT / "code_scores_v1.json"))
    ops = Ops(corpus_path=str(ROOT / "runs/validity_full/v2/press_releases/datapoints.json"))

    prog_dir = pathlib.Path(__file__).parent / "programs"
    report = {}
    for prog in sorted(prog_dir.glob(f"*_{suffix}.py")):
        aid = prog.stem.split("_")[0]
        try:
            mod = load_hybrid(prog)
        except Exception as e:
            report[aid] = {"error": f"load: {e}"}
            continue
        fields = load_fields(aid)
        scores = run_hybrid(mod, items, fields, ops)
        n_err = sum(1 for v in scores.values() if v is None)

        # baseline = best description-compiled flavor on the SAME test items
        best_base, best_fl = float("-inf"), None
        for fl in FLAVORS:
            col = code.get(f"{aid}_{fl}")
            if col is None:
                continue
            sel = [d for d in test if d in judge.get(aid, {})
                   and col.get(d) is not None and scores.get(d) is not None]
            r = spearman([col[d] for d in sel], [judge[aid][d] for d in sel])
            if r == r and r > best_base:
                best_base, best_fl = r, fl

        row = {"fields_loaded": sorted({f for m in fields.values() for f in m}),
               "n_score_errors": n_err,
               "baseline_test": {"flavor": best_fl, "rho": round(best_base, 3)}}
        for name, idset in [("train", train), ("test", test)]:
            sel = [d for d in idset if d in judge.get(aid, {})
                   and scores.get(d) is not None]
            xs, ys = [scores[d] for d in sel], [judge[aid][d] for d in sel]
            sub = {"n": len(sel), "rho": round(spearman(xs, ys), 3)}
            ssel = [d for d in sel if d in in_scope]
            sub["rho_scoped"] = round(spearman([scores[d] for d in ssel],
                                               [judge[aid][d] for d in ssel]), 3)
            sub["n_scoped"] = len(ssel)
            row[name] = sub

        # G3 CF for a86
        if aid == "a86":
            cf_items = {x["datapoint_id"]: x["ctext"]
                        for x in json.load(open(OUT / "cf_items_a86.json"))}
            cf_scores = run_hybrid(mod, cf_items, load_fields("a86cf"), ops)
            deltas = [cf_scores[d] - scores[d] for d in cf_items
                      if cf_scores.get(d) is not None and scores.get(d) is not None]
            cf_judge = json.load(open(OUT / "seam_table_v1.json"))["cf"]["judge_delta"]
            row["cf"] = {"hybrid_delta": round(st.mean(deltas), 3),
                         "judge_delta": round(cf_judge, 3),
                         "pass_G3": st.mean(deltas) <= cf_judge + 0.05}

        g1_thresh = max(best_base + 0.10, 0.60)
        row["gate"] = {"G1_threshold": round(g1_thresh, 3),
                       "pass_G1": row["test"]["rho"] >= g1_thresh}
        report[aid] = row
        json.dump({d: scores[d] for d in scores},
                  open(OUT / f"hybrid_scores_{aid}_{suffix}.json", "w"))

    json.dump(report, open(OUT / f"hybrid_eval_{suffix}.json", "w"), indent=1)
    for aid, row in report.items():
        if "error" in row:
            print(aid, "ERROR", row["error"])
            continue
        cf = row.get("cf")
        print(f"{aid} [{suffix}] fields={row['fields_loaded']} errs={row['n_score_errors']}\n"
              f"   train rho={row['train']['rho']} (scoped {row['train']['rho_scoped']})  "
              f"test rho={row['test']['rho']} (scoped {row['test']['rho_scoped']})  "
              f"baseline_test={row['baseline_test']['rho']} ({row['baseline_test']['flavor']})\n"
              f"   gate G1(>= {row['gate']['G1_threshold']}): "
              f"{'PASS' if row['gate']['pass_G1'] else 'fail'}"
              + (f" | CF hybridΔ={cf['hybrid_delta']} vs judgeΔ={cf['judge_delta']} "
                 f"G3: {'PASS' if cf['pass_G3'] else 'FAIL'}" if cf else ""))


if __name__ == "__main__":
    main()
