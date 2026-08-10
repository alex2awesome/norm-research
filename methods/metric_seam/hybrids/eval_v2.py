"""Wave-2 hybrid evaluation WITH certificates: bootstrap gate + attenuation ceilings +
ceiling-normalized fidelity. Programs in hybrids/programs_v2/<aid>_h0.py."""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(ROOT / "methods/metric_seam/pilot"))
from certificates import (spearman, attenuation_ceiling, ceiling_normalized,
                          bootstrap_gate)
from harness import split_ids, load_scope, load_hybrid, run_hybrid
from analyze_v2 import load_judge_v2, OUT, V1
from ops import Ops

FLAVORS = ["v0_keyword", "v1_structure", "v2_holistic"]
FIELDS_DIR = OUT / "llm_fields"


def load_fields_v2(aid):
    out = {}
    if FIELDS_DIR.exists():
        for f in FIELDS_DIR.glob(f"{aid}__*.json"):
            field = f.stem.split("__", 1)[1]
            for dpid, ans in json.load(open(f)).items():
                out.setdefault(dpid, {})[field] = ans
    return out


def main():
    judge, p1, p2 = load_judge_v2()
    seam = {r["aspect"]: r for r in json.load(open(OUT / "seam_table_v2.json"))}
    train, test = split_ids()
    in_scope, _ = load_scope()
    items = {x["datapoint_id"]: x["ctext"] for x in json.load(open(V1 / "items_v1.json"))}
    code = json.load(open(OUT / "code_scores_v2.json"))
    ops = Ops(corpus_path=str(ROOT / "runs/validity_full/v2/press_releases/datapoints.json"))
    prog_dir = pathlib.Path(__file__).parent / "programs_v2"

    report = {}
    for prog in sorted(prog_dir.glob("*_h0.py")):
        aid = prog.stem.split("_")[0]
        try:
            mod = load_hybrid(prog)
        except Exception as e:
            report[aid] = {"error": str(e)}
            print(f"{aid}: LOAD ERROR {e}")
            continue
        fields = load_fields_v2(aid)
        scores = run_hybrid(mod, items, fields, ops)

        best = (None, float("-inf"), None)
        for fl in FLAVORS:
            col = code.get(f"{aid}_{fl}")
            if col is None:
                continue
            sel = [d for d in test if d in judge.get(aid, {})
                   and col.get(d) is not None and scores.get(d) is not None]
            r = spearman([col[d] for d in sel], [judge[aid][d] for d in sel])
            if r == r and r > best[1]:
                best = (fl, r, col)
        fl, base_rho, base_col = best
        if base_col is None:   # no working description-compiled program (e.g. a119)
            base_col = {d: 0.5 for d in items}
            fl, base_rho = "none(constant)", 0.0
        gate = bootstrap_gate(scores, base_col, judge[aid], list(test), B=2000)
        rel1 = seam.get(aid, {}).get("rel1", 1.0) or 1.0
        ceil = attenuation_ceiling(max(rel1, 0.0), 2)
        sel_tr = [d for d in train if d in judge.get(aid, {}) and scores.get(d) is not None]
        rho_tr = spearman([scores[d] for d in sel_tr], [judge[aid][d] for d in sel_tr])
        ssel = [d for d in test if d in judge.get(aid, {})
                and scores.get(d) is not None and d in in_scope]
        rho_sc = spearman([scores[d] for d in ssel], [judge[aid][d] for d in ssel]) \
            if len(ssel) >= 20 else float("nan")
        row = {"fields": sorted({f for m in fields.values() for f in m}),
               "baseline": {"flavor": fl, "rho_test": round(base_rho, 3)},
               "rho_train": round(rho_tr, 3),
               "gate": gate, "ceiling": round(ceil, 3),
               "rho_over_ceiling": round(ceiling_normalized(
                   gate["rho_mean"], max(rel1, 1e-6), 2), 3),
               "rho_test_scoped": round(rho_sc, 3) if rho_sc == rho_sc else None}
        report[aid] = row
        json.dump(scores, open(OUT / f"hybrid_scores_{aid}_h0.json", "w"))
        print(f"{aid}: test ρ={gate['rho_mean']} CI{gate['rho_ci']} "
              f"(scoped {row['rho_test_scoped']}) base={round(base_rho,3)}({fl}) "
              f"P(gate)={gate['P_gate']} P(>base)={gate['P_beats_baseline']} "
              f"ρ/ceil={row['rho_over_ceiling']}")
    json.dump(report, open(OUT / "hybrid_eval_v2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
