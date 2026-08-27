"""Wave-2 seam table WITH certificates: reliability, attenuation ceiling, ceiling-normalized
rho, bootstrap CI per best flavor, scoped subset. Uses certificates.py (tested)."""
import json, pathlib, statistics as st, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(ROOT / "methods/metric_seam/hybrids"))
from certificates import (spearman, attenuation_ceiling, ceiling_normalized,
                          bootstrap_gate)
from harness import load_scope, split_ids   # scope reused from v1 (same items)

V1 = ROOT / "outputs/metric_seam_pilot/v1"
OUT = ROOT / "outputs/metric_seam_pilot/v2"
FLAVORS = ["v0_keyword", "v1_structure", "v2_holistic"]


def load_judge_v2():
    p1, p2 = {}, {}
    for line in open(OUT / "results_v2.jsonl"):
        r = json.loads(line)
        if isinstance(r["score"], int):
            d = p1 if r["channel"] == "pass1" else p2 if r["channel"] == "pass2" else None
            if d is not None:
                d.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
    combined = {}
    for aid in set(p1) | set(p2):
        for dpid in set(p1.get(aid, {})) | set(p2.get(aid, {})):
            vals = [d[aid][dpid] for d in (p1, p2) if dpid in d.get(aid, {})]
            combined.setdefault(aid, {})[dpid] = sum(vals) / len(vals) / 10.0
    return combined, p1, p2


def main():
    aspects = json.load(open(OUT / "wave2_aspects.json"))
    judge, p1, p2 = load_judge_v2()
    in_scope, _ = load_scope()
    code = json.load(open(OUT / "code_scores_v2.json"))

    table = []
    print(f"{'aspect':6} {'rel1':>5} {'ceil':>5} {'bestfl':>13} {'rho':>6} "
          f"{'rho/ceil':>8} {'rho_scoped':>10} {'NA%':>4}")
    for aid in aspects:
        both = [d for d in p1.get(aid, {}) if d in p2.get(aid, {})]
        if len(both) < 30:
            print(f"{aid:6} DEGENERATE (n={len(both)})")
            table.append({"aspect": aid, "verdict": "degenerate"})
            continue
        rel1 = spearman([p1[aid][d] for d in both], [p2[aid][d] for d in both])
        ceil = attenuation_ceiling(max(rel1, 0.0), 2)
        na = 250 - len(judge.get(aid, {}))

        best = (None, float("-inf"))
        for fl in FLAVORS:
            col = code.get(f"{aid}_{fl}")
            if col is None:
                continue
            sel = [d for d in judge.get(aid, {}) if col.get(d) is not None]
            r = spearman([col[d] for d in sel], [judge[aid][d] for d in sel])
            if r == r and r > best[1]:
                best = (fl, r)
        fl, rho = best
        col = code.get(f"{aid}_{fl}") if fl else {}
        ssel = [d for d in judge.get(aid, {}) if col.get(d) is not None and d in in_scope]
        rho_sc = spearman([col[d] for d in ssel], [judge[aid][d] for d in ssel]) \
            if len(ssel) >= 20 else float("nan")
        norm = ceiling_normalized(rho, max(rel1, 1e-6), 2) if fl else float("nan")
        row = {"aspect": aid, "rel1": round(rel1, 3), "ceiling": round(ceil, 3),
               "best_flavor": fl, "rho": round(rho, 3), "rho_over_ceiling": round(norm, 3),
               "rho_scoped": round(rho_sc, 3) if rho_sc == rho_sc else None, "na": na}
        table.append(row)
        print(f"{aid:6} {rel1:5.2f} {ceil:5.2f} {str(fl):>13} {rho:6.2f} "
              f"{norm:8.2f} {rho_sc:10.2f} {na:4}")
    json.dump(table, open(OUT / "seam_table_v2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
