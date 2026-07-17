"""G-only held-out eval for the legal waterfall extension.

Legal has no certified agentic arm-H program, so the standard eval_final.py `eval`
(which recomputes arm H) does not apply. The S4 waterfall F needs only base + G + ceiling:
  base, ceiling  <- outputs/metric_seam_pilot/tasks/legal_title_vii/hybrid_gate_report.json
  G              <- arm-G held-out test Spearman vs the judge, computed here.

Usage: python3 eval_g_only.py gepa_final_results.jsonl
-> prints per-criterion base/G/ceiling + ceiling-normalized r_base/r_G + seam width,
   writes gepa_g_only_final.json (rows compatible with fig_s4_waterfall).
"""
import json, math, sys, pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "methods/metric_seam/battery"))
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
import battery_common as bc
bc.PROGDIR["legal_title_vii"] = "programs_legal"
from certificates import spearman  # noqa: E402

GATE = json.load(open(ROOT / "outputs/metric_seam_pilot/tasks/legal_title_vii/hybrid_gate_report.json"))
TASK = "legal_title_vii"


def ceiling(rel1, k=2):
    r = max(0.0, min(1.0, rel1))
    relk = k * r / (1 + (k - 1) * r)
    return math.sqrt(relk) if relk > 0 else float("nan")


def clip01(x):
    return max(0.0, min(1.0, x))


def load_arm_g(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        a = r.get("aspect_id", "")
        if not a.endswith(".final"):
            continue
        key = a[:-len(".final")]
        sc = r.get("score")
        if isinstance(sc, int):
            out.setdefault(key, {})[r["datapoint_id"]] = sc
    return out


def main():
    results = sys.argv[1]
    ctx = bc.load_ctx(TASK)
    test_ids = sorted(ctx["test"])
    g_by_key = load_arm_g(results)
    rows = []
    print(f"{'aid':5s} {'base':>6s} {'G':>7s} {'ceil':>6s} | {'r_base':>7s} {'r_G':>6s} {'seamW':>6s}  n")
    for key, col_g in sorted(g_by_key.items()):
        aid = key.split(".")[1]
        judge = ctx["judge"].get(aid, {})
        sel = [d for d in test_ids if d in judge and col_g.get(d) is not None]
        if len(sel) < 20:
            print(f"{aid}: only {len(sel)} scorable test items, skip"); continue
        rho_g = spearman([col_g[d] for d in sel], [judge[d] for d in sel])
        base = GATE[aid]["full"]["rho_baseline"]
        ceil = ceiling(GATE[aid]["judge_rel1"])
        r_base = clip01(base / ceil)
        r_g = clip01(rho_g / ceil)
        seamw = (r_g - r_base) / r_g if r_g > 0 else float("nan")
        rows.append({"task": TASK, "aid": aid, "base": round(base, 4), "G": round(rho_g, 4),
                     "ceil": round(ceil, 4), "r_base": round(r_base, 3), "r_G": round(r_g, 3),
                     "seam_width": round(seamw, 3), "n_test": len(sel)})
        print(f"{aid:5s} {base:6.3f} {rho_g:7.3f} {ceil:6.3f} | {r_base:7.3f} {r_g:6.3f} {seamw:6.3f}  {len(sel)}")

    import statistics as st
    med_r_base = st.median(r["r_base"] for r in rows)
    med_r_G = st.median(r["r_G"] for r in rows)
    F = (med_r_G - med_r_base) / med_r_G if med_r_G else float("nan")
    summ = {"n": len(rows), "med_r_base": round(med_r_base, 3), "med_r_G": round(med_r_G, 3),
            "seam_width_F": round(F, 3)}
    print(f"\nLEGAL domain: med_r_base(V)={summ['med_r_base']} med_r_G(V+A)={summ['med_r_G']} "
          f"-> F=A/(V+A)={summ['seam_width_F']}  (n={summ['n']})")
    json.dump({"rows": rows, "domain_summary": {TASK: summ}},
              open(HERE / "gepa_g_only_final.json", "w"), indent=1)
    print("-> gepa_g_only_final.json")


if __name__ == "__main__":
    main()
