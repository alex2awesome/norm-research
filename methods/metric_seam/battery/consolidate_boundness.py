"""Fleet-wide ALL-METRICS boundness consolidation (2026-07-07).

crossfam_cert.py reports the gate-vs-baseline bootstrap only for the ~30 CERTIFIED
gates. The transport_eval_3fam.json files already carry the family-swap degradation
readout (ratio_fam = td/fm = fraction of field signal lost; P_degrade_fam) for
EVERY gradable hybrid aspect in each of the 5 fleet corpora. This rolls them up into
one table so the boundness claim is stated over all metrics, not just the certified
subset, and flags which aspects are certified gates.

Usage: python3 consolidate_boundness.py
-> outputs/metric_seam_pilot/battery/fleet_boundness.json (+ printed table)
"""
import json, pathlib, statistics as st, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import BASE  # noqa: E402
import certificates  # noqa: E402
spearman = certificates.spearman

TFILES = {"press_releases": BASE / "v2/transport_eval_3fam.json",
          "creative_writing": BASE / "tasks/creative_writing/transport_eval_3fam.json",
          "math": BASE / "tasks/math/transport_eval_3fam.json",
          "humor": BASE / "tasks/humor/transport_eval_3fam.json",
          "legal_title_vii": BASE / "tasks/legal_title_vii/transport_eval_3fam.json"}
CERT_GATES = {"press_releases": ["a119", "a115", "a87", "a103", "a76", "a86", "a110",
                                 "a105", "a112", "a104", "a128", "a67"],
              "creative_writing": ["a144", "a72", "a99", "a90", "a342"],
              "humor": ["a351", "a135", "a153", "a81"],
              "legal_title_vii": ["a44", "a46", "a39", "a15", "a0", "a23", "a18",
                                  "a36", "a5"],
              "math": []}


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[len(xs) // 2], 3) if xs else None


def main():
    rows, summ = {}, {}
    for task, p in TFILES.items():
        if not p.exists():
            summ[task] = {"error": "no transport file"}
            continue
        asp = json.load(open(p))["aspects"]
        gate = set(CERT_GATES.get(task, []))
        rl = [v.get("ratio_llama") for v in asp.values()]
        rq = [v.get("ratio_qwen") for v in asp.values()]
        both = [(v["ratio_llama"], v["ratio_qwen"]) for v in asp.values()
                if v.get("ratio_llama") is not None and v.get("ratio_qwen") is not None]
        # both-swap bound = P_degrade >= .95 under BOTH families
        boundboth = sum(1 for v in asp.values()
                        if (v.get("P_degrade_llama") or 0) >= .95
                        and (v.get("P_degrade_qwen") or 0) >= .95)
        e6 = spearman([b[0] for b in both], [b[1] for b in both]) if len(both) >= 8 \
            else float("nan")
        n_grad = sum(1 for v in asp.values() if v.get("ratio_llama") is not None
                     or v.get("ratio_qwen") is not None)
        summ[task] = {
            "n_aspects": len(asp), "n_ratio_gradable": n_grad,
            "n_certified_gates": len(gate),
            "median_ratio_llama": med(rl), "median_ratio_qwen": med(rq),
            "pct_both_swap_bound": round(boundboth / max(1, len(asp)), 3),
            "E6_within_task": round(e6, 3) if e6 == e6 else None, "E6_n": len(both)}
        rows[task] = {a: {"ratio_llama": v.get("ratio_llama"),
                          "ratio_qwen": v.get("ratio_qwen"),
                          "fm": v.get("field_marginal"),
                          "certified_gate": a in gate}
                      for a, v in asp.items()}
    # fleet-level pooled E6 across all tasks
    allpairs = []
    for task, p in TFILES.items():
        if not p.exists():
            continue
        for v in json.load(open(p))["aspects"].values():
            if v.get("ratio_llama") is not None and v.get("ratio_qwen") is not None:
                allpairs.append((v["ratio_llama"], v["ratio_qwen"]))
    pooled_e6 = spearman([x[0] for x in allpairs], [x[1] for x in allpairs])
    out = {"summary": summ, "per_aspect": rows,
           "pooled_E6_spearman_ratio_l_q": round(pooled_e6, 3),
           "pooled_E6_n": len(allpairs)}
    json.dump(out, open(BASE / "battery/fleet_boundness.json", "w"), indent=1)
    order = sorted(summ, key=lambda t: summ[t].get("median_ratio_llama") or 9)
    print(f"{'task':18s} {'nasp':>4s} {'ngate':>5s} {'med_r_L':>8s} {'med_r_Q':>8s} "
          f"{'%bound':>7s} {'E6':>6s}")
    for t in order:
        s = summ[t]
        if "error" in s:
            print(f"{t:18s} {s['error']}"); continue
        print(f"{t:18s} {s['n_aspects']:>4d} {s['n_certified_gates']:>5d} "
              f"{str(s['median_ratio_llama']):>8s} {str(s['median_ratio_qwen']):>8s} "
              f"{s['pct_both_swap_bound']:>7.2f} {str(s['E6_within_task']):>6s}")
    print(f"POOLED E6 Spearman(ratio_l,ratio_q) = {out['pooled_E6_spearman_ratio_l_q']} "
          f"(n={out['pooled_E6_n']})")
    print(f"-> {BASE / 'battery/fleet_boundness.json'}")


if __name__ == "__main__":
    main()
