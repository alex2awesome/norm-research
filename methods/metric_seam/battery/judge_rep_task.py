"""Judge-family replication, task-generalized (R19 legal → CW/humor scale-out).

Llama-70B second judge (results_llama.jsonl) vs Gemma judge for one task:
(1) cross-judge agreement per aspect (raw + disattenuated), (2) codability-ordering
replication Spearman, (3) gate replication for that task's certified gates.

Usage: python3 judge_rep_task.py <task>
-> outputs/metric_seam_pilot/battery/judge_rep_<task>.json
"""
import json, math, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, load_mod, run_prog, BASE  # noqa: E402
from eval_hybrids_task import load_judge, paired_boot, FLAVORS  # noqa: E402
import certificates  # noqa: E402
spearman = certificates.spearman

CERT_GATES = {"creative_writing": ["a144", "a72", "a99", "a90", "a342"],
              "humor": ["a351", "a135", "a153", "a81"],
              "legal_title_vii": ["a44", "a46", "a39", "a15", "a0", "a23", "a18",
                                  "a36", "a5"],
              "math": [],
              "press_releases": ["a119", "a115", "a87", "a103", "a76", "a86",
                                 "a110", "a105", "a112", "a104", "a128", "a67"]}


def rho_on(ids, col, judge):
    s = [d for d in ids if col.get(d) is not None and d in judge]
    return (spearman([col[d] for d in s], [judge[d] for d in s]), len(s)) \
        if len(s) >= 20 else (float("nan"), len(s))


def best_code(aid, ids, codes, judge):
    best = -2.0
    for fl in FLAVORS:
        col = codes.get(f"{aid}_{fl}") or {}
        r, _ = rho_on(ids, col, judge)
        if r == r and r > best:
            best = r
    return best if best > -2 else float("nan")


def gemma_judge(task, td):
    """(judge, rel) for the Gemma channel — PR lives in v2 machinery (no results.jsonl)."""
    if task != "press_releases":
        return load_judge(td / "results.jsonl")
    from analyze_v2 import load_judge_v2
    _, p1, p2 = load_judge_v2()
    jg, relg = {}, {}
    for a in set(p1) & set(p2):
        both = sorted(set(p1[a]) & set(p2[a]))
        if len(both) < 30:
            continue
        relg[a] = spearman([p1[a][d] for d in both], [p2[a][d] for d in both])
        jg[a] = {d: (p1[a][d] + p2[a][d]) / 2 / 10.0 for d in both}
    return jg, relg


def main():
    task = sys.argv[1]
    ctx = load_ctx(task)
    td = ctx["outdir"]
    jg, relg = gemma_judge(task, td)
    jl, rell = load_judge(td / "results_llama.jsonl")
    cs = td / ("code_scores_v2.json" if task == "press_releases" else "code_scores.json")
    codes = json.load(open(cs)) if cs.exists() else {}
    test = sorted(ctx["test"])
    aids = sorted(set(jg) & set(jl))
    cert = CERT_GATES.get(task, [])

    agree = {}
    for a in aids:
        both = sorted(set(jg[a]) & set(jl[a]))
        if len(both) < 30:
            continue
        r = spearman([jg[a][d] for d in both], [jl[a][d] for d in both])
        rg, rl = relg.get(a, float("nan")), rell.get(a, float("nan"))
        dis = (r / math.sqrt(rg * rl)) if rg == rg and rl == rl and rg > 0 and rl > 0 \
            else float("nan")
        agree[a] = {"raw": round(r, 3) if r == r else None,
                    "disattenuated": round(min(dis, 1.0), 3) if dis == dis else None}

    cg, cl = {}, {}
    for a in aids:
        cg[a] = best_code(a, test, codes, jg[a])
        cl[a] = best_code(a, test, codes, jl[a])
    common = [a for a in aids if cg[a] == cg[a] and cl[a] == cl[a]]
    cod_rho = spearman([cg[a] for a in common], [cl[a] for a in common]) \
        if len(common) >= 8 else float("nan")

    gates = {}
    for aid in cert:
        if aid not in jl or not (ctx["hyb"] / f"{aid}_h0.py").exists():
            gates[aid] = {"error": "missing"}
            continue
        pm = load_mod(ctx["hyb"] / f"{aid}_h0.py")
        col = run_prog(pm.score, ctx["items"], ctx["f_orig"].get(aid, {}), ctx["ops"])
        train = sorted(ctx["train"])
        best_fl, best_tr, base_col = None, -2, None
        for fl in FLAVORS:
            c = codes.get(f"{aid}_{fl}") or {}
            s = [d for d in train if d in jl[aid] and c.get(d) is not None]
            if len(s) < 30:
                continue
            r = spearman([c[d] for d in s], [jl[aid][d] for d in s])
            if r == r and r > best_tr:
                best_fl, best_tr, base_col = fl, r, c
        rh, n = rho_on(test, col, jl[aid])
        row = {"rho_hybrid_llamajudge": round(rh, 3) if rh == rh else None, "n": n,
               "baseline_flavor": best_fl}
        if base_col is not None:
            sel = [d for d in test if d in jl[aid] and col.get(d) is not None
                   and base_col.get(d) is not None]
            pg, _, _ = paired_boot(sel, col, base_col, jl[aid])
            row["P_gate_llamajudge"] = pg
        gates[aid] = row

    def med(xs):
        xs = sorted(x for x in xs if x is not None)
        return round(xs[len(xs) // 2], 3) if xs else None
    out = {"task": task,
           "median_raw_agreement": med([v["raw"] for v in agree.values()]),
           "median_disattenuated": med([v["disattenuated"] for v in agree.values()]),
           "codability_ordering_spearman": round(cod_rho, 3) if cod_rho == cod_rho else None,
           "codability_n": len(common),
           "n_gates_replicating_p95": sum(1 for r in gates.values()
                                          if (r.get("P_gate_llamajudge") or 0) >= .95),
           "n_gates": len(cert), "gate_replication": gates,
           "cross_judge_agreement": agree}
    json.dump(out, open(BASE / f"battery/judge_rep_{task}.json", "w"), indent=1)
    print(f"[{task}] agreement raw {out['median_raw_agreement']} / disatt "
          f"{out['median_disattenuated']}; codability-order rho "
          f"{out['codability_ordering_spearman']} (n={out['codability_n']}); "
          f"gates replicating P>=.95: {out['n_gates_replicating_p95']}/{out['n_gates']}")
    for a, r in gates.items():
        print(f"  {a}: rho_H={r.get('rho_hybrid_llamajudge')} "
              f"P_gate_llama={r.get('P_gate_llamajudge')}")
    print(f"-> {BASE / f'battery/judge_rep_{task}.json'}")


if __name__ == "__main__":
    main()
