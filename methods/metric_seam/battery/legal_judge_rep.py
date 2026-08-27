"""Legal judge-family replication (R19): Llama-70B second judge vs Gemma judge.

Mirrors W1.2 (PR+math): (1) cross-judge agreement per aspect (raw + disattenuated),
(2) codability-ordering replication — Spearman of the code-rung best-flavor rho
computed against each judge over the 20 aspects, (3) gate replication — do the 9
certified hybrid gates still fire when the LLAMA judge is the target?

Usage: python3 legal_judge_rep.py
-> outputs/metric_seam_pilot/battery/legal_judge_rep.json
"""
import json, math, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, load_mod, load_fields, run_prog, BASE  # noqa: E402
from eval_hybrids_task import load_judge, paired_boot, FLAVORS  # noqa: E402
import certificates  # noqa: E402
spearman = certificates.spearman

CERT = ["a44", "a46", "a39", "a15", "a0", "a23", "a18", "a36", "a5"]


def rho_on(ids, col, judge):
    s = [d for d in ids if col.get(d) is not None and d in judge]
    return (spearman([col[d] for d in s], [judge[d] for d in s]), len(s)) \
        if len(s) >= 20 else (float("nan"), len(s))


def best_code(aid, ids, codes, judge):
    best = float("-2")
    for fl in FLAVORS:
        col = codes.get(f"{aid}_{fl}") or {}
        r, n = rho_on(ids, col, judge)
        if r == r and r > best:
            best = r
    return best if best > -2 else float("nan")


def main():
    ctx = load_ctx("legal_title_vii")
    td = ctx["outdir"]
    jg, relg = ctx["judge"], None
    jg, relg = load_judge(td / "results.jsonl")
    jl, rell = load_judge(td / "results_llama.jsonl")
    codes = json.load(open(td / "code_scores.json"))
    test = sorted(ctx["test"])
    aids = sorted(set(jg) & set(jl))

    # (1) cross-judge agreement
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
                    "rel_gemma": round(rg, 3) if rg == rg else None,
                    "rel_llama": round(rl, 3) if rl == rl else None,
                    "disattenuated": round(min(dis, 1.0), 3) if dis == dis else None}

    # (2) codability-ordering replication
    cg = {a: best_code(a, test, codes, jg[a]) for a in aids if a in jg}
    cl = {a: best_code(a, test, codes, jl[a]) for a in aids if a in jl}
    common = [a for a in aids if cg.get(a) == cg.get(a) and cl.get(a) == cl.get(a)]
    cod_rho = spearman([cg[a] for a in common], [cl[a] for a in common])

    # (3) gate replication under the llama judge
    gates = {}
    for aid in CERT:
        if aid not in jl:
            gates[aid] = {"error": "no llama judge"}
            continue
        pm = load_mod(ctx["hyb"] / f"{aid}_h0.py")
        col = run_prog(pm.score, ctx["items"], ctx["f_orig"].get(aid, {}), ctx["ops"])
        # frozen train-best baseline flavor vs the LLAMA judge
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
               "gemma_judge_gate_published": True, "baseline_flavor": best_fl}
        if base_col is not None:
            sel = [d for d in test if d in jl[aid] and col.get(d) is not None
                   and base_col.get(d) is not None]
            pg, _, _ = paired_boot(sel, col, base_col, jl[aid])
            row["P_gate_llamajudge"] = pg
        gates[aid] = row

    out = {"cross_judge_agreement": agree,
           "median_raw_agreement": round(
               sorted(v["raw"] for v in agree.values() if v["raw"] is not None)[len(agree) // 2], 3),
           "median_disattenuated": round(sorted(
               v["disattenuated"] for v in agree.values()
               if v["disattenuated"] is not None)[len(agree) // 2], 3),
           "codability_ordering_spearman": round(cod_rho, 3) if cod_rho == cod_rho else None,
           "codability_n": len(common),
           "gate_replication": gates}
    json.dump(out, open(BASE / "battery/legal_judge_rep.json", "w"), indent=1)
    print(f"cross-judge agreement: median raw {out['median_raw_agreement']} "
          f"disatt {out['median_disattenuated']}")
    print(f"codability ordering Spearman(gemma,llama) = {out['codability_ordering_spearman']} "
          f"(n={out['codability_n']})")
    print("gate replication under llama judge:")
    for a, r in gates.items():
        print(f"  {a}: rho_H={r.get('rho_hybrid_llamajudge')} "
              f"P_gate_llama={r.get('P_gate_llamajudge')}")
    print(f"-> {BASE / 'battery/legal_judge_rep.json'}")


if __name__ == "__main__":
    main()
