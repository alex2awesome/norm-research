"""W1.2: second-judge-family replication — Llama-3.3-70B BF16 on the VERBATIM survey
prompts (same items, same prompt text; only the judge model family changes).

Nothing is retrained or re-selected: code flavors, hybrid programs, field extractions and
baseline-flavor choices are all frozen from the Gemma runs. Three readouts:

A. Instrument stability per aspect: rel1 under each judge, cross-judge channel agreement
   rho(Gemma mean, Llama mean), and the disattenuated construct agreement
   rho / sqrt(rel1_G * rel1_L) (only when both rel1 > 0.05).
B. Code-rung replication: for every aspect x flavor column, rho vs each judge channel on
   the same items; Spearman across cells answers "does the codability ordering replicate?"
C. v1 gate replication (a80/a86/a105/a110): frozen hybrid columns + frozen baseline flavor
   (selected on GEMMA train, original protocol) re-bootstrapped on the v1 test split
   against the LLAMA channel; scoped variant uses the Llama scope channel.

Covers PR waves v1/v2/v3 + math survey. -> outputs/metric_seam_pilot/replication_llama/
"""
import json, pathlib, random, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam/hybrids"))
from harness import spearman, split_ids  # noqa: E402

OUT = ROOT / "outputs/metric_seam_pilot"
REP = OUT / "replication_llama"
REP.mkdir(exist_ok=True)
B = 2000
WAVES = {
    "pr_v1": ("v1/results_v1.jsonl", "v1/results_llama.jsonl", "v1/code_scores_v1.json"),
    "pr_v2": ("v2/results_v2.jsonl", "v2/results_llama.jsonl", "v2/code_scores_v2.json"),
    "pr_v3": ("v3/results_v3.jsonl", "v3/results_llama.jsonl", "v3/code_scores_v3.json"),
    "math": ("tasks/math/results.jsonl", "tasks/math/results_llama.jsonl",
             "tasks/math/code_scores.json"),
}


def load_channels(path):
    """aspect -> (mean channel over both-pass items, rel1); plus scope set if present."""
    p1, p2, sc = {}, {}, {}
    for line in open(path):
        r = json.loads(line)
        if not isinstance(r["score"], int):
            continue
        if r["channel"] == "scope":
            sc[r["datapoint_id"]] = r["score"]
        elif r["channel"] == "pass1":
            p1.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
        elif r["channel"] == "pass2":
            p2.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
    chan, rel = {}, {}
    for aid in set(p1) & set(p2):
        both = sorted(set(p1[aid]) & set(p2[aid]))
        if len(both) < 30:
            continue
        rel[aid] = spearman([p1[aid][d] for d in both], [p2[aid][d] for d in both])
        chan[aid] = {d: (p1[aid][d] + p2[aid][d]) / 2 / 10.0 for d in both}
    return chan, rel, {d for d, s in sc.items() if s >= 7}


def main():
    report = {"stability": {}, "code_rung": {}, "gates_v1": {}}

    # ---- A + B: stability and code-rung replication per wave -------------------
    for wave, (gp, lp, cp) in WAVES.items():
        gch, grel, gscope = load_channels(OUT / gp)
        lch, lrel, lscope = load_channels(OUT / lp)
        code = json.load(open(OUT / cp))
        if "scores" in code and isinstance(code["scores"], dict):
            code = code["scores"]

        rows = {}
        for aid in sorted(set(gch) & set(lch)):
            common = sorted(set(gch[aid]) & set(lch[aid]))
            if len(common) < 30:
                continue
            x = spearman([gch[aid][d] for d in common], [lch[aid][d] for d in common])
            rg, rl = grel[aid], lrel[aid]
            dis = (x / (rg * rl) ** 0.5
                   if rg == rg and rl == rl and rg > 0.05 and rl > 0.05 else None)
            rows[aid] = {"rel1_gemma": round(rg, 3), "rel1_llama": round(rl, 3),
                         "cross_judge_rho": round(x, 3),
                         "disattenuated": round(min(dis, 1.5), 3) if dis else None,
                         "n_common": len(common)}
        med = lambda k: round(sorted(v[k] for v in rows.values()
                                     if v[k] is not None)[max(0, len([v for v in rows.values() if v[k] is not None]) // 2)], 3) if rows else None
        report["stability"][wave] = {
            "aspects": rows,
            "median_rel1_gemma": med("rel1_gemma"),
            "median_rel1_llama": med("rel1_llama"),
            "median_cross_judge": med("cross_judge_rho"),
            "scope_jaccard": (round(len(gscope & lscope) / len(gscope | lscope), 3)
                              if gscope and lscope else None)}

        cells = []
        for key, col in code.items():
            aid = key.rsplit("_v", 1)[0]
            if aid not in gch or aid not in lch or not isinstance(col, dict):
                continue
            sg = [d for d in gch[aid] if col.get(d) is not None]
            sl = [d for d in lch[aid] if col.get(d) is not None]
            if len(sg) < 30 or len(sl) < 30:
                continue
            rg = spearman([col[d] for d in sg], [gch[aid][d] for d in sg])
            rl = spearman([col[d] for d in sl], [lch[aid][d] for d in sl])
            if rg == rg and rl == rl:
                cells.append((key, round(rg, 3), round(rl, 3)))
        order_rho = spearman([c[1] for c in cells], [c[2] for c in cells])
        report["code_rung"][wave] = {
            "n_cells": len(cells),
            "codability_order_replication_rho": round(order_rho, 3),
            "cells": {c[0]: {"rho_gemma": c[1], "rho_llama": c[2]} for c in cells}}
        print(f"[{wave}] {len(rows)} aspects | med rel1 G={report['stability'][wave]['median_rel1_gemma']} "
              f"L={report['stability'][wave]['median_rel1_llama']} | med cross-judge "
              f"{report['stability'][wave]['median_cross_judge']} | code-rung order rho "
              f"{order_rho:+.3f} over {len(cells)} cells")

    # ---- C: v1 gate replication under the Llama channel -------------------------
    gch, grel, _ = load_channels(OUT / "v1/results_v1.jsonl")
    lch, lrel, lscope = load_channels(OUT / "v1/results_llama.jsonl")
    code = json.load(open(OUT / "v1/code_scores_v1.json"))
    train, test = split_ids()
    FLAVORS = ["v0_keyword", "v1_structure", "v2_holistic"]

    def boot(sel, hyb, base, judge, seed=17):
        rng = random.Random(seed)
        n = len(sel)
        pg = pb = used = 0
        for _ in range(B):
            idx = [sel[rng.randrange(n)] for _ in range(n)]
            rh = spearman([hyb[d] for d in idx], [judge[d] for d in idx])
            rb = spearman([base[d] for d in idx], [judge[d] for d in idx])
            if rh != rh or rb != rb:
                continue
            used += 1
            pg += rh >= max(rb + 0.10, 0.60)
            pb += rh > rb
        return (pg / used if used else None, pb / used if used else None)

    for aid, htag in [("a80", "h0"), ("a86", "h0"), ("a105", "h0"), ("a110", "h0"),
                      ("a80", "h1")]:
        hyb = {k: v for k, v in json.load(
            open(OUT / f"v1/hybrid_scores_{aid}_{htag}.json")).items() if v is not None}
        # frozen baseline: best flavor by GEMMA train rho (original selection, no re-pick)
        best_fl, best_tr = None, -2
        for fl in FLAVORS:
            col = code.get(f"{aid}_{fl}") or {}
            sel = [d for d in train if d in gch.get(aid, {}) and col.get(d) is not None]
            if len(sel) < 30:
                continue
            r = spearman([col[d] for d in sel], [gch[aid][d] for d in sel])
            if r == r and r > best_tr:
                best_fl, best_tr = fl, r
        base = code[f"{aid}_{best_fl}"]
        rows = {}
        for jname, jch in [("gemma", gch[aid]), ("llama", lch[aid])]:
            for scope_name, idset in [("full", test),
                                      ("scoped", {d for d in test if d in lscope})]:
                sel = [d for d in idset if d in jch and hyb.get(d) is not None
                       and base.get(d) is not None]
                if len(sel) < 20:
                    rows[f"{jname}_{scope_name}"] = {"n": len(sel)}
                    continue
                rh = spearman([hyb[d] for d in sel], [jch[d] for d in sel])
                rb = spearman([base[d] for d in sel], [jch[d] for d in sel])
                pg, pb = boot(sel, hyb, base, jch)
                rows[f"{jname}_{scope_name}"] = {
                    "n": len(sel), "rho_hybrid": round(rh, 3),
                    "rho_baseline": round(rb, 3), "P_gate": pg, "P_beats": pb}
        report["gates_v1"][f"{aid}_{htag}"] = {"baseline_flavor": best_fl, **rows}
        g, l = rows.get("gemma_full", {}), rows.get("llama_full", {})
        print(f"{aid}_{htag} [{best_fl}] gemma full: hyb {g.get('rho_hybrid')} "
              f"P(gate)={g.get('P_gate')} | llama full: hyb {l.get('rho_hybrid')} "
              f"P(gate)={l.get('P_gate')} P(beats)={l.get('P_beats')}")

    json.dump(report, open(REP / "replication_report.json", "w"), indent=1)
    print(f"-> {REP / 'replication_report.json'}")


if __name__ == "__main__":
    main()
