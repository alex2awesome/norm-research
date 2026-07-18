"""Multi-task GEPA final: freeze argmax-dev prompts -> TEST prompts file ('build'),
then G-only held-out eval with degeneracy guards ('eval') per task.

  python3 final_g_eval.py build                      -> gepa_final_prompts.jsonl
  python3 final_g_eval.py eval gepa_final_results.jsonl
      -> per-task refined F (base+ceiling from each task's gate source) ->
         gepa_multi_refined_final.json
"""
import json, math, sys, pathlib
from collections import Counter

from common import CRITERIA, HERE, ROOT, BASE, build_doc_prompt, crit_key, load_ctx, load_state

sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman  # noqa: E402


def ceiling(rel1, k=2):
    r = max(0.0, min(1.0, rel1)); relk = k * r / (1 + (k - 1) * r)
    return math.sqrt(relk) if relk > 0 else float("nan")


def clip01(x):
    return max(0.0, min(1.0, x))


def variance_ok(vals, min_minority=5):
    c = Counter(vals)
    return len(vals) - c.most_common(1)[0][1] >= min_minority


def gate_row(task, aid):
    """-> (rho_baseline, ceiling) from the task's canonical gate source."""
    if task == "patents_pa":
        h = json.load(open(BASE / "tasks/patents_pa/pa_eval.json"))["hybrids"][aid]
        return h["baseline"]["rho_test"], h["ceiling"]
    g = json.load(open(BASE / f"tasks/{task}/hybrid_gate_report.json"))[aid]
    return g["full"]["rho_baseline"], ceiling(g["judge_rel1"])


def cmd_build():
    state = load_state()
    ctxs = {}
    out_path = HERE / "gepa_final_prompts.jsonl"
    frozen = {}
    n = 0
    with open(out_path, "w") as f:
        for task, aid in CRITERIA:
            key = crit_key(task, aid)
            c = state["criteria"][key]
            best = c["best"] or {"prompt": c["prompt"], "round": c["round"], "rho_dev": None}
            frozen[key] = {"round": best["round"], "rho_dev": best.get("rho_dev")}
            if task not in ctxs:
                ctxs[task] = load_ctx(task)
            items, test_ids = ctxs[task]["items"], sorted(ctxs[task]["test"])
            for dpid in test_ids:
                f.write(json.dumps({"channel": "field", "aspect_id": f"{key}.final",
                                    "datapoint_id": dpid,
                                    "prompt": build_doc_prompt(best["prompt"], items.get(dpid, ""))}) + "\n")
                n += 1
            print(f"{key}: frozen round {best['round']} (dev {best.get('rho_dev')})")
    json.dump(frozen, open(HERE / "gepa_final_frozen_meta.json", "w"), indent=1)
    print(f"wrote {n} rows -> {out_path}")


def cmd_eval(results_path):
    g_by = {}
    for line in open(results_path):
        r = json.loads(line)
        a = r.get("aspect_id", "")
        if not a.endswith(".final"):
            continue
        key = a[:-len(".final")]
        if isinstance(r.get("score"), int):
            g_by.setdefault(key, {})[r["datapoint_id"]] = r["score"]
    ctxs = {}
    per_task = {}
    for task, aid in CRITERIA:
        key = crit_key(task, aid)
        if task not in ctxs:
            ctxs[task] = load_ctx(task)
        ctx = ctxs[task]
        judge = ctx["judge"].get(aid, {})
        col = g_by.get(key, {})
        sel = [d for d in sorted(ctx["test"]) if d in judge and col.get(d) is not None]
        if len(sel) < 20:
            print(f"SKIP {key}: n={len(sel)}"); continue
        gv = [col[d] for d in sel]; jv = [judge[d] for d in sel]
        if not variance_ok(gv) or not variance_ok(jv):
            print(f"SKIP {key}: degenerate"); continue
        rho = spearman(gv, jv)
        if rho != rho:
            print(f"SKIP {key}: nan"); continue
        base, ceil = gate_row(task, aid)
        r_base, r_g = clip01(base / ceil), clip01(rho / ceil)
        per_task.setdefault(task, []).append(
            dict(aid=aid, G=round(rho, 4), r_base=round(r_base, 3), r_G=round(r_g, 3),
                 n=len(sel)))
        print(f"{key}: G={rho:+.3f} r_base={r_base:.3f} r_G={r_g:.3f} n={len(sel)}")
    import statistics as st
    summary = {}
    for task, rows in per_task.items():
        mb = st.median(r["r_base"] for r in rows); mg = st.median(r["r_G"] for r in rows)
        F = (mg - mb) / mg if mg else float("nan")
        summary[task] = dict(n=len(rows), med_r_base=round(mb, 3), med_r_G=round(mg, 3),
                             seam_width_F=round(F, 3), arm="GEPA-refined (argmax-dev)")
        print(f"\n{task}: V={mb:.3f} V+A={mg:.3f} -> refined F={F:.3f} (n={len(rows)})")
    json.dump({"per_task": per_task, "summary": summary},
              open(HERE / "gepa_multi_refined_final.json", "w"), indent=1)
    print("-> gepa_multi_refined_final.json")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        cmd_build()
    else:
        cmd_eval(sys.argv[2])
