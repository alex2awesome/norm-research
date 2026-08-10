"""h1-vs-h0-vs-baseline gate eval on the SAME held-out test split (task-generic).

Promotion discipline (the a80 lesson, kill-switch protocol): h0 stays HEAD unless h1 clears —
we report P(h1 > h0) and the G1 gate for both, paired bootstrap B=2000 on identical items.
h1 fields = h0's stored extractions for unchanged fields + namespaced "<aid>.h1__<field>"
extractions (field_results_h1.jsonl) for new/changed fields.

Usage: python3 eval_h1_task.py <task> <progdir> <aid1,aid2,...> [math]
-> outputs/metric_seam_pilot/tasks/<task>/h1_gate_report.json
"""
import importlib.util, json, pathlib, random, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from certificates import spearman                                        # noqa: E402
from eval_hybrids_task import (split_task_ids, load_judge, load_fields,  # noqa: E402
                               load_mod, run_prog)
from ops import Ops                                                      # noqa: E402

B = 2000


def load_h1_fields(path):
    """field_results_h1.jsonl rows keyed '<aid>.h1__<field>' -> {aid: {dpid: {field: ans}}}."""
    out = {}
    if not path.exists():
        return out
    for line in open(path):
        r = json.loads(line)
        if r.get("channel") != "field":
            continue
        key, field = r["aspect_id"].split("__", 1)
        aid = key.split(".")[0]
        ans = (r.get("raw") or "").strip()
        if ans.upper() == "NONE":
            ans = ""
        out.setdefault(aid, {}).setdefault(r["datapoint_id"], {})[field] = ans
    return out


def main():
    task, progdir, aids = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    use_math = len(sys.argv) > 4 and sys.argv[4] == "math"
    OUT = ROOT / "outputs/metric_seam_pilot/tasks" / task
    HYB = pathlib.Path(__file__).parent / progdir

    items_l = json.load(open(OUT / "items.json"))
    items = {x["datapoint_id"]: x["ctext"] for x in items_l}
    _, test = split_task_ids(items_l)
    judge, rel = load_judge(OUT / "results.jsonl")
    f_h0 = load_fields(OUT / "field_results.jsonl")
    f_h1x = load_h1_fields(OUT / "field_prompts_h1.jsonl".replace("prompts", "results"))
    f_h1x = f_h1x or load_h1_fields(OUT / "field_results_h1.jsonl")
    code = json.load(open(OUT / "code_scores.json"))
    gate_rep = json.load(open(OUT / "hybrid_gate_report.json"))
    if use_math:
        from ops_math import MathOps
        ops = MathOps(corpus_path=str(OUT / "items.json"))
    else:
        ops = Ops(corpus_path=str(OUT / "items.json"))

    report = {}
    for aid in aids:
        m0 = load_mod(HYB / f"{aid}_h0.py")
        m1 = load_mod(HYB / f"{aid}_h1.py")
        f0_names = dict(getattr(m0, "LLM_FIELDS", {}) or {})
        f1_names = dict(getattr(m1, "LLM_FIELDS", {}) or {})
        # h1 per-item fields: unchanged -> h0 store; new/changed -> namespaced h1 store
        fl0 = f_h0.get(aid, {})
        fl1x = f_h1x.get(aid, {})
        fl1 = {}
        for d in items:
            row = {}
            for k, v in f1_names.items():
                if f0_names.get(k) == v:
                    row[k] = fl0.get(d, {}).get(k, "")
                else:
                    row[k] = fl1x.get(d, {}).get(k, "")
            fl1[d] = row
        missing = sum(1 for d in test if any(
            f0_names.get(k) != v and k not in fl1x.get(d, {})
            for k, v in f1_names.items()))
        col0 = run_prog(m0.score, items, fl0, ops)
        col1 = run_prog(m1.score, items, fl1, ops)

        fl_frozen = gate_rep[aid]["baseline_flavor"]
        base = code[f"{aid}_{fl_frozen}"]
        sel = [d for d in test if d in judge.get(aid, {})
               and col0.get(d) is not None and col1.get(d) is not None
               and base.get(d) is not None]
        j = judge[aid]
        r0 = spearman([col0[d] for d in sel], [j[d] for d in sel])
        r1 = spearman([col1[d] for d in sel], [j[d] for d in sel])
        rb = spearman([base[d] for d in sel], [j[d] for d in sel])
        rng = random.Random(17)
        n = len(sel)
        pg0 = pg1 = p10 = used = 0
        for _ in range(B):
            idx = [sel[rng.randrange(n)] for _ in range(n)]
            jj = [j[d] for d in idx]
            b0 = spearman([col0[d] for d in idx], jj)
            b1 = spearman([col1[d] for d in idx], jj)
            bb = spearman([base[d] for d in idx], jj)
            if b0 != b0 or b1 != b1 or bb != bb:
                continue
            used += 1
            pg0 += b0 >= max(bb + .10, .60)
            pg1 += b1 >= max(bb + .10, .60)
            p10 += b1 > b0
        rep = {"n": n, "missing_h1_fields_on_test": missing,
               "rho_h0": round(r0, 3), "rho_h1": round(r1, 3),
               "rho_baseline": round(rb, 3),
               "P_gate_h0": round(pg0 / used, 4), "P_gate_h1": round(pg1 / used, 4),
               "P_h1_gt_h0": round(p10 / used, 4), "boot_used": used,
               "head": "h1" if p10 / used >= 0.8 else "h0"}
        report[aid] = rep
        print(f"{aid}: h0 {r0:+.3f} -> h1 {r1:+.3f} (base {rb:+.3f})  "
              f"P(gate) {rep['P_gate_h0']}->{rep['P_gate_h1']}  "
              f"P(h1>h0)={rep['P_h1_gt_h0']}  HEAD={rep['head']}")
    json.dump(report, open(OUT / "h1_gate_report.json", "w"), indent=1)
    print(f"-> {OUT / 'h1_gate_report.json'}")


if __name__ == "__main__":
    main()
