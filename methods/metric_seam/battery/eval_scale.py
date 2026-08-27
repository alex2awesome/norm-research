"""E3 SCALE (+E4 LOCUS) eval — same-family fm staircase on the frozen programs.

Per task, per h0 program: rho on held-out test under fields from each extractor file,
fm = rho_x - rho_blank. Staircase read ONLY within the Llama family
(3B -> 8B -> 70B, sanctioned primary); Gemma/Qwen reported as unpooled replication
points. E4 columns (8bbase / 8binstr, completion-format twins) appear when their
field_results_e4_* files exist.

T-RET expects: fm monotone in scale on fm-bearing criteria; H_spec predicts near-flat.
E4: base ~ instruct locates competence in pretraining; base ~ blank in post-training.

Usage: python3 eval_scale.py [task ...]     (default: all 5)
-> outputs/metric_seam_pilot/battery/eval_scale.json
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, load_mod, load_fields, run_prog, BASE, ROOT  # noqa
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman  # noqa: E402

TASKS = ["press_releases", "creative_writing", "math", "humor", "legal_title_vii"]
FAMFILES = {"llama3b": "field_results_llama3b.jsonl",
            "llama8b": "field_results_llama8b.jsonl",
            "llama70": "field_results_llama.jsonl",
            "qwen": "field_results_qwen.jsonl",
            "e4_8bbase": "field_results_e4_8bbase.jsonl",
            "e4_8binstr": "field_results_e4_8binstr.jsonl",
            "e4_70bbase": "field_results_e4_70bbase.jsonl",
            "e4_70binstr": "field_results_e4_70binstr.jsonl"}
LADDER = ["llama3b", "llama8b", "llama70"]


def rho_on(sel, col, judge):
    s = [d for d in sel if col.get(d) is not None]
    return spearman([col[d] for d in s], [judge[d] for d in s]) if len(s) >= 20 else float("nan")


def med(xs):
    xs = sorted(x for x in xs if x is not None and x == x)
    return round(xs[len(xs) // 2], 3) if xs else None


def eval_task(task):
    ctx = load_ctx(task)
    fams = {"gemma": ctx["f_orig"]}
    for fam, fn in FAMFILES.items():
        fb = load_fields(ctx["outdir"] / fn)
        if fb:
            # e4 files key by "{aid}.e4"; remap to plain aid
            if fam.startswith("e4_"):
                fb = {k.split(".")[0]: v for k, v in fb.items()}
            fams[fam] = fb
    report = {}
    for prog in sorted(ctx["hyb"].glob("a*_h0.py")):
        aid = prog.stem[:-3]
        judge = ctx["judge"].get(aid, {})
        if not judge:
            continue
        mod = load_mod(prog)
        cols = {"blank": run_prog(mod.score, ctx["items"], {}, ctx["ops"])}
        for fam, fb in fams.items():
            if fb.get(aid):
                cols[fam] = run_prog(mod.score, ctx["items"], fb[aid], ctx["ops"])
        tsel = [d for d in ctx["test"] if d in judge
                and all(cols[c].get(d) is not None for c in cols)]
        if len(tsel) < 30:
            report[aid] = {"error": f"n_test={len(tsel)}"}
            continue
        rho = {c: rho_on(tsel, cols[c], judge) for c in cols}
        row = {"n_test": len(tsel),
               "fm": {c: (round(rho[c] - rho["blank"], 3)
                          if rho[c] == rho[c] and rho["blank"] == rho["blank"] else None)
                      for c in cols if c != "blank"},
               "rho_blank": round(rho["blank"], 3) if rho["blank"] == rho["blank"] else None}
        lad = [row["fm"].get(f) for f in LADDER]
        if all(v is not None for v in lad):
            row["monotone"] = bool(lad[0] <= lad[1] <= lad[2])
        report[aid] = row
    return report


def main():
    tasks = sys.argv[1:] or TASKS
    out = {}
    for task in tasks:
        try:
            rep = eval_task(task)
        except Exception as e:
            out[task] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{task}: ERROR {e}")
            continue
        ok = [v for v in rep.values() if "error" not in v]
        fams = sorted({f for v in ok for f in v["fm"]})
        summ = {"n_criteria": len(ok)}
        for f in fams:
            summ[f"median_fm_{f}"] = med([v["fm"].get(f) for v in ok])
        # headline staircase on fm-bearing criteria only (gemma fm >= .10)
        bear = [v for v in ok if (v["fm"].get("gemma") or 0) >= 0.10]
        summ["n_fm_bearing"] = len(bear)
        for f in LADDER:
            summ[f"bearing_median_fm_{f}"] = med([v["fm"].get(f) for v in bear])
        mono = [v["monotone"] for v in bear if "monotone" in v]
        summ["bearing_frac_monotone"] = (round(sum(mono) / len(mono), 3)
                                         if mono else None)
        out[task] = {"criteria": rep, "summary": summ}
        print(f"{task}: {json.dumps(summ)}")
    path = BASE / "battery/eval_scale.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
