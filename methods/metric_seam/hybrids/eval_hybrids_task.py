"""Task-generic hybrid gate eval (wave-2 protocol), mirrors eval_gate_expansion.py.

Usage: python3 eval_hybrids_task.py <task> <progdir> [math|code]
  <task>    e.g. creative_writing, math    (outputs/metric_seam_pilot/tasks/<task>/)
  <progdir> e.g. programs_cw, programs_math (methods/metric_seam/hybrids/<progdir>/)
  optional 3rd arg "math" -> use MathOps; "code" -> use outcome-blind CodeOps.

Split: SAME as build_packs_task.py (150 train / 100 test, seed 7, sorted datapoint_id).
Frozen: baseline flavor selected on TRAIN by rho (never re-picked at eval time, but this
IS the eval time here — first cert run, so this selection is also the aspect's canonical
baseline choice going forward). Gate G1: rho_test >= max(baseline_rho_test + 0.10, 0.60),
Rung-3 paired bootstrap B=2000 (same-item resample for hybrid vs baseline).
-> outputs/metric_seam_pilot/tasks/<task>/hybrid_gate_report.json
"""
import importlib.util, json, pathlib, random, signal, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from certificates import spearman  # noqa: E402
from ops import Ops                # noqa: E402

FLAVORS = ["v0_keyword", "v1_structure", "v2_holistic"]
B = 2000


def _alarm(sig, frame):
    raise TimeoutError()


signal.signal(signal.SIGALRM, _alarm)


def split_task_ids(items):
    ids = sorted(x["datapoint_id"] for x in items)
    random.Random(7).shuffle(ids)
    return set(ids[:150]), set(ids[150:])


def load_mod(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_judge(results_path):
    p1, p2 = {}, {}
    for line in open(results_path):
        r = json.loads(line)
        if not isinstance(r["score"], int) or r["channel"] not in ("pass1", "pass2"):
            continue
        (p1 if r["channel"] == "pass1" else p2).setdefault(
            r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
    judge, rel = {}, {}
    for aid in set(p1) & set(p2):
        both = sorted(set(p1[aid]) & set(p2[aid]))
        if len(both) < 30:
            continue
        rel[aid] = spearman([p1[aid][d] for d in both], [p2[aid][d] for d in both])
        judge[aid] = {d: (p1[aid][d] + p2[aid][d]) / 2 / 10.0 for d in both}
    return judge, rel


def load_fields(field_results_path):
    out = {}
    if not field_results_path.exists():
        return out
    for line in open(field_results_path):
        r = json.loads(line)
        if r.get("channel") != "field":
            continue
        aid, field = r["aspect_id"].split("__", 1)
        ans = (r.get("raw") or "").strip()
        if ans.upper() == "NONE":
            ans = ""
        out.setdefault(aid, {}).setdefault(r["datapoint_id"], {})[field] = ans
    return out


def run_prog(fn, texts, fields, ops):
    col = {}
    for dpid, t in texts.items():
        try:
            signal.alarm(20)
            col[dpid] = float(fn(t, fields.get(dpid, {}), ops))
        except Exception:
            col[dpid] = None
        finally:
            signal.alarm(0)
    return col


def paired_boot(sel, hyb, base, judge, gate_floor=0.60, margin=0.10, seed=17):
    rng = random.Random(seed)
    n = len(sel)
    p_gate = p_beat = used = 0
    for _ in range(B):
        idx = [sel[rng.randrange(n)] for _ in range(n)]
        rh = spearman([hyb[d] for d in idx], [judge[d] for d in idx])
        rb = spearman([base[d] for d in idx], [judge[d] for d in idx])
        if rh != rh or rb != rb:
            continue
        used += 1
        p_gate += rh >= max(rb + margin, gate_floor)
        p_beat += rh > rb
    return (p_gate / used if used else None, p_beat / used if used else None, used)


def main():
    task, progdir = sys.argv[1], sys.argv[2]
    use_math = len(sys.argv) > 3 and sys.argv[3] == "math"
    use_code = len(sys.argv) > 3 and sys.argv[3] == "code"
    OUT = ROOT / "outputs/metric_seam_pilot/tasks" / task
    HYB = pathlib.Path(__file__).parent / progdir
    CG = ROOT / "runs/validity_full/v2" / task / "codegen_claude"

    items_l = json.load(open(OUT / "items.json"))
    items = {x["datapoint_id"]: x["ctext"] for x in items_l}
    train, test = split_task_ids(items_l)
    judge, rel = load_judge(OUT / "results.jsonl")
    fields = load_fields(OUT / "field_results.jsonl")
    code = json.load(open(OUT / "code_scores.json"))

    if use_math:
        from ops_math import MathOps
        ops = MathOps(corpus_path=str(OUT / "items.json"))
    elif use_code:
        from ops_code import CodeOps
        ops = CodeOps()
    else:
        ops = Ops(corpus_path=str(OUT / "items.json"))

    report = {}
    progs = sorted(HYB.glob("*_h0.py"))
    print(f"{len(progs)} hybrid programs in {HYB}")
    for prog in progs:
        aid = prog.stem.split("_")[0]
        if aid not in judge:
            print(f"{aid}: skip (no judge channel)")
            continue
        try:
            mod = load_mod(prog)
        except Exception as e:
            print(f"{aid}: LOAD FAIL {e}")
            report[aid] = {"error": f"load: {e}"}
            continue

        # frozen baseline: best flavor by TRAIN rho (same recipe as build_packs_task.py)
        best_fl, best_tr, base_col = None, -2, None
        for fl in FLAVORS:
            p = CG / f"{aid}_{fl}.py"
            if not p.exists():
                continue
            col = code.get(f"{aid}_{fl}") or {}
            sel = [d for d in train if d in judge[aid] and col.get(d) is not None]
            if len(sel) < 30:
                continue
            r = spearman([col[d] for d in sel], [judge[aid][d] for d in sel])
            if r == r and r > best_tr:
                best_fl, best_tr, base_col = fl, r, col
        if base_col is None:
            print(f"{aid}: skip (no runnable baseline flavor)")
            continue

        fl_map = fields.get(aid, {})
        hyb_col = run_prog(mod.score, items, fl_map, ops)

        rows = {}
        test_ids = list(test)
        sel = [d for d in test_ids if hyb_col.get(d) is not None
               and base_col.get(d) is not None and d in judge[aid]]
        rh = spearman([hyb_col[d] for d in sel], [judge[aid][d] for d in sel])
        rb = spearman([base_col[d] for d in sel], [judge[aid][d] for d in sel])
        pg, pb, used = paired_boot(sel, hyb_col, base_col, judge[aid])
        rows["full"] = {"n": len(sel), "rho_hybrid": round(rh, 3) if rh == rh else None,
                        "rho_baseline": round(rb, 3) if rb == rb else None,
                        "P_gate": pg, "P_beats_baseline": pb, "boot_used": used}
        has_fields = bool(getattr(mod, "LLM_FIELDS", {}))
        report[aid] = {"baseline_flavor": best_fl, "baseline_train_rho": round(best_tr, 3),
                       "judge_rel1": round(rel.get(aid, float("nan")), 3),
                       "has_llm_fields": has_fields, **rows}
        print(f"{aid} [{best_fl}] n={len(sel)}: hyb {rh:+.3f} vs base {rb:+.3f}  "
              f"P(gate)={pg}  P(beats)={pb}  fields={has_fields}")

    json.dump(report, open(OUT / "hybrid_gate_report.json", "w"), indent=1)
    n_cert = sum(1 for v in report.values() if v.get("full", {}).get("P_gate", 0) and
                v["full"]["P_gate"] >= 0.5)
    print(f"-> {OUT/'hybrid_gate_report.json'}  ({n_cert}/{len(report)} certified P(gate)>=.5)")


if __name__ == "__main__":
    main()
