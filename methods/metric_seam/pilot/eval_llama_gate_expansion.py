"""W1.2 + W1.3 combined: re-bootstrap the v1 hybrid gates on the EXPANDED held-out set
(100 old + 400 new) under the LLAMA-3.3-70B judge channel — second-family replication at
full n=500 power, mirroring eval_gate_expansion.py's Gemma protocol exactly.

Frozen: h0 hybrids, description-compiled baseline flavors (selected on GEMMA train, original
protocol — no re-pick), LLM fields (Gemma extractor, unchanged). Only the JUDGE channel
differs. Needs outputs/metric_seam_pilot/v1/expansion/results_llama_exp.jsonl (queue job 83).
"""
import importlib.util, json, pathlib, random, signal, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam/hybrids"))
from harness import split_ids, spearman  # noqa: E402
from ops import Ops                       # noqa: E402

V1 = ROOT / "outputs/metric_seam_pilot/v1"
EXP = V1 / "expansion"
HYB = ROOT / "methods/metric_seam/hybrids/programs"
CG = ROOT / "runs/validity_full/v2/press_releases/codegen_claude"
ASPECTS = ["a80", "a86", "a105", "a110"]
FLAVORS = ["v0_keyword", "v1_structure", "v2_holistic"]
B = 2000


def _alarm(sig, frame):
    raise TimeoutError()


signal.signal(signal.SIGALRM, _alarm)


def load_mod(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_llama_judge(path):
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
    comb = {}
    for aid in set(p1) | set(p2):
        for dpid in set(p1.get(aid, {})) | set(p2.get(aid, {})):
            vals = [d[aid][dpid] for d in (p1, p2) if dpid in d.get(aid, {})]
            comb.setdefault(aid, {})[dpid] = sum(vals) / len(vals) / 10.0
    return comb, {d for d, s in sc.items() if s >= 7}


def load_exp_fields():
    out = {}
    f = EXP / "field_results_exp.jsonl"
    if not f.exists():
        return out
    for line in open(f):
        r = json.loads(line)
        if r["channel"] != "field":
            continue
        aid, field = r["aspect_id"].split("__", 1)
        ans = (r.get("raw") or "").strip()
        if ans.upper() == "NONE":
            ans = ""
        out.setdefault(aid, {}).setdefault(r["datapoint_id"], {})[field] = ans
    return out


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
    judge_l_v1, _ = load_llama_judge(V1 / "results_llama.jsonl")
    judge_l_exp, scope_exp = load_llama_judge(EXP / "results_llama_exp.jsonl")
    judge_g_v1 = json.load(open(V1 / "results_v1.jsonl")) if False else None  # unused
    items = {x["datapoint_id"]: x["ctext"] for x in json.load(open(V1 / "items_v1.json"))}
    items_exp = {x["datapoint_id"]: x["ctext"]
                 for x in json.load(open(EXP / "items_exp.json"))}
    _, test_v1 = split_ids()
    all_texts = {**items, **items_exp}
    ops = Ops(corpus_path=str(ROOT / "runs/validity_full/v2/press_releases/datapoints.json"))
    exp_fields = load_exp_fields()

    # GEMMA train channel, for the frozen baseline-flavor selection only (original protocol)
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from harness import load_judge as load_gemma_judge  # noqa: E402
    judge_gemma_v1, _, _ = load_gemma_judge()
    train_ids_by_aid = {aid: [d for d in judge_gemma_v1.get(aid, {}) if d not in test_v1]
                        for aid in ASPECTS}

    def fields_for(aid):
        out = {}
        for f in (V1 / "llm_fields").glob(f"{aid}__*.json"):
            field = f.stem.split("__", 1)[1]
            for dpid, ans in json.load(open(f)).items():
                out.setdefault(dpid, {})[field] = ans
        for dpid, fv in exp_fields.get(aid, {}).items():
            out.setdefault(dpid, {}).update(fv)
        return out

    report = {}
    for aid in ASPECTS:
        judge = {**judge_l_v1.get(aid, {}), **judge_l_exp.get(aid, {})}
        test_ids = [d for d in list(test_v1) + list(items_exp) if d in judge]

        best_fl, best_tr, base_col = None, -2, None
        for fl in FLAVORS:
            p = CG / f"{aid}_{fl}.py"
            if not p.exists():
                continue
            try:
                mod = load_mod(p)
            except Exception:
                continue
            col = {}
            for dpid in all_texts:
                try:
                    signal.alarm(15)
                    col[dpid] = float(mod.score(all_texts[dpid]))
                except Exception:
                    col[dpid] = None
                finally:
                    signal.alarm(0)
            sel = [d for d in train_ids_by_aid[aid] if col.get(d) is not None]
            r = spearman([col[d] for d in sel], [judge_gemma_v1[aid][d] for d in sel])
            if r == r and r > best_tr:
                best_fl, best_tr, base_col = fl, r, col

        hyb_mod = load_mod(HYB / f"{aid}_h0.py")
        fl_map = fields_for(aid)
        hyb_col = {}
        for dpid in all_texts:
            try:
                signal.alarm(15)
                hyb_col[dpid] = float(hyb_mod.score(all_texts[dpid],
                                                    fl_map.get(dpid, {}), ops))
            except Exception:
                hyb_col[dpid] = None
            finally:
                signal.alarm(0)

        rows = {}
        for name, idset in [("full", test_ids),
                            ("scoped", [d for d in test_ids if d in scope_exp])]:
            sel = [d for d in idset
                   if hyb_col.get(d) is not None and base_col.get(d) is not None]
            rh = spearman([hyb_col[d] for d in sel], [judge[d] for d in sel])
            rb = spearman([base_col[d] for d in sel], [judge[d] for d in sel])
            pg, pb, used = paired_boot(sel, hyb_col, base_col, judge)
            rows[name] = {"n": len(sel), "rho_hybrid": round(rh, 3),
                          "rho_baseline": round(rb, 3),
                          "P_gate": pg, "P_beats_baseline": pb, "boot_used": used}
            print(f"{aid} [llama:{name}] n={len(sel)}: hyb {rh:+.3f} vs base({best_fl}) "
                  f"{rb:+.3f}  P(gate)={pg}  P(beats)={pb}")
        report[aid] = {"baseline_flavor": best_fl, **rows}

    json.dump(report, open(EXP / "gate_expansion_report_llama.json", "w"), indent=1)
    print(f"-> {EXP/'gate_expansion_report_llama.json'}")


if __name__ == "__main__":
    main()
