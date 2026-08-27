"""W1.3: re-bootstrap the v1 hybrid gates on the EXPANDED held-out set (100 old + 400 new).

Nothing is retrained: h0 hybrids and description-compiled baselines are frozen; the new 400
items are pure held-out. Gate G1 (rho_test >= max(baseline+0.10, 0.60)) and P(beats
baseline) via PAIRED bootstrap (same item resample for both channels), B=2000 — the Rung-3
form that the n=100 run could not resolve for a110 (P=.59) and a80 (P=.31).

Needs: expansion/results_exp.jsonl + expansion/field_results_exp.jsonl (queue jobs 61/62).
"""
import importlib.util, json, pathlib, random, signal, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam/hybrids"))
from harness import load_judge, load_scope, split_ids, spearman  # noqa: E402
from ops import Ops                                              # noqa: E402

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


def run_prog(fn, texts, *extra):
    col = {}
    for dpid, t in texts.items():
        try:
            signal.alarm(15)
            col[dpid] = float(fn(t, *extra(dpid) if callable(extra) else extra))
        except Exception:
            col[dpid] = None
        finally:
            signal.alarm(0)
    return col


def load_exp_judge():
    p1, p2, sc = {}, {}, {}
    for line in open(EXP / "results_exp.jsonl"):
        r = json.loads(line)
        if not isinstance(r["score"], int):
            continue
        if r["channel"] == "scope":
            sc[r["datapoint_id"]] = r["score"]
            continue
        d = p1 if r["channel"] == "pass1" else p2
        d.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
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
        xs = [hyb[d] for d in idx]
        bs = [base[d] for d in idx]
        ys = [judge[d] for d in idx]
        rh, rb = spearman(xs, ys), spearman(bs, ys)
        if rh != rh or rb != rb:
            continue
        used += 1
        p_gate += rh >= max(rb + margin, gate_floor)
        p_beat += rh > rb
    return (p_gate / used if used else None, p_beat / used if used else None, used)


def main():
    judge_v1, _, _ = load_judge()
    scope_v1, _ = load_scope()
    judge_exp, scope_exp = load_exp_judge()
    items = {x["datapoint_id"]: x["ctext"] for x in json.load(open(V1 / "items_v1.json"))}
    items_exp = {x["datapoint_id"]: x["ctext"]
                 for x in json.load(open(EXP / "items_exp.json"))}
    _, test_v1 = split_ids()
    all_texts = {**items, **items_exp}
    ops = Ops(corpus_path=str(ROOT / "runs/validity_full/v2/press_releases/datapoints.json"))

    # v1 fields from llm_fields/, expansion fields from the new batch
    exp_fields = load_exp_fields()

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
        judge = {**judge_v1.get(aid, {}), **judge_exp.get(aid, {})}
        test_ids = [d for d in list(test_v1) + list(items_exp)
                    if d in judge]
        train_ids = [d for d in judge_v1.get(aid, {}) if d not in test_v1]

        # frozen baseline: best flavor by TRAIN rho on v1 train (original protocol)
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
            sel = [d for d in train_ids if col.get(d) is not None]
            r = spearman([col[d] for d in sel], [judge[d] for d in sel])
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
                            ("scoped", [d for d in test_ids
                                        if d in scope_v1 | scope_exp])]:
            sel = [d for d in idset
                   if hyb_col.get(d) is not None and base_col.get(d) is not None]
            rh = spearman([hyb_col[d] for d in sel], [judge[d] for d in sel])
            rb = spearman([base_col[d] for d in sel], [judge[d] for d in sel])
            pg, pb, used = paired_boot(sel, hyb_col, base_col, judge)
            rows[name] = {"n": len(sel), "rho_hybrid": round(rh, 3),
                          "rho_baseline": round(rb, 3),
                          "P_gate": pg, "P_beats_baseline": pb, "boot_used": used}
            print(f"{aid} [{name}] n={len(sel)}: hyb {rh:+.3f} vs base({best_fl}) "
                  f"{rb:+.3f}  P(gate)={pg}  P(beats)={pb}")
        report[aid] = {"baseline_flavor": best_fl, **rows}

    json.dump(report, open(EXP / "gate_expansion_report.json", "w"), indent=1)
    print(f"-> {EXP/'gate_expansion_report.json'}")


if __name__ == "__main__":
    main()
