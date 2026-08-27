"""WS3 eval: op-marginals against the evidence-aware judge M̄(x,Z) (runbook 2026-07-10).

Pre-registered readouts, per evidence-dominant aspect (a26, a34, a60, a35):
  1. 2-pass reliability of the NEW targets (evidence arm = M̄(x,Z); filler arm =
     instruction-load control) — new target = new ceiling, read BEFORE any seam claim.
  2. Op-marginal of the R7.1 PriorArtOps hybrids: gate(hyb_full vs hyb_null) against
     (a) M̄(x,Z)   expect POSITIVE for evidence-dominant criteria,
     (b) M̄(x)     doc-only judge (existing results.jsonl) — expect ~0, the R7.1 null
                   replicates by design (I(M̄(X);Z|X)=0),
     (c) filler   judge saw doc + inert text — expect ~(b); anything else = payload
                   FORMAT (not content) moved the judge.
  3. Descriptive: rho(exposure, judge) per target — does the evidence judge track the
     payload's own novelty-exposure summary? (M̄(x,Z) should; M̄(x) shouldn't.)
Same test split as R7.1 (rng(7), 40%) — apples-to-apples with eval_patents_pa.py.

Usage (laptop, after ws3_evidence_results.jsonl is rsynced): python3 ws3_eval_evidence.py
"""
import importlib.util
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/metric_seam_pilot/tasks/patents_pa"
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from certificates import spearman, attenuation_ceiling, bootstrap_gate  # noqa: E402
from ops_pa import PriorArtOps, NullPriorArtOps                          # noqa: E402
from eval_patents_pa import load_all, clean                              # noqa: E402,F401

ASPECTS = ["a26", "a34", "a60", "a35"]


def load_ws3():
    arms = {}
    for line in open(OUT / "ws3_evidence_results.jsonl"):
        r = json.loads(line)
        if not isinstance(r["score"], int):
            continue
        arm, ch = r["channel"].rsplit("_", 1)
        arms.setdefault(arm, {}).setdefault(ch, {}).setdefault(
            r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
    out = {}
    for arm, passes in arms.items():
        judge, rel1 = {}, {}
        p1, p2 = passes.get("pass1", {}), passes.get("pass2", {})
        for aid in set(p1) | set(p2):
            # v2 (2026-07-10, post external review): judge = INTERSECTION only (items with
            # BOTH passes) — the v1 union mixed 1-pass and 2-pass scores in one target,
            # making the 2-pass attenuation ceiling unjustified for the single-pass slice.
            both = [d for d in p1.get(aid, {}) if d in p2.get(aid, {})]
            rel1[aid] = spearman([p1[aid][d] for d in both], [p2[aid][d] for d in both])
            for dp in both:
                judge.setdefault(aid, {})[dp] = (p1[aid][dp] + p2[aid][dp]) / 2 / 10.0
        out[arm] = {"judge": judge, "rel1": rel1}
    return out


def load_doc_intersection():
    """Doc-only judge rebuilt INTERSECTION-only from results.jsonl (same v2 rule as the
    WS3 arms; the frozen R7.1 loader (eval_patents_pa.load_all) union-averages and is
    left untouched — this is for apples-to-apples within THIS report only)."""
    p1, p2 = {}, {}
    for line in open(OUT / "results.jsonl"):
        r = json.loads(line)
        if r["channel"] not in ("pass1", "pass2") or not isinstance(r["score"], int):
            continue
        d = p1 if r["channel"] == "pass1" else p2
        d.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
    judge, rel1 = {}, {}
    for aid in set(p1) | set(p2):
        both = [d for d in p1.get(aid, {}) if d in p2.get(aid, {})]
        rel1[aid] = spearman([p1[aid][d] for d in both], [p2[aid][d] for d in both])
        for dp in both:
            judge.setdefault(aid, {})[dp] = (p1[aid][dp] + p2[aid][dp]) / 2 / 10.0
    return judge, rel1


def main():
    _, _, _, fields = load_all()
    doc_judge, doc_rel1 = load_doc_intersection()
    ws3 = load_ws3()
    items = json.load(open(OUT / "items.json"))
    texts = {it["datapoint_id"]: it["ctext"] for it in items}
    feats = json.load(open(OUT / "pa_features.json"))
    exposure = {d: 1.0 - f.get("frac_claims_any_disclose", 0.0) for d, f in feats.items()}
    ops_full = PriorArtOps(OUT / "pa_features.json")
    ops_null = NullPriorArtOps(OUT / "pa_features.json")

    ids = sorted(texts)
    rng = random.Random(7)
    test = set(rng.sample(ids, int(0.4 * len(ids))))

    targets = {"evidence": ws3["evidence"]["judge"], "doc_only": doc_judge,
               "filler": ws3["filler"]["judge"]}
    rel1s = {"evidence": ws3["evidence"]["rel1"], "doc_only": doc_rel1,
             "filler": ws3["filler"]["rel1"]}

    report = {}
    for aid in ASPECTS:
        prog = pathlib.Path(__file__).parent / "programs_pa" / f"{aid}_h0.py"
        spec = importlib.util.spec_from_file_location(prog.stem, prog)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fmap = {d: {f.split("__", 1)[1]: fields[f].get(d, "")
                    for f in fields if f.startswith(aid + "__")} for d in ids}
        hyb_full = {d: mod.score(texts[d], fmap[d], ops_full, dpid=d) for d in ids}
        hyb_null = {d: mod.score(texts[d], fmap[d], ops_null, dpid=d) for d in ids}

        row = {}
        for tname, judge in targets.items():
            j = judge.get(aid, {})
            tsel = [d for d in test if d in j]
            r1 = rel1s[tname].get(aid, float("nan"))
            g_op = bootstrap_gate(hyb_full, hyb_null, j, tsel,
                                  margin=0.0, floor=0.0, B=2000, skip_undefined=True)
            sel = [d for d in tsel if d in exposure]
            r_exp = spearman([exposure[d] for d in sel], [j[d] for d in sel])
            rho_full = spearman([hyb_full[d] for d in tsel], [j[d] for d in tsel])
            rho_null = spearman([hyb_null[d] for d in tsel], [j[d] for d in tsel])
            row[tname] = {
                "rel1": round(r1, 3) if r1 == r1 else None,
                "ceiling": (round(attenuation_ceiling(min(max(r1, 0), 1), 2), 3)
                            if r1 == r1 else None),
                "rho_full": round(rho_full, 3), "rho_null": round(rho_null, 3),
                "op_marginal": round(rho_full - rho_null, 3),
                "P_op_beats_null": g_op["P_beats_baseline"],
                "rho_exposure": round(r_exp, 3), "n_test": len(tsel)}
        report[aid] = row

    json.dump(report, open(OUT / "ws3_eval_report.json", "w"), indent=1)
    hdr = f"{'aid':5} {'target':9} {'rel1':>5} {'ceil':>5} {'full':>6} {'null':>6} " \
          f"{'op_m':>6} {'P>null':>6} {'expo':>6}"
    print(hdr)
    for aid, row in report.items():
        for tname, c in row.items():
            v = {k: ("--" if x is None else x) for k, x in c.items()}
            print(f"{aid:5} {tname:9} {v['rel1']:>5} {v['ceiling']:>5} {v['rho_full']:>6} "
                  f"{v['rho_null']:>6} {v['op_marginal']:>6} {v['P_op_beats_null']:>6} "
                  f"{v['rho_exposure']:>6}")
    print(f"-> {OUT / 'ws3_eval_report.json'}")


if __name__ == "__main__":
    main()
