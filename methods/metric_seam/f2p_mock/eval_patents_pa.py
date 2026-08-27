"""Certified eval for the patents_pa prior-art evidence-op experiment (after Gemma lands).

Per prior-art hybrid aspect (a26 non-obviousness, a34 novelty bars, a60 prior-art
differentiation, a35 patentability triad):
  channels: best description-compiled code flavor | hybrid with the PRECOMPUTED prior-art op
            (PriorArtOps) | same hybrid with the op nulled (NullPriorArtOps)
  readouts: held-out rho vs judge + bootstrap gate vs code baseline
            + gate(hybrid_full vs hybrid_noop) = the retrieval+disclosure machinery's
            CERTIFIED MARGINAL
Survey block: rel1 / ceiling / best-flavor full-sample r~ for ALL 8 judged aspects (the 4
hybrid targets + a22 grace-period, a25 statutory overview, a36 clarity, a16 abstract).
Anchor (final outcome `judgement`), descriptive ONLY, pooled Spearman per channel + the raw
payload exposure channel (1 - frac_claims_any_disclose). No within-batch guard is available
(no batch variable was retained) — anchors are flagged accordingly, never gates.

Usage (laptop, after pulling results): python3 eval_patents_pa.py
Needs: outputs/metric_seam_pilot/tasks/patents_pa/{items,pa_features,aspects_used}.json,
       results.jsonl, code_scores.json (run_code_flavors_task.py patents_pa via the
       runs/validity_full/v2/patents_pa -> patents symlink).
"""
import importlib.util, json, math, pathlib, random, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/metric_seam_pilot/tasks/patents_pa"
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from certificates import spearman, attenuation_ceiling, bootstrap_gate
from ops_pa import PriorArtOps, NullPriorArtOps

FLAVORS = ["v0_keyword", "v1_structure", "v2_holistic"]
HYB_ASPECTS = ["a26", "a34", "a60", "a35"]


def clean(raw):
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    line = re.sub(r"^(answer|reply)\s*[:\-]\s*", "", line, flags=re.I).strip()
    return "" if line.upper().startswith("NONE") else line[:200]


def load_all():
    p1, p2, scope, fields = {}, {}, {}, {}
    for line in open(OUT / "results.jsonl"):
        r = json.loads(line)
        if r["channel"] == "field":
            fields.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = clean(r.get("raw", ""))
            continue
        if not isinstance(r["score"], int):
            continue
        d = {"pass1": p1, "pass2": p2}.get(r["channel"])
        if d is not None:
            d.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
        elif r["channel"] == "scope":
            scope[r["datapoint_id"]] = r["score"]
    judge = {}
    for aid in set(p1) | set(p2):
        for dp in set(p1.get(aid, {})) | set(p2.get(aid, {})):
            vals = [m[aid][dp] for m in (p1, p2) if dp in m.get(aid, {})]
            judge.setdefault(aid, {})[dp] = sum(vals) / len(vals) / 10.0
    rel1 = {aid: spearman([p1[aid][d] for d in p1.get(aid, {}) if d in p2.get(aid, {})],
                          [p2[aid][d] for d in p1.get(aid, {}) if d in p2.get(aid, {})])
            for aid in set(p1) & set(p2)}
    return judge, rel1, scope, fields


def main():
    judge, rel1, scope, fields = load_all()
    items = json.load(open(OUT / "items.json"))
    texts = {it["datapoint_id"]: it["ctext"] for it in items}
    outcome = {it["datapoint_id"]: it["judgement"] for it in items}
    feats = json.load(open(OUT / "pa_features.json"))
    code = json.load(open(OUT / "code_scores.json"))
    ops_full = PriorArtOps(OUT / "pa_features.json")
    ops_null = NullPriorArtOps(OUT / "pa_features.json")

    ids = sorted(texts)
    rng = random.Random(7)
    test = set(rng.sample(ids, int(0.4 * len(ids))))

    # raw payload exposure as its own channel (for anchors): high = novelty-clean
    exposure = {d: 1.0 - f.get("frac_claims_any_disclose", 0.0) for d, f in feats.items()}

    # ---- survey block: all 8 aspects, full-sample floors ----
    survey = {}
    for aid in json.load(open(OUT / "aspects_used.json")):
        j = judge.get(aid, {})
        r1 = rel1.get(aid, float("nan"))
        c = attenuation_ceiling(min(max(r1, 0), 1), 2) if r1 == r1 else float("nan")
        row = {"n_judged": len(j), "rel1": round(r1, 3) if r1 == r1 else None,
               "ceiling": round(c, 3) if c == c else None, "flavors": {}}
        for fl in FLAVORS:
            col = code.get(f"{aid}_{fl}")
            if not col:
                continue
            sel = [d for d in j if col.get(d) is not None]
            if len(sel) < 30:
                continue
            r = spearman([col[d] for d in sel], [j[d] for d in sel])
            if r == r:
                row["flavors"][fl] = round(r, 3)
        if row["flavors"] and c == c and c > 0.3:
            best = max(row["flavors"].values())
            row["best_r_tilde"] = round(max(0.0, min(1.0, best / c)), 3)
        survey[aid] = row

    # ---- hybrid block: 4 prior-art aspects, held-out gates + op marginal ----
    report = {"survey": survey, "hybrids": {}, "anchors": {}}
    for aid in HYB_ASPECTS:
        prog = pathlib.Path(__file__).parent / "programs_pa" / f"{aid}_h0.py"
        spec = importlib.util.spec_from_file_location(prog.stem, prog)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fmap = {d: {f.split("__", 1)[1]: fields[f].get(d, "")
                    for f in fields if f.startswith(aid + "__")} for d in ids}
        hyb_full = {d: mod.score(texts[d], fmap[d], ops_full, dpid=d) for d in ids}
        hyb_null = {d: mod.score(texts[d], fmap[d], ops_null, dpid=d) for d in ids}

        j = judge.get(aid, {})
        tsel = [d for d in test if d in j]
        best = (None, float("-inf"), None)
        for fl in FLAVORS:
            col = code.get(f"{aid}_{fl}")
            if not col:
                continue
            sel = [d for d in tsel if col.get(d) is not None]
            if len(sel) < 40:
                continue
            r = spearman([col[d] for d in sel], [j[d] for d in sel])
            if r == r and r > best[1]:
                best = (fl, r, col)
        fl, base_rho, base_col = best
        if base_col is None:
            base_col, fl, base_rho = {d: 0.5 for d in ids}, "none(constant)", 0.0

        g_code = bootstrap_gate(hyb_full, base_col, j, tsel, B=2000)
        g_op = bootstrap_gate(hyb_full, hyb_null, j, tsel, margin=0.0, floor=0.0, B=2000)
        r1 = rel1.get(aid, float("nan"))
        report["hybrids"][aid] = {
            "rel1": round(r1, 3) if r1 == r1 else None,
            "ceiling": (round(attenuation_ceiling(min(max(r1, 0), 1), 2), 3)
                        if r1 == r1 else None),
            "n_test": len(tsel),
            "baseline": {"flavor": fl, "rho_test": round(base_rho, 3)},
            "hybrid_full": g_code,
            "rho_noop": round(spearman([hyb_null[d] for d in tsel],
                                       [j[d] for d in tsel]), 3),
            "op_marginal_gate": {"P_beats_noop": g_op["P_beats_baseline"],
                                 "rho_full": g_op["rho_mean"]},
        }

        # anchors: descriptive pooled Spearman vs final outcome, every channel
        anch = {}
        for ch_name, ch in (("judge", j), ("code", base_col), ("hybrid", hyb_full),
                            ("hybrid_noop", hyb_null), ("pa_exposure", exposure)):
            sel = [d for d in ids if d in outcome and ch.get(d) is not None]
            if len(sel) < 60:
                continue
            anch[ch_name] = round(spearman([ch[d] for d in sel],
                                           [outcome[d] for d in sel]), 3)
        report["anchors"][aid] = {"outcome_pooled": anch,
                                  "note": "descriptive only; no batch guard available"}

    json.dump(report, open(OUT / "pa_eval.json", "w"), indent=1)
    print("== survey (all 8) ==")
    for aid, row in survey.items():
        print(f"{aid}: rel1={row['rel1']} ceil={row['ceiling']} "
              f"flavors={row['flavors']} r~={row.get('best_r_tilde')}")
    print("== hybrids (4) ==")
    for aid, r in report["hybrids"].items():
        g = r["hybrid_full"]
        print(f"{aid}: base[{r['baseline']['flavor']}] {r['baseline']['rho_test']:+.3f}  "
              f"hyb {g['rho_mean']:+.3f} P(gate)={g['P_gate']} "
              f"P(beats code)={g['P_beats_baseline']}  "
              f"noop {r['rho_noop']:+.3f} P(op marginal)={r['op_marginal_gate']['P_beats_noop']}")
    print("== anchors (descriptive) ==")
    for aid, a in report["anchors"].items():
        print(f"{aid}: {a['outcome_pooled']}")
    print(f"-> {OUT / 'pa_eval.json'}")


if __name__ == "__main__":
    main()
