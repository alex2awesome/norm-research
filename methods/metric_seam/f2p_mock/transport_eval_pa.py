"""Interpreter-swap TRANSPORT TEST for the patents_pa prior-art hybrid programs.

Mirrors methods/metric_seam/hybrids/transport_eval_task.py (report/summary structure and key
names match exactly), but reuses the patents_pa-specific recipe from eval_patents_pa.py: the
union-of-passes judge/rel1 loader, the `clean()` field-text cleaner, and the 40%
random.Random(7).sample held-out split (NOT the generic 150/100 hybrids split).

For each of the 4 prior-art hybrid programs (a26, a34, a60, a35), score the SAME frozen
program on the SAME held-out split under three field conditions, ALL with the full
PriorArtOps evidence op enabled (ops_pa.PriorArtOps over pa_features.json):
  gemma  — original field extractions (results.jsonl, channel=="field")      [certified condition]
  llama  — Llama-3.3-70B re-extractions (field_results_llama.jsonl), same    [family swap]
           aspect_id/datapoint_id keying, cleaned with the identical clean()
  blank  — all LLM_FIELDS set to ""                                          [borrowed-meaning
                                                                                ablation]

Per aspect:
  field_marginal   = rho_gemma - rho_blank    (weight of the borrowed enculturated payload)
  transport_delta  = rho_gemma - rho_llama    (certificate loss under interpreter swap)
  P_degrade        = paired-bootstrap P(rho_gemma > rho_llama), B=2000, seed 11
  transport_ratio  = transport_delta / field_marginal, only when |field_marginal| > 0.05

Judge coverage varies (a34 has ~95 judged items with few distinct values): bootstrap draws
that yield NaN spearman (degenerate resample) are skipped, and the used-draw count is
reported (boot_used) rather than assuming all B draws were valid.

Sanity check (printed, not gated): rho_gemma should be close to the certified hybrid_full
rho_mean in pa_eval.json (built with the PriorArtOps-enabled hybrid over the same split);
deviations > 0.08 are flagged. a34's pa_eval.json hybrid_full rho_mean is itself NaN
(degenerate bootstrap upstream, no NaN-skipping there) so it cannot be compared numerically.

Usage: python3 transport_eval_pa.py
-> outputs/metric_seam_pilot/tasks/patents_pa/transport_eval.json
"""
import importlib.util, json, pathlib, random, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/metric_seam_pilot/tasks/patents_pa"
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from certificates import spearman  # noqa: E402
from ops_pa import PriorArtOps      # noqa: E402

B = 2000


def _load_module(path, name=None):
    spec = importlib.util.spec_from_file_location(name or path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# eval_patents_pa.py has a `if __name__ == "__main__": main()` guard, so importing it is safe
# and gives us its `clean()`, `load_all()`, and `HYB_ASPECTS` verbatim (no re-implementation).
epa = _load_module(pathlib.Path(__file__).parent / "eval_patents_pa.py", "eval_patents_pa")
clean = epa.clean
HYB_ASPECTS = epa.HYB_ASPECTS


def load_field_file(path):
    """Same field-loading + clean() recipe eval_patents_pa.load_all uses for results.jsonl,
    applied to a standalone field-results file (llama re-extractions)."""
    fields = {}
    if not path.exists():
        return fields
    for line in open(path):
        r = json.loads(line)
        if r.get("channel") == "field":
            fields.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = clean(r.get("raw", ""))
    return fields


def build_fmap(fields, aid, ids):
    """Identical to eval_patents_pa.main()'s fmap comprehension."""
    return {d: {f.split("__", 1)[1]: fields[f].get(d, "")
                for f in fields if f.startswith(aid + "__")} for d in ids}


def rho_on(sel, col, judge):
    s = [d for d in sel if col.get(d) is not None]
    return spearman([col[d] for d in s], [judge[d] for d in s]) if len(s) >= 20 else float("nan")


def main():
    judge, rel1, _scope, fields_gemma = epa.load_all()
    items = json.load(open(OUT / "items.json"))
    texts = {it["datapoint_id"]: it["ctext"] for it in items}
    ops_full = PriorArtOps(OUT / "pa_features.json")

    fields_llama = load_field_file(OUT / "field_results_llama.jsonl")

    # EXACT split recipe from eval_patents_pa.py: sorted ids, random.Random(7).sample, 40%.
    ids = sorted(texts)
    rng = random.Random(7)
    test = set(rng.sample(ids, int(0.4 * len(ids))))

    # sanity target: certified PriorArtOps-enabled hybrid rho from pa_eval.json
    pa_eval_path = OUT / "pa_eval.json"
    pa_eval = json.load(open(pa_eval_path)) if pa_eval_path.exists() else {"hybrids": {}}

    report = {}
    flags = []
    for aid in HYB_ASPECTS:
        prog = pathlib.Path(__file__).parent / "programs_pa" / f"{aid}_h0.py"
        mod = _load_module(prog)

        fmap_gemma = build_fmap(fields_gemma, aid, ids)
        fmap_llama = build_fmap(fields_llama, aid, ids)
        llm_fields = getattr(mod, "LLM_FIELDS", {})
        fmap_blank = {d: {k: "" for k in llm_fields} for d in ids}

        j = judge.get(aid, {})
        tsel = sorted(d for d in ids if d in test and d in j)
        if len(tsel) < 20:
            report[aid] = {"error": f"n_test={len(tsel)}"}
            continue

        cols = {}
        for cond, fmap in (("gemma", fmap_gemma), ("llama", fmap_llama), ("blank", fmap_blank)):
            cols[cond] = {d: mod.score(texts[d], fmap[d], ops_full, dpid=d) for d in tsel}

        rho = {c: rho_on(tsel, cols[c], j) for c in cols}

        # paired bootstrap P(gemma > llama), B=2000, seed 11; skip NaN draws (degenerate resamples)
        rngb = random.Random(11)
        deg = used = 0
        for _ in range(B):
            s = [tsel[rngb.randrange(len(tsel))] for _ in tsel]
            rg = spearman([cols["gemma"][d] for d in s], [j[d] for d in s])
            rl = spearman([cols["llama"][d] for d in s], [j[d] for d in s])
            if rg == rg and rl == rl:
                used += 1
                deg += rg > rl

        fm = (rho["gemma"] - rho["blank"]) if rho["gemma"] == rho["gemma"] and rho["blank"] == rho["blank"] else float("nan")
        td = (rho["gemma"] - rho["llama"]) if rho["gemma"] == rho["gemma"] and rho["llama"] == rho["llama"] else float("nan")

        report[aid] = {
            "n_test": len(tsel),
            "rel1": round(rel1[aid], 3) if rel1.get(aid, float("nan")) == rel1.get(aid, float("nan")) else None,
            "rho": {c: round(v, 3) if v == v else None for c, v in rho.items()},
            "field_marginal": round(fm, 3) if fm == fm else None,
            "transport_delta": round(td, 3) if td == td else None,
            "transport_ratio": (round(td / fm, 3) if fm == fm and abs(fm) > 0.05
                                and td == td else None),
            "P_degrade": round(deg / used, 4) if used else None,
            "boot_used": used,
        }

        # sanity vs pa_eval.json's certified PriorArtOps-enabled hybrid_full rho
        ref = pa_eval.get("hybrids", {}).get(aid, {}).get("hybrid_full", {}).get("rho_mean")
        sanity = None
        if ref is not None and ref == ref and rho["gemma"] == rho["gemma"]:
            dev = abs(rho["gemma"] - ref)
            sanity = {"pa_eval_rho_mean": ref, "deviation": round(dev, 3)}
            if dev > 0.08:
                flags.append(f"{aid}: rho_gemma={rho['gemma']:.3f} vs pa_eval hybrid_full "
                             f"rho_mean={ref:.3f} (deviation {dev:.3f} > 0.08)")
        elif ref is not None and ref != ref:
            sanity = {"pa_eval_rho_mean": None, "note": "reference is NaN (degenerate bootstrap "
                     "upstream); cannot compare"}
        report[aid]["sanity_vs_pa_eval"] = sanity

        r = report[aid]
        print(f"{aid}: g={r['rho']['gemma']} l={r['rho']['llama']} b={r['rho']['blank']}  "
              f"fm={r['field_marginal']} td={r['transport_delta']} "
              f"ratio={r['transport_ratio']} P_deg={r['P_degrade']} (boot_used={used}/{B})")

    # cross-aspect summary: does transport_delta track field_marginal?
    pairs = [(v["field_marginal"], v["transport_delta"]) for v in report.values()
             if isinstance(v, dict) and v.get("field_marginal") is not None
             and v.get("transport_delta") is not None]
    summ = {"n": len(pairs)}
    if len(pairs) >= 4:
        summ["spearman_fm_td"] = round(spearman([p[0] for p in pairs],
                                                [p[1] for p in pairs]), 3)
        tds = sorted(p[1] for p in pairs)
        summ["median_transport_delta"] = round(tds[len(tds) // 2], 3)
        fms = sorted(p[0] for p in pairs)
        summ["median_field_marginal"] = round(fms[len(fms) // 2], 3)

    json.dump({"aspects": report, "summary": summ, "sanity_flags": flags},
              open(OUT / "transport_eval.json", "w"), indent=1)
    print("summary:", summ)
    if flags:
        print("SANITY FLAGS:")
        for f in flags:
            print(" ", f)
    else:
        print("sanity: no deviations > 0.08 vs pa_eval.json hybrid_full rho_mean")
    print(f"-> {OUT / 'transport_eval.json'}")


if __name__ == "__main__":
    main()
