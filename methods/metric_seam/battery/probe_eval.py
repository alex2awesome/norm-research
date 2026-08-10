"""LOCAL PROBE eval — small-scale battery pilot (PR, 4 e2-bearing criteria, 100 test items).

Readouts, per API model with all-condition extractions (GLM-4.7, GLM-5.2):
  E1-KEY (within model): rho under full / name-only / nonce+definition / blank on the
     frozen programs; surviving fractions frac_x=(rho_x-rho_blank)/(rho_full-rho_blank);
     paired bootstrap P(name > nonce). T-RET: frac_name ~ 1 >> frac_nonce.
  E2-STIP (within model, field level): on items where the deviant stipulated answer
     conflicts with the model's own full-condition answer: compliance vs snap-back.
     KEY COMPARISON: same fields, same conflict rule, different extractor family.
  E3-SCALE (program level): fm = rho_x - rho_blank for Llama-3B/8B (API) alongside the
     local Llama-70B, Gemma-31B, GLM, Qwen extractions.

Usage: python3 probe_eval.py
-> outputs/metric_seam_pilot/battery/probe/probe_eval.json
"""
import json, pathlib, random, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, load_mod, load_fields, run_prog, BASE, ROOT  # noqa
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman  # noqa: E402

SAFE_BUILTINS = {"len": len, "any": any, "sum": sum, "min": min, "max": max, "all": all, "abs": abs}

B = 2000
CRITERIA = ["a76", "a87", "a104", "a112"]
PROBE = BASE / "battery/probe"
FULLCOND_MODELS = {"glm47": "probe_results_glm47.jsonl",
                   "glm52": "probe_results_glm52.jsonl"}
SCALE_ONLY = {"llama3b": "probe_results_llama3b.jsonl",
              "llama8b": "probe_results_llama8b.jsonl"}


def rho_on(sel, col, judge):
    s = [d for d in sel if col.get(d) is not None]
    return spearman([col[d] for d in s], [judge[d] for d in s]) if len(s) >= 20 else float("nan")


def norm(x):
    return (x or "").strip().strip('."’”').lower()


def med(xs):
    xs = sorted(x for x in xs if x is not None and x == x)
    return xs[len(xs) // 2] if xs else None


def eval_model(tag, fields, ctx, variants, scale_cols_by_aid):
    """E1+E2 within one all-condition extractor; returns {aid: row}."""
    report = {}
    for aid in CRITERIA:
        mod = load_mod(ctx["hyb"] / f"{aid}_h0.py")
        judge = ctx["judge"].get(aid, {})
        fmaps = {f"{tag}_full": fields.get(f"{aid}.full", {}),
                 f"{tag}_name": fields.get(f"{aid}.keyname", {}),
                 f"{tag}_nonce": fields.get(f"{aid}.keynonce", {}),
                 "blank": {}}
        if not fmaps[f"{tag}_name"]:
            continue
        probe_items = set(fields.get(f"{aid}.full", {}))
        cols = {c: run_prog(mod.score, ctx["items"], fm, ctx["ops"])
                for c, fm in fmaps.items()}
        cols.update(scale_cols_by_aid.get(aid, {}))
        tsel = [d for d in sorted(probe_items) if d in judge
                and all(cols[c].get(d) is not None for c in cols)]
        if len(tsel) < 30:
            report[aid] = {"error": f"n={len(tsel)}"}
            continue
        rho = {c: rho_on(tsel, cols[c], judge) for c in cols}
        fm_full = rho[f"{tag}_full"] - rho["blank"]
        row = {"n_test": len(tsel),
               "rho": {c: round(v, 3) if v == v else None for c, v in rho.items()}}
        for cond in ("name", "nonce"):
            r = rho[f"{tag}_{cond}"]
            row[f"frac_{cond}"] = (round((r - rho["blank"]) / fm_full, 3)
                                   if abs(fm_full) > 0.05 else None)
        row["fm"] = {c.replace("_full", ""): (round(rho[c] - rho["blank"], 3)
                                              if rho[c] == rho[c] else None)
                     for c in cols if c.endswith("_full") or c in
                     ("gemma", "llama70", "qwen")}
        rng = random.Random(11)
        wins = used = 0
        for _ in range(B):
            s = [tsel[rng.randrange(len(tsel))] for _ in tsel]
            rn = spearman([cols[f"{tag}_name"][d] for d in s], [judge[d] for d in s])
            rz = spearman([cols[f"{tag}_nonce"][d] for d in s], [judge[d] for d in s])
            if rn == rn and rz == rz:
                used += 1
                wins += rn > rz
        row["P_name_gt_nonce"] = round(wins / used, 4) if used else None

        # ---- E2 field-level: compliance vs snap-back on the conflict set ----
        stips = {}
        for key, v in variants.items():
            if not v.get("e2") or v["aid"] != aid:
                continue
            field = v["field"]
            full_ans = {d: norm(fm.get(field))
                        for d, fm in fields.get(f"{aid}.full", {}).items()}
            stip_ans = {d: norm(fm.get(field))
                        for d, fm in fields.get(f"{aid}.stip", {}).items()}
            expr = v["e2"]["checker_expr"]
            comp = snap = other = conflict = 0
            for d, ans in stip_ans.items():
                text = ctx["items"].get(d)
                o = full_ans.get(d)
                if text is None or o is None:
                    continue
                try:
                    truth = norm(str(eval(expr, {"__builtins__": SAFE_BUILTINS, "text": text}, {})))
                except Exception:
                    continue
                if truth == o:
                    continue
                conflict += 1
                if ans == truth:
                    comp += 1
                elif ans == o:
                    snap += 1
                else:
                    other += 1
            if conflict >= 15:
                stips[field] = {"n_conflict": conflict,
                                "compliance": round(comp / conflict, 3),
                                "snapback": round(snap / conflict, 3),
                                "other": round(other / conflict, 3)}
        if stips:
            row["stip"] = stips
        report[aid] = row
        print(f"[{tag}] {aid}: full={row['rho'][f'{tag}_full']} "
              f"name={row['rho'][f'{tag}_name']} nonce={row['rho'][f'{tag}_nonce']} "
              f"blank={row['rho']['blank']} frac_name={row.get('frac_name')} "
              f"frac_nonce={row.get('frac_nonce')} P={row['P_name_gt_nonce']}")
        print(f"     fm: {row['fm']}")
        for f, s in (stips or {}).items():
            print(f"     stip {f}: comply={s['compliance']} snapback={s['snapback']} "
                  f"other={s['other']} (n={s['n_conflict']})")
    return report


def main():
    ctx = load_ctx("press_releases")
    variants = json.load(open(BASE / "battery/variants_press_releases.json"))
    f_llama70 = load_fields(ctx["outdir"] / "field_results_llama.jsonl")
    f_qwen = load_fields(ctx["outdir"] / "field_results_qwen.jsonl")
    scales = {k: load_fields(PROBE / f) for k, f in SCALE_ONLY.items()
              if (PROBE / f).exists()}

    # shared scale/reference columns per aid (gemma/llama70/qwen/llama3b/8b full)
    scale_cols_by_aid = {}
    for aid in CRITERIA:
        mod = load_mod(ctx["hyb"] / f"{aid}_h0.py")
        cols = {}
        for name, fmap in [("gemma", ctx["f_orig"].get(aid, {})),
                           ("llama70", f_llama70.get(aid, {})),
                           ("qwen", f_qwen.get(aid, {}))]:
            cols[name] = run_prog(mod.score, ctx["items"], fmap, ctx["ops"])
        for k, fm in scales.items():
            cols[f"{k}_full"] = run_prog(mod.score, ctx["items"],
                                         fm.get(f"{aid}.full", {}), ctx["ops"])
        scale_cols_by_aid[aid] = cols

    out = {}
    for tag, fname in FULLCOND_MODELS.items():
        p = PROBE / fname
        if not p.exists():
            continue
        fields = load_fields(p)
        rep = eval_model(tag, fields, ctx, variants, scale_cols_by_aid)
        if not rep:
            continue
        ok = [v for v in rep.values() if "error" not in v]
        summ = {"n": len(ok),
                "median_frac_name": med([v.get("frac_name") for v in ok]),
                "median_frac_nonce": med([v.get("frac_nonce") for v in ok])}
        for fam in ("llama3b", "llama8b", "llama70", "gemma",
                    "glm47", "glm52", "qwen"):
            fms = [v["fm"].get(fam) for v in ok if v["fm"].get(fam) is not None]
            if fms:
                summ[f"median_fm_{fam}"] = med(fms)
        comp = [s["compliance"] for v in ok for s in v.get("stip", {}).values()]
        snap = [s["snapback"] for v in ok for s in v.get("stip", {}).values()]
        if comp:
            summ["median_stip_compliance"] = med(comp)
            summ["median_stip_snapback"] = med(snap)
        out[tag] = {"aspects": rep, "summary": summ}
        print(f"[{tag}] summary: {json.dumps(summ)}")

    path = PROBE / "probe_eval.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
