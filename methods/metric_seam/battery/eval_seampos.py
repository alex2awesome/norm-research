"""E5 SEAM-POS eval (pre-reg note §1.3) — seam position on PR certified 12.

CLC (aperture) conditions: frozen program re-scored with fields extracted from a coded
view nu(x) — digest / head / mid / tail. Aperture loss = rho_lcc - rho_digest; per-view
fm = rho_view - rho_blank.
CCL (valuation) condition: LLM aggregator over code signals only ({aid}.ccl SCORE rows)
vs a FITTED mechanical aggregator (ridge on the same signals, fit on train) —
fm_A = rho_ccl_llm - rho_ccl_fit, with paired bootstrap.

Usage: python3 eval_seampos.py
-> outputs/metric_seam_pilot/battery/seampos_eval.json
"""
import json, pathlib, random, sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, load_mod, load_fields, run_prog, BASE, ROOT  # noqa
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman  # noqa: E402

B = 2000
CERTIFIED = ["a103", "a104", "a112", "a2", "a28", "a42",
             "a65", "a66", "a75", "a76", "a87", "a97"]
VIEWS = ["apdigest", "aphead", "apmid", "aptail"]


def rho_on(sel, col, judge):
    s = [d for d in sel if col.get(d) is not None]
    return spearman([col[d] for d in s], [judge[d] for d in s]) if len(s) >= 20 else float("nan")


def ridge_fit_predict(X_tr, y_tr, X_te, lam=1.0):
    """Tiny ridge without sklearn: X lists of feature lists."""
    import statistics
    p = len(X_tr[0])
    mu = [statistics.mean(col) for col in zip(*X_tr)]
    sd = [statistics.pstdev(col) or 1.0 for col in zip(*X_tr)]
    Z = [[(r[j] - mu[j]) / sd[j] for j in range(p)] for r in X_tr]
    ym = statistics.mean(y_tr)
    yc = [y - ym for y in y_tr]
    # solve (Z'Z + lam I) w = Z'y via Gauss elimination
    A = [[sum(Z[i][a] * Z[i][b] for i in range(len(Z))) + (lam if a == b else 0.0)
          for b in range(p)] for a in range(p)]
    v = [sum(Z[i][a] * yc[i] for i in range(len(Z))) for a in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(A[r][c]))
        A[c], A[piv] = A[piv], A[c]
        v[c], v[piv] = v[piv], v[c]
        if abs(A[c][c]) < 1e-12:
            continue
        for r in range(c + 1, p):
            f = A[r][c] / A[c][c]
            for k in range(c, p):
                A[r][k] -= f * A[c][k]
            v[r] -= f * v[c]
    w = [0.0] * p
    for c in reversed(range(p)):
        if abs(A[c][c]) < 1e-12:
            continue
        w[c] = (v[c] - sum(A[c][k] * w[k] for k in range(c + 1, p))) / A[c][c]
    return [ym + sum(((r[j] - mu[j]) / sd[j]) * w[j] for j in range(p)) for r in X_te]


def main():
    ctx = load_ctx("press_releases")
    from harness import split_ids
    train, test = split_ids()
    code = json.load(open(ctx["outdir"] / "code_scores_v2.json"))
    fb = load_fields(ctx["outdir"] / "seampos_results.jsonl")
    ccl_raw = {}
    for line in open(ctx["outdir"] / "seampos_results.jsonl"):
        r = json.loads(line)
        if r.get("channel") == "ccl" and r.get("score") is not None \
                and r["score"] != "NA":
            aid = r["aspect_id"].split(".")[0]
            ccl_raw.setdefault(aid, {})[r["datapoint_id"]] = r["score"] / 10.0

    report = {}
    for aid in CERTIFIED:
        prog = ctx["hyb"] / f"{aid}_h0.py"
        mod = load_mod(prog)
        judge = ctx["judge"].get(aid, {})
        fmaps = {"lcc": ctx["f_orig"].get(aid, {}), "blank": {}}
        for v in VIEWS:
            fmaps[v] = fb.get(f"{aid}.{v}", {})
        cols = {c: run_prog(mod.score, ctx["items"], fm, ctx["ops"])
                for c, fm in fmaps.items()}
        tsel = [d for d in test if d in judge
                and all(cols[c].get(d) is not None for c in cols)]
        if len(tsel) < 30:
            report[aid] = {"error": f"n_test={len(tsel)}"}
            continue
        rho = {c: rho_on(tsel, cols[c], judge) for c in cols}
        fm_R = rho["lcc"] - rho["blank"]
        row = {"n_test": len(tsel),
               "rho": {c: round(v, 3) if v == v else None for c, v in rho.items()},
               "fm_R": round(fm_R, 3)}
        for v in VIEWS:
            row[f"fm_{v}"] = round(rho[v] - rho["blank"], 3)
        row["aperture_loss"] = round(rho["lcc"] - rho["apdigest"], 3)
        row["aperture_frac_kept"] = (round((rho["apdigest"] - rho["blank"]) / fm_R, 3)
                                     if abs(fm_R) > 0.05 else None)

        # ---- CCL: LLM aggregator vs fitted mechanical aggregator on same signals ----
        feats = {}
        fields = sorted({fn for d in ctx["f_orig"].get(aid, {})
                         for fn in ctx["f_orig"][aid][d]})
        # top-6 field values by train frequency, one-hot
        fvals = {fn: [val for val, _ in Counter(
            (ctx["f_orig"][aid].get(d, {}).get(fn) or "").strip().lower()
            for d in train).most_common(6)] for fn in fields}
        for d in ctx["items"]:
            x = []
            for fl in ["v0_keyword", "v1_structure", "v2_holistic"]:
                val = (code.get(f"{aid}_{fl}") or {}).get(d)
                x.append(float(val) if val is not None else 0.5)
            for fn in fields:
                raw = (ctx["f_orig"][aid].get(d, {}).get(fn) or "").strip().lower()
                x += [1.0 if raw == v else 0.0 for v in fvals[fn]]
            feats[d] = x
        tr = [d for d in train if d in judge]
        te = [d for d in tsel if d in ccl_raw.get(aid, {})]
        if len(tr) >= 40 and len(te) >= 30:
            preds = ridge_fit_predict([feats[d] for d in tr], [judge[d] for d in tr],
                                      [feats[d] for d in te])
            fit_col = dict(zip(te, preds))
            rho_fit = rho_on(te, fit_col, judge)
            rho_llm = rho_on(te, ccl_raw[aid], judge)
            row["rho_ccl_fit"] = round(rho_fit, 3) if rho_fit == rho_fit else None
            row["rho_ccl_llm"] = round(rho_llm, 3) if rho_llm == rho_llm else None
            if rho_fit == rho_fit and rho_llm == rho_llm:
                row["fm_A"] = round(rho_llm - rho_fit, 3)
                rng = random.Random(11)
                wins = used = 0
                for _ in range(B):
                    s = [te[rng.randrange(len(te))] for _ in te]
                    rl = spearman([ccl_raw[aid][d] for d in s], [judge[d] for d in s])
                    rf = spearman([fit_col[d] for d in s], [judge[d] for d in s])
                    if rl == rl and rf == rf:
                        used += 1
                        wins += rl > rf
                row["P_llm_agg_gt_fit"] = round(wins / used, 4) if used else None
        report[aid] = row
        print(f"{aid}: lcc={row['rho']['lcc']} digest={row['rho']['apdigest']} "
              f"h/m/t={row['rho']['aphead']}/{row['rho']['apmid']}/{row['rho']['aptail']} "
              f"blank={row['rho']['blank']} kept={row.get('aperture_frac_kept')} "
              f"cclL={row.get('rho_ccl_llm')} cclF={row.get('rho_ccl_fit')} "
              f"P(A)={row.get('P_llm_agg_gt_fit')}")

    ok = [v for v in report.values() if "error" not in v]
    med = lambda xs: (sorted(xs)[len(xs) // 2] if xs else None)
    summ = {"n": len(ok),
            "median_aperture_frac_kept": med([v["aperture_frac_kept"] for v in ok
                                              if v.get("aperture_frac_kept") is not None]),
            "median_fm_R": med([v["fm_R"] for v in ok]),
            "median_fm_digest": med([v["fm_apdigest"] for v in ok]),
            "median_fm_A": med([v["fm_A"] for v in ok if v.get("fm_A") is not None]),
            "n_P95_llm_agg": sum(1 for v in ok
                                 if (v.get("P_llm_agg_gt_fit") or 0) >= .95)}
    out = BASE / "battery/seampos_eval.json"
    json.dump({"aspects": report, "summary": summ}, open(out, "w"), indent=1)
    print("summary:", json.dumps(summ))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
