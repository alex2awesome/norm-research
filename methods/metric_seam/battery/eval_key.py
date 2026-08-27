"""E1 KEY-DEPRIVATION eval (pre-reg note §2.3).

Per E1-selected criterion, score the FROZEN hybrid on the FROZEN held-out split under:
  full  — certified Gemma extraction (original instruction)
  name  — name-only instruction extraction   ({aid}.keyname__* in battery results)
  nonce — nonce-name + full-definition extraction ({aid}.keynonce__*)
  blank — fields empty
T-RET predicts rho_name ≈ rho_full ≫ rho_nonce; H_spec predicts nonce ≈ full.
Surviving-fraction readout: frac_x = (rho_x - rho_blank) / (rho_full - rho_blank).

Usage: python3 eval_key.py <task>
-> outputs/metric_seam_pilot/battery/key_eval_<task>.json
"""
import json, pathlib, random, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, load_mod, load_fields, run_prog, BASE, ROOT  # noqa
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman  # noqa: E402

B = 2000


def rho_on(sel, col, judge):
    s = [d for d in sel if col.get(d) is not None]
    return spearman([col[d] for d in s], [judge[d] for d in s]) if len(s) >= 20 else float("nan")


def main():
    task = sys.argv[1]
    ctx = load_ctx(task)
    inv = json.load(open(BASE / "battery/inventory.json"))[task]
    fb = load_fields(ctx["outdir"] / "battery_results.jsonl")

    report = {}
    for aid, meta in sorted(inv.items()):
        if not meta.get("E1_selected"):
            continue
        prog = ctx["hyb"] / f"{aid}_h0.py"
        try:
            mod = load_mod(prog)
        except Exception as e:
            report[aid] = {"error": str(e)}
            continue
        fmaps = {"full": ctx["f_orig"].get(aid, {}),
                 "name": fb.get(f"{aid}.keyname", {}),
                 "nonce": fb.get(f"{aid}.keynonce", {}),
                 "blank": {}}
        cols = {c: run_prog(mod.score, ctx["items"], fm, ctx["ops"])
                for c, fm in fmaps.items()}
        judge = ctx["judge"].get(aid, {})
        tsel = [d for d in ctx["test"] if d in judge
                and all(cols[c].get(d) is not None for c in cols)]
        if len(tsel) < 30:
            report[aid] = {"error": f"n_test={len(tsel)}"}
            continue
        rho = {c: rho_on(tsel, cols[c], judge) for c in cols}
        fm_full = rho["full"] - rho["blank"]
        row = {"n_test": len(tsel), "fm": meta.get("fm"),
               "rho": {c: round(v, 3) if v == v else None for c, v in rho.items()}}
        for c in ("name", "nonce"):
            row[f"d_{c}"] = round(rho["full"] - rho[c], 3)
            row[f"frac_{c}"] = (round((rho[c] - rho["blank"]) / fm_full, 3)
                                if abs(fm_full) > 0.05 else None)
        rng = random.Random(11)
        wins = used = 0
        for _ in range(B):
            s = [tsel[rng.randrange(len(tsel))] for _ in tsel]
            rn = spearman([cols["name"][d] for d in s], [judge[d] for d in s])
            rz = spearman([cols["nonce"][d] for d in s], [judge[d] for d in s])
            if rn == rn and rz == rz:
                used += 1
                wins += rn > rz
        row["P_name_gt_nonce"] = round(wins / used, 4) if used else None
        report[aid] = row
        print(f"{aid}: full={row['rho']['full']} name={row['rho']['name']} "
              f"nonce={row['rho']['nonce']} blank={row['rho']['blank']} "
              f"frac_name={row['frac_name']} frac_nonce={row['frac_nonce']} "
              f"P(name>nonce)={row['P_name_gt_nonce']}")

    ok = [v for v in report.values() if "error" not in v]
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
    summ = {"n": len(ok),
            "median_frac_name": med([v["frac_name"] for v in ok if v["frac_name"] is not None]),
            "median_frac_nonce": med([v["frac_nonce"] for v in ok if v["frac_nonce"] is not None]),
            "median_d_name": med([v["d_name"] for v in ok]),
            "median_d_nonce": med([v["d_nonce"] for v in ok]),
            "n_P95_name_gt_nonce": sum(1 for v in ok
                                       if (v.get("P_name_gt_nonce") or 0) >= .95)}
    pairs = [(v["fm"], v["d_nonce"]) for v in ok if v.get("fm") is not None]
    if len(pairs) >= 5:
        summ["spearman_fm_dnonce"] = round(
            spearman([p[0] for p in pairs], [p[1] for p in pairs]), 3)
    out = BASE / "battery" / f"key_eval_{task}.json"
    json.dump({"aspects": report, "summary": summ}, open(out, "w"), indent=1)
    print("summary:", json.dumps(summ))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
