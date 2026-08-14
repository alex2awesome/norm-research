"""Projected-ceiling table for the app:backtest appendix (2026-08-14, user-directed).

Fits both backtested functional forms on each benchmark's FULL canonical prefix trajectory
(same SOURCES + fitters as ceiling_backtest_all.py) and reports y(k_max), y(k_max+{10,20,50}),
and the fitted asymptote a (the k->infinity ceiling). Units are the VALUE-CURVE's own
(validation-side greedy prefix means, the objects of fig:missing-value and the backtest) —
NOT Table-1 test scores; only the rising/flat verdict transfers across that boundary.

A row is CENSORED (rising) if the last-quintile mean slope exceeds eps=.002/unit under the
power-law fit — for censored rows the asymptote is reported as a lower bound only.
Writes runs/ceiling_projection_v1.json and prints the LaTeX rows.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ceiling_backtest_all import SOURCES, fit_powerlaw, fit_satexp, load_curve  # noqa: E402

EPS_SLOPE = 0.002


def main():
    rows = []
    for bench, src in SOURCES.items():
        curve = load_curve(src)
        ks = sorted(curve)
        ys = [curve[k] for k in ks]
        kmax = ks[-1]
        out = {"bench": bench, "k_max": kmax, "y_kmax": round(ys[-1], 4), "src": src}
        preds = {}
        for name, fit in (("powerlaw", fit_powerlaw), ("satexp", fit_satexp)):
            f = fit(ks, ys)                      # prediction lambda; params in __defaults__
            a = f.__defaults__[0]
            preds[name] = {"a": round(a, 4),
                           **{f"+{d}": round(f(kmax + d), 4) for d in (10, 20, 50)}}
        # censoring: fitted power-law slope over the last observed quintile
        pl = fit_powerlaw(ks, ys)
        q = max(2, len(ks) // 5)
        slope = (pl(kmax) - pl(kmax - q)) / q
        out["forms"] = preds
        out["last_quintile_slope_per_unit"] = round(slope, 5)
        out["verdict"] = "rising-censored" if slope > EPS_SLOPE else "reaches"
        rows.append(out)
    Path("runs/ceiling_projection_v1.json").write_text(json.dumps(rows, indent=1))
    print(f"{'bench':10s} {'k':>3s} {'y(k)':>6s} | {'+10':>6s} {'+20':>6s} {'+50':>6s} "
          f"{'a(pl)':>6s} {'a(se)':>6s} | verdict")
    for r in rows:
        p, s = r["forms"]["powerlaw"], r["forms"]["satexp"]
        print(f"{r['bench']:10s} {r['k_max']:3d} {r['y_kmax']:6.3f} | {p['+10']:6.3f} "
              f"{p['+20']:6.3f} {p['+50']:6.3f} {p['a']:6.3f} {s['a']:6.3f} | {r['verdict']}"
              f"  (slope {r['last_quintile_slope_per_unit']:+.4f}/unit)")


if __name__ == "__main__":
    main()
