"""SIV compression waterfall (R20): per-criterion rho staircase + ceiling-normalized
domain summary for the GEPA-H2H set.

Rungs per criterion: frozen codegen baseline -> certified hybrid HEAD (H arm) ->
GEPA single-prompt (G arm, ~= re-running the judge) -> attenuation ceiling. The
normalized read r~ = clip01(rho/ceiling) makes the fidelity cost of compression
(r~_G - r~_H) comparable across domains: this is the domain-graded gradient quoted
in the results note SGEPA-H2H, now as a figure + table.

Usage: python3 fig_s4_waterfall.py
-> figures/s4_compression_waterfall.{png,pdf} + battery/s4_waterfall.json
"""
import json, math, pathlib, statistics as st, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import BASE, ROOT  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASK_COLOR = {"press_releases": "#1f77b4", "creative_writing": "#d62728",
              "math": "#2ca02c", "humor": "#9467bd"}
TASK_SHORT = {"press_releases": "PR", "creative_writing": "CW",
              "math": "math", "humor": "humor"}


def ceiling(rel1, k=2):
    r = max(0.0, min(1.0, rel1))
    relk = k * r / (1 + (k - 1) * r)
    return math.sqrt(relk) if relk > 0 else float("nan")


def clip01(x):
    return max(0.0, min(1.0, x))


def main():
    gh = json.load(open(ROOT / "methods/metric_seam/battery/gepa_h2h/gepa_h2h_final.json"))
    pr = json.load(open(BASE / "v2/hybrid_eval_v2.json"))
    reports = {}
    rows = []
    for key, v in gh.items():
        task, aid = key.split(".")
        if task == "press_releases":
            e = pr[aid]
            base, ceil = e["baseline"]["rho_test"], e["ceiling"]
        else:
            if task not in reports:
                reports[task] = json.load(open(BASE / "tasks" / task / "hybrid_gate_report.json"))
            r = reports[task][aid]
            base, ceil = r["full"]["rho_baseline"], ceiling(r["judge_rel1"])
        H, G = v["rho_test"]["H"], v["rho_test"]["G"]
        rows.append({"task": task, "aid": aid, "base": base, "H": H, "G": G,
                     "ceil": ceil, "h_source": v.get("arm_h_source"),
                     "r_base": round(clip01(base / ceil), 3),
                     "r_H": round(clip01(H / ceil), 3),
                     "r_G": round(clip01(G / ceil), 3)})
    rows.sort(key=lambda r: (list(TASK_COLOR).index(r["task"]), -r["H"]))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.55, 1]})

    # Panel A: per-criterion staircase (raw rho)
    ys = range(len(rows))
    for y, r in zip(ys, rows):
        c = TASK_COLOR[r["task"]]
        ax1.plot([r["base"], r["G"]], [y, y], color=c, lw=1, alpha=.35, zorder=1)
        ax1.scatter([r["base"]], [y], marker="o", facecolors="white",
                    edgecolors=c, s=38, zorder=3, label=None)
        ax1.scatter([r["H"]], [y], marker="o", color=c, s=44, zorder=3)
        ax1.scatter([r["G"]], [y], marker="D", color=c, s=34, zorder=3)
        ax1.plot([r["ceil"], r["ceil"]], [y - .32, y + .32], color="k",
                 lw=1.4, alpha=.75, zorder=2)
    ax1.set_yticks(list(ys))
    ax1.set_yticklabels([f'{TASK_SHORT[r["task"]]} {r["aid"]}'
                         f'{"*" if r["h_source"] != "h0" else ""}' for r in rows],
                        fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel(r"test Spearman $\rho$ vs judge")
    ax1.set_title("code $\\circ$ $\\to$ hybrid $\\bullet$ $\\to$ prompt $\\diamond$ "
                  "$\\to$ ceiling $|$", fontsize=10)
    ax1.set_xlim(-0.25, 1.02)
    ax1.grid(axis="x", alpha=.25)

    # Panel B: ceiling-normalized rungs per domain (medians)
    summ = {}
    for t in TASK_COLOR:
        tr = [r for r in rows if r["task"] == t]
        summ[t] = {"n": len(tr),
                   "med_r_base": round(st.median(x["r_base"] for x in tr), 3),
                   "med_r_H": round(st.median(x["r_H"] for x in tr), 3),
                   "med_r_G": round(st.median(x["r_G"] for x in tr), 3)}
        summ[t]["fidelity_cost"] = round(summ[t]["med_r_G"] - summ[t]["med_r_H"], 3)
    xs = [0, 1, 2]
    for t, s in summ.items():
        vals = [s["med_r_base"], s["med_r_H"], s["med_r_G"]]
        ax2.plot(xs, vals, marker="o", color=TASK_COLOR[t],
                 label=f'{TASK_SHORT[t]} (cost {s["fidelity_cost"]:+.2f}, n={s["n"]})')
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["code", "certified\nhybrid", "single prompt\n(~judge)"])
    ax2.set_ylabel(r"median $\tilde r$ = clip$[0,1]$($\rho$/ceiling)")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1.0, color="k", lw=.8, alpha=.4)
    ax2.set_title("fidelity cost of compression, by domain", fontsize=10)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(alpha=.25)

    fig.suptitle("Compression waterfall: what thinning costs, where (GEPA-H2H set, "
                 "held-out test)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    figdir = ROOT / "figures"
    figdir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"s4_compression_waterfall.{ext}", dpi=200)
    out = BASE / "battery/s4_waterfall.json"
    json.dump({"rows": rows, "domain_summary": summ}, open(out, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print(f"-> figures/s4_compression_waterfall.png + {out}")


if __name__ == "__main__":
    main()
