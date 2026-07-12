"""Plot the articulation-budget (L) axis: I(m;m^) vs L per metric + per-task mean, and the realized
rule length vs L (to show the budget now binds). Saves PNGs to tmp_vinfo."""
import json
import sys
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/lfs/skampere3/0/alexspan/tmp_vinfo"
fname = sys.argv[1] if len(sys.argv) > 1 else "recon_lsweep_gepa_v2.json"
rows = json.load(open(f"{OUT}/{fname}"))


def nz(x):
    return float("nan") if x is None else x


def rule_len(r):
    lens = [d.get("rubric_len") for d in (r.get("induced") or []) if d.get("rubric_len")]
    return float(np.mean(lens)) if lens else float("nan")


COL = {"math_se": "#1f77b4", "creative_writing": "#d62728"}
by = collections.defaultdict(dict)       # (task,mid,name) -> {L:(T,lo,hi,len)}
for r in rows:
    L = r.get("l_cap")
    ci = r.get("iv_transmission_ci") or [None, None]
    by[(r["task"], r["metric_id"], r.get("name", "")[:22])][L] = (
        nz(r.get("iv_transmission")), nz(ci[0]), nz(ci[1]), rule_len(r))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# panel 1: I vs L (faint per-metric lines + bold per-task mean with CI)
task_curves = collections.defaultdict(lambda: collections.defaultdict(list))
for (task, mid, name), d in by.items():
    Ls = sorted(d)
    ys = [d[L][0] for L in Ls]
    ax1.plot(Ls, ys, color=COL.get(task, "gray"), alpha=0.25, lw=1, marker="o", ms=3)
    for L in Ls:
        task_curves[task][L].append(d[L][0])
for task, byL in task_curves.items():
    Ls = sorted(byL)
    mean = [np.nanmean(byL[L]) for L in Ls]
    ax1.plot(Ls, mean, color=COL.get(task, "k"), lw=2.6, marker="o", ms=6, label=f"{task} (mean)")
ax1.set_xscale("log")
ax1.set_xlabel("articulation budget L (target tokens)")
ax1.set_ylabel("I(m; m^)  transmission (bits)")
ax1.set_title("Articulability vs articulation budget  I_{V_{L,E}}(X->m)")
ax1.axhline(0, color="gray", lw=0.6, ls=":")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.25)

# panel 2: realized rule length vs L per task (does the budget bind?)
len_curves = collections.defaultdict(lambda: collections.defaultdict(list))
for (task, mid, name), d in by.items():
    for L in d:
        if np.isfinite(d[L][3]):
            len_curves[task][L].append(d[L][3])
for task, byL in len_curves.items():
    Ls = sorted(byL)
    ax2.plot(Ls, [np.nanmean(byL[L]) for L in Ls], color=COL.get(task, "k"), lw=2.4,
             marker="s", ms=6, label=task)
ax2.plot([20, 700], [20 * 4, 700 * 4], color="gray", ls="--", lw=1, label="len = L (approx, 4 char/tok)")
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("articulation budget L (target tokens)")
ax2.set_ylabel("realized rule length (chars)")
ax2.set_title("Does the budget bind?  realized length vs L")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.25)

fig.tight_layout()
out = f"{OUT}/lsweep_{fname.replace('.json','')}.png"
fig.savefig(out, dpi=130)
print("wrote", out)
