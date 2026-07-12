"""L-axis: transmission I(m;m^) vs articulation budget L, with REALIZED rule length (to check the
cap actually binds)."""
import json
import collections
import numpy as np

OUT = "/lfs/skampere3/0/alexspan/tmp_vinfo"
import sys
fname = sys.argv[1] if len(sys.argv) > 1 else "recon_lsweep_gepa_v2.json"
rows = json.load(open(f"{OUT}/{fname}"))


def nz(x):
    return float("nan") if x is None else x


def rule_len(r):
    lens = [d.get("rubric_len") for d in (r.get("induced") or []) if d.get("rubric_len")]
    return float(np.mean(lens)) if lens else float("nan")


by = collections.defaultdict(dict)  # (task,metric_id,name) -> {L: (T, ci, len)}
for r in rows:
    key = (r["task"], r["metric_id"], r.get("name", "")[:26])
    L = r.get("l_cap")
    by[key][L] = (nz(r.get("iv_transmission")), r.get("iv_transmission_ci"), rule_len(r))

print("=== I(m;m^) vs L  (T | realized rule_len in chars) ===")
for (task, mid, name) in sorted(by):
    d = by[(task, mid, name)]
    Ls = sorted(d)
    cells = "  ".join(f"L={L}:{d[L][0]:.3f}(len{d[L][2]:.0f})" for L in Ls)
    print(f"  {task:16s} {name:26s} {cells}")

print("\n=== per-task mean transmission by L ===")
tb = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    tb[r["task"]][r.get("l_cap")].append(nz(r.get("iv_transmission")))
for task in sorted(tb):
    Ls = sorted(tb[task])
    print(f"  {task:16s} " + "  ".join(f"L={L}:{np.nanmean(tb[task][L]):.3f}" for L in Ls))
