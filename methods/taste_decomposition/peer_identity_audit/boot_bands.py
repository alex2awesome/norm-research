#!/usr/bin/env python3
"""Paired bootstrap CIs for the per-year-band residuals (d)-(c) of the memorization
battery. Rows are independent (1 paper = 1 group), so a plain paired row bootstrap
within band is valid."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
TD = HERE.parent
def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[alias] = m
    spec.loader.exec_module(m); return m
F2 = _mod(TD / "fusion/f2_deconf.py", "f2m2")
meta, ids_E, y, groups, dense, t0col = F2.load_E("peer_revealed")
ad = F2.F2C.ADAPTERS["peer_revealed"]()
bank, nuis, join = F2.align("peer_revealed", ad, ids_E, y, groups)
r = F2.fit_arm(meta["family"], np.column_stack([bank, nuis]), dense, y, groups)
oof_c, oof_d = r["_oof_VA_nl0"], r["_oof_VAT_nl0"]

rows = [json.loads(l) for l in open(TD.parents[1] / "datasets/peer-review/vat_3y/revealed.jsonl")]
yr = {rr["ntitle"]: int(rr["year"]) for rr in rows}
year = np.array([yr.get(str(nt), -1) for nt in ids_E])

out = {}
rng = np.random.default_rng(20260812)
for name, m in {"2013-2019": (year>=2013)&(year<=2019), "2020-2021": (year>=2020)&(year<=2021),
                "2022-2023": (year>=2022)&(year<=2023)}.items():
    idx = np.where(m)[0]
    est = roc_auc_score(y[idx], oof_d[idx]) - roc_auc_score(y[idx], oof_c[idx])
    bs = []
    for _ in range(4000):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(set(y[s].tolist())) < 2: continue
        bs.append(roc_auc_score(y[s], oof_d[s]) - roc_auc_score(y[s], oof_c[s]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = float(np.mean(np.array(bs) > 0))
    out[name] = {"n": int(len(idx)), "residual": float(est), "ci95": [float(lo), float(hi)], "p_gt_0": p}
    print(name, out[name], flush=True)
json.dump(out, open(HERE / "band_residual_cis.json", "w"), indent=1)
