#!/usr/bin/env python3
"""CPU readout for the recognition probe: does the peer_revealed residual (d)-(c)
concentrate in RECOGNIZED rows (low base-model NLL) within each year band?

Memorization/recognition predicts: residual(low-NLL half) >> residual(high-NLL half),
within bands. Bank-era-failure without memorization predicts no NLL split.
Descriptive; paired bootstrap CIs per half."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
TD = HERE.parent

def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[alias] = m
    spec.loader.exec_module(m); return m

F2 = _mod(TD / "fusion/f2_deconf.py", "f2m3")
meta, ids_E, y, groups, dense, t0col = F2.load_E("peer_revealed")
ad = F2.F2C.ADAPTERS["peer_revealed"]()
bank, nuis, join = F2.align("peer_revealed", ad, ids_E, y, groups)
r = F2.fit_arm(meta["family"], np.column_stack([bank, nuis]), dense, y, groups)
oof_c, oof_d = r["_oof_VA_nl0"], r["_oof_VAT_nl0"]

nll = {}
for line in open(HERE / "peer_recognition_nll.jsonl"):
    d = json.loads(line)
    nll[d["ntitle"]] = d["mean_nll"]
NLL = np.array([nll.get(str(nt), np.nan) for nt in ids_E])

rows = [json.loads(l) for l in open(TD.parents[1] / "datasets/peer-review/vat_3y/revealed.jsonl")]
yr = {rr["ntitle"]: int(rr["year"]) for rr in rows}
year = np.array([yr.get(str(nt), -1) for nt in ids_E])

oa = {json.loads(l)["work_id"]: json.loads(l) for l in open(HERE / "openalex_authorships.jsonl")}
wid_by_nt = {rr["ntitle"]: rr["id"].rsplit("/", 1)[-1] for rr in rows}
cites = np.array([(oa.get(wid_by_nt.get(str(nt), ""), {}) or {}).get("cited_by_count") or np.nan
                  for nt in ids_E], dtype=float)

fin = np.isfinite(NLL)
rep = {"n_with_nll": int(fin.sum()),
       "sanity_spearman_nll_vs_own_cites": float(spearmanr(NLL[fin & np.isfinite(cites)],
                                                           cites[fin & np.isfinite(cites)]).statistic),
       "sanity_spearman_nll_vs_year": float(spearmanr(NLL[fin], year[fin]).statistic),
       "bands": {}}

rng = np.random.default_rng(20260812)
def boot_resid(idx, n=4000):
    est = roc_auc_score(y[idx], oof_d[idx]) - roc_auc_score(y[idx], oof_c[idx])
    bs = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(set(y[s].tolist())) < 2:
            continue
        bs.append(roc_auc_score(y[s], oof_d[s]) - roc_auc_score(y[s], oof_c[s]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": int(len(idx)), "residual": float(est), "ci95": [float(lo), float(hi)],
            "p_gt_0": float(np.mean(np.array(bs) > 0))}

BANDS = {"2013-2019": (year >= 2013) & (year <= 2019),
         "2020-2021": (year >= 2020) & (year <= 2021),
         "2022-2023": (year >= 2022) & (year <= 2023),
         "ALL": year > 0}
for name, m in BANDS.items():
    mm = m & fin
    if mm.sum() < 50:
        continue
    med = np.median(NLL[mm])
    lo_idx = np.where(mm & (NLL <= med))[0]   # low NLL = RECOGNIZED
    hi_idx = np.where(mm & (NLL > med))[0]
    band = {"median_nll": float(med),
            "recognized_lowNLL": boot_resid(lo_idx),
            "unrecognized_highNLL": boot_resid(hi_idx)}
    band["spearman_nll_vs_cites_in_band"] = (
        float(spearmanr(NLL[mm & np.isfinite(cites)], cites[mm & np.isfinite(cites)]).statistic)
        if (mm & np.isfinite(cites)).sum() > 20 else None)
    rep["bands"][name] = band
    print(name, json.dumps(band), flush=True)

json.dump(rep, open(HERE / "recognition_readout.json", "w"), indent=1)
print("RECOG_READOUT_DONE", flush=True)
