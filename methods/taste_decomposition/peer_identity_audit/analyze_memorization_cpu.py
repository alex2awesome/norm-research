#!/usr/bin/env python3
"""peer_revealed memorization-suspicion battery (CPU leg) — user-ordered audit
continuation, 2026-08-12. All readouts DESCRIPTIVE (report results, not verdicts).

Logic: if the dense model's edge comes from PRETRAINING MEMORIZATION of these ICLR
papers (Llama-3.1, cutoff ~Dec 2023), the residual should
  (a) concentrate in OLD years (2013-2019: deeply represented, citation standing
      settled in the pretraining corpus) and vanish in 2022-2023 (standing unsettled
      at pretraining time);
  (b) concentrate in the FAMOUS slice (papers the model has certainly seen);
  (c) make dense track REALIZED fame within a label class (Spearman dense-prob vs
      own cited_by_count within y=1 and within y=0) far better than the bank+nuis
      stack does — knowing outcomes, not quality proxies.
The label is top-vs-bottom quartile of citation percentile WITHIN venue x year, so
year-alone should be ~chance by construction — verified empirically on E.

Inputs: fusion/t0_rows/peer_revealed.npz (E-rows), F2 machinery for the (c)/(d)
OOF vectors (same frozen stack as f2_deconf/f2_identity), OpenAlex pull for
year + cited_by_count. Writes memorization_cpu_report.json.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
TD = HERE.parent
REPO = TD.parents[1]

def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m

F2 = _mod(TD / "fusion/f2_deconf.py", "f2_deconf_mod")
CELL = "peer_revealed"

meta, ids_E, y, groups, dense, t0col = F2.load_E(CELL)
ad = F2.F2C.ADAPTERS[CELL]()
bank, nuis, join = F2.align(CELL, ad, ids_E, y, groups)
bn = np.column_stack([bank, nuis])
r = F2.fit_arm(meta["family"], bn, dense, y, groups)
oof_c, oof_d = r["_oof_VA_nl0"], r["_oof_VAT_nl0"]

# ---- join year + own cited_by_count by ntitle
rows = [json.loads(l) for l in open(REPO / "datasets/peer-review/vat_3y/revealed.jsonl")]
oa = {json.loads(l)["work_id"]: json.loads(l)
      for l in open(HERE / "openalex_authorships.jsonl")}
by_nt = {}
for rr in rows:
    wid = rr["id"].rsplit("/", 1)[-1]
    m = oa.get(wid, {})
    by_nt[rr["ntitle"]] = {"year": int(rr["year"]), "cites": m.get("cited_by_count")}

year = np.array([by_nt.get(str(nt), {}).get("year", -1) for nt in ids_E])
cites = np.array([by_nt.get(str(nt), {}).get("cites") if by_nt.get(str(nt), {}).get("cites") is not None
                  else np.nan for nt in ids_E], dtype=float)
rep = {"n_E": int(len(y)), "joined_year": int((year > 0).sum()),
       "joined_cites": int(np.isfinite(cites).sum())}

def auc_or_none(yy, ss):
    yy, ss = np.asarray(yy), np.asarray(ss)
    if len(set(yy.tolist())) < 2:
        return None
    return float(roc_auc_score(yy, ss))

# ---- construction check: year alone
rep["year_alone_auc"] = auc_or_none(y, year.astype(float))

# ---- (a) residual by year band
bands = {"2013-2019": (year >= 2013) & (year <= 2019),
         "2020-2021": (year >= 2020) & (year <= 2021),
         "2022-2023": (year >= 2022) & (year <= 2023)}
rep["residual_by_year_band"] = {}
for name, m in bands.items():
    if m.sum() < 30:
        continue
    rep["residual_by_year_band"][name] = {
        "n": int(m.sum()), "pos_rate": float(y[m].mean()),
        "auc_c_bank_nuis": auc_or_none(y[m], oof_c[m]),
        "auc_d_plus_T": auc_or_none(y[m], oof_d[m]),
        "auc_dense_raw": auc_or_none(y[m], dense[m]),
    }
    b = rep["residual_by_year_band"][name]
    if b["auc_c_bank_nuis"] is not None and b["auc_d_plus_T"] is not None:
        b["residual_d_minus_c"] = b["auc_d_plus_T"] - b["auc_c_bank_nuis"]

# ---- (b) residual by own-citation fame band (WITHIN year quartile label -> use
# raw cited_by_count terciles within year band to respect construction)
rep["residual_by_fame_band"] = {}
fin = np.isfinite(cites)
for name, m in bands.items():
    mm = m & fin
    if mm.sum() < 60:
        continue
    med = np.nanmedian(cites[mm])
    for tag, sel in (("low_fame", mm & (cites <= med)), ("high_fame", mm & (cites > med))):
        if sel.sum() < 25:
            continue
        e = {"n": int(sel.sum()), "median_cites": float(np.nanmedian(cites[sel])),
             "auc_c": auc_or_none(y[sel], oof_c[sel]),
             "auc_d": auc_or_none(y[sel], oof_d[sel])}
        if e["auc_c"] is not None and e["auc_d"] is not None:
            e["residual"] = e["auc_d"] - e["auc_c"]
        rep["residual_by_fame_band"][f"{name}/{tag}"] = e

# ---- extreme slices
for tag, sel in (("cites_gt_1000", fin & (cites > 1000)),
                 ("cites_lt_100", fin & (cites < 100))):
    e = {"n": int(sel.sum()), "pos_rate": float(y[sel].mean()) if sel.sum() else None,
         "auc_c": auc_or_none(y[sel], oof_c[sel]) if sel.sum() >= 20 else None,
         "auc_d": auc_or_none(y[sel], oof_d[sel]) if sel.sum() >= 20 else None}
    rep[tag] = e

# ---- (c) within-class realized-fame tracking
rep["within_class_fame_tracking_spearman"] = {}
for cls in (1, 0):
    m = (y == cls) & fin
    if m.sum() < 30:
        continue
    rep["within_class_fame_tracking_spearman"][f"y={cls}"] = {
        "n": int(m.sum()),
        "dense_raw_vs_own_cites": float(spearmanr(dense[m], cites[m]).statistic),
        "oof_d_vs_own_cites": float(spearmanr(oof_d[m], cites[m]).statistic),
        "oof_c_bank_nuis_vs_own_cites": float(spearmanr(oof_c[m], cites[m]).statistic),
    }

json.dump(rep, open(HERE / "memorization_cpu_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
print("MEMO_CPU_DONE", flush=True)
