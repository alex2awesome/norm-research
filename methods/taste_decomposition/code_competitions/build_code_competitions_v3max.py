#!/usr/bin/env python3
"""V3-MAX for code competitions (user directive 2026-08-13: "why so few training
examples? train with more... make sure the V3 sees VA+VA_new scores as well as the
full coding training data").

Upgrades over the landed v3_aug (which read .6554 at ~680 train rows/fold):
  * TRAINING POOL = the full labeled four-platform population, 6,353 rows
    (AC strict-L1 2,495 + LC 1,995 + CC 995 + CF 868) — ~5,080 train rows/fold.
  * BLOCK = ALL VA + VA_new scores, not a top-10: 27 deterministic V/V_new features
    (v_features, computable on every row) + all 139 judged bank criteria with their
    registry names (real scores where Gemma-scored: AC-999 + CF-869; "NA" elsewhere,
    matching judge-NA semantics).
  * READOUT stays same-rows AC-999 (each row's prediction comes from the fold that
    held its problem group out; groups are platform-prefixed canonical_pid so no
    problem straddles train/test). Secondary: all-platform OOF.
ESTIMAND unchanged: fused V+A+T arm, max-of-variants VAT column ONLY.

  python3 build_code_competitions_v3max.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

R = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = HERE / "dense_crossfit_v3max"
sys.path.insert(0, str(HERE))
from build_v_and_readout import v_features  # noqa: E402

V2 = R / "outputs/v2_analysis"
NAMES = json.loads((R / "methods/taste_decomposition/closure/code_v3/abank_rescore/"
                    "aspect_names.json").read_text())

# ---- population: four platforms, AC uses the strict-L1 labels
cells = {
    "ac": pd.read_parquet(V2 / "dense_ceiling/cell_ac_l1.parquet"),
    "cf": pd.read_parquet(V2 / "dense_ceiling/cell_cf.parquet"),
    "lc": pd.read_parquet(V2 / "dense_ceiling/cell_lc.parquet"),
    "cc": pd.read_parquet(V2 / "dense_ceiling/cell_cc_v2_pooled.parquet"),
}
rows = []
for plat, d in cells.items():
    d = d[["pair_id", "canonical_pid", "candidate_code", "label"]].copy()
    d["platform"] = plat
    rows.append(d)
pop = pd.concat(rows, ignore_index=True).dropna(subset=["label"])
pop["label"] = pop.label.astype(int)
pop["group"] = pop.platform + ":" + pop.canonical_pid.astype(str)
assert pop.pair_id.is_unique
print(f"[pop] {len(pop)} labeled rows, {pop.group.nunique()} groups, "
      f"pos_rate {pop.label.mean():.3f}, per-platform "
      f"{pop.platform.value_counts().to_dict()}", flush=True)

# ---- A scores where they exist (AC + CF banks, same 139-column schema)
banks = []
for f in ["comp_fourplatform_cells/ac_bank_scores.parquet",
          "comp_fourplatform_cells/cf_bank_scores.parquet"]:
    banks.append(pd.read_parquet(V2 / f))
a_cols = [c for c in banks[0].columns if c.endswith("_score")]
assert all(set(a_cols) <= set(b.columns) for b in banks)
A = pd.concat([b[["pair_id"] + a_cols] for b in banks], ignore_index=True)
A = A.drop_duplicates("pair_id").set_index("pair_id")
n_scored = pop.pair_id.isin(A.index).sum()
print(f"[bank] {len(a_cols)} criteria; judged coverage {n_scored}/{len(pop)}", flush=True)

# ---- language inference for v_features (cpp vs python-ish)
def infer_lang(code):
    return "cpp" if ("#include" in code or re.search(r"\bstd::|int main\s*\(", code)) else "py"

def fmt(v):
    if v is None or v != v:
        return "NA"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return f"{v:.3f}".rstrip("0").rstrip(".")

def block_for(code, pair_id):
    V = v_features(code, infer_lang(code))
    lines = ["VA metrics:"]
    lines += [f"    {k}: {fmt(float(v))}" for k, v in V.items()]
    arow = A.loc[pair_id] if pair_id in A.index else None
    for c in a_cols:
        aid = c[:-len("_score")]
        name = NAMES.get(aid, f"criterion {aid}")
        val = float(arow[c]) if arow is not None else float("nan")
        lines.append(f"    {name}: {fmt(val)}")
    return "\n".join(lines)

print("[block] rendering 6,353 blocks (27 V + 139 A lines each)...", flush=True)
pop = pop.reset_index(drop=True)
texts = [block_for(c, p) + "\n\nSUBMISSION CODE:\n" + str(c)
         for c, p in zip(pop.candidate_code, pop.pair_id)]
lens = np.array([len(t) for t in texts])
print(f"[block] total chars median {int(np.median(lens))} p99 {int(np.percentile(lens,99))} "
      f"max {lens.max()}", flush=True)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
folds = list(sgkf.split(np.zeros(len(pop)), pop.label.values, pop.group.values))
rng = np.random.default_rng(20260813)
man = {"design_id": "code_competitions_v3max",
       "estimand": "FUSED V+A+T arm, max-of-variants VAT column ONLY",
       "pool": {p: int(n) for p, n in pop.platform.value_counts().items()},
       "n": int(len(pop)), "n_groups": int(pop.group.nunique()),
       "judged_A_coverage": int(n_scored), "block": "ALL 27 V/V_new + ALL 139 named A "
       "criteria (NA where unscored)", "folds_protocol":
       "StratifiedGroupKFold(5,shuffle,rs=0) by platform-prefixed canonical_pid over "
       "the UNION pool; readout = same-rows AC-999 OOF", "max_len": 6144, "folds": {}}

for k, (tr, te) in enumerate(folds):
    gtr = pop.group.values[tr]
    tr_groups = np.unique(gtr)
    ev_groups = set(rng.choice(tr_groups, size=max(1, int(len(tr_groups) * 0.12)),
                               replace=False))
    is_ev = np.array([g in ev_groups for g in gtr])
    d = OUT / "arm_a" / f"fold{k}"
    (d / "split").mkdir(parents=True, exist_ok=True)
    def df_of(idx):
        return pd.DataFrame({"text": [texts[i] for i in idx],
                             "judgement": pop.label.values[idx],
                             "group": pop.group.values[idx],
                             "row_id": pop.pair_id.values[idx]})
    df_tr, df_ev, df_te = df_of(tr[~is_ev]), df_of(tr[is_ev]), df_of(te)
    pd.concat([df_tr, df_ev, df_te]).to_csv(d / "data.csv", index=False)
    df_tr.to_csv(d / "split/train.csv", index=False)
    df_ev.to_csv(d / "split/eval.csv", index=False)
    df_te.to_csv(d / "split/test.csv", index=False)
    man["folds"][f"fold{k}"] = {"n_train": int(len(df_tr)), "n_eval": int(len(df_ev)),
                                "n_test": int(len(df_te))}
    print(f"[fold{k}] train {len(df_tr)} eval {len(df_ev)} test {len(df_te)}", flush=True)

(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
print("BUILD_DONE", OUT, flush=True)
