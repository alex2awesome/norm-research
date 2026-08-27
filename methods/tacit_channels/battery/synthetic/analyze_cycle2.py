"""Cycle-2 calibration analysis: negation-wording A/B, E5 replacement, known-mixture
graded holistic (end-to-end estimator validation), second composed pair."""
from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np

from methods.tacit_channels.battery.stats import holistic_residual
from methods.tacit_channels.battery.synthetic.constructs import (
    CONSTRUCTS, ITEMS, NEG_DIRECT, blend_oracle_score, oracle_vector,
)

RESULTS = "outputs/tacit_channels/battery_calibration/results_cycle2.jsonl"
YESNO = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)
INT = re.compile(r"\b(\d{1,3})\b")

by = defaultdict(dict)
for line in open(RESULTS):
    r = json.loads(line)
    cid, variant = r["aspect_id"].rsplit("::", 1)
    by[(cid, variant)][int(r["datapoint_id"])] = r["raw"]

def yn(raw):
    m = YESNO.search(raw or "")
    return None if not m else m.group(1).upper() == "YES"

def integer(raw):
    for m in INT.finditer(raw or ""):
        v = int(m.group(1))
        if 0 <= v <= 10:
            return float(v)
    return float("nan")

def acc(cid, variant, truth):
    raws = by.get((cid, variant), {})
    pairs = [(yn(raws[j]), truth[j]) for j in raws if yn(raws[j]) is not None]
    return (float(np.mean([p == t for p, t in pairs])), len(pairs)) if pairs else (None, 0)

print("== negation wording A/B (accuracy vs NOT oracle) ==")
for cid in NEG_DIRECT:
    inv = [not x for x in oracle_vector(cid)]
    fx, _ = acc(cid, "neg_fx", inv)
    dr, _ = acc(cid, "neg_direct", inv)
    print(f"  {cid:12s} neg_fx {fx:.3f}  neg_direct {dr:.3f}")

print("== E5 replacement construct ==")
v5 = oracle_vector("E5_qmark")
print("  tf", acc("E5_qmark", "tf", v5), " exclusion_fx",
      acc("E5_qmark", "exclusion_fx", [not x for x in v5]))
craw = by.get(("E5_qmark", "confidence"), {})
cv = [INT.search(craw[j]) for j in craw]
vals = [float(m.group(1)) for m in cv if m and 0 <= float(m.group(1)) <= 100]
print(f"  confidence mean {np.mean(vals):.1f} n_unique {len(set(vals))}")

print("== composed E3&&E4 ==")
va, vb = oracle_vector("E3_animal"), oracle_vector("E4_digit")
print("  ", acc("E3_animal&&E4_digit", "composed", [x and y for x, y in zip(va, vb)]))

print("== known-mixture graded holistic (end-to-end estimator validation) ==")
braw = by.get(("BLEND", "graded"), {})
y = np.array([integer(braw.get(j, "")) for j in range(len(ITEMS))])
truth = np.array([blend_oracle_score(t) for t in ITEMS])
ok = np.isfinite(y)
from methods.tacit_channels.channels.common import spearman
print(f"  parse rate {ok.mean():.2f}; rho(GLM blend rating, oracle blend) "
      f"{spearman(y[ok], truth[ok]):+.3f}")
X = np.column_stack([np.array(oracle_vector(c), float) for c in
                     ("E1_exclaim", "E3_animal", "E4_digit")])
Xbad = np.column_stack([np.array(oracle_vector(c), float) for c in
                        ("E5_qmark", "G1_formal", "G2_excited")])
idx = np.where(ok)[0]
fit = np.zeros(len(ITEMS), bool); ev = np.zeros(len(ITEMS), bool)
fit[idx[::2]] = True; ev[idx[1::2]] = True
r_good = holistic_residual(np.nan_to_num(y), X, fit, ev, y_std_floor=0.3)
r_bad = holistic_residual(np.nan_to_num(y), Xbad, fit, ev, y_std_floor=0.3)
print(f"  span-recovery R2 on TRUE predictors: {r_good}")
print(f"  R2 on WRONG predictors (should be lower): {r_bad}")

print("== H1_charming graded scale use ==")
hraw = by.get(("H1_charming", "graded"), {})
hv = np.array([integer(hraw.get(j, "")) for j in range(len(ITEMS))])
hv = hv[np.isfinite(hv)]
print(f"  n {len(hv)} mean {hv.mean():.2f} std {hv.std():.2f} n_unique {len(set(hv))}")
