#!/usr/bin/env python3
"""RUNG 2 READOUT v2 — Addendum C (2026-08-24): floor-based statistics with
the CERTIFIED full dense model as arbiter; no own-argmax comparisons.

  loss_on_bank_channel(dense policy) = bank_rank(SEL_dense) − bank_rank(SEL_random)
  loss_on_dense_channel(bank policy) = arb_rank(SEL_bank)   − arb_rank(SEL_random)
  A' = former − latter   (prereg soft: ≈0 on THESE homogeneous pools)

Plus: within-pool agreement of certified arbiter with half-A/half-B
(homogeneity mechanism check) and blindness AUCs. CPU, mac.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
SEED = 12345


def load(name, val):
    return {r["cand_id"]: float(r[val]) for r in csv.DictReader(open(HERE / name))}


bank = load("rung2_bank_selector_scores_cw.csv", "bank_score")
dA = load("rung2_dense_scores_cw_halfA.csv", "dense_halfA_prob")
dB = load("rung2_dense_scores_cw_halfB.csv", "dense_halfB_prob")
dF = load("rung2_dense_scores_cw_full.csv", "dense_full_prob")
cands = list(csv.DictReader(open(HERE / "rung2_candidates_cw_community_full.csv")))
by_prompt = {}
for r in cands:
    by_prompt.setdefault(r["prompt_id"], []).append(r["cand_id"])


def wrank(scorer, ids, pick):
    s = np.array([scorer[i] for i in ids])
    return float((s < scorer[pick]).mean() + 0.5 * (s == scorer[pick]).mean())


rows = []
for pid, ids in sorted(by_prompt.items()):
    sel_bank = max(ids, key=lambda i: bank[i])
    sel_dense = max(ids, key=lambda i: dF[i])       # selector = certified model
    sel_rand = ids[int(hashlib.md5(f"rand::{pid}".encode()).hexdigest(), 16) % len(ids)]
    rows.append(dict(
        pid=pid, agree=sel_bank == sel_dense,
        bank_rank_dense=wrank(bank, ids, sel_dense),
        bank_rank_rand=wrank(bank, ids, sel_rand),
        arb_rank_bank=wrank(dF, ids, sel_bank),
        arb_rank_rand=wrank(dF, ids, sel_rand),
        cand_bank=sel_bank, cand_dense=sel_dense,
    ))

lb = np.array([r["bank_rank_dense"] - r["bank_rank_rand"] for r in rows])
ld = np.array([r["arb_rank_bank"] - r["arb_rank_rand"] for r in rows])
Ap = lb - ld
rng = np.random.default_rng(SEED)
boots = {"lb": [], "ld": [], "Ap": []}
n = len(rows)
for _ in range(2000):
    ix = rng.integers(0, n, n)
    boots["lb"].append(lb[ix].mean())
    boots["ld"].append(ld[ix].mean())
    boots["Ap"].append(Ap[ix].mean())
ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
      for k, v in boots.items()}

dis = [r for r in rows if not r["agree"]]
yb = [1] * len(dis) + [0] * len(dis)
blind = {}
for label, sc in (("bank", bank), ("certified_arbiter", dF)):
    v = [sc[r["cand_dense"]] for r in dis] + [sc[r["cand_bank"]] for r in dis]
    blind[label] = float(roc_auc_score(yb, v)) if dis else None

# homogeneity mechanism check: certified vs halves, within-pool + global
def wp_corr(x, ycol):
    out = []
    for ids in by_prompt.values():
        a = np.array([x[i] for i in ids]); b = np.array([ycol[i] for i in ids])
        if a.std() > 0 and b.std() > 0:
            out.append(spearmanr(a, b).statistic)
    return float(np.mean(out))


allids = sorted(dF)
gl = {p: spearmanr([dF[i] for i in allids], [q[i] for i in allids]).statistic
      for p, q in (("halfA", dA), ("halfB", dB), ("bank", bank))}

out = {
    "design": "ADDENDUM C, notes/2026-08-21__rung12_design_gap_consequences.md",
    "arbiter": "certified wp_clean full dense (AUC .786)",
    "n_prompts": n, "agreement_rate": float(np.mean([r["agree"] for r in rows])),
    "dense_policy_signal_on_bank_channel_vs_floor": {
        "mean": float(lb.mean()), "ci95": ci["lb"]},
    "bank_policy_signal_on_dense_channel_vs_floor": {
        "mean": float(ld.mean()), "ci95": ci["ld"]},
    "FLOOR_ASYMMETRY_Aprime": {"mean": float(Ap.mean()), "ci95": ci["Ap"],
        "p_gt_0": float(np.mean(np.array(boots["Ap"]) > 0))},
    "blindness_auc_disagreement_sets": blind,
    "n_disagreement_prompts": len(dis),
    "agreement_check": {
        "global_rho_certified_vs": gl,
        "within_pool_rho_certified_vs": {
            "halfA": wp_corr(dF, dA), "halfB": wp_corr(dF, dB),
            "bank": wp_corr(dF, bank)},
    },
}
(HERE / "rung2_readout_cw_v2.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
