#!/usr/bin/env python3
"""RUNG 2 READOUT — cw_community (design §2.5, frozen 2026-08-21).

Selection policies over the SAME K=16 candidate pool per prompt:
  SEL_bank   = argmax of the frozen articulated-bank selector (fullfit on real E)
  SEL_dense  = argmax of dense half-A (selector)
  SEL_random = stable-hash pick (floor)
Scoring arms: bank selector score; dense half-B (ARBITER — zero training rows
shared with half-A). Diagonal cells are never quoted (§0); Δ_articulated's
sign is forced ≤0 by construction (bank is its own argmax) — its MAGNITUDE is
the readout, per §2.5.1.

All Δs on the within-prompt rank scale [0,1] (rank of the pick among the
prompt's K candidates under the scoring arm), paired per prompt, 2,000-draw
prompt bootstrap. CPU, mac.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 12345


def load_csv(name, key, val):
    d = {}
    for r in csv.DictReader(open(HERE / name)):
        d[r[key]] = float(r[val])
    return d


bank = load_csv("rung2_bank_selector_scores_cw.csv", "cand_id", "bank_score")
dA = load_csv("rung2_dense_scores_cw_halfA.csv", "cand_id", "dense_halfA_prob")
dB = load_csv("rung2_dense_scores_cw_halfB.csv", "cand_id", "dense_halfB_prob")
cands = list(csv.DictReader(open(HERE / "rung2_candidates_cw_community_full.csv")))
nchars = {r["cand_id"]: float(r["n_chars"]) for r in cands}
by_prompt = {}
for r in cands:
    by_prompt.setdefault(r["prompt_id"], []).append(r["cand_id"])
K = len(next(iter(by_prompt.values())))
print(f"{len(by_prompt)} prompts x K={K}; scored bank/{len(bank)} "
      f"dA/{len(dA)} dB/{len(dB)}", flush=True)


def within_rank(scorer, ids, pick):
    s = np.array([scorer[i] for i in ids])
    return float((s < scorer[pick]).mean() + 0.5 * (s == scorer[pick]).mean())


rows = []
for pid, ids in sorted(by_prompt.items()):
    sel_bank = max(ids, key=lambda i: bank[i])
    sel_dense = max(ids, key=lambda i: dA[i])
    sel_rand = ids[int(hashlib.md5(f"rand::{pid}".encode()).hexdigest(), 16) % len(ids)]
    rows.append(dict(
        pid=pid, agree=sel_bank == sel_dense,
        # off-diagonal Δs on within-prompt rank scale
        d_artic=within_rank(bank, ids, sel_dense) - within_rank(bank, ids, sel_bank),
        d_dense=within_rank(dB, ids, sel_dense) - within_rank(dB, ids, sel_bank),
        d_dense_rand=within_rank(dB, ids, sel_rand),
        arbB_rank_bankpick=within_rank(dB, ids, sel_bank),
        arbB_rank_densepick=within_rank(dB, ids, sel_dense),
        bank_rank_densepick=within_rank(bank, ids, sel_dense),
        n_bank=nchars[sel_bank], n_dense=nchars[sel_dense],
        s_bank_bankpick=bank[sel_bank], s_bank_densepick=bank[sel_dense],
        cand_bank=sel_bank, cand_dense=sel_dense,
    ))

A = np.array([r["d_artic"] for r in rows])
D = np.array([r["d_dense"] for r in rows])
asym = D + A          # = D - |A| since A<=0 by construction
agree = np.mean([r["agree"] for r in rows])

rng = np.random.default_rng(SEED)
boots = {k: [] for k in ("d_artic", "d_dense", "asym")}
n = len(rows)
for _ in range(2000):
    ix = rng.integers(0, n, n)
    boots["d_artic"].append(A[ix].mean())
    boots["d_dense"].append(D[ix].mean())
    boots["asym"].append(asym[ix].mean())
ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
      for k, v in boots.items()}

# blindness AUCs on the two selected SETS (disagreement prompts only —
# agreement prompts contribute identical items to both sets)
from sklearn.metrics import roc_auc_score
dis = [r for r in rows if not r["agree"]]
yb = [1] * len(dis) + [0] * len(dis)
blind = {}
for label, scorer in (("bank", bank), ("arbiter_halfB", dB), ("n_chars", nchars)):
    v = [scorer[r["cand_dense"]] for r in dis] + [scorer[r["cand_bank"]] for r in dis]
    blind[label] = float(roc_auc_score(yb, v))

out = {
    "cell": "cw_community", "n_prompts": n, "K": K,
    "design": "notes/2026-08-21__rung12_design_gap_consequences.md §2.5",
    "agreement_rate": float(agree),
    "DELTA_articulated_rank": {"mean": float(A.mean()), "ci95": ci["d_artic"],
        "note": "bank rank of dense pick minus 1.0 (sign forced <=0; magnitude is the readout)"},
    "DELTA_dense_rank": {"mean": float(D.mean()), "ci95": ci["d_dense"],
        "p_gt_0": float(np.mean(np.array(boots["d_dense"]) > 0))},
    "ASYMMETRY": {"mean": float(asym.mean()), "ci95": ci["asym"],
        "p_gt_0": float(np.mean(np.array(boots["asym"]) > 0)),
        "prereg": "A > 0"},
    "arbiter_rank_of_picks": {
        "dense_pick_mean": float(np.mean([r["arbB_rank_densepick"] for r in rows])),
        "bank_pick_mean": float(np.mean([r["arbB_rank_bankpick"] for r in rows])),
        "random_floor_mean": float(np.mean([r["d_dense_rand"] for r in rows]))},
    "blindness_auc_disagreement_sets": blind,
    "n_disagreement_prompts": len(dis),
    "nuisance_quick_screen": {
        "len_bankpick_mean": float(np.mean([r["n_bank"] for r in rows])),
        "len_densepick_mean": float(np.mean([r["n_dense"] for r in rows])),
        "len_auc_between_sets": blind["n_chars"]},
}
(HERE / "rung2_readout_cw.json").write_text(json.dumps(out, indent=2))
per_rows = [{k: r[k] for k in ("pid", "agree", "d_artic", "d_dense",
                               "cand_bank", "cand_dense")} for r in rows]
(HERE / "rung2_readout_cw_perprompt.json").write_text(json.dumps(per_rows, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "design"}, indent=1))
