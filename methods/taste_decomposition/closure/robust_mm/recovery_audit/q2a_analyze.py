#!/usr/bin/env python3
"""Post-process q2a_perconcept_depletion.json: does per-concept slice movement (or
AUC drop) predict rediscovery?  Under the prior-sampling model it should not."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RMM = HERE.parent

dep = json.loads((HERE / "q2a_perconcept_depletion.json").read_text())
recall = json.loads((RMM / "m3_recall.json").read_text())
matched = {(r["rep"], r["concept"]): r["match_primary"]
           for r in recall["records"] if r["kind"] == "heldout"}

rows = []
for c in dep["concepts"]:
    rows.append({**c, "rediscovered": matched[(c["rep"], c["concept"])]})

churn = np.array([r["n_rows_entered"] for r in rows], float)
drop = np.array([r["drop_honest_1244"] for r in rows], float)
strength = np.array([abs(r["alone_auc_fitmine"] - .5) for r in rows], float)
m = np.array([r["rediscovered"] for r in rows], float)


def mw_auc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    return float(sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
                 / (len(pos) * len(neg)))


def spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


out = {
    "n": len(rows),
    "churn_summary": {"min": int(churn.min()), "median": float(np.median(churn)),
                      "max": int(churn.max()), "mean": float(churn.mean())},
    "eight_concept_churn_reference": [11, 10, 15],
    "drop_summary": {"min": float(drop.min()), "median": float(np.median(drop)),
                     "max": float(drop.max()), "mean": float(drop.mean()),
                     "n_inside_pm_.007_noise_band": int((np.abs(drop) < .007).sum())},
    "predicts_rediscovery": {
        "rank_auc_churn": mw_auc(churn[m == 1], churn[m == 0]),
        "rank_auc_drop": mw_auc(drop[m == 1], drop[m == 0]),
        "rank_auc_strength_reference": mw_auc(strength[m == 1], strength[m == 0]),
    },
    "spearman": {"churn_vs_strength": spear(churn, strength),
                 "drop_vs_strength": spear(drop, strength),
                 "churn_vs_drop": spear(churn, drop)},
    "table": [{k: r[k] for k in ("rep", "concept", "stratum", "alone_auc_fitmine",
                                 "drop_honest_1244", "n_rows_entered",
                                 "concept_nonnull_frac_entered_rows",
                                 "concept_nonnull_frac_overall", "rediscovered")}
              for r in sorted(rows, key=lambda r: -r["n_rows_entered"])],
}

# permutation p for the churn->rediscovery rank AUC
rng = np.random.default_rng(0)
obs = out["predicts_rediscovery"]["rank_auc_churn"]
perm = []
for _ in range(10000):
    s = rng.permutation(m)
    perm.append(mw_auc(churn[s == 1], churn[s == 0]))
out["predicts_rediscovery"]["rank_auc_churn_perm_p"] = float(
    np.mean([abs(pp - .5) >= abs(obs - .5) for pp in perm]))

(HERE / "q2a_analysis.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "table"}, indent=1))
print("\nper-concept (sorted by churn):")
for r in out["table"]:
    print(f"  churn={r['n_rows_entered']:2d} drop={r['drop_honest_1244']:+.4f} "
          f"{'REDISC' if r['rediscovered'] else '      '} "
          f"[{r['stratum']:4s} {r['alone_auc_fitmine']:.3f}] {r['concept'][:52]}")
