#!/usr/bin/env python3
"""ROUND-0 structural diagnostic: how much of this cell is COMPANY IDENTITY?

The N&C campaign's hardest structural fact was docket identity (docket-alone AUC .916).
The press cell's analogue is the issuing organisation, and the brief names
"company size/prominence fingerprints" as an upstream Track-B prior, so the size of
that floor has to be measured before any channel is mined against it.

Three readouts, all on saved predictions plus one cheap fit:

  1. COMPANY-IDENTITY-ALONE AUC.  Grouped-OOF is impossible for a company-identity
     model by construction (the group IS the feature), so the honest version is the
     LEAVE-ONE-COMPANY-OUT-INFEASIBLE case: report instead the in-sample company mean
     label rate as a ceiling, and the between-company variance decomposition.
  2. WITHIN-COMPANY PAIR CONCORDANCE.  AUC on a binary label is exactly pair
     concordance, so pool every (positive, negative) pair that lies inside ONE company
     and read T and VA_nl on exactly those pairs.  This is the N&C 7.3 enlargement and
     it answers "is the residual a between-company effect?" without needing a
     per-company AUC.
  3. CONCENTRATION.  What share of the HONEST population the largest companies carry --
     the number that governs how much any 45-company jackknife can be trusted.

CPU only.  Usage: python company_structure.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

import cells as C
import closure_core as L

HERE = Path(__file__).resolve().parent


def within_group_pairs(y, g, scores):
    """Pair concordance inside each group, pooled. scores: {name: vector}."""
    num = {k: 0.0 for k in scores}
    tot = 0
    per_group = []
    for c in sorted(set(g.tolist())):
        m = g == c
        yy = y[m]
        if yy.sum() == 0 or (yy == 0).sum() == 0:
            continue
        pi, ni = np.where(yy == 1)[0], np.where(yy == 0)[0]
        n_pairs = len(pi) * len(ni)
        rec = {"company": c, "n": int(m.sum()), "n_pairs": int(n_pairs)}
        for k, s in scores.items():
            ss = s[m]
            diff = ss[pi][:, None] - ss[ni][None, :]
            conc = float((diff > 0).sum() + 0.5 * (diff == 0).sum())
            num[k] += conc
            rec[k] = conc / n_pairs
        per_group.append(rec)
        tot += n_pairs
    out = {k: num[k] / tot for k in scores}
    out["n_pairs"] = int(tot)
    out["n_companies_with_both_labels"] = len(per_group)
    return out, per_group


def main():
    C.sklearn_guard()
    d = C.load()
    z = np.load(HERE / "press_verdict_r0_preds.npz", allow_pickle=True)
    va, dense = z["va_nl"], d["dense"]
    y, g = d["y"], d["groups"]
    held = np.isin(d["dense_split"], ["eval", "test"])

    out = {"cell": "press_verdict"}

    # --- 1. how much label variance sits BETWEEN companies -------------------
    rates, sizes = [], []
    for c in sorted(set(g.tolist())):
        m = g == c
        rates.append(float(y[m].mean()))
        sizes.append(int(m.sum()))
    rates, sizes = np.array(rates), np.array(sizes)
    grand = float(y.mean())
    ss_between = float((sizes * (rates - grand) ** 2).sum())
    ss_total = float(((y - grand) ** 2).sum())
    out["company_label_structure"] = {
        "n_companies": len(rates),
        "share_of_label_variance_between_companies": ss_between / ss_total,
        "n_companies_all_positive": int((rates == 1).sum()),
        "n_companies_all_negative": int((rates == 0).sum()),
        "n_companies_mixed": int(((rates > 0) & (rates < 1)).sum()),
        "rows_in_single_label_companies": int(sizes[(rates == 0) | (rates == 1)].sum()),
        "note": "a company that is entirely positive or entirely negative contributes NO "
                "within-company pair, so it can only be predicted by company-level "
                "information. This is the press analogue of the N&C docket-identity floor.",
    }

    # --- 2. within-company pair concordance ---------------------------------
    for label, mask in (("HONEST", held), ("FULL_population_bank_only", np.ones(len(y), bool))):
        sc = {"VA_nl": va}
        if label == "HONEST":
            sc["T"] = dense
        pooled = {k: L.auc(y[mask], v[mask]) for k, v in sc.items()
                  if np.isfinite(v[mask]).all()}
        wg, per = within_group_pairs(y[mask], g[mask],
                                     {k: v[mask] for k, v in sc.items()
                                      if np.isfinite(v[mask]).all()})
        out[f"within_company_{label}"] = {"pooled": pooled, "within_company": wg,
                                          "top_companies": sorted(
                                              per, key=lambda r: -r["n_pairs"])[:12]}

    # --- 3. concentration ----------------------------------------------------
    cnt = Counter(g[held].tolist())
    tot = sum(cnt.values())
    top = cnt.most_common(12)
    out["HONEST_concentration"] = {
        "n_rows": tot, "n_companies": len(cnt),
        "top12": [{"company": c, "n": n, "share": n / tot, "pos_rate": float(y[held & (g == c)].mean())}
                  for c, n in top],
        "share_in_top9": sum(n for _, n in cnt.most_common(9)) / tot,
        "eval_companies": sorted(set(g[d["dense_split"] == "eval"].tolist())),
        "test_companies": sorted(set(g[d["dense_split"] == "test"].tolist())),
    }

    (HERE / "company_structure.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "within_company_FULL_population_bank_only"},
                     indent=1, default=float)[:6000])
    print("\nFULL-population within-company:",
          json.dumps(out["within_company_FULL_population_bank_only"]["within_company"], default=float))


if __name__ == "__main__":
    main()
