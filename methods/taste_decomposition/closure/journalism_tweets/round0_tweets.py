#!/usr/bin/env python3
"""Journalism tweets (V9) — Layer-3 closure ROUND 0. APPENDIX CELL.

Prereg: notes/2026-08-05__layer3-closure-prereg.md + FREEZE DECLARATION + ADDENDA.
Campaign note: notes/2026-08-12__closure_journalism_tweets.md.
Layer-1 ledger: methods/taste_decomposition/results/journalism_tweets_ledger.json.

SCOPE IS BOUNDED BY RULING (user, via coordinator 2026-08-12): tweets is the
journalism column's APPENDIX cell, so this script runs round 0 and the gate readout
ONLY. Rounds do not follow automatically even on a gate pass — appendix cells do not
earn fleet spend without an explicit go, because the sealed fleet's one blind look is
the scarce resource.

ORDER OF OPERATIONS, per the coordinator's caution: the ε-resolvability power check
runs BEFORE anything is read off the curve. Homepage curation failed that check at
n=622 MONITOR rows on the same day with the same code; this cell's MONITOR will land
near ~1,250 rows (20% of 6,226 dense-held-out), between homepage's 622 and BBC's 2,060,
so its resolvability is genuinely open. A resolution-bound terminal is an acceptable
outcome and is reported as such — "closed at this cell's resolution", never "saturated".

Cell-specific note: this cell's y is a WITHIN-GROUP MEDIAN SPLIT, so group identity
alone sits at exactly .5000 (Layer-1 measured it) and pooled ≈ within-group by
construction. That is the opposite of BBC most-read (.5814) and it means the pooled
MONITOR tier is the honest primary here.

  python3 round0_tweets.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "methods/taste_decomposition"))
sys.path.insert(0, str(HERE.parent / "bbc_mostread"))
sys.path.insert(0, str(HERE.parent / "homepage_curation"))
import scaleupC_layer1 as SC  # noqa: E402
from round0_bbc import paired_seed_noise, fit_va, swap_pair  # noqa: E402
from round0_homepage import group_boot_ci  # noqa: E402

CELL_BANK = "journalism_tweets"
VA_DIR = REPO / "datasets/journalism-tweets/va"
DENSE = VA_DIR / "dense_standard_journalism_tweets"
BANK_OUT = REPO / "outputs/va_gemma_banks_journalism_tweets"
LEDGER = (REPO / "methods/taste_decomposition/results"
          / "journalism_tweets_ledger.json")
SALT = "journalism-tweets-closure-v1|"
MONITOR_FRAC = 0.20
EPS = 0.005
GATE = 0.02
SEEDS = (42, 1, 2)


def h(g) -> float:
    return int(hashlib.sha256((SALT + str(g)).encode()).hexdigest()[:16], 16) / 2**64


def main():
    argparse.ArgumentParser().parse_args()
    res = {"cell": "journalism_tweets", "scope": "APPENDIX — round 0 + gate only",
           "sklearn_here": sklearn.__version__, "epsilon": EPS,
           "gate_threshold": GATE, "salt": SALT}
    led = json.loads(LEDGER.read_text())
    res["sklearn_layer1"] = led.get("sklearn_version")
    res["sklearn_drift"] = res["sklearn_here"] != res["sklearn_layer1"]

    pop = pd.read_csv(VA_DIR / "population.csv.gz")
    pop["row_id"] = pop.row_id.astype(str)
    pop["group"] = pop.group.astype(str)

    # ---- 1. dense join gate (preds carry no row_id -> ORDER join, proven)
    dj = {"legs": {}}
    ids, per_seed = [], {s: [] for s in SEEDS}
    for leg in ("eval", "test"):
        sp = pd.read_csv(DENSE / "split" / f"{leg}.csv")
        for s in SEEDS:
            p = pd.read_csv(DENSE / f"rm_out_seed{s}" / f"preds_{leg}.csv")
            if len(p) != len(sp):
                raise SystemExit(f"DENSE JOIN GATE FAIL {leg}/{s}: len")
            if not (p["judgement"].values == sp["judgement"].values).all():
                raise SystemExit(f"DENSE JOIN GATE FAIL {leg}/{s}: judgement sequence")
            if "group" in p and "group" in sp:
                if not (p["group"].astype(str).values
                        == sp["group"].astype(str).values).all():
                    raise SystemExit(f"DENSE JOIN GATE FAIL {leg}/{s}: group sequence")
            per_seed[s].append(p["prob"].values.astype(float))
        rng = np.random.default_rng(3)
        sh = per_seed[42][-1].copy()
        rng.shuffle(sh)
        dj["legs"][leg] = {
            "n": int(len(sp)),
            "auc_seed42": float(roc_auc_score(sp["judgement"], per_seed[42][-1])),
            "auc_shuffled_counterfactual": float(roc_auc_score(sp["judgement"], sh)),
            "sequence_match_all_seeds": True}
        ids += sp["row_id"].astype(str).tolist()
    dense_ho = {s: np.concatenate(v) for s, v in per_seed.items()}
    dj["n_dense_heldout"] = len(ids)
    dj["ids_unique"] = bool(len(set(ids)) == len(ids))
    if not dj["ids_unique"]:
        raise SystemExit("DENSE JOIN GATE FAIL: duplicate dense row_ids")
    dj["passes"] = True
    res["dense_join_gate"] = dj
    print("[dense join gate] PASS", json.dumps(dj["legs"]), flush=True)

    # ---- bank
    meta, A, V, groups_bank, shard, bids = SC.load_scaleupC_bank(CELL_BANK, out=BANK_OUT)
    idl = [str(i) for i in bids]
    byid = pop.set_index("row_id")
    y_all = byid.loc[idl, "judgement"].values.astype(int)
    grp = byid.loc[idl, "group"].values.astype(str)

    # ---- 2. splits
    ho = set(ids)
    in_ho = np.array([i in ho for i in idl])
    mon_g = {g for g in set(grp[in_ho]) if h(g) < MONITOR_FRAC}
    split3 = np.where(np.isin(grp, list(mon_g)) & in_ho, "MONITOR", "FITMINE")
    isM = (split3 == "FITMINE") & in_ho
    res["splits"] = {
        "rule": "stable-hash sha256(salt+outlet|day) < .20 AND row in dense-held-out",
        "n_MONITOR": int((split3 == "MONITOR").sum()),
        "n_FITMINE": int((split3 == "FITMINE").sum()),
        "n_mining_slice_M": int(isM.sum()),
        "MONITOR_groups": len(mon_g),
        "MONITOR_pos_rate": float(y_all[split3 == "MONITOR"].mean()),
        "FITMINE_pos_rate": float(y_all[split3 == "FITMINE"].mean()),
        "group_disjoint": bool(pd.Series(split3).groupby(grp).nunique().max() == 1),
    }
    assert res["splits"]["group_disjoint"], "a group spans MONITOR and FIT+MINE"
    print("[splits]", json.dumps(res["splits"]), flush=True)

    # ---- 3. item view (byte-identical across arms on this cell)
    iv = {"identical_string": True,
          "note": "V, A and dense all read 'HEADLINE: ' + anchor headline, "
                  "byte-identical, so no view asymmetry is possible (the SO lesson)."}
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        n = [len(tk.encode(t, add_special_tokens=True))
             for t in byid.loc[idl, "text"].head(4000)]
        iv["dense_tokens"] = {"max": int(max(n)), "p99": int(np.percentile(n, 99)),
                              "frac_over_1024": float(np.mean(np.array(n) > 1024))}
    except Exception as e:
        iv["dense_tokens"] = f"SKIPPED ({type(e).__name__})"
    res["item_view"] = iv

    # ---- 4. anchor + epsilon (epsilon FIRST in the report, per the caution)
    X = np.column_stack([V, A])
    fm, mo = split3 == "FITMINE", split3 == "MONITOR"
    info, mon_lin, mon_nl = fit_va(X[fm], X[mo], y_all[fm], grp[fm])
    y_m, g_m = y_all[mo], grp[mo]
    hopos = {r: i for i, r in enumerate(ids)}
    midx = np.array([hopos[i] for i in np.array(idl)[mo]])
    t_mon = {str(s): float(roc_auc_score(y_m, dense_ho[s][midx])) for s in SEEDS}
    va_nl = float(np.mean([roc_auc_score(y_m, mon_nl[s]) for s in mon_nl]))

    res["epsilon_resolvability"] = paired_seed_noise(y_m, g_m, mon_nl)
    print("[eps power]", json.dumps({k: v for k, v in
          res["epsilon_resolvability"].items() if k != "interpretation"}), flush=True)

    res["round0"] = {
        "n_fitmine": int(fm.sum()), "n_monitor": int(mo.sum()),
        "VA_lin_MONITOR": float(roc_auc_score(y_m, mon_lin)),
        "VA_nl_MONITOR_per_seed": {str(s): float(roc_auc_score(y_m, mon_nl[s]))
                                   for s in mon_nl},
        "VA_nl_MONITOR": va_nl,
        "T_MONITOR_per_seed": t_mon,
        "T_MONITOR": float(np.mean(list(t_mon.values()))),
        "T_MONITOR_seed_spread": float(max(t_mon.values()) - min(t_mon.values())),
        "VA_lin_oof_within_FITMINE": info["lin_oof_fitmine"],
        "gbm_oof_fitmine_per_seed": info["gbm_oof_fitmine_per_seed"],
    }
    res["round0"]["Delta_0_MONITOR"] = (res["round0"]["T_MONITOR"] - va_nl)

    dmean = np.mean([dense_ho[s][midx] for s in SEEDS], axis=0)
    vamean = np.mean([mon_nl[s] for s in mon_nl], axis=0)
    res["residual_resolvability"] = {
        "MONITOR_T_minus_VAnl": group_boot_ci(y_m, g_m, dmean, vamean),
        "layer1_same_rows_eval": led["journalism_tweets_extras"]["same_rows"]["eval"][
            "Delta_beyond"],
        "layer1_same_rows_test": led["journalism_tweets_extras"]["same_rows"]["test"][
            "Delta_beyond"],
    }
    res["swap_baseline"] = swap_pair(y_m, dmean, vamean)

    d0 = res["round0"]["Delta_0_MONITOR"]
    eps_ok = res["epsilon_resolvability"]["resolvable"]
    spans0 = res["residual_resolvability"]["MONITOR_T_minus_VAnl"]["spans_zero"]
    if d0 <= GATE:
        verdict = "STOP AT ROUND 0 (terminal, gate)"
    elif not eps_ok or spans0:
        verdict = "STOP AT ROUND 0 (terminal, resolution-bound)"
    else:
        verdict = "GATE PASS + RESOLVABLE -> report to coordinator before any rounds"
    res["gate"] = {"threshold": GATE, "Delta_0_MONITOR": d0,
                   "epsilon_resolvable": eps_ok, "residual_spans_zero": spans0,
                   "verdict": verdict,
                   "rule": "APPENDIX cell: rounds require an explicit go even on a "
                           "gate pass; a resolution-bound stop is reported as "
                           "'closed at this cell's resolution', never 'saturated'."}
    (HERE / "round0_results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in
                      ("round0", "epsilon_resolvability", "residual_resolvability",
                       "swap_baseline", "gate")}, indent=2, default=str)[:2600])


if __name__ == "__main__":
    main()
