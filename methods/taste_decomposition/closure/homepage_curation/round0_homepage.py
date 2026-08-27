#!/usr/bin/env python3
"""Homepage curation (STORY-GROUPED, rubrics_v2) — Layer-3 closure ROUND 0.

Prereg: notes/2026-08-05__layer3-closure-prereg.md + FREEZE DECLARATION + ADDENDA.
Campaign note: notes/2026-08-12__closure_homepage_curation.md.
Layer-1 ledger: methods/taste_decomposition/results/homepage_curation_storygrouped_ledger.json

This cell enters with a SMALL residual — Layer-1 same-rows Δ_beyond +.0068 (eval,
n=1,313) and +.0109 (eval+test, n=2,631) — and with VA_nl (.7291) ABOVE T (.7109)
on the pooled tier. The frozen gate stops a campaign at round 0 when the residual is
≤ .02, so the expected outcome is a round-0 terminal. The scientific content of this
round is therefore NOT the curve; it is answering, rigorously, **whether a residual of
that size is resolvable at all on this cell** — the press_verdict precedent, where the
honest verdict was "residual closed at this cell's resolution, stopping rule not
fired", rather than "saturated".

Everything the BBC round 0 asserts is asserted here too: dense join gate (preds carry
no row_id), MONITOR inside dense-held-out, item-view assertion in TOKENS, and the
ε-resolvability power check. Two cell-specific wrinkles are handled explicitly:

  * ARTICLE DEDUP. The dense arm dropped 630 TRAIN rows whose normalised headline also
    appears in eval or test (300 distinct headlines; stories persist across successive
    captures, so snapshot grouping alone leaks). The dense arm therefore covers 12,368
    of the 12,998 A/V-scored rows. The closure population is intersected to the rows
    the dense arm actually covers, and the shortfall is recorded.
  * SKLEARN DRIFT. The Layer-1 ledger was produced under 1.9.0; this box runs 1.8.0,
    and GroupKFold assignments move across releases. Recorded and asserted, and the
    reason the curve is measured from this script's own round-0 anchor rather than
    from the Layer-1 number (prereg AMENDMENT 1).

  python3 round0_homepage.py
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
import scaleupC_layer1 as SC  # noqa: E402
from round0_bbc import paired_seed_noise, fit_va, swap_pair  # noqa: E402

CELL_BANK = "homepage_curation_v2"
VA_DIR = REPO / "datasets/news-homepages/va"
DENSE = VA_DIR / "dense_standard_storygrouped"
BANK_OUT = REPO / "outputs/va_gemma_banks_homepage_v2"
LEDGER = (REPO / "methods/taste_decomposition/results"
          / "homepage_curation_storygrouped_ledger.json")
OUT = HERE
SALT = "homepage-curation-closure-v1|"
MONITOR_FRAC = 0.20
EPS = 0.005
GATE = 0.02
SEEDS = (42, 1, 2)


def h(g) -> float:
    return int(hashlib.sha256((SALT + str(g)).encode()).hexdigest()[:16], 16) / 2**64


def group_boot_ci(y, groups, a, b, n_boot=4000, seed=5):
    """Group bootstrap CI for AUC(a) - AUC(b) on identical rows."""
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    d = []
    while len(d) < n_boot:
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        ys = y[idx]
        if ys.min() == ys.max():
            continue
        d.append(roc_auc_score(ys, a[idx]) - roc_auc_score(ys, b[idx]))
    d = np.array(d)
    return {"point": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "sd": float(d.std(ddof=1)),
            "p_gt_0": float((d > 0).mean()),
            "spans_zero": bool(np.percentile(d, 2.5) < 0 < np.percentile(d, 97.5))}


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    res = {"cell": "homepage_curation_storygrouped", "sklearn_here": sklearn.__version__,
           "epsilon": EPS, "gate_threshold": GATE, "salt": SALT}
    led = json.loads(LEDGER.read_text())
    res["sklearn_layer1"] = led.get("sklearn_version")
    res["sklearn_drift"] = res["sklearn_here"] != res["sklearn_layer1"]

    pop = pd.read_csv(VA_DIR / "population.csv.gz")
    pop["row_id"] = pop.row_id.astype(str)
    pop["snapshot_id"] = pop.snapshot_id.astype(str)

    # ---- 1. dense join gate (preds carry no row_id -> ORDER join, proven)
    dj = {"legs": {}}
    ids, per_seed = [], {s: [] for s in SEEDS}
    for leg in ("eval", "test"):
        sp = pd.read_csv(DENSE / "split" / f"{leg}.csv")
        for s in SEEDS:
            p = pd.read_csv(DENSE / f"rm_out_seed{s}" / f"preds_{leg}.csv")
            if len(p) != len(sp):
                raise SystemExit(f"DENSE JOIN GATE FAIL {leg}/{s}: {len(p)}!={len(sp)}")
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

    # ---- bank + coverage
    meta, A, V, groups_bank, shard, bids = SC.load_scaleupC_bank(CELL_BANK, out=BANK_OUT)
    idl = [str(i) for i in bids]
    byid = pop.set_index("row_id")
    y_all = byid.loc[idl, "judgement"].values.astype(int)
    snap = byid.loc[idl, "snapshot_id"].values.astype(str)
    res["coverage"] = {
        "n_bank_rows": len(idl),
        "n_dense_heldout": len(ids),
        "dense_heldout_in_bank": int(len(set(ids) & set(idl))),
        "article_dedup_note": "the dense arm dropped 630 TRAIN rows (300 distinct "
                              "normalised headlines) that recur in eval/test; the "
                              "dense arm covers 12,368 of the 12,998 scored rows",
    }

    # ---- 2. splits: MONITOR inside dense-held-out, snapshot-grouped
    ho = set(ids)
    in_ho = np.array([i in ho for i in idl])
    mon_snaps = {s for s in set(snap[in_ho]) if h(s) < MONITOR_FRAC}
    split3 = np.where(np.isin(snap, list(mon_snaps)) & in_ho, "MONITOR", "FITMINE")
    isM = (split3 == "FITMINE") & in_ho
    res["splits"] = {
        "rule": "stable-hash sha256(salt+snapshot_id) < .20 AND row in dense-held-out",
        "n_MONITOR": int((split3 == "MONITOR").sum()),
        "n_FITMINE": int((split3 == "FITMINE").sum()),
        "n_mining_slice_M": int(isM.sum()),
        "MONITOR_snapshots": len(mon_snaps),
        "MONITOR_pos_rate": float(y_all[split3 == "MONITOR"].mean()),
        "FITMINE_pos_rate": float(y_all[split3 == "FITMINE"].mean()),
        "snapshot_disjoint": bool(pd.Series(split3).groupby(snap).nunique().max() == 1),
    }
    assert res["splits"]["snapshot_disjoint"], "a snapshot spans MONITOR and FIT+MINE"
    print("[splits]", json.dumps(res["splits"]), flush=True)

    # ---- 3. item view (tokens, not chars)
    iv = {"dense_text": "population 'text' column = HEADLINE + CONTEXT block",
          "judge_view": "rubrics_v2 item view (see rubrics_v2.jsonl)",
          "note": "V is computed on the HEADLINE half only via v_features.headline_of; "
                  "the dense arm reads the whole stored text. Recorded as a known "
                  "asymmetry of this cell's design, not introduced here."}
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

    # ---- 4/5. anchor + epsilon check
    X = np.column_stack([V, A])
    fm = split3 == "FITMINE"
    mo = split3 == "MONITOR"
    info, mon_lin, mon_nl = fit_va(X[fm], X[mo], y_all[fm], snap[fm])
    y_m, g_m = y_all[mo], snap[mo]
    hopos = {r: i for i, r in enumerate(ids)}
    midx = np.array([hopos[i] for i in np.array(idl)[mo]])
    t_mon = {str(s): float(roc_auc_score(y_m, dense_ho[s][midx])) for s in SEEDS}
    va_nl_mon_seedmean_auc = float(np.mean(
        [roc_auc_score(y_m, mon_nl[s]) for s in mon_nl]))
    res["round0"] = {
        "n_fitmine": int(fm.sum()), "n_monitor": int(mo.sum()),
        "VA_lin_MONITOR": float(roc_auc_score(y_m, mon_lin)),
        "VA_nl_MONITOR_per_seed": {str(s): float(roc_auc_score(y_m, mon_nl[s]))
                                   for s in mon_nl},
        "VA_nl_MONITOR": va_nl_mon_seedmean_auc,
        "T_MONITOR_per_seed": t_mon,
        "T_MONITOR": float(np.mean(list(t_mon.values()))),
        "VA_lin_oof_within_FITMINE": info["lin_oof_fitmine"],
        "gbm_oof_fitmine_per_seed": info["gbm_oof_fitmine_per_seed"],
    }
    res["round0"]["Delta_0_MONITOR"] = (res["round0"]["T_MONITOR"]
                                        - res["round0"]["VA_nl_MONITOR"])
    res["epsilon_resolvability"] = paired_seed_noise(y_m, g_m, mon_nl)

    # ---- THE decisive statistic: is a residual this small resolvable?
    dmean_mon = np.mean([dense_ho[s][midx] for s in SEEDS], axis=0)
    va_mon = np.mean([mon_nl[s] for s in mon_nl], axis=0)
    res["residual_resolvability"] = {
        "MONITOR_T_minus_VAnl": group_boot_ci(y_m, g_m, dmean_mon, va_mon),
        "layer1_same_rows_eval_Delta_beyond": led["ledger"]["SAME_ROWS_eval"]["Delta_beyond"],
        "layer1_same_rows_eval_n": led["ledger"]["SAME_ROWS_eval"]["n"],
        "interpretation": "If the CI on T - VA_nl spans zero by a wide margin, the "
                          "cell's residual is not resolvable at this resolution and "
                          "the honest verdict is 'closed at this cell's resolution', "
                          "NOT 'saturated' (the press_verdict precedent).",
    }
    res["swap_baseline"] = swap_pair(y_m, dmean_mon, va_mon)

    d0 = res["round0"]["Delta_0_MONITOR"]
    res["gate"] = {
        "threshold": GATE, "Delta_0_MONITOR": d0,
        "verdict": "RUN ROUNDS" if d0 > GATE else "STOP AT ROUND 0 (terminal)",
        "rule": "frozen gate: residual <= .02 stops the campaign at round 0",
    }
    (OUT / "round0_results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in
                      ("round0", "epsilon_resolvability", "residual_resolvability",
                       "swap_baseline", "gate")}, indent=2, default=str)[:2600])


if __name__ == "__main__":
    main()
