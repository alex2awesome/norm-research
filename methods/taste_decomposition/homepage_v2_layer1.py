#!/usr/bin/env python3
"""Layer-1 V / A / VA_lin / VA_nl / T ledger for the REBUILT homepage-curation cell.

Cell:  homepage_curation_storygrouped
A:     outputs/va_gemma_banks_homepage_v2/  (rubrics_v2.jsonl, 29 criteria,
       Gemma-4-31B label-blind, K>=50 anchors + per-criterion coherence battery)
T:     datasets/news-homepages/va/dense_standard_storygrouped/  (3 seeds, verified
       in results/samerows_T_homepage_storygrouped.json)

Everything is the frozen Layer-1 protocol, machinery IMPORTED (never re-typed) from
layer1_gemma_cells.py:
  * linear    = family1 (SimpleImputer(median, add_indicator) + StandardScaler +
                LogisticRegression(C=1, liblinear, max_iter 2000, rs 20260728)),
                GroupKFold(5) on snapshot_id
  * nonlinear = HistGradientBoosting, frozen grid {15,31} leaves / lr .06 /
                max_iter 400 + early stopping, grid by inner GroupKFold(3) INSIDE
                each outer train fold, per-fold imputation identical to the linear leg
  * FREEZE CHANGE 1: VA_nl / V_nl = mean over GBM seeds {0,1,2}, spread reported
  * FREEZE CHANGE 2: T same-rows -- and here that is ENFORCED, not asserted (see below)
  * FREEZE CHANGE 3: Delta_interact CI = GROUP-level bootstrap

THREE THINGS THIS DRIVER DOES THAT THE SCALEUP-C DRIVER DID NOT
---------------------------------------------------------------
1. SAME-ROWS Delta_beyond, ENFORCED.
   samerows_T_press.json records a Delta_beyond of +.0486 RETRACTED because T was
   measured on 288 eval rows while VA_nl was pooled over 2,956. Here the OOF VA vector
   is RESTRICTED BY ROW ID to the dense arm's own held-out rows before differencing.
   Both the eval-only (n=1,313) and the eval+test union (n=2,631) versions are reported,
   and the pooled-population number is kept only as clearly-labelled context.

2. APPLICABILITY-MASK ABLATION (the press genre-detector diagnosis, run on this cell).
   notes/2026-08-10__closure_press.md 2.2: on press_verdict the applicability mask ALONE
   reached .7322 while the 40 judged levels were worth .0014 over it -- the bank was a
   genre fingerprint. The v2 homepage prompt is designed so NA means "empty input" and
   never "wrong section", which should drive the mask to degeneracy. That is a PREDICTION
   and it is measured here, in the press cell's own five-block form.

3. STORY-TYPE-STRATIFIED READOUT (the within-story-type requirement, measured).
   A pooled AUC can be earned by separating story types rather than by ranking headlines
   within one. Story type is assigned by a DETERMINISTIC, LABEL-BLIND keyword map over
   the headline (no judge, no y, so no circularity), and the A/V/VA AUCs are recomputed
   (a) pooled and (b) as a within-type average weighted by type size. Per the standing
   stratified-readout rule, the STRATIFIED number is the honest one; pooled is
   composition-diluted and is never the headline on its own.

CPU only. No GPU, no new judging.

  python3 methods/taste_decomposition/homepage_v2_layer1.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer1_gemma_cells as L  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HP2_OUT = Path(os.environ.get("VA_OUT_HP2",
                              str(REPO / "outputs/va_gemma_banks_homepage_v2")))
DENSE_DIR = REPO / "datasets/news-homepages/va/dense_standard_storygrouped"
SLUG = "homepage_curation_storygrouped"
BANK = "homepage_curation_v2"
ORDER_GATE_TOL = 1e-9
DENSE_SEEDS = ("seed42", "seed1", "seed2")


# ------------------------------------------------------------------ loading -
def load_bank(name=BANK, out=None):
    out = Path(out) if out is not None else HP2_OUT
    meta = json.loads((out / f"{name}_meta.json").read_text())
    Xs, Vs, ids, grp, shard_of = [], [], [], [], []
    si = 0
    while (out / f"{name}_shard{si}.npz").exists():
        z = np.load(out / f"{name}_shard{si}.npz", allow_pickle=True)
        Xs.append(z["X"])
        Vs.append(z["V"])
        ids += z["ids"].tolist()
        grp += z["groups"].tolist()
        shard_of += [si] * len(z["ids"])
        si += 1
    if not Xs:
        raise FileNotFoundError(f"no shards for {name} under {out}")
    X = np.vstack(Xs)
    V = np.vstack(Vs)
    order = {d: i for i, d in enumerate(ids)}
    idx = np.array([order[d] for d in meta["item_ids"]])
    return (meta, X[idx], V[idx], np.array(meta["item_groups"], dtype=object),
            np.array(shard_of)[idx], np.array(meta["item_ids"], dtype=object))


def load_dense_rows():
    """Row-level dense predictions, keyed by row_id, for every held-out row.

    ALIGNMENT GATE: score_eval_dense_v4 writes preds_{split}.csv in the split file's
    own row order but carries no id column, so the join is positional -- and is only
    admissible because the `group` and `judgement` sequences of the two files are
    asserted identical, row for row, for every seed.
    """
    out = {}
    checks = []
    for split in ("eval", "test"):
        sdf = pd.read_csv(DENSE_DIR / "split" / f"{split}.csv")
        per_seed = {}
        for seed in DENSE_SEEDS:
            p = DENSE_DIR / f"rm_out_{seed}" / f"preds_{split}.csv"
            pdf = pd.read_csv(p)
            ok_n = len(pdf) == len(sdf)
            ok_g = bool((pdf["group"].astype(str).values
                         == sdf["group"].astype(str).values).all())
            ok_y = bool((pdf["judgement"].astype(int).values
                         == sdf["judgement"].astype(int).values).all())
            checks.append({"split": split, "seed": seed, "n_match": ok_n,
                           "group_seq_match": ok_g, "y_seq_match": ok_y,
                           "n": int(len(pdf)),
                           "auc": float(roc_auc_score(pdf["judgement"].astype(int),
                                                      pdf["prob"]))})
            assert ok_n and ok_g and ok_y, f"dense preds misaligned: {split}/{seed}"
            per_seed[seed] = pdf["prob"].to_numpy(float)
        out[split] = {"row_id": sdf["row_id"].astype(str).to_numpy(),
                      "y": sdf["judgement"].astype(int).to_numpy(),
                      "per_seed": per_seed,
                      "mean3": np.mean([per_seed[s] for s in DENSE_SEEDS], axis=0)}
    return out, checks


# ------------------------------------------------- deterministic story types -
# Label-blind, no judge, no y. Coarse on purpose: it exists to STRATIFY, not to
# classify. First matching bucket wins; order is fixed here and never tuned.
STORY_TYPE_RULES = [
    ("conflict_security", r"\b(war|troops?|military|missiles?|strikes?|airstrike|"
                          r"soldier|army|navy|invasion|ceasefire|hostages?|militant|"
                          r"nato|drones?|nuclear|rebels?|insurgen\w*|ukraine|"
                          r"russia\w*|gaza|israel\w*|hamas|hezbollah|taliban|"
                          r"iran\w*|bomb\w*|attack\w*|conflict|peace deal|"
                          r"weapons?|terror\w*)\b"),
    ("crime_justice", r"\b(police|arrest\w*|charged?|murder\w*|killed|shooting|"
                      r"courts?|trial|jury|sentenc\w*|convict\w*|lawsuits?|"
                      r"prosecut\w*|jail|prison|fraud|guilty|indict\w*|"
                      r"investigat\w*|abuse|assault|theft|smuggl\w*|"
                      r"supreme court|judge|verdict|sheriff|fbi)\b"),
    ("politics_govt", r"\b(president\w*|minister|senat\w*|congress\w*|parliament\w*|"
                      r"election\w*|voters?|voted?|votes|campaign|governor|"
                      r"white house|policy|bill|laws?|tariffs?|sanction\w*|treaty|"
                      r"cabinet|democrat\w*|republican\w*|labour|tory|governmen\w*|"
                      r"trump|biden|harris|obama|administration|immigration|"
                      r"migrants?|deport\w*|border|ice\b|federal|gop|ballot|"
                      r"redistrict\w*|diplomat\w*|ambassador|summit|un\b|eu\b|"
                      r"state department|shutdown|impeach\w*|secretary)\b"),
    ("business_econ", r"\b(markets?|stocks?|shares?|inflation|economy|economic|jobs?|"
                      r"unemploy\w*|banks?|profits?|revenue|earnings|merger|ipo|"
                      r"prices?|wages?|budget|deficit|trade|compan\w*|ceo|startup|"
                      r"airlines?|tech|meta|apple|google|amazon|tesla|openai|"
                      r"layoffs?|investors?|billion|million|rebate|oil|"
                      r"housing|mortgage|crypto|bitcoin)\b"),
    ("health_science", r"\b(health\w*|hospitals?|patients?|doctors?|virus|covid|"
                       r"vaccine|disease|cancer|drugs?|stud(?:y|ies)|research\w*|"
                       r"scientists?|climate|nasa|space|ai\b|technolog\w*|"
                       r"surgery|pregnan\w*|birth|mental|sleep|brain|therapy|"
                       r"medical|medicine|obesity|diet)\b"),
    ("disaster_weather", r"\b(storm|hurricane|floods?|earthquake|wildfires?|fire|"
                         r"tornado|heat ?wave|drought|evacuat\w*|tsunami|landslide|"
                         r"crash\w*|derail\w*|collision|explosion|rescue|"
                         r"survivors?|weather|snow|blizzard)\b"),
    ("sport", r"\b(match|games?|seasons?|leagues?|cup|finals?|coach|players?|club|"
              r"goals?|scored?|championship|olympics?|nfl|nba|mlb|nhl|"
              r"premier league|f1|tennis|cricket|world cup|golf|boxing|boxer|"
              r"wimbledon|marathon|rugby|athletics|tournament|medal|fans?|"
              r"stadium|transfer|manager united|formula one)\b"),
    ("culture_celebrity", r"\b(films?|movies?|tv|series|albums?|songs?|music|actors?|"
                          r"actress|singer|stars?|celebrit\w*|awards?|oscars?|"
                          r"grammy|netflix|books?|artists?|fashion|royals?|"
                          r"museum|theatre|theater|festival|streaming|podcast|"
                          r"comedian|died|dies|obituary)\b"),
    ("lifestyle_service", r"\b(how to|best|tips|guide|recipes?|review|travel|quiz|"
                          r"puzzle|wirecutter|shopping|deals?|gift|holiday|"
                          r"what to|things to|your\b|you\b)\b"),
]
_ST_COMPILED = [(n, re.compile(p, re.I)) for n, p in STORY_TYPE_RULES]


def story_type(headline: str) -> str:
    h = headline or ""
    for name, rx in _ST_COMPILED:
        if rx.search(h):
            return name
    return "other"


def headline_of_text(t):
    t = str(t or "")
    return t.split("\n\nCONTEXT: ", 1)[0].removeprefix("HEADLINE: ").strip()


# ------------------------------------------------------ group-level bootstrap
def group_bootstrap_delta(y, groups, lin_pred, nl_pred, n_boot=2000, seed=12345):
    point = float(roc_auc_score(y, nl_pred) - roc_auc_score(y, lin_pred))
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    deltas, tries = [], 0
    while len(deltas) < n_boot and tries < n_boot * 4:
        tries += 1
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        ys = y[idx]
        if ys.min() == ys.max():
            continue
        deltas.append(float(roc_auc_score(ys, nl_pred[idx])
                            - roc_auc_score(ys, lin_pred[idx])))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"estimate": point, "ci95": [float(lo), float(hi)],
            "p_gt_0": float((deltas > 0).mean()), "n_boot_used": int(len(deltas)),
            "note": "GROUP-level paired bootstrap over snapshots (FREEZE CHANGE 3), "
                    "linear vs GBM seed 0."}


def group_bootstrap_auc(y, groups, pred, n_boot=2000, seed=999):
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    vals, tries = [], 0
    while len(vals) < n_boot and tries < n_boot * 4:
        tries += 1
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        ys = y[idx]
        if ys.min() == ys.max():
            continue
        vals.append(float(roc_auc_score(ys, pred[idx])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


# ------------------------------------------------------ stratified readouts --
def stratified_auc(y, pred, strata, min_n=60):
    """Within-stratum AUC, size-weighted. The composition-free readout."""
    rows = []
    for s in sorted(set(strata.tolist())):
        m = strata == s
        if m.sum() < min_n or len(set(y[m].tolist())) < 2:
            rows.append({"stratum": s, "n": int(m.sum()), "auc": None,
                         "pos_rate": float(y[m].mean()) if m.sum() else None})
            continue
        rows.append({"stratum": s, "n": int(m.sum()),
                     "auc": float(roc_auc_score(y[m], pred[m])),
                     "pos_rate": float(y[m].mean())})
    used = [r for r in rows if r["auc"] is not None]
    tot = sum(r["n"] for r in used)
    wavg = (float(sum(r["auc"] * r["n"] for r in used) / tot) if tot else float("nan"))
    return {"pooled": float(roc_auc_score(y, pred)),
            "stratified_weighted": wavg,
            "n_strata_used": len(used), "n_rows_used": int(tot),
            "per_stratum": rows}


# ------------------------------------------------------------------- main ---
def run(args):
    t0 = time.time()
    meta, A, V, groups, shard_of, ids = load_bank()
    y = np.array(meta["ys"]["top_half_placement"], dtype=int)
    n = len(y)
    VA = np.column_stack([V, A])
    mats = {"V": V, "A": A, "VA": VA}
    folds = L.outer_folds(n, groups, n_splits=5)
    names = {"V": list(meta["v_names"]), "A": list(meta["a_names"])}
    names["VA"] = names["V"] + names["A"]

    print(f"=== {SLUG} === n={n} pos={y.mean():.4f} snapshots={len(np.unique(groups))} "
          f"V={V.shape[1]}c A={A.shape[1]}c NA={np.isnan(A).mean():.4f}")

    res = {"cell": SLUG,
           "title": "journalism homepage curation (spatial placement) -- STORY-GROUPED, "
                    "REBUILT A bank (rubrics_v2, 29 criteria)",
           "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n": int(n), "pos_rate": float(y.mean()),
           "n_groups": int(len(np.unique(groups))),
           "group_column": "snapshot_id (STORY-GROUPED)",
           "matrix": "outputs/va_gemma_banks_homepage_v2/homepage_curation_v2_shard*.npz",
           "bank": "datasets/news-homepages/va/rubrics_v2.jsonl",
           "dense_dir": str(DENSE_DIR.relative_to(REPO)),
           "n_features": {k: int(v.shape[1]) for k, v in mats.items()},
           "sklearn_version": sklearn.__version__,
           "first_fit": True,
           "linear": {}, "nonlinear": {}}

    # ---- linear + nonlinear legs -------------------------------------------
    lin_oof, nl_oof = {}, {}
    for k in ["V", "A", "VA"]:
        auc, oof = L.linear_oof_family1(mats[k], y, groups, folds)
        res["linear"][k] = auc
        lin_oof[k] = oof
        print(f"  linear  {k:2s}: {auc:.4f}")
    for k in ["V", "VA"]:
        res["nonlinear"][k] = {}
        for seed in L.GBM_SEEDS:
            r = L.gbm_oof_family1(mats[k], y, groups, folds, seed)
            nl_oof[(k, seed)] = r.pop("oof")
            res["nonlinear"][k][str(seed)] = r
            print(f"  gbm {k:2s} seed {seed}: {r['auc']:.4f} "
                  f"(train {r['train_auc_mean']:.4f}, picks {r['picks']})")

    va_seeds = [res["nonlinear"]["VA"][str(s)]["auc"] for s in L.GBM_SEEDS]
    v_seeds = [res["nonlinear"]["V"][str(s)]["auc"] for s in L.GBM_SEEDS]
    VA_nl, V_nl = float(np.mean(va_seeds)), float(np.mean(v_seeds))
    va_nl_mean_oof = np.mean([nl_oof[("VA", s)] for s in L.GBM_SEEDS], axis=0)
    res["seed_spread"] = {"VA": va_seeds, "V": v_seeds}
    res["seed_spread_range"] = {"VA": float(max(va_seeds) - min(va_seeds)),
                                "V": float(max(v_seeds) - min(v_seeds))}
    res["overfit_gap"] = {
        k: {str(s): res["nonlinear"][k][str(s)]["train_auc_mean"]
            - res["nonlinear"][k][str(s)]["auc"] for s in L.GBM_SEEDS}
        for k in ["V", "VA"]}

    # ---- T, and the SAME-ROWS Delta ----------------------------------------
    dense, align_checks = load_dense_rows()
    res["dense_alignment_gate"] = {
        "checks": align_checks,
        "PASS": bool(all(c["n_match"] and c["group_seq_match"] and c["y_seq_match"]
                         for c in align_checks)),
        "note": "preds_{split}.csv carries no id column; the positional join to "
                "split/{split}.csv is admissible only because group and judgement "
                "sequences match row-for-row in every seed. Asserted, not assumed."}
    T_eval = float(np.mean([roc_auc_score(dense["eval"]["y"],
                                          dense["eval"]["per_seed"][s])
                            for s in DENSE_SEEDS]))
    T_test = float(np.mean([roc_auc_score(dense["test"]["y"],
                                          dense["test"]["per_seed"][s])
                            for s in DENSE_SEEDS]))
    res["T_dense"] = round(T_eval, 4)
    res["T_info"] = {
        "T": T_eval, "T_test": T_test,
        "per_seed_eval": {s: float(roc_auc_score(dense["eval"]["y"],
                                                 dense["eval"]["per_seed"][s]))
                          for s in DENSE_SEEDS},
        "per_seed_test": {s: float(roc_auc_score(dense["test"]["y"],
                                                 dense["test"]["per_seed"][s]))
                          for s in DENSE_SEEDS},
        "recorded_in": "methods/taste_decomposition/results/"
                       "samerows_T_homepage_storygrouped.json",
        "historic_do_not_conflate": {
            "T_storygrouped_historic": 0.824,
            "why": "a provisional snapshot-grouped sweep on an older 4,400-row split "
                   "with a 70B-judged A side. Real, but a different split over a "
                   "different row set; never averaged or differenced with this T."},
        "outlet_held_out_RETIRED": 0.4322}

    id_pos = {d: i for i, d in enumerate(ids)}
    same_rows = {}
    for tag, splits in (("eval", ("eval",)), ("eval_plus_test", ("eval", "test"))):
        rid = np.concatenate([dense[s]["row_id"] for s in splits])
        yy = np.concatenate([dense[s]["y"] for s in splits])
        dd = np.concatenate([dense[s]["mean3"] for s in splits])
        miss = [r for r in rid if r not in id_pos]
        pos = np.array([id_pos[r] for r in rid if r in id_pos])
        keep = np.array([r in id_pos for r in rid])
        blk = {"n_dense_rows": int(len(rid)), "n_missing_from_A_matrix": len(miss),
               "n": int(keep.sum()), "pos_rate": float(yy[keep].mean()),
               "y_identical_between_dense_and_bank": bool(
                   (yy[keep] == y[pos]).all()),
               "T": float(np.mean([roc_auc_score(
                   np.concatenate([dense[s]["y"] for s in splits]),
                   np.concatenate([dense[s]["per_seed"][sd] for s in splits]))
                   for sd in DENSE_SEEDS])),
               "V_lin": float(roc_auc_score(yy[keep], lin_oof["V"][pos])),
               "A_lin": float(roc_auc_score(yy[keep], lin_oof["A"][pos])),
               "VA_lin": float(roc_auc_score(yy[keep], lin_oof["VA"][pos])),
               "V_nl": float(np.mean([roc_auc_score(yy[keep], nl_oof[("V", s)][pos])
                                      for s in L.GBM_SEEDS])),
               "VA_nl": float(roc_auc_score(yy[keep], va_nl_mean_oof[pos])),
               "VA_nl_per_seed": [float(roc_auc_score(yy[keep], nl_oof[("VA", s)][pos]))
                                  for s in L.GBM_SEEDS]}
        blk["Delta_total"] = blk["T"] - blk["VA_lin"]
        blk["Delta_beyond"] = blk["T"] - blk["VA_nl"]
        same_rows[tag] = blk
        print(f"  [same-rows {tag}] n={blk['n']} T={blk['T']:.4f} "
              f"VA_nl={blk['VA_nl']:.4f} Delta_beyond={blk['Delta_beyond']:+.4f}")
    res["same_rows"] = same_rows

    ledger = {"V_lin": res["linear"]["V"], "V_nl_mean": V_nl, "V_nl_seeds": v_seeds,
              "A_lin": res["linear"]["A"], "VA_lin": res["linear"]["VA"],
              "VA_nl_mean": VA_nl, "VA_nl_seeds": va_seeds,
              "T": same_rows["eval"]["T"],
              "Delta_interact": VA_nl - res["linear"]["VA"],
              "V_interact": V_nl - res["linear"]["V"],
              "SAME_ROWS_eval": {
                  "n": same_rows["eval"]["n"],
                  "T": same_rows["eval"]["T"],
                  "VA_lin": same_rows["eval"]["VA_lin"],
                  "VA_nl": same_rows["eval"]["VA_nl"],
                  "Delta_total": same_rows["eval"]["Delta_total"],
                  "Delta_beyond": same_rows["eval"]["Delta_beyond"]},
              "SAME_ROWS_eval_plus_test": {
                  "n": same_rows["eval_plus_test"]["n"],
                  "T": same_rows["eval_plus_test"]["T"],
                  "VA_lin": same_rows["eval_plus_test"]["VA_lin"],
                  "VA_nl": same_rows["eval_plus_test"]["VA_nl"],
                  "Delta_total": same_rows["eval_plus_test"]["Delta_total"],
                  "Delta_beyond": same_rows["eval_plus_test"]["Delta_beyond"]},
              "POOLED_CONTEXT_ONLY": {
                  "VA_nl_pooled_all_12998": VA_nl,
                  "Delta_beyond_if_pooled": same_rows["eval"]["T"] - VA_nl,
                  "warning": "NOT a same-rows quantity. The press cell retracted a "
                             "+.0486 computed exactly this way. Context only."}}
    res["ledger"] = ledger

    res["group_bootstrap_delta_interact"] = group_bootstrap_delta(
        y, groups, lin_oof["VA"], nl_oof[("VA", 0)])
    res["group_bootstrap_ci95"] = {
        "V_lin": group_bootstrap_auc(y, groups, lin_oof["V"]),
        "A_lin": group_bootstrap_auc(y, groups, lin_oof["A"]),
        "VA_lin": group_bootstrap_auc(y, groups, lin_oof["VA"]),
        "VA_nl_seed0": group_bootstrap_auc(y, groups, nl_oof[("VA", 0)])}

    # ---- APPLICABILITY-MASK ABLATION (press form) --------------------------
    mask = np.isfinite(A).astype(float)          # 1 = judged, 0 = NA
    var_mask = mask[:, mask.std(0) > 0]
    levels = np.where(np.isfinite(A), A, np.nan)
    levels_imp = np.where(np.isfinite(A), A, np.nanmedian(
        np.where(np.isfinite(A), A, np.nan), axis=0))   # mask erased
    const05 = np.where(np.isfinite(A), A, 0.5)           # the press "layer1" form
    blocks = {
        "A_mask_only": var_mask,
        "A_levels_only_median_imputed": levels_imp,
        "A_layer1_const05": const05,
        "A_mask_plus_levels": np.column_stack([var_mask, levels_imp])
        if var_mask.shape[1] else levels_imp,
        "V_only": V,
        "V_plus_mask": np.column_stack([V, var_mask]) if var_mask.shape[1] else V,
        "V_plus_A_primary": VA,
    }
    abl = {"n_criteria": int(A.shape[1]),
           "na_rate_overall": float(np.isnan(A).mean()),
           "n_mask_columns_with_variance": int(var_mask.shape[1]),
           "per_criterion_na_rate": {nm: float(np.isnan(A[:, i]).mean())
                                     for i, nm in enumerate(names["A"])},
           "blocks": {}}
    ev_pos = np.array([id_pos[r] for r in dense["eval"]["row_id"] if r in id_pos])
    ev_y = np.array([yy for r, yy in zip(dense["eval"]["row_id"], dense["eval"]["y"])
                     if r in id_pos])
    for bn, Xb in blocks.items():
        if Xb.shape[1] == 0:
            abl["blocks"][bn] = {"n_features": 0, "auc_pooled": None,
                                 "auc_same_rows_eval": None,
                                 "note": "block is empty -- no mask column has variance"}
            continue
        auc_b, oof_b = L.linear_oof_family1(Xb, y, groups, folds)
        nlb = L.gbm_oof_family1(Xb, y, groups, folds, 0)
        abl["blocks"][bn] = {
            "n_features": int(Xb.shape[1]),
            "lin_pooled": auc_b,
            "nl_seed0_pooled": nlb["auc"],
            "lin_same_rows_eval": float(roc_auc_score(ev_y, oof_b[ev_pos])),
            "nl_same_rows_eval": float(roc_auc_score(ev_y, nlb["oof"][ev_pos]))
            if "oof" in nlb else None}
        print(f"  [ablation] {bn:32s} k={Xb.shape[1]:3d} "
              f"lin {auc_b:.4f} nl {nlb['auc']:.4f}")
    if abl["blocks"]["A_mask_only"]["n_features"]:
        m_alone = abl["blocks"]["A_mask_only"]["lin_pooled"]
        vm = abl["blocks"]["V_plus_mask"]["lin_pooled"]
        va = abl["blocks"]["V_plus_A_primary"]["lin_pooled"]
        abl["mask_alone"] = m_alone
        abl["levels_worth_over_mask"] = va - vm
        abl["press_comparison"] = {
            "press_mask_alone": 0.7322, "press_levels_worth_over_mask": 0.0014,
            "press_source": "notes/2026-08-10__closure_press.md 2.2"}
    else:
        abl["mask_alone"] = None
        abl["levels_worth_over_mask"] = None
        abl["verdict_no_mask"] = (
            "The v2 prompt reserves NA for empty input, so no applicability bit has "
            "variance and the genre-detector channel the press cell found is CLOSED BY "
            "CONSTRUCTION on this cell. The diagnosis is therefore carried by the "
            "story-type-stratified readout below, which measures the same worry "
            "(is the bank separating story types rather than ranking within them?) "
            "without relying on the missingness pattern.")
    res["applicability_mask_ablation"] = abl

    # ---- STORY-TYPE STRATIFIED READOUT -------------------------------------
    pop = pd.read_csv(REPO / "datasets/news-homepages/va/population.csv.gz")
    hmap = dict(zip(pop["row_id"].astype(str),
                    pop["text"].map(headline_of_text)))
    st = np.array([story_type(hmap.get(str(i), "")) for i in ids], dtype=object)
    strat = {"assignment": "deterministic keyword map over the HEADLINE only "
                           "(label-blind, no judge, first-match-wins); coarse by "
                           "design -- it exists to stratify, not to classify",
             "rules_order": [n for n, _ in STORY_TYPE_RULES] + ["other"],
             "counts": {s: int((st == s).sum()) for s in sorted(set(st.tolist()))},
             "pos_rate_by_type": {s: float(y[st == s].mean())
                                  for s in sorted(set(st.tolist()))},
             "readouts": {}}
    # does story type itself predict placement? if it does, pooled AUC is buyable
    st_codes = pd.get_dummies(pd.Series(st)).to_numpy(float)
    st_auc, _ = L.linear_oof_family1(st_codes, y, groups, folds)
    strat["story_type_alone_auc"] = st_auc
    for nm, pred in (("V_lin", lin_oof["V"]), ("A_lin", lin_oof["A"]),
                     ("VA_lin", lin_oof["VA"]), ("VA_nl_mean3", va_nl_mean_oof)):
        strat["readouts"][nm] = stratified_auc(y, pred, st)
        r = strat["readouts"][nm]
        print(f"  [story-type] {nm:12s} pooled {r['pooled']:.4f} -> "
              f"stratified {r['stratified_weighted']:.4f} "
              f"({r['n_strata_used']} strata, {r['n_rows_used']} rows)")
    strat["interpretation_rule"] = (
        "pooled minus stratified is the part of the readout that comes from separating "
        "story TYPES rather than ranking headlines WITHIN a type. A bank designed to "
        "rank within story-type should lose little. Per feedback_threshold_free_readouts "
        "and the mention-AUC stratified rule, the stratified number is the honest one.")
    res["story_type_stratified"] = strat

    # ---- assembled-order gate ----------------------------------------------
    headline = {"V_lin": res["linear"]["V"], "A_lin": res["linear"]["A"],
                "VA_lin": res["linear"]["VA"],
                "VA_nl_seed0": res["nonlinear"]["VA"]["0"]["auc"]}
    oof_store = {"V_lin": lin_oof["V"], "A_lin": lin_oof["A"],
                 "VA_lin": lin_oof["VA"], "VA_nl_seed0": nl_oof[("VA", 0)]}
    meta2, A2, V2, groups2, _, ids2 = load_bank()
    gate = {"matrix_A_identical": bool(np.array_equal(np.nan_to_num(A2, nan=-9e9),
                                                      np.nan_to_num(A, nan=-9e9))),
            "matrix_V_identical": bool(np.array_equal(np.nan_to_num(V2, nan=-9e9),
                                                      np.nan_to_num(V, nan=-9e9))),
            "ids_identical": bool(np.array_equal(ids2, ids)),
            "groups_identical": bool(np.array_equal(groups2, groups))}
    y2 = np.array(meta2["ys"]["top_half_placement"], dtype=int)
    rng = np.random.default_rng(20260809)
    perm = rng.permutation(len(ids2))
    by_key = {k: dict(zip(ids, v)) for k, v in oof_store.items()}
    recomputed, diffs = {}, {}
    for key, val in headline.items():
        vec = np.array([by_key[key][i] for i in ids2[perm]])
        recomputed[key] = float(roc_auc_score(y2[perm], vec))
        diffs[key] = abs(recomputed[key] - val)
    gate.update({"recomputed": recomputed, "abs_diff": diffs,
                 "max_abs_diff": float(max(diffs.values())),
                 "tolerance": ORDER_GATE_TOL})
    gate["PASS"] = bool(all(gate[k] for k in ("matrix_A_identical", "matrix_V_identical",
                                              "ids_identical", "groups_identical"))
                        and gate["max_abs_diff"] < ORDER_GATE_TOL)
    gate["note"] = ("Headline AUCs recomputed from id-keyed OOF vectors after an "
                    "independent re-assembly and a random row permutation.")
    res["assembled_order_gate"] = gate
    print(f"  [order gate] PASS={gate['PASS']} max|diff|={gate['max_abs_diff']:.2e}")

    # ---- descriptive screens ------------------------------------------------
    sys.path.insert(0, str(REPO / "datasets/va_gemma_banks"))
    import readout_va_gemma as rvg
    uni = rvg.univariate(A, y, names["A"])
    res["univariate_A"] = uni
    res["na_rate_overall"] = float(np.isnan(A).mean())
    res["collapsed_criteria"] = [r["criterion"] for r in uni if r["near_constant"]]
    for fn, key in ((HP2_OUT / "anchor_battery.json", "anchor_battery"),
                    (HP2_OUT / "anchor_battery_percriterion.json",
                     "anchor_battery_percriterion"),
                    (HP2_OUT / "distribution_check.json",
                     "judge_distribution_check")):
        res[key] = json.loads(fn.read_text()).get(BANK) if fn.exists() else None
    res["anchor_reports"] = meta.get("anchor_reports")

    bad = [r["shard"] for r in (meta.get("anchor_reports") or []) if not r["valid"]]
    res["invalid_shards"] = bad
    if bad:
        sh = np.asarray(shard_of)
        keep = np.flatnonzero(~np.isin(sh, bad))
        if len(keep) > 200 and len(np.unique(y[keep])) > 1:
            f3 = L.outer_folds(len(keep), groups[keep], n_splits=5)
            sens = {"dropped_shards": bad, "n": int(len(keep))}
            for k in ["V", "A", "VA"]:
                sens[f"{k}_lin"], _ = L.linear_oof_family1(
                    mats[k][keep], y[keep], groups[keep], f3)
            sens["VA_nl_seed0"] = L.gbm_oof_family1(
                mats["VA"][keep], y[keep], groups[keep], f3, 0)["auc"]
            res["invalid_shard_sensitivity"] = sens

    # secondary grouping: OUTLET (the design that failed, kept as a descriptive)
    out_of_item = np.array(meta["meta"]["outlet_of_item"], dtype=object)
    id_row = {d: i for i, d in enumerate(meta["item_ids"])}
    g2 = np.array([out_of_item[id_row[d]] for d in ids], dtype=object)
    f2 = L.outer_folds(n, g2, n_splits=5)
    sec = {"group_column": "outlet", "n_groups": int(len(np.unique(g2)))}
    for k in ["V", "A", "VA"]:
        sec[f"{k}_lin"], _ = L.linear_oof_family1(mats[k], y, g2, f2)
    sec["VA_nl_seed0"] = L.gbm_oof_family1(mats["VA"], y, g2, f2, 0)["auc"]
    sec["caveat"] = ("outlet-grouped CV at k=8 is the design the registry retired as "
                     "unpowered; descriptive only")
    res["secondary_grouping"] = sec

    res["weak_instrument_flag"] = meta["meta"].get("weak_instrument_flag")
    res["prior_instrument_context_only"] = {
        "census_bank_A_lin_outlet_grouped": 0.5979,
        "census_bank_VA_nl_outlet_grouped": 0.5562,
        "census_bank_coherent_vs_scrambled": 0.3869,
        "note": "the census bank is a DIFFERENT instrument (14 topic/entity criteria, "
                "NA=wrong-section prompt) on a DIFFERENT grouping (outlet-held-out). "
                "Its numbers are context only and are never differenced against these."}
    res["protocol_notes"] = [
        "FIRST-FIT cell for this bank: no prior V+A stack of this construction exists, "
        "so the linear leg is the first fit; the assembled-order gate stands in for a "
        "reproduction gate (cw_expert precedent).",
        "VA_nl / V_nl = mean over GBM seeds {0,1,2} (FREEZE CHANGE 1); read "
        "Delta_interact only against seed_spread_range.",
        "Delta_beyond is SAME-ROWS: the OOF vector is restricted BY ROW ID to the dense "
        "arm's held-out rows before differencing (FREEZE CHANGE 2, enforced not "
        "asserted). The pooled figure is carried only under POOLED_CONTEXT_ONLY.",
        "Delta_interact CI is a GROUP-level bootstrap over snapshots (FREEZE CHANGE 3).",
        "Every number carries the weak-instrument flag: y is spatial placement.",
    ]
    res["runtime_sec"] = time.time() - t0

    (RESULTS_DIR / f"{SLUG}_ledger.json").write_text(
        json.dumps(res, indent=2, default=str))
    np.savez_compressed(
        RESULTS_DIR / f"{SLUG}_oof.npz",
        ids=ids, groups=groups, y=y, story_type=st,
        secondary_groups=g2,
        V_lin=lin_oof["V"], A_lin=lin_oof["A"], VA_lin=lin_oof["VA"],
        VA_nl_seed0=nl_oof[("VA", 0)], VA_nl_seed1=nl_oof[("VA", 1)],
        VA_nl_seed2=nl_oof[("VA", 2)], VA_nl_mean3=va_nl_mean_oof,
        V_nl_seed0=nl_oof[("V", 0)], V_nl_seed1=nl_oof[("V", 1)],
        V_nl_seed2=nl_oof[("V", 2)])
    print("wrote", RESULTS_DIR / f"{SLUG}_ledger.json")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="check inputs exist and shapes line up, then exit")
    a = ap.parse_args()
    if a.dry_run:
        meta, A, V, groups, shard_of, ids = load_bank()
        y = np.array(meta["ys"]["top_half_placement"])
        print(f"bank OK: A{A.shape} V{V.shape} n={len(y)} "
              f"groups={len(set(groups.tolist()))} NA={np.isnan(A).mean():.4f}")
        d, checks = load_dense_rows()
        print(f"dense OK: eval {len(d['eval']['y'])} test {len(d['test']['y'])}; "
              f"alignment {all(c['group_seq_match'] and c['y_seq_match'] for c in checks)}")
        return
    r = run(a)
    Lg = r["ledger"]
    print("\n=== LEDGER (pooled OOF) ===")
    for k in ("V_lin", "V_nl_mean", "A_lin", "VA_lin", "VA_nl_mean", "Delta_interact"):
        print(f"  {k:16s} {Lg[k]:+.4f}")
    print("=== LEDGER (SAME-ROWS, eval) ===")
    for k, v in Lg["SAME_ROWS_eval"].items():
        print(f"  {k:16s} {v:+.4f}" if isinstance(v, float) else f"  {k:16s} {v}")


if __name__ == "__main__":
    main()
