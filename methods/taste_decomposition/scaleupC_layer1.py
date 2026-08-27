#!/usr/bin/env python3
"""Full V / A / VA_lin / VA_nl / T ledger for the scale-up-wave-C instrument
builds (task D7): reddit-jokes community, math.SE (multi-y), AoPS curation,
homepage curation.

These are FIRST-FIT cells: no earlier V+A stack of the same construction exists,
so the linear leg IS the first fit and there is no reproduction gate to pass
(the press-verdict precedent, design note S4b wave 3). Everything else follows
the frozen Layer-1 protocol (design note S1 + the three FREEZE CHANGES):

  * linear  = family1 pipeline, imported from datasets/va_gemma_banks/
    readout_va_gemma.py: SimpleImputer(median, add_indicator) + StandardScaler +
    LogisticRegression(C=1, liblinear, max_iter 2000, random_state 20260728),
    GroupKFold(5) on the cell's own grouping unit.
  * nonlinear (VA_nl / V_nl) = HistGradientBoostingClassifier, frozen grid
    (max_leaf_nodes in {15,31}, lr .06, max_iter 400 + early stopping), grid
    chosen by inner GroupKFold(3) INSIDE each outer train fold only, same outer
    folds as the linear leg, per-fold imputation identical to the linear leg.
  * FREEZE CHANGE 1: VA_nl / V_nl := mean over seeds {0,1,2}, spread reported.
  * FREEZE CHANGE 2: T is a SAME-ROWS readout -- the dense arm is trained on the
    identical population, so its eval rows are by construction a subset of the
    A/V-scored rows.
  * FREEZE CHANGE 3: Delta_interact CI uses a GROUP-LEVEL bootstrap (resample
    groups, not rows), mandatory for coarse-grouped cells.

CPU only. No GPU, no new judging.

  python methods/taste_decomposition/scaleupC_layer1.py --cell jokes_community
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer1_gemma_cells as L  # noqa: E402  (same directory)

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SCALEUPC_OUT = REPO / "outputs/va_gemma_banks_scaleupC"


# ------------------------------------------------------------- loading ------
def load_scaleupC_bank(name, out=SCALEUPC_OUT):
    """Same layout as readout_va_gemma.load_bank but rooted at the scale-up-C
    output dir (so the wave never touches the frozen va_gemma_banks matrices)."""
    meta = json.loads((out / f"{name}_meta.json").read_text())
    Xs, Vs, ids, grp, shard_of = [], [], [], [], []
    si = 0
    while (out / f"{name}_shard{si}.npz").exists():
        z = np.load(out / f"{name}_shard{si}.npz", allow_pickle=True)
        Xs.append(z["X"]); Vs.append(z["V"])
        ids += z["ids"].tolist(); grp += z["groups"].tolist()
        shard_of += [si] * len(z["ids"])
        si += 1
    if not Xs:
        raise FileNotFoundError(f"no shards for {name} under {out}")
    X = np.vstack(Xs); V = np.vstack(Vs)
    order = {d: i for i, d in enumerate(ids)}
    idx = np.array([order[d] for d in meta["item_ids"]])
    return (meta, X[idx], V[idx], np.array(meta["item_groups"], dtype=object),
            np.array(shard_of)[idx], np.array(meta["item_ids"], dtype=object))


def dense_T(dense_dir: Path, split="eval"):
    """Seed-mean dense AUC from score_eval_dense_v4.py's results json."""
    p = dense_dir / "eval_pass_results.json"
    if not p.exists():
        return None, {}
    d = json.loads(p.read_text())
    runs = d.get("runs", d)
    aucs = {}
    for k, v in runs.items():
        if isinstance(v, dict) and f"{split}_auc" in v:
            aucs[k] = float(v[f"{split}_auc"])
    if not aucs:
        return None, d
    return float(np.mean(list(aucs.values()))), {"per_seed": aucs, "raw": d}


# --------------------------------------------------- group-level bootstrap --
def group_bootstrap_delta(y, groups, lin_pred, nl_pred, n_boot=2000, seed=12345):
    """FREEZE CHANGE 3: resample GROUPS with replacement (not rows)."""
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
            "note": "GROUP-level paired bootstrap (resamples grouping units, "
                    "FREEZE CHANGE 3), linear vs GBM seed 0."}


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


# ------------------------------------------------------------ cell registry -
def cell_jokes_community():
    meta, A, V, groups, shard, ids = load_scaleupC_bank("jokes_community")
    y = np.array(meta["ys"]["crowd_top_quartile"])
    dense = REPO / "datasets/humor/reddit_jokes/dense_standard"
    T, Tinfo = dense_T(dense)
    return dict(
        title="reddit-jokes community (r/Jokes crowd upvote quartile split)",
        A=A, V=V, y=y, groups=groups, ids=ids, meta=meta, shard_of=shard,
        group_column="LDA topic (50)", T=T, T_info=Tinfo,
        matrix="outputs/va_gemma_banks_scaleupC/jokes_community_shard*.npz",
        dense_dir=str(dense.relative_to(REPO)),
        prior_published={"V": 0.574, "VA": 0.564,
                         "note": "May-2026 floor-harness numbers (dagger) -- NOT "
                                 "comparable to a mature Gemma bank; recorded for "
                                 "context only, never used as a gate."},
    )


def _mathse(y_key, dense_sub, title):
    def load():
        meta, A, V, groups, shard, ids = load_scaleupC_bank("mathse_multiy")
        y = np.array(meta["ys"][y_key], dtype=float)
        keep = np.isfinite(y)          # vote_score is undefined on median-tied rows
        dense = REPO / "datasets/math-stackexchange/v2_va" / dense_sub
        T, Tinfo = dense_T(dense)
        return dict(title=title, A=A[keep], V=V[keep], y=y[keep].astype(int),
                    groups=groups[keep], ids=ids[keep], meta=meta,
                    shard_of=shard[keep], group_column="question_id", T=T, T_info=Tinfo,
                    matrix="outputs/va_gemma_banks_scaleupC/mathse_multiy_shard*.npz "
                           f"(y = {y_key}; the two y's share ONE scored matrix and are "
                           "never merged)",
                    dense_dir=str(dense),
                    prior_published={"V": 0.565, "VA": 0.673, "T": 0.794,
                                     "note": "the OLD BINARIZED math.SE cell "
                                             "(accepted AND score>=3 vs score<=0, "
                                             "Qwen-scored A). Different population, "
                                             "different y, different judge -- context "
                                             "only, never a gate and never differenced "
                                             "against these numbers."})
    return load


def cell_aops_curation():
    """T is REUSED, not retrained: the AoPS cell already had a grouped
    Llama-3.1-8B dense arm with row-level predictions, and this cell's
    population IS that arm's held-out rows, so T is same-rows by construction."""
    meta, A, V, groups, shard, ids = load_scaleupC_bank("aops_curation")
    y = np.array(meta["ys"]["same_approach"])
    man = json.loads((REPO / "datasets/math/aops/va/population_manifest.json").read_text())
    T = man["T_dense_pooled"]
    Tinfo = {"reused": True, "by_split": man["T_dense_by_split"],
             "provenance": man["T_provenance"], "source": man["reuse"]}
    return dict(
        title="AoPS curation (same-approach-as-editorial)",
        A=A, V=V, y=y, groups=groups, ids=ids, meta=meta, shard_of=shard,
        group_column="problem", T=T, T_info=Tinfo,
        matrix="outputs/va_gemma_banks_scaleupC/aops_curation_shard*.npz",
        dense_dir="runs/aops_same_approach_dense_llama8b (REUSED, not retrained)",
        prior_published={"V": 0.706, "T": 0.769,
                         "note": "published V .706 used the lexicon set PLUS a learned "
                                 "out-of-fold editorial-register score (p_ed) and two "
                                 "LLM-judge fields (is_correct, has_answer); this build's "
                                 "V is deterministic-surface only, so the two V numbers "
                                 "are not the same instrument."})


def cell_homepage_curation():
    meta, A, V, groups, shard, ids = load_scaleupC_bank("homepage_curation")
    y = np.array(meta["ys"]["top_half_placement"])
    dense = REPO / "datasets/news-homepages/va/dense_standard"
    T, Tinfo = dense_T(dense)
    id_pos = {d: i for i, d in enumerate(meta["item_ids"])}
    snap = np.array(meta["meta"]["snapshot_ids"], dtype=object)
    order = np.array([id_pos[d] for d in meta["item_ids"]])
    return dict(
        title="journalism homepage curation (spatial placement, OUTLET-HELD-OUT)",
        A=A, V=V, y=y, groups=groups, ids=ids, meta=meta, shard_of=shard,
        group_column="outlet (OUTLET-HELD-OUT)", T=T, T_info=Tinfo,
        secondary_groups=snap[order], secondary_group_column="snapshot_id",
        matrix="outputs/va_gemma_banks_scaleupC/homepage_curation_shard*.npz",
        dense_dir=str(dense),
        weak_instrument_flag=meta["meta"].get("weak_instrument_flag"),
        prior_published={"A_census": 0.593, "A_llm_mined": 0.531, "T": 0.824,
                         "note": "prior numbers came from a 4,400-row snapshot-grouped "
                                 "sample judged by a 70B model (a different instrument "
                                 "from the Gemma-4-31B A-bank standard) and a "
                                 "provisional snapshot-grouped dense sweep; context "
                                 "only, never a gate."})


def cell_snl_asr():
    """SNL cut-for-time verdict, ASR lane (2026-08-22). PILOT-n: 72 rows
    (36 cut + 36 within-season length-matched aired), 6 season groups —
    grouped-OOF CIs only; the eval/test bakeoff is NOT run (33/3/4-per-class
    split slices are too thin to select on). Dense T: NOT instrumentable at
    this n (declared, not attempted) — cell enters the master ladder flagged
    PILOT. Confound declared everywhere: cut = dress-rehearsal recordings."""
    meta, A, V, groups, shard, ids = load_scaleupC_bank("snl_asr")
    y = np.array(meta["ys"]["aired"])
    return dict(
        title="SNL cut-for-time verdict, ASR lane (aired=1 vs cut=0) — PILOT-n",
        A=A, V=V, y=y, groups=groups, ids=ids, meta=meta, shard_of=shard,
        group_column="season (6 groups)", T=None,
        T_info={"note": "dense T not instrumentable at PILOT-n=72: a LoRA/dense "
                        "arm has ~58 train rows under grouped folds — below every "
                        "dense arm ever certified in this ladder; declared rather "
                        "than attempted (recommendation logged in registry note "
                        "2026-08-22)"},
        matrix="outputs/va_gemma_banks_scaleupC/snl_asr_shard*.npz",
        dense_dir="NONE (PILOT-n; see T_info)",
        weak_instrument_flag="PILOT-n=72, 6 coarse season groups: all readouts "
                             "are grouped-OOF with group-bootstrap CIs; treat as "
                             "pilot row in the master ladder, never as a "
                             "certified-n cell",
        prior_published=None,
    )


def cell_rr_community():
    """U1 COMMUNITY: RoyalRoad reader engagement on fiction DESCRIPTIONS."""
    meta, A, V, groups, shard, ids = load_scaleupC_bank("rr_community")
    y = np.array(meta["ys"]["followers_above_stratum_median"])
    return dict(
        title="RoyalRoad community (followers > within-(genre,year)-stratum median; "
              "X = author blurb)",
        A=A, V=V, y=y, groups=groups, ids=ids, meta=meta, shard_of=shard,
        group_column="stratum (genre::year, 149)", T=None,
        T_info={"note": "dense T not yet trained; decision gated on this ladder "
                        "(registry 2026-08-23)"},
        matrix="outputs/va_gemma_banks_scaleupC/rr_community_shard*.npz",
        dense_dir="NONE yet",
        weak_instrument_flag="K=50 battery pos-vs-neg AUC .5076 (means ordered "
                             ".564>.552>.0005 scram; coherence 1.0): the CW craft "
                             "bank barely separates high/low-follower blurbs on "
                             "anchors — read A numbers with this flag",
        prior_published=None,
    )


def cell_rr_magazine():
    """U1 CURATED: Community Magazine contest picks — PILOT (26 positives).
    Unlabeled editions (2022-01/06, no winner announcement) are EXCLUDED:
    their y=0 rows would be silent false negatives."""
    meta, A, V, groups, shard, ids = load_scaleupC_bank("rr_magazine")
    y = np.array(meta["ys"]["magazine_winner"])
    keep = ~np.isin(np.asarray(groups, dtype=object), ["2022-01", "2022-06"])
    return dict(
        title="RoyalRoad Community Magazine curated (ranked judge picks, 3-5 per "
              "edition) — PILOT: 26 positives",
        A=A[keep], V=V[keep], y=y[keep], groups=groups[keep], ids=ids[keep],
        meta=meta, shard_of=shard[keep], group_column="edition (8 labeled)",
        T=None,
        T_info={"note": "dense T NOT RECOMMENDED at 26 positives (grouped folds "
                        "put ~20 pos in train) — below every certified dense arm; "
                        "declared rather than attempted (SNL-ASR precedent)"},
        matrix="outputs/va_gemma_banks_scaleupC/rr_magazine_shard*.npz",
        dense_dir="NONE (PILOT; see T_info)",
        weak_instrument_flag="PILOT: 26 positives / 2,012 rows, 8 coarse edition "
                             "groups; every readout grouped-OOF with group-"
                             "bootstrap CI; pre-kill checklist applies; never a "
                             "certified-n cell",
        prior_published=None,
    )


CELLS = {"jokes_community": cell_jokes_community,
         "rr_community": cell_rr_community,
         "rr_magazine": cell_rr_magazine,
         "snl_asr": cell_snl_asr,
         "aops_curation": cell_aops_curation,
         "homepage_curation": cell_homepage_curation,
         "mathse_accepted_verdict": _mathse(
             "accepted_verdict", "dense_standard_mathse_accepted_verdict",
             "math.SE accepted verdict (asker's accept, un-binarized rebuild)"),
         "mathse_vote_score": _mathse(
             "vote_score", "dense_standard_mathse_vote_score",
             "math.SE vote score (within-question median split, un-binarized rebuild)")}


# ------------------------------------------------------------------- main ---
def run_cell(slug):
    t0 = time.time()
    d = CELLS[slug]()
    A, V, y, groups = d["A"], d["V"], d["y"], d["groups"]
    VA = np.column_stack([V, A])
    mats = {"V": V, "A": A, "VA": VA}
    n = len(y)
    folds = L.outer_folds(n, groups, n_splits=5)
    names = {"V": list(d["meta"]["v_names"]), "A": list(d["meta"]["a_names"])}
    names["VA"] = names["V"] + names["A"]
    print(f"=== {slug} === n={n} pos={y.mean():.4f} groups={len(np.unique(groups))} "
          f"V={V.shape[1]}c A={A.shape[1]}c")

    res = {"cell": slug, "title": d["title"], "n": int(n), "pos_rate": float(y.mean()),
           "n_groups": int(len(np.unique(groups))), "group_column": d["group_column"],
           "matrix": d["matrix"], "dense_dir": d["dense_dir"],
           "n_features": {k: int(v.shape[1]) for k, v in mats.items()},
           "T_dense": d["T"], "T_info": d["T_info"],
           "sklearn_version": sklearn.__version__,
           "first_fit": True,
           "prior_published": d.get("prior_published"),
           "linear": {}, "nonlinear": {}}

    lin_oof = {}
    for k in ["V", "A", "VA"]:
        auc, oof = L.linear_oof_family1(mats[k], y, groups, folds)
        res["linear"][k] = auc
        lin_oof[k] = oof
        print(f"  linear  {k:2s}: {auc:.4f}")

    nl_oof = {}
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
    res["seed_spread"] = {"VA": va_seeds, "V": v_seeds}
    res["seed_spread_range"] = {"VA": float(max(va_seeds) - min(va_seeds)),
                                "V": float(max(v_seeds) - min(v_seeds))}
    VA_nl, V_nl = float(np.mean(va_seeds)), float(np.mean(v_seeds))
    ledger = {"V_lin": res["linear"]["V"], "V_nl_mean": V_nl, "V_nl_seeds": v_seeds,
              "A_lin": res["linear"]["A"], "VA_lin": res["linear"]["VA"],
              "VA_nl_mean": VA_nl, "VA_nl_seeds": va_seeds, "T": d["T"],
              "Delta_interact": VA_nl - res["linear"]["VA"],
              "V_interact": V_nl - res["linear"]["V"]}
    if d["T"] is not None:
        ledger["Delta_total"] = d["T"] - res["linear"]["VA"]
        ledger["Delta_beyond"] = d["T"] - VA_nl
    res["ledger"] = ledger
    res["overfit_gap"] = {
        k: {str(s): res["nonlinear"][k][str(s)]["train_auc_mean"]
            - res["nonlinear"][k][str(s)]["auc"] for s in L.GBM_SEEDS}
        for k in ["V", "VA"]}

    res["group_bootstrap_delta_interact"] = group_bootstrap_delta(
        y, groups, lin_oof["VA"], nl_oof[("VA", 0)])
    res["group_bootstrap_ci95"] = {
        "V_lin": group_bootstrap_auc(y, groups, lin_oof["V"]),
        "A_lin": group_bootstrap_auc(y, groups, lin_oof["A"]),
        "VA_lin": group_bootstrap_auc(y, groups, lin_oof["VA"]),
        "VA_nl_seed0": group_bootstrap_auc(y, groups, nl_oof[("VA", 0)]),
    }
    b = res["group_bootstrap_delta_interact"]
    print(f"  Delta_interact seed0 = {b['estimate']:+.4f} "
          f"95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] P(>0)={b['p_gt_0']:.2f}")

    # univariate descriptive screen (never used for selection)
    import sys
    sys.path.insert(0, str(REPO / "datasets/va_gemma_banks"))
    import readout_va_gemma as rvg
    res["univariate_A"] = rvg.univariate(A, y, names["A"])[:15]
    res["na_rate_overall"] = float(np.isnan(A).mean())
    res["collapsed_criteria"] = [r["criterion"] for r in rvg.univariate(A, y, names["A"])
                                 if r["near_constant"]]
    bat = SCALEUPC_OUT / "anchor_battery.json"
    res["anchor_battery"] = json.loads(bat.read_text()).get(slug) if bat.exists() else None
    res["anchor_reports"] = d["meta"].get("anchor_reports")
    # invalid-shard sensitivity (style_invitational precedent): a shard whose
    # 3-row blinded anchor draw never ordered pos > neg > scrambled within 4
    # independent draws is INVALID. Its rows stay in the headline readout (a
    # re-draw cannot change temperature-0 item scores) and a leave-those-shards-
    # out readout is reported beside it.
    bad = [r["shard"] for r in (d["meta"].get("anchor_reports") or []) if not r["valid"]]
    if bad and d.get("shard_of") is not None:
        sh = np.asarray(d["shard_of"])
        keep = np.flatnonzero(~np.isin(sh, bad))
        if len(keep) > 200 and len(np.unique(y[keep])) > 1:
            f3 = L.outer_folds(len(keep), groups[keep], n_splits=5)
            sens = {"dropped_shards": bad, "n": int(len(keep))}
            for k in ["V", "A", "VA"]:
                auc3, _ = L.linear_oof_family1(mats[k][keep], y[keep], groups[keep], f3)
                sens[f"{k}_lin"] = auc3
            sens["VA_nl_seed0"] = L.gbm_oof_family1(
                mats["VA"][keep], y[keep], groups[keep], f3, 0)["auc"]
            res["invalid_shard_sensitivity"] = sens
            print(f"  [invalid-shard sensitivity, dropped {bad}] n={sens['n']} "
                  f"V {sens['V_lin']:.4f} A {sens['A_lin']:.4f} VA {sens['VA_lin']:.4f} "
                  f"VA_nl {sens['VA_nl_seed0']:.4f}")
    res["invalid_shards"] = bad

    # optional secondary grouping readout (linear only -- descriptive)
    if d.get("secondary_groups") is not None:
        g2 = np.asarray(d["secondary_groups"])
        f2 = L.outer_folds(n, g2, n_splits=5)
        sec = {}
        for k in ["V", "A", "VA"]:
            auc2, _ = L.linear_oof_family1(mats[k], y, g2, f2)
            sec[f"{k}_lin"] = auc2
        r2 = L.gbm_oof_family1(mats["VA"], y, g2, f2, 0)
        sec["VA_nl_seed0"] = r2["auc"]
        sec["group_column"] = d["secondary_group_column"]
        sec["n_groups"] = int(len(np.unique(g2)))
        res["secondary_grouping"] = sec
        print(f"  [secondary grouping {sec['group_column']}] "
              f"V {sec['V_lin']:.4f} A {sec['A_lin']:.4f} VA {sec['VA_lin']:.4f} "
              f"VA_nl {sec['VA_nl_seed0']:.4f}")

    if d.get("weak_instrument_flag"):
        res["weak_instrument_flag"] = d["weak_instrument_flag"]

    res["protocol_notes"] = [
        "FIRST-FIT cell: no prior V+A stack of this construction exists, so the "
        "linear leg is the first fit and no reproduction gate applies (press-verdict "
        "precedent, design note S4b wave 3).",
        "VA_nl / V_nl = mean over GBM seeds {0,1,2} (FREEZE CHANGE 1); read "
        "Delta_interact only against seed_spread_range.",
        "T is same-rows by construction: the dense arm trains on the identical "
        "frozen population, so its eval rows are a subset of the A/V-scored rows "
        "(FREEZE CHANGE 2 satisfied without a separate rescore).",
        "Delta_interact CI is a GROUP-level bootstrap (FREEZE CHANGE 3).",
    ]
    res["runtime_sec"] = time.time() - t0
    (RESULTS_DIR / f"{slug}_ledger.json").write_text(json.dumps(res, indent=2, default=str))
    np.save(RESULTS_DIR / f"{slug}_va_nl_oof_seed0.npy", nl_oof[("VA", 0)])
    np.save(RESULTS_DIR / f"{slug}_va_nl_oof_mean3.npy",
            np.mean([nl_oof[("VA", s)] for s in L.GBM_SEEDS], axis=0))
    print("wrote", RESULTS_DIR / f"{slug}_ledger.json")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    a = ap.parse_args()
    r = run_cell(a.cell)
    Lg = r["ledger"]
    print("\n=== LEDGER ===")
    for k in ("V_lin", "V_nl_mean", "A_lin", "VA_lin", "VA_nl_mean", "T",
              "Delta_interact", "Delta_total", "Delta_beyond"):
        if k in Lg and Lg[k] is not None:
            print(f"  {k:16s} {Lg[k]:+.4f}")


if __name__ == "__main__":
    main()
