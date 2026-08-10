#!/usr/bin/env python3
"""V8 -- N&C CO-SIGNING cell: full V / A / VA_lin / VA_nl / T ledger.

The third y of the regulatory field (grid column VOTE/REVEALED). y and its
rationale: datasets/notice-and-comment/cosigning/build_cosign_y.py +
notes/2026-08-08__v8_cosigning_build.md.

FIRST-FIT cell: no earlier V+A stack of this construction exists, so the linear
leg IS the first fit and there is no published number to gate against (press
precedent, design note S4b wave 3). Everything else is the frozen Layer-1
protocol, reusing nc_layer1_stack.py verbatim for every estimator so the
co-signing cell is comparable to nc responded / outcome / agree:

  * matrices  : the SAME frozen 198-rubric pre-GEPA Gemma-4-31B A-bank
                (nc_scores_shard0..4.npz + nc_scores_unmatched.npz) and the
                SAME 27 V extractors (aggregate_vat_nc.v_features), on the SAME
                rows. NO new judging -- only a new y is attached.
  * linear    : StandardScaler + LogisticRegression(C=1), GroupKFold(5) docket
  * nonlinear : HistGradientBoostingClassifier, frozen grid {15,31} leaves,
                lr .06, 400 iters + early stopping, inner GroupKFold(3) grid
                pick inside train folds only
  * FREEZE CHANGE 1 : V_nl/A_nl/VA_nl = mean over seeds {0,1,2}, spread reported
  * FREEZE CHANGE 2 : T is a SAME-ROWS readout -- the dense arm is trained on
                the identical 9,520-unit population, so its eval/test rows are
                by construction a subset of the A/V-scored rows
  * FREEZE CHANGE 3 : Delta_interact CI = docket-level (group) bootstrap

Beyond Layer 1, because this corpus's docket-identity leak is severe (Layer 2
found identity-alone .86-.92 on the sibling N&C cells), the ledger carries the
Layer-2(a) grouped-transfer table beside every pooled number: within-docket
AUC (pair-weighted over the 156 mixed dockets) and docket-identity-alone.

CPU only. Usage:
  python methods/taste_decomposition/nc_cosigning_layer1.py            # primary y
  python methods/taste_decomposition/nc_cosigning_layer1.py --y nearby # sensitivity
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[2]
NC = REPO / "datasets" / "notice-and-comment" / "v4"
COS = REPO / "datasets" / "notice-and-comment" / "cosigning"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(NC))
import nc_layer1_stack as L  # noqa: E402  (frozen protocol, reused verbatim)
from aggregate_vat_nc import v_features, V_NAMES  # noqa: E402


# --------------------------------------------------------------- readouts ---
def within_group_auc(y, score, groups):
    """Pair-weighted mean of within-docket AUCs (threshold-free, per the
    stratified-readout rule). Weight = n_pos*n_neg = the number of orderable
    pairs the docket contributes, so this equals the pooled AUC restricted to
    within-docket pairs."""
    tot_w, acc, used = 0.0, 0.0, []
    for g in np.unique(groups):
        m = groups == g
        yy, ss = y[m], score[m]
        if len(np.unique(yy)) < 2:
            continue
        w = float(yy.sum() * (len(yy) - yy.sum()))
        a = roc_auc_score(yy, ss)
        acc += w * a
        tot_w += w
        used.append({"docket": str(g), "n": int(m.sum()), "n_pos": int(yy.sum()), "auc": float(a)})
    return (acc / tot_w if tot_w else float("nan")), len(used), used


def group_identity_alone_auc(y, groups, n_splits=5, seed=0):
    """How much of y is explained by docket identity ALONE. Deliberately a
    RANDOM (non-grouped) OOF split -- under the grouped folds used everywhere
    else a held-out docket is unseen and this is chance by construction, which
    is precisely why the leak needs its own non-grouped readout. Feature =
    out-of-fold smoothed docket mean of y (target encoding, m=5)."""
    y = np.asarray(y, dtype=float)
    oof = np.zeros(len(y))
    prior = y.mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(len(y)), y):
        s, c = defaultdict(float), defaultdict(int)
        for i in tr:
            s[groups[i]] += y[i]
            c[groups[i]] += 1
        for i in te:
            g = groups[i]
            oof[i] = (s[g] + 5.0 * prior) / (c[g] + 5.0)
    return float(roc_auc_score(y, oof))


def single_feature_aucs(V, y, names):
    out = {}
    for j, n in enumerate(names):
        col = V[:, j]
        if np.std(col) == 0:
            continue
        out[n] = float(roc_auc_score(y, col))
    return dict(sorted(out.items(), key=lambda kv: -abs(kv[1] - .5))[:8])


# ------------------------------------------------------------------- data ---
def load_population(y_name: str):
    units = [json.loads(l) for l in open(COS / "cosign_population.jsonl")]
    if y_name == "nearby":
        units = [u for u in units if u["y_nearby"] >= 0]
        ykey = "y_nearby"
    elif y_name == "campaign":
        ykey = "y_campaign"
    else:
        ykey = "y_cosign"

    X_m, docket_m, _, a_names = L.load_shard_scores(L.MATCHED_SHARDS)
    X_u, docket_u, _, _ = L.load_shard_scores([str(L.UNMATCHED_NPZ)])
    A_by_id = dict(X_u)
    A_by_id.update(X_m)          # matched wins on overlap, as in NCData

    keep = [u for u in units if u["doc_id"] in A_by_id]
    dropped = len(units) - len(keep)
    y = np.array([u[ykey] for u in keep])
    A = np.array([A_by_id[u["doc_id"]] for u in keep], dtype=float)
    # one v_features() call per document, not per (document, feature name)
    V = np.array([[f[n] for n in V_NAMES]
                  for f in (v_features(u["text"]) for u in keep)], dtype=float)
    groups = np.array([u["docket"] for u in keep])
    splits = np.array([u["split"] for u in keep])
    doc_ids = np.array([u["doc_id"] for u in keep])

    Ac, a_keep = L.clean_cols(A)
    Vc, v_keep = L.clean_cols(V)
    VA = np.column_stack([Vc, Ac])
    names = {"V": [V_NAMES[j] for j in v_keep], "A": [a_names[j] for j in a_keep]}
    names["VA"] = names["V"] + names["A"]
    return ({"V": Vc, "A": Ac, "VA": VA}, names, y, groups, splits, doc_ids,
            dropped, V, keep)


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y", default="cosign", choices=["cosign", "nearby", "campaign"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()

    mats, names, y, groups, splits, doc_ids, dropped, V_raw, units = load_population(args.y)
    folds = L.outer_folds(len(y), groups)
    n_groups = len(np.unique(groups))
    print(f"y={args.y} n={len(y)} pos={y.mean():.4f} ({int(y.sum())}) groups={n_groups} "
          f"V={mats['V'].shape[1]}c A={mats['A'].shape[1]}c VA={mats['VA'].shape[1]}c "
          f"(no-A-score dropped={dropped})", flush=True)

    res = {
        "cell": "nc cosigning (V8)",
        "y_variant": args.y,
        "y_definition": ("exact within-docket normalised-text duplicate count >= 2 "
                         "(number of submitters who signed on to this text)"
                         if args.y == "cosign" else
                         "shipped MinHash near-dup family >= 2 (PARTIAL COVERAGE sensitivity arm)"
                         if args.y == "nearby" else
                         "exact within-docket duplicate count >= 10 (organised-campaign scale)"),
        "n": int(len(y)), "pos_rate": float(y.mean()), "n_pos": int(y.sum()),
        "n_groups": int(n_groups), "group_column": "docket",
        "population": str(COS / "cosign_population.jsonl"),
        "a_bank": ("REUSED VERBATIM: pre-GEPA, 198 rubrics, Gemma-4-31B "
                   "(nc_scores_shard0..4.npz + nc_scores_unmatched.npz) -- no new judging"),
        "v_features": "REUSED VERBATIM: aggregate_vat_nc.v_features, 27 extractors",
        "n_features": {k: int(v.shape[1]) for k, v in mats.items()},
        "first_fit": True,
        "gate": {"kind": "first-fit: no published V+A stack of this construction exists; "
                         "no reproduction gate applicable (press precedent)"},
        "linear": {}, "nonlinear": {},
    }

    lin_oof = {}
    for k in ("V", "A", "VA"):
        auc, oof = L.linear_oof(mats[k], y, groups, folds)
        res["linear"][k] = auc
        lin_oof[k] = oof
        print(f"  linear  {k:2s}: {auc:.4f}", flush=True)

    nl = {"V": {}, "A": {}, "VA": {}}
    for k in ("V", "A", "VA"):
        for s in (0, 1, 2):
            r = L.gbm_oof(mats[k], y, groups, folds, seed=s)
            nl[k][s] = r
            print(f"  gbm     {k:2s} seed {s}: {r['auc']:.4f} (train {r['train_auc_mean']:.4f})",
                  flush=True)

    nl_oof_mean = {}
    for k in ("V", "A", "VA"):
        aucs = [nl[k][s]["auc"] for s in (0, 1, 2)]
        nl_oof_mean[k] = np.mean([nl[k][s]["oof"] for s in (0, 1, 2)], axis=0)
        res["nonlinear"][k] = {
            "seed_aucs": {str(s): nl[k][s]["auc"] for s in (0, 1, 2)},
            "mean_auc": float(np.mean(aucs)),
            "spread": float(max(aucs) - min(aucs)),
            "auc_of_seedmean_oof": float(roc_auc_score(y, nl_oof_mean[k])),
            "train_auc_mean_seed0": nl[k][0]["train_auc_mean"],
            "picks_seed0": nl[k][0]["picks"],
        }

    ledger = {
        "V_lin": res["linear"]["V"], "V_nl_mean": res["nonlinear"]["V"]["mean_auc"],
        "A_lin": res["linear"]["A"], "A_nl_mean": res["nonlinear"]["A"]["mean_auc"],
        "VA_lin": res["linear"]["VA"], "VA_nl_mean": res["nonlinear"]["VA"]["mean_auc"],
        "V_interact": res["nonlinear"]["V"]["mean_auc"] - res["linear"]["V"],
        "Delta_interact": res["nonlinear"]["VA"]["mean_auc"] - res["linear"]["VA"],
    }
    res["ledger"] = ledger
    res["overfit_gap"] = {k: res["nonlinear"][k]["train_auc_mean_seed0"] - res["nonlinear"][k]["mean_auc"]
                          for k in ("V", "A", "VA")}

    res["delta_interact_bootstrap_docket_PRIMARY"] = L.bootstrap_delta_interact_docket(
        lin_oof["VA"], nl_oof_mean["VA"], y, groups)
    res["delta_interact_bootstrap_row_secondary"] = L.bootstrap_delta_interact(
        lin_oof["VA"], nl_oof_mean["VA"], y)

    # ---- Layer-2(a) grouped transfer, mandatory beside every pooled number --
    gt = {"docket_identity_alone_auc": group_identity_alone_auc(y, groups)}
    for k in ("V", "A", "VA"):
        wl, ndk, per = within_group_auc(y, lin_oof[k], groups)
        wn, _, _ = within_group_auc(y, nl_oof_mean[k], groups)
        gt[k] = {"pooled_lin": res["linear"][k], "within_docket_lin": wl,
                 "pooled_nl": res["nonlinear"][k]["auc_of_seedmean_oof"],
                 "within_docket_nl": wn, "n_mixed_dockets": ndk}
        if k == "VA":
            gt["per_docket_VA_nl"] = sorted(per, key=lambda d: -d["n"])[:15]
    res["grouped_transfer"] = gt

    # ---- nuisance quick-look (the genre-detection caveat is testable) -------
    res["nuisance"] = {
        "char_len_alone_auc": float(roc_auc_score(y, V_raw[:, V_NAMES.index("v_char_len")])),
        "top_single_V_feature_aucs": single_feature_aucs(V_raw, y, V_NAMES),
    }

    # ---- cross-y contrast with the verdict column --------------------------
    resp = np.array([u["is_matched"] for u in units])
    res["cross_y"] = {
        "responded_rate_given_cosign1": float(resp[y == 1].mean()),
        "responded_rate_given_cosign0": float(resp[y == 0].mean()),
        "phi": float(np.corrcoef(resp, y)[0, 1]),
    }

    res["split_audit"] = {s: {"n": int((splits == s).sum()),
                              "pos_rate": float(y[splits == s].mean())}
                          for s in ("train", "eval", "test")}
    res["runtime_sec"] = time.time() - t0
    out = args.out or str(RESULTS_DIR / (f"nc_cosigning_layer1_{args.y}.json"))
    json.dump(res, open(out, "w"), indent=1)
    np.save(RESULTS_DIR / f"nc_cosigning_{args.y}_va_nl_oof_mean3.npy", nl_oof_mean["VA"])
    np.save(RESULTS_DIR / f"nc_cosigning_{args.y}_va_lin_oof.npy", lin_oof["VA"])
    np.save(RESULTS_DIR / f"nc_cosigning_{args.y}_doc_ids.npy", doc_ids)
    print(json.dumps({"ledger": ledger, "grouped_transfer": {k: v for k, v in gt.items()
                                                             if k != "per_docket_VA_nl"},
                      "nuisance": res["nuisance"], "cross_y": res["cross_y"]}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
