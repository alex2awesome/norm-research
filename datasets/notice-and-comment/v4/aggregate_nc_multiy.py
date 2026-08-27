#!/usr/bin/env python3
"""N&C multi-y VAT aggregation (laptop / base env with sklearn).

Replicates the academia peer-review "multi-y within-domain contrast"
(../../peer-review/vat_3y/aggregate_3y.py, read that file first — this is a
verbatim port of its frozen design) on notice-and-comment: items scored ONCE
with a 198-rubric A-bank (Gemma-4-31B, pre-GEPA) + 27 regex V-features, then
three different preference labels y attached to the same items:

  1. outcome-majority : majority MADE(1) vs NONE(0) over a comment's matched
                         response labels, ties dropped.
  2. agree-vs-disagree: majority accepted/agree(1) vs disagree(0) response_type.
  3. responded-or-not : matched-with-any-label(1) vs genuinely-unmatched(0).

Definitions copied EXACTLY from y_audit_nc.py (outcome_maj / agree-vs-disagree)
and nc_responded_or_not.json's build (matched-sample vs nc_unmatched_sample.jsonl).
NEVER any-MADE union (retired, confounded by n_labels — see notes/2026-07-15__nc-vat-run.md).

Design (frozen, copied from vat_3y/aggregate_3y.py):
  - A = 198 rubric scores (pre-GEPA nc_scores_shard*.npz); NA -> column median
    over non-NA (drop all-NA cols).
  - V = 27 regex features (aggregate_vat_nc.py::v_features), same treatment.
  - Degeneracy guard: drop any A/V column with <5 off-modal values or zero variance.
  - Model: StandardScaler + LogisticRegression(C=1, max_iter=2000).
  - AUC = roc_auc on out-of-fold cross_val_predict, GroupKFold(5), group = docket.
  - V, A, V+A each fit on the SAME rows so V/A/VA are apples-to-apples within a y.
  - Apples-to-apples: strict-common items (identical comments labeled under BOTH
    y's) restrict to identical rows/features; only y changes.

Run (matched shards + unmatched shard must already be on disk):
  python3 aggregate_nc_multiy.py
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
from aggregate_vat_nc import v_features, V_NAMES  # noqa: E402

MATCHED_SHARDS = sorted(glob.glob(str(D / "nc_scores_shard*.npz")))
GEPA_SHARDS = sorted(glob.glob(str(D / "nc_scores_gepa_shard*.npz")))
UNMATCHED_NPZ = D / "nc_scores_unmatched.npz"
SAMPLE_JSONL = D / "nc_vat_sample.jsonl"
UNMATCHED_JSONL = D / "nc_unmatched_sample.jsonl"
LABELS_FULL = D / "nc_vat_sample_labels_full.json"

AGREE = {"accepted", "agree"}
DISAGREE = {"disagree"}
MADE = {"MADE"}
NONE_ = {"NONE"}


# ---------------------------------------------------------------- loaders ---
def load_shard_scores(paths):
    """doc_id -> A-score row (198,), doc_id -> docket, doc_id -> agency. Anchors dropped."""
    X_by_id, docket_by_id, agency_by_id = {}, {}, {}
    a_names = None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        a_names = [str(x) for x in d["a_names"]]
        for i, did in enumerate(d["doc_id"]):
            did = str(did)
            if str(d["agency"][i]) == "__ANCHOR":
                continue
            X_by_id[did] = d["X"][i]
            docket_by_id[did] = str(d["docket"][i])
            agency_by_id[did] = str(d["agency"][i])
    return X_by_id, docket_by_id, agency_by_id, a_names


def load_jsonl_texts(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["doc_id"]] = r.get("text", "")
    return out


def maj(vals, pos, neg):
    p = sum(v in pos for v in vals)
    n = sum(v in neg for v in vals)
    if p == n:
        return -1
    return int(p > n)


def load_label_ys(path):
    """doc_id -> {'outcome_majority': -1/0/1, 'agree_vs_disagree': -1/0/1}"""
    labs = json.load(open(path))
    out = {}
    for doc_id, ls in labs.items():
        oc = [l["outcome_collapsed"] for l in ls]
        rt = [l["response_type"] for l in ls]
        out[doc_id] = {
            "outcome_majority": maj(oc, MADE, NONE_),
            "agree_vs_disagree": maj(rt, AGREE, DISAGREE),
        }
    return out


# ------------------------------------------------------------- VAT design ---
def clean_cols(M):
    """Drop degenerate columns; median-impute NA. Verbatim from vat_3y/aggregate_3y.py."""
    keep = []
    out = []
    for j in range(M.shape[1]):
        col = M[:, j].astype(float)
        nonna = col[~np.isnan(col)]
        if len(nonna) == 0:
            continue
        med = np.median(nonna)
        c = np.where(np.isnan(col), med, col)
        vals, counts = np.unique(c, return_counts=True)
        modal = counts.max()
        offmodal = len(c) - modal
        if offmodal < 5 or c.std() == 0:  # degeneracy guard
            continue
        keep.append(j)
        out.append(c)
    if not out:
        return np.zeros((M.shape[0], 0)), keep
    return np.column_stack(out), keep


def auc_cv(Xf, y, groups):
    if Xf.shape[1] == 0 or len(np.unique(y)) < 2:
        return float("nan")
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    gkf = GroupKFold(n_splits=n_splits)
    proba = cross_val_predict(clf, Xf, y, cv=gkf, groups=groups, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def build_row(doc_id, y, X_by_id, docket_by_id, text_by_id):
    a = X_by_id[doc_id].astype(float)
    v = np.array([v_features(text_by_id.get(doc_id, ""))[n] for n in V_NAMES], dtype=float)
    return y, a, v, docket_by_id[doc_id]


def fit_row(items):
    """items: list of (y, a_vec, v_vec, docket). Returns dict with n/pos/V/A/VA."""
    if len(items) < 20:
        return None
    y = np.array([it[0] for it in items])
    A = np.array([it[1] for it in items], dtype=float)
    V = np.array([it[2] for it in items], dtype=float)
    groups = np.array([it[3] for it in items])
    if len(np.unique(y)) < 2:
        return {"n": len(items), "pos": float(y.mean()), "V": float("nan"),
                "A": float("nan"), "VA": float("nan"), "A_minus_V": float("nan"),
                "note": "single-class (degenerate y within this subset)"}
    Ac, _ = clean_cols(A)
    Vc, _ = clean_cols(V)
    VA = np.column_stack([Vc, Ac]) if Vc.shape[1] and Ac.shape[1] else (Vc if Vc.shape[1] else Ac)
    out = {"n": len(items), "pos": float(y.mean()),
           "V": auc_cv(Vc, y, groups), "A": auc_cv(Ac, y, groups),
           "VA": auc_cv(VA, y, groups)}
    out["A_minus_V"] = out["A"] - out["V"] if np.isfinite(out["A"]) and np.isfinite(out["V"]) else float("nan")
    return out


def main():
    for p in (SAMPLE_JSONL, UNMATCHED_JSONL, LABELS_FULL, UNMATCHED_NPZ):
        assert p.exists(), f"missing {p}"
    assert MATCHED_SHARDS, "missing nc_scores_shard*.npz"

    X_m, docket_m, agency_m, a_names = load_shard_scores(MATCHED_SHARDS)
    X_u, docket_u, agency_u, _ = load_shard_scores([str(UNMATCHED_NPZ)])
    text_m = load_jsonl_texts(SAMPLE_JSONL)
    text_u = load_jsonl_texts(UNMATCHED_JSONL)
    label_ys = load_label_ys(LABELS_FULL)

    # ---- de-conflict the 34 doc_ids that appear scored in BOTH matched and
    # unmatched shards (an artifact of independent sampling draws) — keep them
    # on the matched side only, so responded-or-not never double-counts an id
    # with contradictory labels.
    overlap = set(X_m) & set(X_u)
    for did in overlap:
        del X_u[did]
        del docket_u[did]
        del agency_u[did]

    print(f"[inventory] pre-GEPA matched-shard scored ids (non-anchor): {len(X_m)}")
    print(f"[inventory] pre-GEPA unmatched-shard scored ids (non-anchor, "
          f"after dropping {len(overlap)} matched/unmatched id overlap): {len(X_u)}")
    print(f"[inventory] label_lists coverage of matched ids: {len(set(label_ys) & set(X_m))}/{len(X_m)}")

    # ---------------------------------------------------------- y universes
    # outcome-majority / agree-vs-disagree: only defined on the matched+labeled side.
    y_out_by_id = {did: label_ys[did]["outcome_majority"] for did in X_m if did in label_ys}
    y_agr_by_id = {did: label_ys[did]["agree_vs_disagree"] for did in X_m if did in label_ys}
    valid_out = {did for did, y in y_out_by_id.items() if y in (0, 1)}
    valid_agr = {did for did, y in y_agr_by_id.items() if y in (0, 1)}
    # responded-or-not: matched (y=1, any doc_id scored+sampled in the matched draw,
    # regardless of whether outcome/agree resolved to a tie) vs unmatched (y=0).
    valid_resp = set(X_m) | set(X_u)
    y_resp_by_id = {did: 1 for did in X_m}
    y_resp_by_id.update({did: 0 for did in X_u})

    inv_rows = [
        ("outcome-majority", len(valid_out), float(np.mean([y_out_by_id[d] for d in valid_out]))),
        ("agree-vs-disagree", len(valid_agr), float(np.mean([y_agr_by_id[d] for d in valid_agr]))),
        ("responded-or-not", len(valid_resp), float(np.mean([y_resp_by_id[d] for d in valid_resp]))),
    ]
    pair_out_agr = valid_out & valid_agr
    pair_out_resp = valid_out & valid_resp   # == valid_out (RESP y constant=1 here)
    pair_agr_resp = valid_agr & valid_resp   # == valid_agr (RESP y constant=1 here)
    triple = valid_out & valid_agr & valid_resp

    inventory = {
        "scored_matched_n": len(X_m),
        "scored_unmatched_n": len(X_u),
        "matched_unmatched_id_overlap_dropped": len(overlap),
        "per_y": [{"y": n, "n_valid_labeled_and_scored": k, "pos_rate": p} for n, k, p in inv_rows],
        "pairwise_strict_common_n": {
            "outcome-majority & agree-vs-disagree": len(pair_out_agr),
            "outcome-majority & responded-or-not": len(pair_out_resp),
            "agree-vs-disagree & responded-or-not": len(pair_agr_resp),
        },
        "triple_strict_common_n": len(triple),
        "caveat": ("responded-or-not is defined over a DIFFERENT item universe (matched U "
                   "unmatched) than outcome-majority/agree-vs-disagree (matched-labeled only, "
                   "a subset of 'matched'). Any strict-common set built against "
                   "responded-or-not therefore only pulls from the matched side, where "
                   "responded-or-not is constant y=1 -> AUC is undefined there (single class); "
                   "reported as such below, not tuned away."),
    }

    print("\n[inventory table]")
    print("| y | n valid (scored+labeled) | pos rate |")
    print("|---|---|---|")
    for n, k, p in inv_rows:
        print(f"| {n} | {k} | {p:.3f} |")
    print("\n| strict-common pair | n |")
    print("|---|---|")
    for k, v in inventory["pairwise_strict_common_n"].items():
        print(f"| {k} | {v} |")
    print(f"| triple (all three) | {inventory['triple_strict_common_n']} |")

    # -------------------------------------------------------- full-pool rows
    def rows_for(ids, y_by_id, X_by_id, docket_by_id, text_by_id):
        return [build_row(did, y_by_id[did], X_by_id, docket_by_id, text_by_id) for did in ids]

    full_pool = {}
    full_pool["outcome-majority"] = fit_row(rows_for(valid_out, y_out_by_id, X_m, docket_m, text_m))
    full_pool["agree-vs-disagree"] = fit_row(rows_for(valid_agr, y_agr_by_id, X_m, docket_m, text_m))
    resp_items = (rows_for(set(X_m), y_resp_by_id, X_m, docket_m, text_m)
                  + rows_for(set(X_u), y_resp_by_id, X_u, docket_u, text_u))
    full_pool["responded-or-not"] = fit_row(resp_items)

    print("\n### Full-pool per-y (pre-GEPA A-bank)")
    print("| y | n | pos | V | A | V+A | A-V |")
    print("|---|---|---|---|---|---|---|")
    for name, r in full_pool.items():
        if r:
            print(f"| {name} | {r['n']} | {r['pos']:.3f} | {r['V']:.3f} | {r['A']:.3f} | "
                  f"{r['VA']:.3f} | {r['A_minus_V']:+.3f} |")

    # ------------------------------------------------------ sanity gates ---
    gates = {
        "outcome-majority": {"V": 0.595, "A": 0.592},
        "agree-vs-disagree": {"A": 0.612, "V": 0.612},
        "responded-or-not": {"A": 0.636, "VA": 0.646},
    }
    TOL = 0.03
    sanity = {}
    print("\n### Sanity gates (tol ±0.03 vs prior per-y VAT runs)")
    print("| y | metric | prior | observed | delta | verdict |")
    print("|---|---|---|---|---|---|")
    for name, checks in gates.items():
        r = full_pool.get(name)
        sanity[name] = {}
        for metric, prior in checks.items():
            obs = r[metric] if r else float("nan")
            delta = obs - prior if np.isfinite(obs) else float("nan")
            verdict = "PASS" if np.isfinite(delta) and abs(delta) <= TOL else "FAIL"
            sanity[name][metric] = {"prior": prior, "observed": obs, "delta": delta, "verdict": verdict}
            print(f"| {name} | {metric} | {prior:.3f} | {obs:.3f} | {delta:+.3f} | {verdict} |")

    # -------------------------------------------------- apples-to-apples ---
    a2a = {}
    r_out_common = fit_row(rows_for(pair_out_agr, y_out_by_id, X_m, docket_m, text_m))
    r_agr_common = fit_row(rows_for(pair_out_agr, y_agr_by_id, X_m, docket_m, text_m))
    a2a["outcome-majority & agree-vs-disagree"] = {
        "common_n": len(pair_out_agr),
        "outcome-majority": r_out_common,
        "agree-vs-disagree": r_agr_common,
    }
    print(f"\n### Apples-to-apples: outcome-majority & agree-vs-disagree "
          f"(identical {len(pair_out_agr)} comments labeled under BOTH, same A/V features)")
    print("| preference y | n | pos | V | A | V+A | A-V |")
    print("|---|---|---|---|---|---|---|")
    for name, r in (("outcome-majority", r_out_common), ("agree-vs-disagree", r_agr_common)):
        if r:
            print(f"| {name} | {r['n']} | {r['pos']:.3f} | {r['V']:.3f} | {r['A']:.3f} | "
                  f"{r['VA']:.3f} | {r['A_minus_V']:+.3f} |")

    for label_, common_set, y_by_id, X_by_id, docket_by_id, text_by_id in (
        ("outcome-majority & responded-or-not", pair_out_resp, y_resp_by_id, X_m, docket_m, text_m),
        ("agree-vs-disagree & responded-or-not", pair_agr_resp, y_resp_by_id, X_m, docket_m, text_m),
    ):
        r = fit_row(rows_for(common_set, y_by_id, X_by_id, docket_by_id, text_by_id))
        a2a[label_] = {"common_n": len(common_set), "responded-or-not": r}
        print(f"\n### Apples-to-apples: {label_} (n={len(common_set)}) — "
              f"responded-or-not is CONSTANT (y=1) within this set (matched-only subset); "
              f"AUC is structurally undefined, reported as such:")
        print(json.dumps(r, indent=2))

    triple_note = {
        "common_n": len(triple),
        "note": "identical to outcome-majority & agree-vs-disagree common set restricted further "
                "by responded-or-not's (trivially-satisfied, matched-only) universe -> same n as "
                "the pairwise outcome&agree common set whenever pair_out_agr subset of matched.",
    }

    # -------------------------------------------------- selection-effect audit
    selection = {
        "pos_rate_outcome_majority_full_pool": full_pool["outcome-majority"]["pos"] if full_pool["outcome-majority"] else None,
        "pos_rate_outcome_majority_in_common_with_agree": (
            float(np.mean([y_out_by_id[d] for d in pair_out_agr])) if pair_out_agr else None),
        "pos_rate_agree_full_pool": full_pool["agree-vs-disagree"]["pos"] if full_pool["agree-vs-disagree"] else None,
        "pos_rate_agree_in_common_with_outcome": (
            float(np.mean([y_agr_by_id[d] for d in pair_out_agr])) if pair_out_agr else None),
    }
    print("\n### Selection-effect audit (pos-rate shift when restricting to strict-common items)")
    print(json.dumps(selection, indent=2))

    # -------------------------------------------------- secondary GEPA table
    gepa_table = {}
    if GEPA_SHARDS:
        Xg, docketg, agencyg, _ = load_shard_scores(GEPA_SHARDS)
        for name, y_by_id, valid in (("outcome-majority", y_out_by_id, valid_out),
                                      ("agree-vs-disagree", y_agr_by_id, valid_agr)):
            ids = [d for d in valid if d in Xg]
            r = fit_row(rows_for(ids, y_by_id, Xg, docketg, text_m))
            gepa_table[name] = r
        print("\n### SECONDARY (non-headline): post-GEPA A-bank, same y's/rows-where-scored")
        print("| y | n | pos | V | A | V+A | A-V |")
        print("|---|---|---|---|---|---|---|")
        for name, r in gepa_table.items():
            if r:
                print(f"| {name} | {r['n']} | {r['pos']:.3f} | {r['V']:.3f} | {r['A']:.3f} | "
                      f"{r['VA']:.3f} | {r['A_minus_V']:+.3f} |")
        print("(GEPA has no unmatched-side shard -> responded-or-not omitted from this table.)")

    res = {
        "design": "verbatim port of ../../peer-review/vat_3y/aggregate_3y.py; GroupKFold(5) "
                  "group=docket, StandardScaler+LogisticRegression(C=1,max_iter=2000), "
                  "clean_cols degeneracy guard (<5 off-modal or zero-var dropped), NA->median.",
        "a_bank": "pre-GEPA, 198 rubrics, Gemma-4-31B (nc_scores_shard0..4.npz)",
        "inventory": inventory,
        "full_pool": full_pool,
        "sanity_gates": sanity,
        "apples_to_apples": a2a,
        "triple_common": triple_note,
        "selection_effect_audit": selection,
        "secondary_gepa": gepa_table,
    }
    out_path = D / "nc_multiy_results.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
