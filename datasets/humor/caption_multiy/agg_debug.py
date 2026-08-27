#!/usr/bin/env python3
"""Caption-contest multi-y VAT aggregation (laptop / base env with sklearn).

Third within-domain preference-variable contrast (after academia 3-y and N&C 3-y):
same captions scored ONCE (364-rubric standup A-bank, Gemma-4-31B + 16 V-feats on
v2-normalized text), two preference labels attached:

  1. finalist-B (curation): editor-picked finalist(1) vs crowd-loved-but-not-picked
     neg_hard(0). HEADLINE pool = hard-negative build ONLY (finalist vs neg_hard);
     finalist-vs-all-negs is reported as SECONDARY (known length/topic-inflated —
     the .938-style number, never headline).
  2. crowd-C (revealed): within-contest crowd_mean median split, captions with
     crowd_votes >= 100.

Frozen design copied from aggregate_nc_multiy.py / vat_3y/aggregate_3y.py:
median-impute + degeneracy guard (<5 off-modal or zero-var dropped),
StandardScaler + LogisticRegression(C=1, max_iter=2000), AUC on out-of-fold
cross_val_predict, GroupKFold(5), group = contest.

PRE-DECLARED estimator-design secondary (N&C 2026-07-26 lesson: CV stratification
was load-bearing for docket-clustered y): every row is ALSO computed under
StratifiedGroupKFold(5, shuffle, seed 0), reported side-by-side, never mixed.

crowd_mean / crowd_votes are used ONLY to build y — never as features.
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = Path(__file__).resolve().parent
SHARDS = sorted(glob.glob(str(D / "cap_scores_shard*.npz")))
POOL = D / "caption_contest_v2.jsonl"
MIN_VOTES = 100


def load_scores():
    X_by_id, V_by_id, contest_by_id, role_by_id = {}, {}, {}, {}
    a_names = v_names = None
    anchor_report = defaultdict(list)
    for p in SHARDS:
        d = np.load(p, allow_pickle=True)
        a_names = [str(x) for x in d["a_names"]]
        v_names = [str(x) for x in d["v_names"]]
        for i, did in enumerate(d["doc_id"]):
            did = str(did)
            if did.startswith("__ANCHOR"):
                anchor_report[did].append(float(np.nanmean(d["X"][i])))
                continue
            X_by_id[did] = d["X"][i]
            V_by_id[did] = d["V"][i]
            contest_by_id[did] = str(d["contest"][i])
            role_by_id[did] = str(d["role"][i])
    return X_by_id, V_by_id, contest_by_id, role_by_id, a_names, v_names, dict(anchor_report)


def load_pool():
    import hashlib
    rows = [json.loads(l) for l in open(POOL) if l.strip()]
    meta = {}
    for r in rows:
        did = f"{r['contest']}_{hashlib.sha1(r['text'].encode()).hexdigest()[:12]}"
        meta[did] = r
    return meta


def clean_cols(M):
    keep, out = [], []
    for j in range(M.shape[1]):
        col = M[:, j].astype(float)
        nonna = col[~np.isnan(col)]
        if len(nonna) == 0:
            continue
        med = np.median(nonna)
        c = np.where(np.isnan(col), med, col)
        vals, counts = np.unique(c, return_counts=True)
        offmodal = len(c) - counts.max()
        if offmodal < 5 or c.std() == 0:
            continue
        keep.append(j)
        out.append(c)
    if not out:
        return np.zeros((M.shape[0], 0)), keep
    return np.column_stack(out), keep


def auc_cv(Xf, y, groups, cv):
    if Xf.shape[1] == 0 or len(np.unique(y)) < 2:
        return float("nan")
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    if cv == "gkf":
        splitter = GroupKFold(n_splits=n_splits)
        proba = cross_val_predict(clf, Xf, y, cv=splitter, groups=groups,
                                  method="predict_proba")[:, 1]
    else:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
        proba = np.full(len(y), np.nan)
        for tr, te in splitter.split(np.zeros(len(y)), y, groups):
            clf.fit(Xf[tr], y[tr])
            proba[te] = clf.predict_proba(Xf[te])[:, 1]
        m = ~np.isnan(proba)
        return float(roc_auc_score(y[m], proba[m]))
    return float(roc_auc_score(y, proba))


def fit_row(ids, y_by_id, X_by_id, V_by_id, contest_by_id):
    ids = [d for d in ids if d in X_by_id]
    if len(ids) < 20:
        return None
    y = np.array([y_by_id[d] for d in ids])
    A = np.array([X_by_id[d] for d in ids], dtype=float)
    V = np.array([V_by_id[d] for d in ids], dtype=float)
    groups = np.array([contest_by_id[d] for d in ids])
    if len(np.unique(y)) < 2:
        return {"n": len(ids), "pos": float(y.mean()), "note": "single-class"}
    Ac, _ = clean_cols(A)
    Vc, _ = clean_cols(V)
    VA = np.column_stack([Vc, Ac]) if Vc.shape[1] and Ac.shape[1] else (Vc if Vc.shape[1] else Ac)
    out = {"n": len(ids), "pos": float(y.mean())}
    for cv in ("gkf", "sgkf"):
        out[cv] = {"V": auc_cv(Vc, y, groups, cv), "A": auc_cv(Ac, y, groups, cv),
                   "VA": auc_cv(VA, y, groups, cv)}
        out[cv]["A_minus_V"] = (out[cv]["A"] - out[cv]["V"]
                                if np.isfinite(out[cv]["A"]) and np.isfinite(out[cv]["V"])
                                else float("nan"))
    return out


def prow(name, r):
    if not r:
        print(f"| {name} | - | - | (skipped) |")
        return
    if "gkf" not in r:
        print(f"| {name} | {r['n']} | {r['pos']:.3f} | single-class |")
        return
    g, s = r["gkf"], r["sgkf"]
    print(f"| {name} | {r['n']} | {r['pos']:.3f} | {g['V']:.3f} | {g['A']:.3f} | {g['VA']:.3f} | "
          f"{g['A_minus_V']:+.3f} | {s['V']:.3f} | {s['A']:.3f} | {s['VA']:.3f} | {s['A_minus_V']:+.3f} |")


HDR = ("| y / pool | n | pos | V(gkf) | A(gkf) | V+A(gkf) | A-V(gkf) | V(sgkf) | A(sgkf) | "
       "V+A(sgkf) | A-V(sgkf) |")
SEP = "|---|---|---|---|---|---|---|---|---|---|---|"


import resource, sys as _s
def _t(msg):
    print(f'[trace] {msg} mem={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.0f}MB', flush=True)
def main():
    _t('main start')
    X_by_id, V_by_id, contest_by_id, role_by_id, a_names, v_names, anchors = load_scores()
    _t('scores loaded')
    meta = load_pool()
    _t('pool loaded')
    print(f"[inventory] scored non-anchor captions: {len(X_by_id)}")
    print(f"[inventory] anchors (mean A over shards): "
          + json.dumps({k: round(float(np.mean(v)), 3) for k, v in sorted(anchors.items())}))

    # ---------------------------------------------------------------- y's ---
    y_fin, y_crowd = {}, {}
    by_contest_crowd = defaultdict(list)
    for did in X_by_id:
        m = meta.get(did)
        if m is None:
            continue
        role = role_by_id[did]
        if role in ("finalist", "neg_hard", "neg_random"):
            y_fin[did] = 1 if role == "finalist" else 0
        if m.get("crowd_mean") is not None and (m.get("crowd_votes") or 0) >= MIN_VOTES:
            by_contest_crowd[m["contest"]].append((did, m["crowd_mean"]))
    for c, items in by_contest_crowd.items():
        if len(items) < 6:
            continue
        med = float(np.median([v for _, v in items]))
        for did, v in items:
            if v == med:
                continue  # drop exact-median ties
            y_crowd[did] = int(v > med)

    hardneg_ids = {d for d in y_fin if role_by_id[d] in ("finalist", "neg_hard")}
    crowd_ids = set(y_crowd)
    common = hardneg_ids & crowd_ids

    print("\n[inventory table]")
    print("| y | n valid | pos rate |")
    print("|---|---|---|")
    print(f"| finalist-B (hardneg pool) | {len(hardneg_ids)} | "
          f"{np.mean([y_fin[d] for d in hardneg_ids]):.3f} |")
    print(f"| finalist-B (full pool, SECONDARY) | {len(y_fin)} | "
          f"{np.mean(list(y_fin.values())):.3f} |")
    print(f"| crowd-C (median split, votes>={MIN_VOTES}) | {len(y_crowd)} | "
          f"{np.mean(list(y_crowd.values())):.3f} |")
    print(f"| strict-common (hardneg & crowd-labeled) | {len(common)} | - |")

    res = {"anchors": {k: [float(x) for x in v] for k, v in anchors.items()}}

    # ------------------------------------------------------------ full pool ---
    print("\n### Full-pool per-y (364-rubric standup bank; gkf = frozen primary, sgkf = pre-declared secondary)")
    print(HDR); print(SEP)
    rows = {}
    rows["finalist-B (hardneg pool)"] = fit_row(hardneg_ids, y_fin, X_by_id, V_by_id, contest_by_id)
    rows["finalist-B (full pool, SECONDARY)"] = fit_row(set(y_fin), y_fin, X_by_id, V_by_id, contest_by_id)
    rows["crowd-C (full crowd-labeled)"] = fit_row(crowd_ids, y_crowd, X_by_id, V_by_id, contest_by_id)
    for k, r in rows.items():
        prow(k, r)
    res["full_pool"] = rows

    # ------------------------------------------------- apples-to-apples ---
    print(f"\n### Apples-to-apples: identical {len(common)} captions (hardneg pool ∩ crowd-labeled), only y changes")
    print(HDR); print(SEP)
    a2a = {}
    a2a["finalist-B"] = fit_row(common, y_fin, X_by_id, V_by_id, contest_by_id)
    a2a["crowd-C"] = fit_row(common, y_crowd, X_by_id, V_by_id, contest_by_id)
    for k, r in a2a.items():
        prow(k, r)
    res["apples_to_apples"] = {"common_n": len(common), **a2a}

    # selection audit
    sel = {
        "pos_fin_hardneg_pool": float(np.mean([y_fin[d] for d in hardneg_ids])),
        "pos_fin_in_common": float(np.mean([y_fin[d] for d in common])) if common else None,
        "pos_crowd_full": float(np.mean(list(y_crowd.values()))),
        "pos_crowd_in_common": float(np.mean([y_crowd[d] for d in common])) if common else None,
        "note": ("common set = finalists + neg_hard with >=100 crowd votes; neg_hard are BY "
                 "CONSTRUCTION top-20 crowd per contest, so crowd-C pos rate in common is "
                 "structurally high — report, don't reweight."),
    }
    print("\n### Selection-effect audit")
    print(json.dumps(sel, indent=2))
    res["selection_effect_audit"] = sel

    res["design"] = ("A/V from cap_scores_shard*.npz (v2-normalized text; crowd fields never "
                     "features). Frozen: clean_cols guard, StandardScaler+LR(C=1), "
                     "GroupKFold(5) group=contest (gkf). Pre-declared secondary: "
                     "StratifiedGroupKFold(5,shuffle,seed0) (sgkf). MIN_VOTES=%d." % MIN_VOTES)
    (D / "cap_multiy_results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {D / 'cap_multiy_results.json'}")


if __name__ == "__main__":
    main()
