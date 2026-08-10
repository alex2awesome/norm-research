#!/usr/bin/env python3
"""VAT recovery readout for the claim-matching subtask — "how solved is claim-fetching?"

Given metric scores from one or more backends (scores_<tag>.jsonl: {metric_id,domain,uid,y,score}),
computes the V/A decomposition on the clean gold-vs-filler testbed:
  V (mechanical) : recovery from lexical/length features alone (computed here, CPU) — the code floor.
  A (articulated): recovery from the articulated claim-matching metric bank, per backend model, both
                   per-metric and COMBINED (logistic over all metrics, app-hash CV).
  ceiling        : the strongest backend's combined recovery (frontier anchor) — lower-bounds T=ceil-A.
Readouts per BEST-PRACTICES: threshold-free (AUC + within-claim paired accuracy), MI/H(Y), domain
split, constancy already filtered upstream. Within-claim paired accuracy = fraction of claims where
the metric scores the examiner's gold span above the filler (ties .5) — the honest matching metric.

Codex-audit fixes (2026-07-10): app_id CV folds; train-fold-only imputation; v2 testbed default.
Codex round-2 fixes (2026-07-12): --patch is TAG-SCOPED (tag=path; a bare path is an error — a
12b patch must never overwrite 4b/27b rows); constant metrics dropped INSIDE each training fold;
new diagnostics per tag: per-metric tie mass, identical-feature-vector mass, and a pair-difference
readout (logistic on x_gold - x_filler, app-folds) that targets the paired endpoint directly.

  python scripts/claim_matching_recovery.py --tags gemma3_12b --patch gemma3_12b=outputs/claim_matching/scores_gemma3_12b_v2negs.jsonl
"""
import argparse, json, re, hashlib, glob, os, collections
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE = "/lfs/skampere3/0/alexspan/norm-research"
TESTBED = f"{BASE}/datasets/claim-matching/testbed/pair_testbed_v2.jsonl"
OUTDIR = f"{BASE}/outputs/claim_matching"
WORD = re.compile(r"[a-z]{3,}")


def toks(s):
    return set(WORD.findall((s or "").lower()))


def within_claim_acc(uids, y, score):
    """fraction of claims where score(gold) > score(filler); ties .5."""
    byu = collections.defaultdict(dict)
    for u, yy, s in zip(uids, y, score):
        byu[u][yy] = s
    acc, n = 0.0, 0
    for u, d in byu.items():
        if 1 in d and 0 in d:
            n += 1
            acc += 1.0 if d[1] > d[0] else 0.5 if d[1] == d[0] else 0.0
    return acc / max(1, n), n


def mi_binned(y, s, bins=6):
    y = np.asarray(y, int); s = np.asarray(s, float)
    edges = np.unique(np.quantile(s, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.unique(s)
        if len(edges) < 2:
            return 0.0
        b = np.searchsorted(edges, s)
    else:
        b = np.clip(np.digitize(s, edges[1:-1]), 0, len(edges) - 2)
    def H(p):
        p = p[p > 0]; return float(-(p * np.log2(p)).sum())
    Hy = H(np.bincount(y) / len(y))
    hc = sum((b == v).mean() * H(np.bincount(y[b == v], minlength=2) / (b == v).sum())
             for v in np.unique(b))
    return max(0.0, Hy - hc)


def app_fold(app, k=5):
    return int(hashlib.md5(f"cv::{app}".encode()).hexdigest(), 16) % k


def cv_combined(Mraw, y, apps):
    """5-fold CV grouped by app_id hash; NaN scores imputed with the TRAIN fold's medians only;
    metrics constant in the TRAIN fold are dropped for that fold (Codex #8)."""
    folds = np.array([app_fold(a) for a in apps])
    oof = np.zeros(len(y))
    for f in range(5):
        te = folds == f; tr = ~te
        if te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        med = np.nanmedian(Mraw[tr], axis=0)
        med[np.isnan(med)] = 0.0
        Xtr = np.where(np.isnan(Mraw[tr]), med, Mraw[tr])
        Xte = np.where(np.isnan(Mraw[te]), med, Mraw[te])
        keep = Xtr.std(axis=0) > 0
        if not keep.any():
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(Xtr[:, keep], y[tr]); oof[te] = clf.predict_proba(Xte[:, keep])[:, 1]
    return oof


def pairdiff_cv(Mraw, y, uids, apps):
    """Paired-endpoint readout (Codex #3): per claim d = x_gold - x_filler (train-fold-imputed),
    symmetrized logistic on {(d,1),(-d,0)} with app folds; within-claim acc = mean(P(d)>0.5),
    ties (P==0.5) get half credit."""
    idx = collections.defaultdict(dict)
    for i, (u, yy) in enumerate(zip(uids, y)):
        idx[u][yy] = i
    both = [(u, d[1], d[0]) for u, d in idx.items() if 1 in d and 0 in d]
    u2a = {u: a for u, a in zip(uids, apps)}
    folds = np.array([app_fold(u2a[u]) for u, _, _ in both])
    correct = np.zeros(len(both))
    for f in range(5):
        te = folds == f; tr = ~te
        if te.sum() == 0 or tr.sum() < 10:
            continue
        rows_tr = [i for (u, gi, fi), t in zip(both, tr) if t for i in (gi, fi)]
        med = np.nanmedian(Mraw[rows_tr], axis=0)
        med[np.isnan(med)] = 0.0
        M = np.where(np.isnan(Mraw), med, Mraw)
        D = np.array([M[gi] - M[fi] for u, gi, fi in both])
        Xtr = np.vstack([D[tr], -D[tr]])
        ytr = np.r_[np.ones(tr.sum()), np.zeros(tr.sum())]
        keep = Xtr.std(axis=0) > 0
        if not keep.any():
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(Xtr[:, keep], ytr)
        p = clf.predict_proba(D[te][:, keep])[:, 1]
        correct[te] = np.where(p > 0.5, 1.0, np.where(p == 0.5, 0.5, 0.0))
    return float(correct.mean()), len(both)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--testbed", default=TESTBED, help="pair testbed (default v2 = multiple-gold fix)")
    ap.add_argument("--patch", nargs="*", default=[],
                    help="TAG-SCOPED overlays, format tag=path (Codex #2: a patch from one model "
                         "must never overwrite another tag's rows)")
    a = ap.parse_args()
    patches = {}
    for p in a.patch:
        if "=" not in p:
            raise SystemExit(f"--patch must be tag=path (got {p!r}); refusing un-scoped patch")
        t, path = p.split("=", 1)
        patches.setdefault(t, []).append(path)

    # testbed lookup: lexical V baseline + uid->app_id for grouped CV
    tb, u2app = {}, {}
    for ln in open(a.testbed):
        r = json.loads(ln)
        tb[(r["uid"], r["y"])] = r
        u2app[r["uid"]] = str(r.get("app_id") or r["uid"])

    for tag in a.tags:
        fp = f"{OUTDIR}/scores_{tag}.jsonl"
        if not os.path.exists(fp):
            print(f"[{tag}] no scores file, skip", flush=True); continue
        rows = [json.loads(l) for l in open(fp)]
        over = {}
        for pf in patches.get(tag, []):
            for ln in open(pf):
                r = json.loads(ln)
                over[(r["uid"], r["y"], r["metric_id"])] = r["score"]
        if over:
            npatch = 0
            for r in rows:
                k = (r["uid"], r["y"], r["metric_id"])
                if k in over:
                    r["score"] = over[k]; npatch += 1
            print(f"[{tag}] patched {npatch} score rows from {len(patches[tag])} file(s)", flush=True)
        # organize: metric_id -> list of (uid,y,score); and the pair index
        mids = sorted({r["metric_id"] for r in rows})
        dom = {r["metric_id"]: r["domain"] for r in rows}
        # unique pairs in stable order
        pairkeys = sorted({(r["uid"], r["y"]) for r in rows})
        pidx = {k: i for i, k in enumerate(pairkeys)}
        uids = np.array([k[0] for k in pairkeys])
        y = np.array([k[1] for k in pairkeys])
        # raw score matrix [n_pairs x n_metrics]; NaN = unparsed. CV imputes train-fold-only;
        # the globally-imputed copy M is used ONLY for per-metric descriptives (no fitting).
        Mraw = np.full((len(pairkeys), len(mids)), np.nan)
        mcol = {m: j for j, m in enumerate(mids)}
        for r in rows:
            if r["score"] is not None:
                Mraw[pidx[(r["uid"], r["y"])], mcol[r["metric_id"]]] = r["score"]
        M = Mraw.copy()
        for j in range(M.shape[1]):
            col = M[:, j]
            med = np.nanmedian(col) if not np.all(np.isnan(col)) else 0.0
            col[np.isnan(col)] = med
        apps = np.array([u2app.get(u, u) for u in uids])

        print(f"\n===== BACKEND {tag}  ({len(pairkeys)} pairs, {len(mids)} metrics) =====", flush=True)
        # per-metric recovery
        perm = []
        for j, m in enumerate(mids):
            s = M[:, j]
            wc, npair = within_claim_acc(uids, y, s)
            auc = roc_auc_score(y, s) if len(set(s)) > 1 else 0.5
            perm.append((m, dom[m], wc, auc, mi_binned(y, s)))
        perm.sort(key=lambda t: -t[2])
        print("[top metrics by within-claim accuracy]", flush=True)
        for m, d, wc, auc, mi in perm[:12]:
            print(f"  {m} [{d:10s}] within={wc:.3f} pooledAUC={auc:.3f} MI={mi:.3f}", flush=True)
        # domain-mean within-claim
        bydom = collections.defaultdict(list)
        for m, d, wc, auc, mi in perm:
            bydom[d].append(wc)
        print("[within-claim acc by source domain]  " +
              "  ".join(f"{d}={np.mean(v):.3f}(n{len(v)})" for d, v in sorted(bydom.items())), flush=True)

        # V baseline (lexical) on the same pairs
        contain = np.array([len(toks(tb[k]["element"]) & toks(tb[k]["span"])) /
                            max(1, len(toks(tb[k]["element"]))) if k in tb else 0.0
                            for k in pairkeys])
        vwc, _ = within_claim_acc(uids, y, contain)
        print(f"\n[V lexical baseline] within-claim acc={vwc:.3f}  pooledAUC={roc_auc_score(y, contain):.3f}",
              flush=True)

        # tie-mass diagnostics (Codex #3): how much paired signal does the 0-4 scale destroy?
        byu = collections.defaultdict(dict)
        for u, yy, i in zip(uids, y, range(len(uids))):
            byu[u][yy] = i
        pairs2 = [(d[1], d[0]) for d in byu.values() if 1 in d and 0 in d]
        if pairs2:
            # ties computed on RAW scores (both sides present and equal) — imputation would
            # manufacture artificial equalities from missing values (Codex round-2 drift note)
            G = Mraw[[g for g, _ in pairs2]]; F = Mraw[[f for _, f in pairs2]]
            both = ~np.isnan(G) & ~np.isnan(F)
            tie_per_metric = np.where(both, G == F, False).sum(axis=0) / np.maximum(1, both.sum(axis=0))
            ident = (np.where(both, G == F, True)).all(axis=1).mean()
            print(f"[tie mass] per-metric median {np.median(tie_per_metric):.2f} "
                  f"(min {tie_per_metric.min():.2f} / max {tie_per_metric.max():.2f}); "
                  f"identical-full-vector claims {ident:.1%} (raw, non-missing both sides)", flush=True)

        # A combined (all articulated metrics, app-grouped CV logistic)
        oof = cv_combined(Mraw, y, apps)
        awc, _ = within_claim_acc(uids, y, oof)
        aauc = roc_auc_score(y, oof)
        print(f"[A combined bank]    within-claim acc={awc:.3f}  pooledAUC={aauc:.3f}  "
              f"MI={mi_binned(y, oof):.3f}  best-single-metric within={perm[0][2]:.3f}", flush=True)
        pd_acc, pd_n = pairdiff_cv(Mraw, y, uids, apps)
        print(f"[A pair-difference]  within-claim acc={pd_acc:.3f}  (n={pd_n}; logistic on "
              f"x_gold-x_filler, app-folds — paired-endpoint-aligned readout)", flush=True)
        # invented-only combined (if present)
        inv = [j for j, m in enumerate(mids) if dom[m] == "invented"]
        if inv:
            oofi = cv_combined(Mraw[:, inv], y, apps)
            iwc, _ = within_claim_acc(uids, y, oofi)
            print(f"[A invented-only]    within-claim acc={iwc:.3f}  pooledAUC={roc_auc_score(y, oofi):.3f}",
                  flush=True)
        json.dump({"tag": tag, "n_pairs": len(pairkeys), "V_lexical_within": vwc,
                   "A_combined_within": awc, "A_combined_auc": aauc,
                   "A_pairdiff_within": pd_acc, "A_pairdiff_n": pd_n,
                   "best_metric": perm[0][0], "best_within": perm[0][2],
                   "by_domain": {d: float(np.mean(v)) for d, v in bydom.items()}},
                  open(f"{OUTDIR}/recovery_{tag}.json", "w"), indent=1)
    print("\nRECOVERY_DONE", flush=True)


if __name__ == "__main__":
    main()
