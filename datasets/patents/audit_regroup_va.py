#!/usr/bin/env python3
"""Audit: re-measure the patents claim-level V/A grouped AUCs (the hardcoded .591/.616).

The notebook (2026-07-01__patents-laws-VA-decomposition) reports PAT_V=.591/PAT_A=.616 as
constants "measured on sk3 (grouped-5fold by app)" — the computing script was never checked in,
and patents_va_features.csv lacks app_id so the grouping is unreproducible from the CSV alone.
This script closes that hole:
  1. row-aligns the CSV with option3_claims_gemma_scale.jsonl (verified on shared fields),
     attaching app_id + a claim-element dedup key;
  2. recomputes grouped-5fold-by-app AUC with the notebook's own grouped_auc code, verbatim;
  3. recomputes after dedup (12.2% exact-duplicate rows; drops contradictory-label dup groups).

Run ON sk3 (CPU only): python3 datasets/patents/audit_regroup_va.py
Needs: notebooks/data/patents_va_features.csv rsynced to the sk3 mirror.
"""
import csv, hashlib, json, pathlib
import numpy as np

BASE = pathlib.Path("/lfs/skampere3/0/alexspan/norm-research")
CSV = BASE / "notebooks/data/patents_va_features.csv"
JL = BASE / "datasets/patents/processed/option3_claims_gemma_scale.jsonl"

V_COLS = ['v_max_lexoverlap', 'v_mean_lexoverlap', 'v_count_lexhit', 'v_element_wordlen',
          'v_n_refs', 'v_max_spanlen', 'v_mean_spanlen']
A_COLS = V_COLS + ['a_n_disclose', 'a_any_disclose', 'a_frac_disclose', 'a_max_disclose_overlap']


# ---- notebook's own auc/grouped_auc, verbatim ----
def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    order = np.argsort(s, kind='mergesort'); sr = s[order]
    rk = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sr[j + 1] == sr[i]:
            j += 1
        rk[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    p = (y == 1).sum(); n = (y == 0).sum()
    return (rk[y == 1].sum() - p * (p + 1) / 2) / (p * n)


def grouped_auc(X, y, g):
    uniq = np.array(sorted(set(g))); folds = np.array_split(uniq, 5)
    sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    oof = np.zeros(len(y))
    for k in range(5):
        te = set(folds[k]); m = np.array([x in te for x in g]); tr = ~m
        Xt = X[tr]; mu = Xt.mean(0); sd = Xt.std(0) + 1e-8
        Xb = np.c_[np.ones(len(Xt)), (Xt - mu) / sd]; w = np.zeros(Xb.shape[1])
        for _ in range(2500):
            p = sig(Xb @ w); w -= 0.3 * (Xb.T @ (p - y[tr]) / len(Xt) + 1e-2 * np.r_[0, w[1:]])
        oof[m] = sig(np.c_[np.ones(int(m.sum())), (X[m] - mu) / sd] @ w)
    return auc(y, oof)


def main():
    rows = list(csv.DictReader(open(CSV)))
    print(f"[csv] {len(rows)} rows", flush=True)

    jkeys = []  # (app_id, claim_num, element-md5, label, rejtype, n_refs, n_disclose, gold_disclose)
    with open(JL) as fh:
        for ln in fh:
            r = json.loads(ln)
            jkeys.append((str(r["app_id"]), str(r["claim_num"]),
                          hashlib.md5(r["element"].encode()).hexdigest(),
                          r["label"], str(r.get("rejection_type")),
                          int(r["n_refs"]), int(r["n_disclose"]), bool(r["gold_disclose"])))
    print(f"[jsonl] {len(jkeys)} rows", flush=True)
    assert len(rows) == len(jkeys), "row-count mismatch — alignment impossible"

    # alignment check on every shared field
    mism = 0
    for c, j in zip(rows, jkeys):
        ok = (int(float(c["fell"])) == (1 if j[3] == "pos" else 0)
              and int(float(c["v_n_refs"])) == j[5]
              and int(float(c["a_n_disclose"])) == j[6]
              and int(float(c["gold_disclose"])) == int(j[7]))
        mism += not ok
    print(f"[align] mismatched rows: {mism}/{len(rows)} "
          f"({'ALIGNED — app_id attach is valid' if mism == 0 else 'NOT ALIGNED — stop'})", flush=True)
    if mism:
        return

    X_all = np.array([[float(c[col]) for col in A_COLS] for c in rows])
    y = np.array([float(c["fell"]) for c in rows])
    g = np.array([j[0] for j in jkeys])
    iV = [A_COLS.index(c) for c in V_COLS]

    print("\n=== FULL DATA (as originally measured, now grouped-by-app for real) ===", flush=True)
    v_full = grouped_auc(X_all[:, iV], y, g)
    a_full = grouped_auc(X_all, y, g)
    print(f"  V={v_full:.3f} (reported .591)   A={a_full:.3f} (reported .616)   gap={a_full - v_full:+.3f}",
          flush=True)

    # dedup on (app_id, claim_num, element): keep first; drop contradictory-label groups entirely
    from collections import defaultdict
    groups = defaultdict(list)
    for i, j in enumerate(jkeys):
        groups[(j[0], j[1], j[2])].append(i)
    keep, dropped_dup, dropped_contra = [], 0, 0
    for idxs in groups.values():
        labels = {jkeys[i][3] for i in idxs}
        if len(labels) > 1:
            dropped_contra += len(idxs)
            continue
        keep.append(idxs[0])
        dropped_dup += len(idxs) - 1
    keep = np.array(sorted(keep))
    print(f"\n=== DEDUPED (drop {dropped_dup} exact dups + {dropped_contra} contradictory-label rows) ===",
          flush=True)
    print(f"  n={len(keep)} (from {len(rows)}), pos={int(y[keep].sum())}", flush=True)
    v_dd = grouped_auc(X_all[keep][:, iV], y[keep], g[keep])
    a_dd = grouped_auc(X_all[keep], y[keep], g[keep])
    print(f"  V={v_dd:.3f}   A={a_dd:.3f}   gap={a_dd - v_dd:+.3f}", flush=True)

    print("\n=== univariate sanity (expect v_mean_lexoverlap .583, a_n_disclose .571) ===", flush=True)
    for col in ("v_mean_lexoverlap", "a_n_disclose"):
        s = X_all[:, A_COLS.index(col)]
        print(f"  {col:22s} full={auc(y, s):.3f}  dedup={auc(y[keep], s[keep]):.3f}", flush=True)
    print("AUDIT_REGROUP_DONE", flush=True)


if __name__ == "__main__":
    main()
