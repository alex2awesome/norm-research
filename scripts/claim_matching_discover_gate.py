#!/usr/bin/env python3
"""Two-stage gate for discovered claim-matching metrics (BEST-PRACTICES [metric inference]).

Inputs (all in outputs/claim_matching/):
  scores_discovered.jsonl  : discovered candidates scored (Gemma) on the 800-claim probe
  scores_gemma3_4b.jsonl   : the 80-metric BANK scored on the same probe (for bank-combined + residual)
  discover_splits.json     : app_id -> propose|stage1|stage2  (app-disjoint)
Gate:
  STAGE-1 (suggestive): candidate within-claim accuracy on stage1 claims (and on the stage1 RESIDUAL
                        = claims the bank ranks wrong) — in-run, overfits its split, not a finding.
  STAGE-2 (evidence): strictly-disjoint replication; binomial p vs 0.5 per candidate, Bonferroni over
                      the candidate set. A metric counts only if it replicates here.
  RESIDUAL-OVER-BANK: does adding discovered metrics to the bank raise held-out (stage2) within-claim
                      accuracy beyond the bank alone? (else it just rewords bank signal). Also reports
                      recovery specifically on stage2-residual claims (can discovery FIX bank misses?).
Run on sk3 (CPU)."""
import json, collections, os
import numpy as np
from scipy.stats import binomtest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import hashlib

BASE = "/lfs/skampere3/0/alexspan/norm-research"
OUTDIR = f"{BASE}/outputs/claim_matching"
TESTBED = f"{BASE}/datasets/claim-matching/testbed/pair_testbed.jsonl"


def load_scores(fp):
    """metric_id -> {(uid,y): score}; also uid set."""
    d = collections.defaultdict(dict)
    for ln in open(fp):
        r = json.loads(ln)
        if r["score"] is not None:
            d[r["metric_id"]][(r["uid"], r["y"])] = r["score"]
    return d


def within_acc(claims, scoref):
    """claims: list of uid. scoref: (uid,y)->score. returns (acc, n)."""
    acc, n = 0.0, 0
    for u in claims:
        g, f = scoref.get((u, 1)), scoref.get((u, 0))
        if g is None or f is None:
            continue
        n += 1
        acc += 1.0 if g > f else 0.5 if g == f else 0.0
    return acc / max(1, n), n


def wins(claims, scoref):
    """count (wins, ties, n) for binomial test (ties excluded)."""
    w = t = n = 0
    for u in claims:
        g, f = scoref.get((u, 1)), scoref.get((u, 0))
        if g is None or f is None:
            continue
        if g > f:
            w += 1
        elif g == f:
            t += 1
        n += 1
    return w, t, n


def main():
    uid_app = {}
    for ln in open(TESTBED):
        r = json.loads(ln); uid_app[r["uid"]] = r["app_id"]
    splits = json.load(open(f"{OUTDIR}/discover_splits.json"))  # app -> split
    disc = load_scores(f"{OUTDIR}/scores_discovered.jsonl")
    bank = load_scores(f"{OUTDIR}/scores_gemma3_4b.jsonl")

    # bank combined score per (uid,y) = mean over bank metrics
    bankcomb = {}
    allpairs = set()
    for m in bank:
        for k in bank[m]:
            allpairs.add(k)
    for k in allpairs:
        vs = [bank[m][k] for m in bank if k in bank[m]]
        if vs:
            bankcomb[k] = float(np.mean(vs))

    claims = sorted({u for (u, y) in bankcomb})
    split_of = lambda u: splits.get(uid_app.get(u, ""), "propose")
    s1 = [u for u in claims if split_of(u) == "stage1"]
    s2 = [u for u in claims if split_of(u) == "stage2"]
    bank_hard = lambda u: bankcomb.get((u, 1), 0) <= bankcomb.get((u, 0), 0)
    s1r = [u for u in s1 if bank_hard(u)]
    s2r = [u for u in s2 if bank_hard(u)]
    print(f"[gate] stage1 {len(s1)} ({len(s1r)} bank-hard)  stage2 {len(s2)} ({len(s2r)} bank-hard)",
          flush=True)
    print(f"[gate] {len(disc)} discovered candidates\n", flush=True)

    # per-candidate stage-1 then stage-2, Bonferroni over candidates
    m = len(disc)
    alpha = 0.05 / max(1, m)
    survivors = []
    print(f"{'metric':8s} {'s1_all':>7s} {'s1_hard':>7s} {'s2_all':>7s} {'s2_hard':>7s} {'s2_p':>9s} {'verdict'}")
    for mid in sorted(disc):
        sf = disc[mid]
        a1, _ = within_acc(s1, sf); a1h, _ = within_acc(s1r, sf)
        a2, _ = within_acc(s2, sf); a2h, _ = within_acc(s2r, sf)
        w, t, n = wins(s2, sf)
        p = binomtest(w, n - t, 0.5, alternative="greater").pvalue if (n - t) > 0 else 1.0
        ok = (a1 > 0.52) and (p < alpha) and (a2 > 0.5)
        if ok:
            survivors.append(mid)
        print(f"{mid:8s} {a1:7.3f} {a1h:7.3f} {a2:7.3f} {a2h:7.3f} {p:9.2e} "
              f"{'CONFIRM' if ok else ''}", flush=True)
    print(f"\n[gate] {len(survivors)}/{m} survive stage-2 Bonferroni (alpha={alpha:.2e}): {survivors}",
          flush=True)

    # residual-over-bank: bank vs bank+discovered combined, held-out stage2 (app-CV logistic)
    def fold(u):
        return int(hashlib.md5(f"g::{u}".encode()).hexdigest(), 16) % 5

    def combined_within(metric_sets, claim_list):
        # build [pair x feature] for gold(1) and filler(0), predict prob, within-claim acc via OOF
        feats = list(metric_sets)
        rows, ys, us = [], [], []
        for u in claim_list:
            for y in (1, 0):
                vec = []
                for (src, mid) in feats:
                    vec.append(src.get(mid, {}).get((u, y), np.nan))
                rows.append(vec); ys.append(y); us.append(u)
        X = np.array(rows, float)
        col_med = np.nanmedian(np.where(np.isnan(X), np.nan, X), axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_med, inds[1])
        y = np.array(ys); folds = np.array([fold(u) for u in us])
        oof = np.zeros(len(y))
        for f in range(5):
            te = folds == f; tr = ~te
            if len(set(y[tr])) < 2:
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
        scoref = {(us[i], ys[i]): oof[i] for i in range(len(us))}
        return within_acc(claim_list, scoref)

    bank_feats = [(bank, mid) for mid in bank]
    disc_feats_all = [(disc, mid) for mid in disc]
    disc_feats_surv = [(disc, mid) for mid in survivors]
    ba, _ = combined_within(bank_feats, s2)
    bda, _ = combined_within(bank_feats + disc_feats_all, s2)
    print(f"\n[residual-over-bank | stage2 held-out]", flush=True)
    print(f"  bank alone           within-claim acc = {ba:.3f}", flush=True)
    print(f"  bank + ALL discovered                 = {bda:.3f}  (delta {bda-ba:+.3f})", flush=True)
    if survivors:
        bds, _ = combined_within(bank_feats + disc_feats_surv, s2)
        print(f"  bank + CONFIRMED disc                 = {bds:.3f}  (delta {bds-ba:+.3f})", flush=True)
    # can discovery fix bank misses? within-acc of discovered-combined on stage2-residual
    if disc_feats_all and s2r:
        dr, nr = combined_within(disc_feats_all, s2r)
        print(f"  discovered-combined on stage2-RESIDUAL (bank-hard, n={nr}) = {dr:.3f} "
              f"(0.5=no rescue)", flush=True)
    print("DISCOVER_GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
