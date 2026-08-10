#!/usr/bin/env python3
"""ROUND-0 AUDIT, part 3: is the y definition clean for a PRIOR-ART reading task?

y ("fell") = 1 iff the claim element was rejected under SOME ground. The text the
dense reader is given is 8 candidate PRIOR-ART references. But only §102
(anticipation) and §103 (obviousness) rejections are prior-art rejections at all:
a §112 rejection is about claim definiteness/enablement, a §101 rejection is about
patent-eligible subject matter, and a double-patenting rejection is over the
applicant's OWN earlier claims. For those positives the 8 retrieved references are
irrelevant to the label BY CONSTRUCTION.

If the dense model separates non-prior-art positives from negatives just as well as
it separates prior-art positives, it is not reading the references for disclosure --
it is reading something about the claim itself.

Also computes:
  * dense AUC matched on claim ordinal number alone (exact-match strata)
  * dense AUC within claim_num strata, per rejection family
  * a claim-number-only "no text" benchmark inside each rejection family

CPU only. Run on sk3.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path("/lfs/skampere3/0/alexspan/norm-research")
JL = BASE / "datasets/patents/processed/option3_claims_gemma_scale.jsonl"
DS = BASE / "datasets/patents/dense_standard"
OUT = Path(__file__).resolve().parent
csv.field_size_limit(sys.maxsize)
PRIOR_ART = {"102", "103"}
NON_PRIOR_ART = {"112", "101", "DoublePatent", "Other", "NONE"}


def build_text(r):
    parts = [f"CLAIM ELEMENT:\n{r['element']}"]
    for i, ref in enumerate(r.get("refs") or []):
        parts.append(f"REFERENCE {i + 1} (patent {ref.get('doc_id', '?')}):\n"
                     f"{' '.join(ref.get('spans') or [])}")
    return "\n\n".join(parts)


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float("nan")


def exact_match_auc(y, score, key, n_draw=200000, seed=0):
    """Concordance over (pos, neg) pairs drawn from the SAME key stratum."""
    rng = np.random.default_rng(seed)
    by = defaultdict(lambda: ([], []))
    for i, (yy, k) in enumerate(zip(y, key)):
        by[k][int(yy)].append(i)
    keys = [k for k, (n, p) in by.items() if n and p]
    if not keys:
        return None, 0
    w = np.array([len(by[k][0]) * len(by[k][1]) for k in keys], float)
    w /= w.sum()
    pick = rng.choice(len(keys), n_draw, p=w)
    a = np.array([by[keys[j]][1][rng.integers(len(by[keys[j]][1]))] for j in pick])
    b = np.array([by[keys[j]][0][rng.integers(len(by[keys[j]][0]))] for j in pick])
    s = np.asarray(score, float)
    conc = (s[a] > s[b]).astype(float) + 0.5 * (s[a] == s[b])
    return round(float(conc.mean()), 4), int(n_draw)


def main():
    jrows = [json.loads(l) for l in open(JL) if l.strip()]
    th = defaultdict(list)
    for i, r in enumerate(jrows):
        th[hashlib.sha1(build_text(r).encode()).hexdigest()].append(i)
    ptr = defaultdict(int)
    R = {"note": "positives partitioned by statutory rejection ground; the 8 candidate "
                 "references are relevant only to 102/103."}
    for split in ("eval", "test"):
        d = pd.read_csv(DS / "split" / f"{split}.csv")
        ix = []
        for t in d["text"].astype(str).values:
            h = hashlib.sha1(t.encode()).hexdigest(); lst = th[h]; k = ptr[h]
            ix.append(lst[k] if k < len(lst) else lst[-1]); ptr[h] = k + 1
        rt = np.array([str(jrows[j].get("rejection_type")) for j in ix])
        cn = np.array([int(jrows[j]["claim_num"]) if str(jrows[j]["claim_num"]).lstrip("-").isdigit()
                       else -1 for j in ix])
        nd = np.array([int(jrows[j].get("n_disclose") or 0) for j in ix])
        gd = np.array([int(bool(jrows[j].get("gold_disclose"))) for j in ix])
        y = d["judgement"].to_numpy()
        p = pd.read_csv(DS / f"rm_out_seed42/preds_{split}.csv")["prob"].to_numpy()
        neg = y == 0
        S = {"n": int(len(y)), "AUC_all": round(auc(y, p), 4)}
        for nm, keep in (("prior_art_102_103", np.isin(rt, list(PRIOR_ART))),
                         ("NON_prior_art_112_101_DP_other", np.isin(rt, list(NON_PRIOR_ART)))):
            m = keep | neg
            S[nm] = {"n_pos": int(keep.sum()), "n_neg": int(neg.sum()),
                     "dense_AUC": round(auc(y[m], p[m]), 4),
                     "claim_num_only_AUC": round(auc(y[m], -cn[m]), 4),
                     "dense_AUC_matched_on_claim_num": exact_match_auc(y[m], p[m], cn[m])[0],
                     "mean_claim_num_pos": round(float(cn[keep].mean()), 2)}
        for t in ("102", "103", "112", "101", "DoublePatent", "Other"):
            k = rt == t
            if k.sum() < 40:
                continue
            m = k | neg
            S[f"only_{t}"] = {"n_pos": int(k.sum()),
                              "dense_AUC": round(auc(y[m], p[m]), 4),
                              "claim_num_only_AUC": round(auc(y[m], -cn[m]), 4),
                              "dense_AUC_matched_on_claim_num": exact_match_auc(y[m], p[m], cn[m])[0]}
        # whole-split claim-number-matched readouts
        S["dense_AUC_matched_on_claim_num"] = exact_match_auc(y, p, cn)[0]
        S["dense_AUC_matched_on_claim_num_x_rejfamily"] = None
        # does the model track the ACTUAL disclosure evidence at all?
        S["disclosure_evidence"] = {
            "n_disclose_alone_AUC": round(auc(y, nd), 4),
            "gold_disclose_alone_AUC": round(auc(y, gd), 4),
            "spearman_dense_vs_n_disclose": round(float(
                pd.Series(p).corr(pd.Series(nd), method="spearman")), 4),
            "spearman_dense_vs_claim_num": round(float(
                pd.Series(p).corr(pd.Series(cn), method="spearman")), 4),
            "dense_AUC_among_pos_only__predicting_gold_disclose": round(
                auc(gd[y == 1], p[y == 1]), 4),
        }
        # positives with ZERO disclosing references: pure "no evidence" positives
        z = (y == 1) & (nd == 0)
        S["positives_with_zero_disclosing_refs"] = {
            "n": int(z.sum()), "frac_of_pos": round(float(z.sum() / max((y == 1).sum(), 1)), 4),
            "mean_dense_prob": round(float(p[z].mean()), 4),
            "mean_dense_prob_on_negatives": round(float(p[neg].mean()), 4),
            "AUC_these_pos_vs_all_neg": round(auc(np.r_[np.ones(z.sum()), np.zeros(neg.sum())],
                                                  np.r_[p[z], p[neg]]), 4)}
        R[split] = S
    json.dump(R, open(OUT / "round0_ydef_probe.json", "w"), indent=2)
    print(json.dumps(R, indent=2), flush=True)
    print("ROUND0_YDEF_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
