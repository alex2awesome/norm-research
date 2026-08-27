#!/usr/bin/env python3
"""Confound hunt for the gold-bib-soft AUC .799 (matched vs SHIFT-mismatched):
(1) TOPIC confound: shift-mismatched pairs are cross-topic -> grader may score topical
    proximity, not anticipation. Test: HARD negatives = for each claim, the most
    lexically-similar OTHER gold. If AUC collapses on hard negatives, .799 is topic.
(2) LEXICAL name-leak: claims often name the prior method (Grad-CAM). Test: CODE
    overlap-only AUC (claim tokens in gold title+abstract). If overlap alone ~.8,
    the LLM adds nothing beyond string match.
Run on sk3: python -m methods.claim_verification.run_gold_bib_confound"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from claim_verification.core import Cache
from claim_verification.run_gold_bib_soft import build_resolved, grade, OUTD
from claim_verification.run_gold_openalex import GENERIC
from claim_verification.seam_metrics import _toks

def overlap(claim, gold, title_only=False):
    gt = str(gold["title"]) if title_only else f"{gold['title']} {gold.get('abstract','')}"
    ct = [w for w in _toks(claim) if len(w) >= 4 and w not in GENERIC]
    gtk = set(_toks(gt))
    return sum(1 for w in ct if w in gtk) / max(len(ct), 1)

def main():
    cache = Cache(f"{OUTD}/oa_cache2.jsonl")
    gcache = Cache(f"{OUTD}/graded_soft_cache.jsonl")
    R = build_resolved(cache)
    n = len(R)
    print(f"[conf] resolved pairs: {n}", flush=True)
    half = max(1, n // 2)
    arms = {}
    arms["matched"] = [(i, i) for i in range(n)]
    arms["shift_mm"] = [(i, (i + half) % n) for i in range(n)]
    # hard negatives: most lexically similar OTHER gold
    hard = []
    for i in range(n):
        sims = [(overlap(R[i]["claim"], R[j]), j) for j in range(n) if j != i]
        hard.append((i, max(sims)[1]))
    arms["hard_mm"] = hard
    res = {}
    for arm, pairs in arms.items():
        scores, ovs, ovt = [], [], []
        for i, j in pairs:
            g = grade(R[i]["claim"], R[j], gcache)
            if g["score"] >= 0:
                scores.append(g["score"])
                ovs.append(overlap(R[i]["claim"], R[j]))
                ovt.append(overlap(R[i]["claim"], R[j], title_only=True))
        res[arm] = (scores, ovs, ovt)
        print(f"  {arm:9} n={len(scores):3d} score mean={np.mean(scores):.2f} "
              f"med={np.median(scores):.1f} | overlap(abs) mean={np.mean(ovs):.3f} "
              f"| overlap(title) mean={np.mean(ovt):.3f}", flush=True)
    m_s, m_o, m_t = res["matched"]
    for mm in ("shift_mm", "hard_mm"):
        s, o, t = res[mm]
        y = [1] * len(m_s) + [0] * len(s)
        print(f"\n  vs {mm}:", flush=True)
        print(f"    LLM graded score      AUC={roc_auc_score(y, m_s + s):.4f}", flush=True)
        print(f"    CODE overlap(abstract) AUC={roc_auc_score(y, m_o + o):.4f}", flush=True)
        print(f"    CODE overlap(title)    AUC={roc_auc_score(y, m_t + t):.4f}", flush=True)
    rho = spearmanr(m_s, m_o).statistic
    print(f"\n  within-matched corr(score, overlap) rho={rho:+.3f}", flush=True)
    # does the LLM score separate matched from hard_mm AMONG LOW-OVERLAP pairs only?
    lo_m = [s for s, o in zip(m_s, m_o) if o < np.median(m_o)]
    h_s, h_o, _ = res["hard_mm"]
    lo_h = [s for s, o in zip(h_s, h_o) if o < np.median(m_o)]
    if len(lo_m) > 5 and len(lo_h) > 5:
        y = [1] * len(lo_m) + [0] * len(lo_h)
        print(f"  low-overlap-only LLM AUC (matched vs hard) = "
              f"{roc_auc_score(y, lo_m + lo_h):.4f} (n={len(lo_m)}+{len(lo_h)})", flush=True)
    print("GOLD_BIB_CONFOUND_DONE", flush=True)

if __name__ == "__main__":
    main()
