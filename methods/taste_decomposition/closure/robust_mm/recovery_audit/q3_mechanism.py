#!/usr/bin/env python3
"""Recovery-audit Q3 + Q4(i): is the fleet a static-prior sampler, and what does
raising P buy?

Tests of the prior-sampling model (proposers draw criteria from a stable personal
prior over plausible quality criteria, insensitive to the bank's actual gaps):

  T1  SPONTANEOUS RECURRENCE OF REDISCOVERIES.  For each judge-matched held-out
      concept, take its matched proposal text and count in how many of the OTHER
      fleets (different slice, different bank state, incl. round5 where nothing
      was depleted) some proposal sits >= tau in the SAME register.  Static prior
      => rediscoveries recur regardless of depletion state.

  T2  SAME-MODEL CROSS-TAG SELF-RECAPTURE vs WITHIN-TAG CROSS-MODEL RECAPTURE.
      Each proposer slot ran on 4 different slices (rep1/2/3, round5).  If slice
      content drove proposals, same-slice-different-model overlap should exceed
      different-slice-same-model overlap.  Static personal priors predict the
      reverse.

  T3  Accumulation shape (from m2_fleet_richness) read against the prior model.

Q4(i): sensitivity-vs-P subset curve from the recall instrument's matched_pids +
beta-binomial extrapolation: what P reaches 70% / 80% on high-strength concepts?

CPU only, embeddings cached.
"""
from __future__ import annotations

import itertools
import json
import sys
from math import lgamma
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RMM = HERE.parent
sys.path.insert(0, str(RMM))
from embed_lib import embed, crit_text, TAU  # noqa: E402

TAGS = ("rep1", "rep2", "rep3", "round5")
recall = json.loads((RMM / "m3_recall.json").read_text())

allp = []
for tag in TAGS:
    d = json.loads((RMM / f"proposals_{tag}.json").read_text())
    for p in d["proposals"]:
        allp.append({"tag": tag, "pid": p["pid"], "proposer": p["proposer"],
                     "family": p["family"], "name": p["name"],
                     "text": crit_text(p["name"], p["instruction"])})

E = embed([p["text"] for p in allp], verbose=True)
S = E @ E.T
idx = {(p["tag"], p["pid"]): i for i, p in enumerate(allp)}

out = {"tau": TAU, "n_proposals_total": len(allp)}

# ---- T1: do the matched proposals recur in fleets that had no depletion? ----
t1 = []
for r in recall["records"]:
    if r["kind"] != "heldout" or not r["match_primary"]:
        continue
    for pid in r["matched_pids"]:
        i = idx[(r["rep"], pid)]
        rec = {"concept": r["concept"], "rep": r["rep"], "pid": pid,
               "recurs_in": {}}
        for tag in TAGS:
            if tag == r["rep"]:
                continue
            js = [j for j, q in enumerate(allp) if q["tag"] == tag]
            best = float(S[i, js].max())
            rec["recurs_in"][tag] = {"max_cos": round(best, 3), "ge_tau": bool(best >= TAU)}
        rec["n_other_fleets_ge_tau"] = sum(v["ge_tau"] for v in rec["recurs_in"].values())
        rec["recurs_in_round5_fullbank"] = rec["recurs_in"]["round5"]["ge_tau"]
        t1.append(rec)

out["T1_spontaneous_recurrence"] = {
    "n_matched_proposals": len(t1),
    "frac_recur_ge1_other_fleet": float(np.mean([r["n_other_fleets_ge_tau"] >= 1 for r in t1])),
    "frac_recur_in_round5_fullbank": float(np.mean([r["recurs_in_round5_fullbank"] for r in t1])),
    "mean_max_cos_round5": float(np.mean([r["recurs_in"]["round5"]["max_cos"] for r in t1])),
    "detail": t1,
}

# ---- T2: same-model-cross-slice vs same-slice-cross-model recapture ---------
# recapture(a -> B): frac of a's proposals with a >= tau neighbour in set B
def recap(ai, bi):
    if not len(ai) or not len(bi):
        return None
    sub = S[np.ix_(ai, bi)]
    return float((sub.max(axis=1) >= TAU).mean())

slots = sorted({p["proposer"] for p in allp})
same_model_cross_tag, cross_model_same_tag = [], []
for s in slots:
    for ta, tb in itertools.permutations(TAGS, 2):
        ai = [i for i, p in enumerate(allp) if p["proposer"] == s and p["tag"] == ta]
        bi = [i for i, p in enumerate(allp) if p["proposer"] == s and p["tag"] == tb]
        r = recap(ai, bi)
        if r is not None:
            same_model_cross_tag.append(r)
for tag in TAGS:
    for sa, sb in itertools.permutations(slots, 2):
        ai = [i for i, p in enumerate(allp) if p["proposer"] == sa and p["tag"] == tag]
        bi = [i for i, p in enumerate(allp) if p["proposer"] == sb and p["tag"] == tag]
        r = recap(ai, bi)
        if r is not None:
            cross_model_same_tag.append(r)

out["T2_recapture"] = {
    "same_model_DIFFERENT_slice_mean": float(np.mean(same_model_cross_tag)),
    "same_model_DIFFERENT_slice_n": len(same_model_cross_tag),
    "different_model_SAME_slice_mean": float(np.mean(cross_model_same_tag)),
    "different_model_SAME_slice_n": len(cross_model_same_tag),
    "note": "static personal priors predict same-model-cross-slice >= cross-model-same-slice; "
            "slice-driven (gap-tracking) proposals predict the reverse",
}

# per-slot self-recapture (personal-prior stability by model)
out["T2_per_slot_self_recapture_cross_tag"] = {}
for s in slots:
    vals = []
    for ta, tb in itertools.permutations(TAGS, 2):
        ai = [i for i, p in enumerate(allp) if p["proposer"] == s and p["tag"] == ta]
        bi = [i for i, p in enumerate(allp) if p["proposer"] == s and p["tag"] == tb]
        r = recap(ai, bi)
        if r is not None:
            vals.append(r)
    if vals:
        out["T2_per_slot_self_recapture_cross_tag"][s] = float(np.mean(vals))

# ---- Q4(i): sensitivity vs P, subset curve + beta-binomial extrapolation ----
P_SLOTS = ["claude_sonnet", "claude_opus", "codex_luna_a", "codex_luna_b"]
targets = [r for r in recall["records"]]
def catch_curve(rows_sel):
    curve = {}
    for p in range(1, 5):
        vals = []
        for combo in itertools.combinations(P_SLOTS, p):
            cs = set(combo)
            vals.append(np.mean([bool(set(r["proposers"]) & cs) for r in rows_sel]))
        curve[p] = float(np.mean(vals))
    return curve

hi_held = [r for r in targets if r["kind"] == "heldout" and r["stratum"] == "high"]
hi_all = [r for r in targets if r["stratum"] == "high"]
all_held = [r for r in targets if r["kind"] == "heldout"]
out["Q4i_subset_curves"] = {
    "heldout_high_n9": catch_curve(hi_held),
    "all_high_pooled_n18": catch_curve(hi_all),
    "heldout_all_n24": catch_curve(all_held),
}

# beta-binomial MLE on per-target counts k/4 (distinct catching proposers)
def bb_fit(ks, n=4):
    def ll(a, b):
        s = 0.0
        for k in ks:
            s += (lgamma(a + k) + lgamma(b + n - k) - lgamma(a + b + n)
                  + lgamma(a + b) - lgamma(a) - lgamma(b))
        return s
    best, arg = -1e18, None
    for la in np.linspace(-4, 6, 201):
        for lb in np.linspace(-2, 8, 201):
            v = ll(np.exp(la), np.exp(lb))
            if v > best:
                best, arg = v, (np.exp(la), np.exp(lb))
    return arg


def zib_fit(ks, n=4):
    """Zero-inflated binomial: with prob pi the concept is nameable (catch prob q
    per proposer), else structurally unnameable (q=0)."""
    from math import comb, log
    best, arg = -1e18, None
    for pi in np.linspace(0.05, 1.0, 96):
        for q in np.linspace(0.01, 0.99, 99):
            s = 0.0
            for k in ks:
                p_k = pi * comb(n, k) * q ** k * (1 - q) ** (n - k)
                if k == 0:
                    p_k += (1 - pi)
                s += log(max(p_k, 1e-300))
            if s > best:
                best, arg = s, (pi, q)
    return arg

def sens_at_P(a, b, P):
    # 1 - E[(1-q)^P], q ~ Beta(a,b)
    return 1.0 - np.exp(lgamma(b + P) + lgamma(a + b) - lgamma(b) - lgamma(a + b + P))

for label, rows_sel in (("high_pooled_n18", hi_all), ("heldout_high_n9", hi_held)):
    ks = [len(set(r["proposers"])) for r in rows_sel]
    a, b = bb_fit(ks)
    proj = {P: float(sens_at_P(a, b, P)) for P in (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64)}
    need70 = next((P for P in range(1, 500) if sens_at_P(a, b, P) >= .70), None)
    need80 = next((P for P in range(1, 500) if sens_at_P(a, b, P) >= .80), None)
    out[f"Q4i_betabinom_{label}"] = {
        "counts_k_of_4": sorted(ks, reverse=True), "alpha": float(a), "beta": float(b),
        "mean_q": float(a / (a + b)),
        "projected_sensitivity_by_P": proj,
        "P_needed_70pct": need70, "P_needed_80pct": need80,
        "caveat": "extrapolation assumes proposers exchangeable with the observed 4 and "
                  "catch events independent across proposers; concepts with q=0 (never "
                  "nameable from this corpus register) bound the asymptote below 1.0",
    }
    pi, q = zib_fit(ks)
    zproj = {P: float(pi * (1 - (1 - q) ** P)) for P in (1, 2, 4, 6, 8, 12, 16, 24, 32, 64)}
    zneed70 = next((P for P in range(1, 500) if pi * (1 - (1 - q) ** P) >= .70), None)
    zneed80 = next((P for P in range(1, 500) if pi * (1 - (1 - q) ** P) >= .80), None)
    out[f"Q4i_zeroinflated_{label}"] = {
        "pi_nameable": float(pi), "q_per_proposer": float(q),
        "asymptote_P_inf": float(pi),
        "projected_sensitivity_by_P": zproj,
        "P_needed_70pct": zneed70, "P_needed_80pct": zneed80,
        "note": "alternative model: a pi-fraction of concepts is nameable at all; "
                "raising P saturates at pi.  The P=4 data cannot separate this from "
                "the beta-binomial; the miss autopsy (q2b) arbitrates which zeros are "
                "structural (out-of-register) vs sampling zeros.",
    }

# empirical asymptote bound: frac of high concepts with k=0 across all 4 proposers
for label, rows_sel in (("high_pooled_n18", hi_all), ("heldout_high_n9", hi_held),
                        ("heldout_all_n24", all_held)):
    k0 = float(np.mean([len(r["proposers"]) == 0 for r in rows_sel]))
    out.setdefault("Q4i_zero_catch_frac", {})[label] = k0

(HERE / "q3_mechanism.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items()
                  if k not in ("T1_spontaneous_recurrence",)}, indent=1))
print("\nT1 summary:", json.dumps({k: v for k, v in out["T1_spontaneous_recurrence"].items()
                                   if k != "detail"}, indent=1))
for r in out["T1_spontaneous_recurrence"]["detail"]:
    print(f"  {r['rep']} {r['pid']:18s} recurs_ge_tau in {r['n_other_fleets_ge_tau']}/3 "
          f"(round5 full-bank: {r['recurs_in_round5_fullbank']}, cos {r['recurs_in']['round5']['max_cos']}) "
          f"| {r['concept'][:50]}")
