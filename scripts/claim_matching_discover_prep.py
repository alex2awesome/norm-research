#!/usr/bin/env python3
"""Prep for metric DISCOVERY on claim-matching: find the residual the current bank misses, and
build disjoint propose/stage1/stage2 splits + a contrastive proposer input.

Discovery discipline (BEST-PRACTICES [metric inference]):
  - Discover on the RESIDUAL: claims where the 80-metric bank's combined score fails to rank the
    examiner's gold span above the filler -> that is where new metrics must add signal.
  - Splits are app-disjoint: PROPOSE (proposer sees these labeled pairs) / STAGE1 (in-run gate) /
    STAGE2 (strictly-held-out replication). No app crosses splits.
  - The proposer may see labeled contrastive pairs (to hypothesize discriminators); the resulting
    metric is applied label-free at scoring and must pass the residual-over-bank gate. Content guard
    is stated in the proposer prompt (no length/formatting/metadata).

Reads scores_gemma3_4b.jsonl (bank per-pair scores) + the testbed. Writes:
  outputs/claim_matching/discover_splits.json   (app_id -> propose|stage1|stage2)
  outputs/claim_matching/proposer_input.json    (hard+easy contrastive claims for the proposer)
Run on sk3 (CPU)."""
import json, hashlib, collections, os
import numpy as np

BASE = "/lfs/skampere3/0/alexspan/norm-research"
TESTBED = f"{BASE}/datasets/claim-matching/testbed/pair_testbed.jsonl"
SCORES = f"{BASE}/outputs/claim_matching/scores_gemma3_4b.jsonl"
OUTDIR = f"{BASE}/outputs/claim_matching"


def split_of(app):
    h = int(hashlib.md5(f"disc::{app}".encode()).hexdigest(), 16) % 100
    return "propose" if h < 40 else "stage1" if h < 70 else "stage2"


def main():
    # testbed: uid -> {element, spans by y, app}
    byuid = collections.defaultdict(dict)
    meta = {}
    for ln in open(TESTBED):
        r = json.loads(ln)
        byuid[r["uid"]][r["y"]] = r["span"]
        meta[r["uid"]] = {"element": r["element"], "app": r["app_id"]}

    # bank combined score per (uid,y): mean of the 80 metric scores (proxy for "bank thinks match")
    sc = collections.defaultdict(list)
    for ln in open(SCORES):
        r = json.loads(ln)
        if r["score"] is not None:
            sc[(r["uid"], r["y"])].append(r["score"])
    bankscore = {k: float(np.mean(v)) for k, v in sc.items() if v}

    # per claim: does the bank rank gold>filler? residual = it does NOT
    rows = []
    for uid, d in byuid.items():
        if 0 not in d or 1 not in d:
            continue
        g, f = bankscore.get((uid, 1)), bankscore.get((uid, 0))
        if g is None or f is None:
            continue
        rows.append({"uid": uid, "app": meta[uid]["app"], "element": meta[uid]["element"],
                     "gold_span": d[1], "filler_span": d[0], "bank_gold": g, "bank_filler": f,
                     "bank_correct": g > f, "margin": g - f, "split": split_of(meta[uid]["app"])})
    n = len(rows)
    hard = [r for r in rows if not r["bank_correct"]]
    print(f"[prep] {n} claims scored; bank ranks gold>filler on {sum(r['bank_correct'] for r in rows)} "
          f"({np.mean([r['bank_correct'] for r in rows]):.3f}) -> residual (hard) = {len(hard)}",
          flush=True)
    for s in ("propose", "stage1", "stage2"):
        ss = [r for r in rows if r["split"] == s]
        print(f"  {s}: {len(ss)} claims ({sum(not r['bank_correct'] for r in ss)} hard)", flush=True)

    # proposer input: from PROPOSE split, sample hard contrastive claims (bank fails) + a few easy
    # (bank succeeds) so the proposer sees BOTH what fools the bank and what the true match looks like
    prop = [r for r in rows if r["split"] == "propose"]
    prop_hard = sorted([r for r in prop if not r["bank_correct"]], key=lambda r: r["margin"])[:40]
    prop_easy = sorted([r for r in prop if r["bank_correct"]], key=lambda r: -r["margin"])[:12]

    def fmt(r):
        return {"claim_element": r["element"][:400],
                "TRUE_MATCH_reference": r["gold_span"][:400],
                "DISTRACTOR_reference": r["filler_span"][:400]}
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump({"hard_cases_bank_fails": [fmt(r) for r in prop_hard],
               "easy_cases_bank_succeeds": [fmt(r) for r in prop_easy]},
              open(f"{OUTDIR}/proposer_input.json", "w"), indent=1)
    json.dump({r["app"]: r["split"] for r in rows}, open(f"{OUTDIR}/discover_splits.json", "w"))
    print(f"[prep] proposer_input.json: {len(prop_hard)} hard + {len(prop_easy)} easy contrastive claims",
          flush=True)
    print("DISCOVER_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
