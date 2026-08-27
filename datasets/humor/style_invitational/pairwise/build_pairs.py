#!/usr/bin/env python3
"""STYLE INVITATIONAL PAIRWISE PROBE -- pair construction.

CHARGE (user-approved, coordinator 2026-08-11).  The v2 bank failed certification:
K=50 winner-vs-honorable-mention reads AUC .483 sign-corrected / .509 raw, and 0 of 33
criteria clear |AUC-.5| >= .05.  That failure is CONFOUNDED between two very different
diagnoses:

  (i)  the criteria are wrong -- they do not name what separates a winner from an HM;
  (ii) Gemma-4-31B cannot SEE the distinction when asked for an absolute item-level
       score, because the editor's construct is inherently COMPARATIVE.

An absolute-scoring instrument cannot tell these apart.  A pairwise frontier judge can:
show it two entries FROM THE SAME CONTEST WEEK -- so the prompt, the topic, the era and
the editor are all held fixed -- and ask which one better exemplifies a criterion, and
separately which one the editor picked.

DESIGN DECISIONS, fixed here before any judge call.

1. WITHIN-WEEK ONLY.  Both members of a pair come from the same `week_id`, so the contest
   prompt is identical and every between-contest nuisance is differenced out by
   construction.  This is the pairwise analogue of the within-container readout the
   programme uses everywhere else.

2. POSITIVE = winner, NEGATIVE = honorable mention.  `runnerup` rows are EXCLUDED from
   the probe: the cell's y pools winner+runnerup, but the sharpest available contrast is
   winner-vs-HM and a probe should be run at the sharpest contrast before a blunter one.
   Recorded as a scope limit, not a silent choice.

3. THE LENGTH CONFOUND IS DESIGNED FOR, NOT DISCOVERED.  Winner median 104.5 chars vs HM
   89.  A pairwise judge that simply prefers the longer entry would score above chance
   for no interesting reason.  Two arms are therefore built and BOTH are always reported:
     MATCHED  -- |log length ratio| <= .20 (roughly +/-20% characters). The primary arm.
     FREE     -- sampled without a length caliper. The secondary arm.
   `longer_wins` (the pick-the-longer-entry baseline) is computed on the same pairs and
   printed beside every judge number. A judge result is only interesting to the extent it
   exceeds that baseline on the MATCHED arm.

4. ORDER BALANCE AND A POSITION-BIAS ARM.  Which entry is shown first is decided by a
   stable sha256 of the pair id, and the assignment is asserted to be within a few points
   of 50/50.  A SWAP arm re-asks a subsample of pairs with the order reversed; comparing
   the two answers measures position bias directly, which is the pairwise analogue of an
   anchor and the failure mode most likely to fake a positive result.

5. PLANTED KNOWN-DIRECTION ANCHOR PAIRS (the pairwise anchor battery).  Two kinds, both
   corpus-native so no synthetic text enters the judge's view:
     ANCHOR_SCRAM    a real entry vs a word-scrambled version of another entry from the
                     same week. A judge that cannot win this is not reading.
     ANCHOR_FRAGMENT a real entry vs one of the 1,574 parse-artifact rows the population
                     audit identified (an orphan byline, a section header). Known
                     direction, and it doubles as a check that the probe's own plumbing
                     shows the judge real text.
   Anchors are interleaved with the real pairs and are indistinguishable in the prompt.

6. TOKENS, NOT CHARACTERS.  Truncation is a guard applied in tokens; the longest clean
   entry is ~596 tokens, so it fires on nothing, exactly as in the bank scoring pass.

7. ITEM VIEW.  Each pair shows ONE contest prompt (shared) and the two entries' text
   only. The trailing "(Name, City)" byline is archive metadata; it is left in place
   because the bank's own item view leaves it in place and the criteria that care about
   it say so explicitly. Recorded so the two instruments stay comparable.

CPU only.  Usage: python build_pairs.py --n-matched 200 --n-free 100 --n-swap 60
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
POP = HERE.parent / "va_v2" / "population.csv.gz"
SALT = "si-pairwise-probe-2026-08-11|"
MATCH_CALIPER = 0.20          # |log(len ratio)| <= .20


def h(s):
    return int(hashlib.sha256((SALT + s).encode()).hexdigest(), 16)


def scramble(text, rng):
    """Word-shuffle a real entry, byline included, so the anchor is corpus-native."""
    w = text.split()
    rng.shuffle(w)
    return " ".join(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-matched", type=int, default=200)
    ap.add_argument("--n-free", type=int, default=100)
    ap.add_argument("--n-swap", type=int, default=60)
    ap.add_argument("--n-anchor", type=int, default=30)
    a = ap.parse_args()

    d = pd.read_csv(POP)
    clean = d[~d.is_fragment].copy()
    frags = d[d.is_fragment].copy()

    win = clean[clean.tier == "winner"]
    hm = clean[clean.tier == "honorable_mention"]
    weeks = sorted(set(win.week_id) & set(hm.week_id))

    cand = []
    for wk in weeks:
        ws = win[win.week_id == wk]
        hs = hm[hm.week_id == wk]
        prompt = ws.iloc[0].contest_prompt
        for _, w in ws.iterrows():
            for _, m in hs.iterrows():
                lr = abs(math.log(max(w.char_len, 1) / max(m.char_len, 1)))
                cand.append({
                    "week_id": int(wk), "prompt": prompt,
                    "pos_id": w.row_id, "neg_id": m.row_id,
                    "pos_text": w.entry_text, "neg_text": m.entry_text,
                    "pos_len": int(w.char_len), "neg_len": int(m.char_len),
                    "abs_log_len_ratio": lr,
                })
    for c in cand:
        c["pair_id"] = "P" + hashlib.sha256(
            (SALT + str(c["pos_id"]) + "|" + str(c["neg_id"])).encode()).hexdigest()[:10]

    # deterministic, week-spread selection: at most 1 pair per week per arm, ordered by
    # stable hash so the draw is unseeded and reproducible
    def pick(pool, n, seen_weeks):
        out = []
        for c in sorted(pool, key=lambda c: h(c["pair_id"])):
            if c["week_id"] in seen_weeks:
                continue
            out.append(c)
            seen_weeks.add(c["week_id"])
            if len(out) >= n:
                break
        return out

    matched_pool = [c for c in cand if c["abs_log_len_ratio"] <= MATCH_CALIPER]
    seen = set()
    matched = pick(matched_pool, a.n_matched, seen)
    free = pick([c for c in cand], a.n_free, seen)
    for c in matched:
        c["arm"] = "MATCHED"
    for c in free:
        c["arm"] = "FREE"

    pairs = matched + free
    rng = random.Random(20260811)

    # ---- planted anchors ----------------------------------------------------
    anchors = []
    wk_list = sorted(set(clean.week_id))
    for k in range(a.n_anchor // 2):
        wk = wk_list[h(f"anchor_scram{k}") % len(wk_list)]
        sub = clean[clean.week_id == wk]
        if len(sub) < 2:
            continue
        r1, r2 = sub.iloc[0], sub.iloc[min(1, len(sub) - 1)]
        anchors.append({
            "pair_id": f"AS{k+1:02d}", "arm": "ANCHOR_SCRAM", "week_id": int(wk),
            "prompt": r1.contest_prompt,
            "pos_id": r1.row_id, "neg_id": f"SCRAM::{r2.row_id}",
            "pos_text": r1.entry_text, "neg_text": scramble(r2.entry_text, rng),
            "pos_len": int(r1.char_len), "neg_len": int(r2.char_len),
            "abs_log_len_ratio": 0.0, "known_direction": "pos",
        })
    fl = frags.reset_index(drop=True)
    for k in range(a.n_anchor - len(anchors)):
        f = fl.iloc[h(f"anchor_frag{k}") % len(fl)]
        sub = clean[clean.week_id == f.week_id]
        if len(sub) == 0:
            sub = clean
        r1 = sub.iloc[h(f"anchor_frag_pos{k}") % len(sub)]
        anchors.append({
            "pair_id": f"AF{k+1:02d}", "arm": "ANCHOR_FRAGMENT", "week_id": int(r1.week_id),
            "prompt": r1.contest_prompt,
            "pos_id": r1.row_id, "neg_id": f.row_id,
            "pos_text": r1.entry_text, "neg_text": f.entry_text,
            "pos_len": int(r1.char_len), "neg_len": int(f.char_len),
            "abs_log_len_ratio": 0.0, "known_direction": "pos",
        })

    allp = pairs + anchors
    # ---- order assignment: which side the POSITIVE is shown on ---------------
    for c in allp:
        c["pos_side"] = "A" if h("side|" + c["pair_id"]) % 2 == 0 else "B"
        c["entry_A"] = c["pos_text"] if c["pos_side"] == "A" else c["neg_text"]
        c["entry_B"] = c["neg_text"] if c["pos_side"] == "A" else c["pos_text"]

    # ---- swap arm: same pairs, order reversed --------------------------------
    swap_src = sorted([c for c in pairs if c["arm"] == "MATCHED"],
                      key=lambda c: h("swap|" + c["pair_id"]))[:a.n_swap]
    swaps = []
    for c in swap_src:
        s = dict(c)
        s["pair_id"] = c["pair_id"] + "R"
        s["arm"] = "SWAP"
        s["swap_of"] = c["pair_id"]
        s["pos_side"] = "B" if c["pos_side"] == "A" else "A"
        s["entry_A"], s["entry_B"] = c["entry_B"], c["entry_A"]
        swaps.append(s)

    out = allp + swaps
    side_pos = sum(1 for c in out if c["pos_side"] == "A") / len(out)

    rep = {
        "salt": SALT,
        "n_total": len(out),
        "by_arm": {k: sum(1 for c in out if c["arm"] == k)
                   for k in sorted({c["arm"] for c in out})},
        "n_weeks_used": len({c["week_id"] for c in out}),
        "match_caliper_abs_log_len_ratio": MATCH_CALIPER,
        "pos_shown_as_A_share": side_pos,
        "candidate_pool": {"n_candidate_within_week_pairs": len(cand),
                           "n_weeks_with_winner_and_HM": len(weeks),
                           "n_matched_candidates": len(matched_pool)},
        "length_baseline_pick_the_longer": {
            arm: float(np.mean([
                1.0 if c["pos_len"] > c["neg_len"] else (0.5 if c["pos_len"] == c["neg_len"] else 0.0)
                for c in out if c["arm"] == arm]))
            for arm in sorted({c["arm"] for c in out})},
        "median_abs_log_len_ratio": {
            arm: float(np.median([c["abs_log_len_ratio"] for c in out if c["arm"] == arm]))
            for arm in sorted({c["arm"] for c in out})},
        "scope_limit": "runnerup rows EXCLUDED; this probe is the winner-vs-HM contrast only",
    }
    assert 0.40 <= side_pos <= 0.60, f"order assignment unbalanced: {side_pos}"
    (HERE / "si_pairs.json").write_text(json.dumps(out, indent=1))
    (HERE / "si_pairs_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
