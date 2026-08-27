#!/usr/bin/env python3
"""SI PAIRWISE PROBE -- readout and verdict.

Pairwise accuracy is the natural AUC here: with one positive and one negative per pair,
P(judge picks the winner) IS the pairwise AUC, so "accuracy" and "AUC" are the same number
and no thresholding is involved (the programme's threshold-free rule is satisfied by
construction).

WHAT IS REPORTED, and why each one is needed
  by arm            MATCHED is primary (length caliper); FREE secondary; the two ANCHOR
                    arms certify the judge is reading at all.
  longer baseline   "pick the longer entry" on the SAME pairs. A judge number is only
                    interesting to the extent it exceeds this.
  length split      accuracy on MATCHED pairs where the winner is LONGER vs SHORTER. This
                    is the sharp length diagnostic: a length-driven judge is above chance
                    on the first and below chance on the second. A caliper cannot show
                    that; this split can.
  position bias     the SWAP arm re-asks matched pairs with A/B reversed. `consistency` =
                    share of swap pairs answered the same way (same underlying entry)
                    under both orders; `side_A_rate` = how often the judge picks whichever
                    entry is shown first, pooled.
  criteria          per-criterion accuracy, with NEGATIVELY-oriented criteria reported
                    BOTH raw and sign-corrected (for a negative criterion the winner is
                    expected to exemplify it LESS, so raw accuracy near .5 is the null and
                    the sign-corrected reading is 1 - raw).
  binomial CI       Wilson 95% on every accuracy, because n is a few hundred and the
                    verdict turns on whether an interval clears .5 / clears the baseline.

VERDICT RULE, fixed before the numbers are read (coordinator's (b)/(c) fork):
  SEPARATES      holistic MATCHED accuracy's Wilson lower bound > .5 AND its point
                 estimate exceeds the longer-entry baseline on the same pairs, OR >= 2
                 criteria clear the same bar.  -> spec the full pairwise instrument.
  DOES NOT       otherwise -> the scope-limited terminal: "the editor's cut is not
                 text-recoverable at current judge capability", bounded by the anchors
                 (which say whether the judge could read the text at all).

CPU only.  Usage: python readout_pairs.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def acc_block(hits, n, base=None):
    if n == 0:
        return {"n": 0}
    lo, hi = wilson(hits, n)
    out = {"n": n, "acc": hits / n, "wilson_lo": lo, "wilson_hi": hi,
           "beats_chance": bool(lo > 0.5)}
    if base is not None:
        out["longer_baseline"] = base
        out["excess_over_length_baseline"] = hits / n - base
        out["beats_baseline"] = bool(hits / n > base and lo > 0.5)
    return out


def main():
    pairs = {p["pair_id"]: p for p in json.loads((HERE / "si_pairs.json").read_text())}
    man = json.loads((HERE / "si_prompt_manifest.json").read_text())
    jobs = {j["tag"]: j for j in man["jobs"]}

    # question -> pair_id -> chosen side
    ans = defaultdict(dict)
    missing, jobs_read = [], 0
    for f in sorted((HERE / "out").glob("*.json")):
        tag = f.stem
        if tag not in jobs:
            continue
        d = json.loads(f.read_text())
        q = jobs[tag]["question"]
        got = set()
        for a in d.get("answers", []):
            pid = str(a.get("pair_id", "")).strip()
            ch = str(a.get("choice", "")).strip().upper()
            if pid in pairs and ch in ("A", "B"):
                ans[q][pid] = ch
                got.add(pid)
        jobs_read += 1
        missing += [p for p in jobs[tag]["pair_ids"] if p not in got]

    def correct(q, pid):
        """1 if the judge picked the POSITIVE (winner / real entry) side."""
        return 1 if ans[q].get(pid) == pairs[pid]["pos_side"] else 0

    def base_longer(pids):
        v = []
        for p in pids:
            pp = pairs[p]
            v.append(1.0 if pp["pos_len"] > pp["neg_len"]
                     else (0.5 if pp["pos_len"] == pp["neg_len"] else 0.0))
        return sum(v) / len(v) if v else None

    res = {"judge": "gpt-5.6-sol (codex exec, effort high) -- family recorded; Claude "
                    "subagent budget exhausted this session",
           "n_jobs_read": jobs_read, "n_jobs_total": man["n_jobs"],
           "n_unanswered_pair_slots": len(missing),
           "readout_note": "one positive and one negative per pair, so accuracy IS the "
                           "pairwise AUC; no threshold is involved"}

    # ---------------- holistic --------------------------------------------------
    hol = {}
    for arm in ("MATCHED", "FREE", "SWAP", "ANCHOR_SCRAM", "ANCHOR_FRAGMENT"):
        pids = [p for p, v in pairs.items() if v["arm"] == arm and p in ans["holistic"]]
        hits = sum(correct("holistic", p) for p in pids)
        hol[arm] = acc_block(hits, len(pids), base_longer(pids))
    # length split inside MATCHED
    for lab, cond in (("MATCHED_winner_longer", lambda v: v["pos_len"] > v["neg_len"]),
                      ("MATCHED_winner_shorter", lambda v: v["pos_len"] < v["neg_len"])):
        pids = [p for p, v in pairs.items()
                if v["arm"] == "MATCHED" and cond(v) and p in ans["holistic"]]
        hol[lab] = acc_block(sum(correct("holistic", p) for p in pids), len(pids))
    res["holistic"] = hol

    # ---------------- position bias --------------------------------------------
    swaps = [(p, v) for p, v in pairs.items() if v["arm"] == "SWAP"]
    cons, both, sideA = 0, 0, 0
    for p, v in swaps:
        o = v.get("swap_of")
        if p in ans["holistic"] and o in ans["holistic"]:
            both += 1
            # same underlying entry chosen under both orders?
            picked_pos_swapped = ans["holistic"][p] == v["pos_side"]
            picked_pos_orig = ans["holistic"][o] == pairs[o]["pos_side"]
            cons += int(picked_pos_swapped == picked_pos_orig)
    allq = [p for p in pairs if p in ans["holistic"]]
    sideA = sum(1 for p in allq if ans["holistic"][p] == "A")
    res["position_bias"] = {
        "n_swap_pairs_with_both_orders": both,
        "consistency": (cons / both) if both else None,
        "note": "consistency = share of swap pairs where the SAME underlying entry was "
                "chosen under both presentation orders; .5 means the judge is answering "
                "position, 1.0 means position is irrelevant",
        "side_A_pick_rate_all_holistic": (sideA / len(allq)) if allq else None,
    }

    # ---------------- criteria ---------------------------------------------------
    crit = {}
    for j in man["jobs"]:
        q = j["question"]
        if q == "holistic" or q in crit:
            continue
        pids = [p for p, v in pairs.items()
                if v["arm"] == "MATCHED" and p in ans[q]]
        hits = sum(correct(q, p) for p in pids)
        blk = acc_block(hits, len(pids), base_longer(pids))
        blk["criterion"] = j.get("criterion")
        blk["orientation"] = j.get("orientation")
        if j.get("orientation") == "negative" and blk.get("n"):
            blk["acc_sign_corrected"] = 1 - blk["acc"]
            lo, hi = wilson(len(pids) - hits, len(pids))
            blk["wilson_lo_sign_corrected"], blk["wilson_hi_sign_corrected"] = lo, hi
            blk["beats_chance_sign_corrected"] = bool(lo > 0.5)
        # anchors for this criterion
        apids = [p for p, v in pairs.items()
                 if v["arm"].startswith("ANCHOR") and p in ans[q]]
        blk["anchor_acc"] = (sum(correct(q, p) for p in apids) / len(apids)) if apids else None
        crit[q] = blk
    res["criteria"] = crit

    # ---------------- verdict ----------------------------------------------------
    hm = hol.get("MATCHED", {})
    hol_sep = bool(hm.get("beats_baseline"))
    crit_sep = [c for c, b in crit.items()
                if b.get("beats_baseline") or b.get("beats_chance_sign_corrected")]
    res["verdict"] = {
        "rule": "SEPARATES if holistic MATCHED Wilson-lo > .5 AND point estimate exceeds "
                "the longer-entry baseline, OR >= 2 criteria clear the same bar",
        "holistic_separates": hol_sep,
        "criteria_separating": crit_sep,
        "SEPARATES": bool(hol_sep or len(crit_sep) >= 2),
        "anchor_gate": {
            "scram": hol.get("ANCHOR_SCRAM", {}).get("acc"),
            "fragment": hol.get("ANCHOR_FRAGMENT", {}).get("acc"),
            "note": "if the anchors are also at chance the probe is uninformative -- the "
                    "judge could not read the items, and no verdict about the construct "
                    "may be drawn",
        },
    }
    (HERE / "si_pairwise_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
