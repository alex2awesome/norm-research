#!/usr/bin/env python3
"""SI PAIRWISE — PHASE 1: sparse within-week comparison graph (approved 2026-08-11).

SCOPE, chosen so §6.5 is commensurable BY CONSTRUCTION rather than by a later argument.
The graph covers the dense arm's **eval ∪ test** rows only — 810 + 797 = 1,607 items over
80 weeks, with **zero week overlap between the splits** (the split is week-grouped), so
every within-week comparison stays inside one split. Those are exactly the rows on which
the rebuild note already reports same-rows `VA_nl` (eval .6165 / test .6042) and `T`
(eval .6241 / test .6237), so θ drops straight into that table with no re-derivation.
Covering all 8,063 rows would cost ~5× the judge budget and would NOT be commensurable
with T, which only exists on the held-out rows.

GRAPH. Per week, items are placed in a stable-hash permutation and connected as a
circulant C_n(1, 2, ⌊n/2⌋) — degree 5 for even n (4 for odd, where the ⌊n/2⌋ chord
doubles back), guaranteed connected, fully deterministic, no seeding. Connectivity is a
hard requirement: Bradley-Terry θ is only identified within a connected component. Weeks
with n ≤ 5 get a full round-robin instead. Connectivity is asserted per week.

CONTROLS carried from the probe (notes/2026-08-11__si_pairwise_probe.md §6.4):
  * A/B order set by stable hash per comparison; realised balance asserted near .5.
  * SWAP arm: ~10% of comparisons re-asked with the order reversed, for the consistency
    figure and to feed the side term in the likelihood.
  * ANCHOR_FRAGMENT planted pairs interleaved (a real entry vs a parse-artifact row).
    **No scramble anchors** — retired for this corpus (§4 of the probe note: SI entries
    are short enough that a shuffle stays near-coherent, and the corpus runs
    rearrangement contests).
  * The judge is never told the tiers, the splits, the anchors or the swap arm.

CPU only.  Usage: python build_graph.py [--degree 5] [--swap-frac .10] [--batch 25]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
POP = HERE.parent / "va_v2" / "population.csv.gz"
ROOT = HERE / "phase1"
SALT = "si-bt-phase1-2026-08-11|"

HOLISTIC = (
    "For each pair below, two entries were submitted to the SAME humour contest, in "
    "answer to the SAME prompt. Both were good enough to be published.\n\n"
    "For each pair, say which entry the contest's editor rated more highly."
)
TAIL = (
    "\n\nOUTPUT. Emit exactly one JSON object and nothing else:\n"
    '{"answers": [{"pair_id": "<id>", "choice": "A" or "B", '
    '"confidence": "high"/"medium"/"low"}, ...]}\n'
    "One entry per pair, using the pair_id shown, covering EVERY pair listed. "
    "You must choose A or B for every pair; there is no tie option.\n\nPAIRS:\n\n"
)


def h(s):
    return int(hashlib.sha256((SALT + s).encode()).hexdigest(), 16)


def week_edges(ids, degree):
    n = len(ids)
    order = sorted(range(n), key=lambda i: h(f"perm|{ids[i]}"))
    if n <= 5:
        return [(ids[order[a]], ids[order[b]])
                for a in range(n) for b in range(a + 1, n)]
    offs = {1, 2, max(1, n // 2)} if degree >= 5 else {1, 2}
    e = set()
    for k in range(n):
        for o in offs:
            a, b = order[k], order[(k + o) % n]
            if a != b:
                e.add(tuple(sorted((ids[a], ids[b]))))
    return sorted(e)


def connected(ids, edges):
    par = {i: i for i in ids}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    return len({find(i) for i in ids}) == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--swap-frac", type=float, default=0.10)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--n-anchor", type=int, default=40)
    a = ap.parse_args()

    d = pd.read_csv(POP)
    clean = d[~d.is_fragment]
    frags = d[d.is_fragment].reset_index(drop=True)
    pool = clean[clean.split.isin(["eval", "test"])].copy()
    txt = dict(zip(pool.row_id, pool.entry_text))
    prm = dict(zip(pool.row_id, pool.contest_prompt))
    wk = dict(zip(pool.row_id, pool.week_id))
    spl = dict(zip(pool.row_id, pool.split))

    comps, degs = [], []
    for w, g in pool.groupby("week_id"):
        ids = list(g.row_id)
        e = week_edges(ids, a.degree)
        assert connected(ids, e), f"week {w} comparison graph is disconnected"
        degs.append(2 * len(e) / len(ids))
        for i, j in e:
            comps.append({"week_id": int(w), "split": spl[i], "i": i, "j": j,
                          "prompt": prm[i]})
    for c in comps:
        c["pair_id"] = "G" + hashlib.sha256(
            (SALT + str(c["i"]) + "|" + str(c["j"])).encode()).hexdigest()[:10]
        c["arm"] = "GRAPH"

    # anchors: real entry vs parse-artifact row, same week where possible
    anchors = []
    for k in range(a.n_anchor):
        f = frags.iloc[h(f"frag{k}") % len(frags)]
        sub = pool[pool.week_id == f.week_id]
        if len(sub) == 0:
            sub = pool
        r = sub.iloc[h(f"fragpos{k}") % len(sub)]
        anchors.append({"pair_id": f"AF{k+1:03d}", "arm": "ANCHOR_FRAGMENT",
                        "week_id": int(r.week_id), "split": spl[r.row_id],
                        "i": r.row_id, "j": f.row_id, "prompt": r.contest_prompt,
                        "known_direction": "i"})
        txt.setdefault(f.row_id, f.entry_text)
        txt.setdefault(r.row_id, r.entry_text)

    allc = comps + anchors
    for c in allc:
        c["i_side"] = "A" if h("side|" + c["pair_id"]) % 2 == 0 else "B"
        c["entry_A"] = txt[c["i"]] if c["i_side"] == "A" else txt[c["j"]]
        c["entry_B"] = txt[c["j"]] if c["i_side"] == "A" else txt[c["i"]]

    n_swap = int(len(comps) * a.swap_frac)
    swaps = []
    for c in sorted(comps, key=lambda c: h("swap|" + c["pair_id"]))[:n_swap]:
        s = dict(c)
        s["pair_id"] = c["pair_id"] + "R"
        s["arm"] = "SWAP"
        s["swap_of"] = c["pair_id"]
        s["i_side"] = "B" if c["i_side"] == "A" else "A"
        s["entry_A"], s["entry_B"] = c["entry_B"], c["entry_A"]
        swaps.append(s)

    out = allc + swaps
    side = sum(1 for c in out if c["i_side"] == "A") / len(out)
    assert 0.45 <= side <= 0.55, f"order balance {side}"

    ROOT.mkdir(exist_ok=True)
    (ROOT / "prompts").mkdir(exist_ok=True)
    (ROOT / "out").mkdir(exist_ok=True)
    (ROOT / "si_bt_comparisons.json").write_text(json.dumps(out, indent=1))

    jobs = []
    ch = [out[i:i + a.batch] for i in range(0, len(out), a.batch)]
    for b, batch in enumerate(ch):
        tag = f"bt_b{b:03d}"
        body = "\n\n".join(
            f"--- pair_id={p['pair_id']} ---\nCONTEST PROMPT: {p['prompt']}\n"
            f"ENTRY A: {p['entry_A']}\nENTRY B: {p['entry_B']}" for p in batch)
        (ROOT / "prompts" / f"{tag}.txt").write_text(HOLISTIC + TAIL + body)
        jobs.append({"tag": tag, "question": "holistic", "n_pairs": len(batch),
                     "pair_ids": [p["pair_id"] for p in batch]})
    man = {"n_jobs": len(jobs), "batch": a.batch, "jobs": jobs}
    (ROOT / "si_prompt_manifest.json").write_text(json.dumps(man, indent=1))

    rep = {"scope": "dense eval + test rows only (commensurable with same-rows VA_nl / T)",
           "n_items": len(pool), "n_weeks": int(pool.week_id.nunique()),
           "n_items_by_split": pool.split.value_counts().to_dict(),
           "mean_degree": sum(degs) / len(degs),
           "n_graph_comparisons": len(comps), "n_anchors": len(anchors),
           "n_swap": len(swaps), "n_total": len(out), "n_jobs": len(jobs),
           "i_shown_as_A_share": side,
           "all_weeks_connected": True,
           "anchors": "ANCHOR_FRAGMENT only; scramble anchors retired for this corpus"}
    (ROOT / "si_bt_graph_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
