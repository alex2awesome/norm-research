#!/usr/bin/env python3
"""Build the WIGLEAF PAIRWISE PROBE packet.

WHY: the mature absolute-scoring bank could not certify this cell. Its K=50 anchor
battery INVERTED (pos .8798 < neg .9016, pos-vs-neg AUC .498) while
coherent-vs-scrambled stayed at .993 -- i.e. the judge reads craft fine but
SATURATES on the published pool: 83% of all 70,560 responses were 1.0, mean .899.
Both classes are already-published literary flash fiction, so an absolute 3-point
scale has no headroom left to express the editor's cut.

HYPOTHESIS: a COMPARATIVE question ("which of these two better exemplifies X?")
restores discrimination that absolute scoring lost to ceiling effects.

DESIGN
  * 200 real pairs, each = one Top-50 positive vs one longlist negative, both
    PUBLISHED (within-pool contrast -- no easy prose/non-prose cue).
  * Confound control by matched sampling, in priority order: same magazine ->
    same year -> unmatched. Venue and era are the two obvious non-craft cues, and
    magazine is where the old absolute bank's signal could have hidden.
  * Position randomised per pair by stable hash (never a seeded shuffle), so
    position bias adds noise but cannot bias the accuracy estimate. The A-choice
    rate is reported as a diagnostic.
  * 40 of the 200 are ALSO issued in FLIPPED order as separate pair_ids -> a
    direct order-consistency measurement (the pairwise analogue of retest
    reliability).
  * 30 ANCHOR pairs = a real piece vs a scrambled word-salad of itself. The judge
    must prefer the real one. This is the blinded known-label anchor battery that
    every judging batch must carry (feedback_anchor_test_annotation_passes); it is
    the pairwise analogue of the coherent-vs-scrambled control the absolute bank
    passed at .993.
  * Dimensions = the mature bank's 45 GEPA-phrased criteria, reused verbatim,
    plus one holistic "overall editorial preference" question.

Labels never appear in a prompt. Anchors are indistinguishable from real pairs in
the rendered prompt.

  python datasets/creative-writing/build_wigleaf_pairwise_packet.py
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
CW = REPO / "datasets/creative-writing"
POP = CW / "wigleaf/va/population.csv.gz"
BANK = CW / "va_bank_v2/rubrics_initial.jsonl"
OUT = CW / "wigleaf/pairwise"
SEED = 20260811
N_PAIRS = 200
N_FLIPPED = 40
N_ANCHORS = 30
PIECE_TOKEN_CAP = 1200          # two pieces per comparison; keeps prompts sane

sys.path.insert(0, str(REPO / "datasets/va_gemma_banks"))


def h(s: str) -> int:
    return int(hashlib.sha256(f"wigleaf-pairwise|{s}".encode()).hexdigest(), 16)


def main():
    import score_cw_expert_banks as SC   # token_trunc + tokenizer, reused verbatim
    tok = SC._tokenizer()

    def cut(t):
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) <= PIECE_TOKEN_CAP:
            return t.strip()
        return (tok.decode(ids[:int(PIECE_TOKEN_CAP * .6)], skip_special_tokens=True)
                + SC.TRUNC_MARKER
                + tok.decode(ids[-int(PIECE_TOKEN_CAP * .4):], skip_special_tokens=True))

    df = pd.read_csv(POP)
    df["text"] = df["text"].astype(str)
    pos = df[df.judgement == 1].to_dict("records")
    neg = df[df.judgement == 0].to_dict("records")
    print(f"[pool] {len(pos)} Top-50 positives / {len(neg)} longlist negatives")

    neg_by_mag = defaultdict(list)
    neg_by_year = defaultdict(list)
    for r in neg:
        neg_by_mag[r["magazine"]].append(r)
        neg_by_year[r["year"]].append(r)

    # deterministic order over positives; greedy matched sampling without reuse
    pos_order = sorted(pos, key=lambda r: h(r["row_id"]))
    used_neg, pairs, match_kind = set(), [], defaultdict(int)

    def take(cands, salt):
        cands = [c for c in cands if c["row_id"] not in used_neg]
        if not cands:
            return None
        c = min(cands, key=lambda r: h(r["row_id"] + salt))
        used_neg.add(c["row_id"])
        return c

    for p in pos_order:
        if len(pairs) >= N_PAIRS:
            break
        m = take(neg_by_mag.get(p["magazine"], []), p["row_id"])
        kind = "same_magazine"
        if m is None:
            m = take(neg_by_year.get(p["year"], []), p["row_id"])
            kind = "same_year"
        if m is None:
            m = take(neg, p["row_id"])
            kind = "unmatched"
        if m is None:
            break
        match_kind[kind] += 1
        pairs.append((p, m, kind))
    print(f"[match] {dict(match_kind)}")

    rubrics = [json.loads(l) for l in open(BANK) if l.strip()]
    print(f"[bank] {len(rubrics)} criteria reused verbatim")

    items, meta = [], {}
    for i, (p, n, kind) in enumerate(pairs):
        flip = bool(h(p["row_id"] + "|pos") % 2)           # stable position randomisation
        A, B = (n, p) if flip else (p, n)
        pid = f"wp{i:04d}"
        items.append({"pair_id": pid, "A": cut(A["text"]), "B": cut(B["text"])})
        meta[pid] = {"pos_side": "B" if flip else "A", "match": kind,
                     "pos_id": p["row_id"], "neg_id": n["row_id"],
                     "magazine_pos": p["magazine"], "magazine_neg": n["magazine"],
                     "year_pos": int(p["year"]), "year_neg": int(n["year"]),
                     "kind": "real"}

    # order-consistency replicates: same content, swapped sides, new pair_id
    for i, (p, n, kind) in enumerate(pairs[:N_FLIPPED]):
        base = f"wp{i:04d}"
        flip = not (meta[base]["pos_side"] == "B")
        A, B = (n, p) if flip else (p, n)
        pid = f"wpflip{i:04d}"
        items.append({"pair_id": pid, "A": cut(A["text"]), "B": cut(B["text"])})
        meta[pid] = {"pos_side": "B" if flip else "A", "match": kind,
                     "pos_id": p["row_id"], "neg_id": n["row_id"],
                     "kind": "flip_replicate", "replicate_of": base}

    # blinded anchors: real piece vs scrambled word-salad of itself
    rng = random.Random(SEED)
    anchors = []
    anchor_src = sorted(df.to_dict("records"), key=lambda r: h(r["row_id"] + "|anc"))[:N_ANCHORS]
    for i, r in enumerate(anchor_src):
        scr = SC.S.scramble([r["text"][:4000]], rng, n_words=220)
        flip = bool(h(r["row_id"] + "|anc2") % 2)
        A, B = (scr, r["text"]) if flip else (r["text"], scr)
        pid = f"wpanc{i:03d}"
        anchors.append({"pair_id": pid, "A": cut(A), "B": cut(B)})
        meta[pid] = {"real_side": "B" if flip else "A", "kind": "anchor",
                     "source_id": r["row_id"]}

    OUT.mkdir(parents=True, exist_ok=True)
    packet = {
        "cell": "cw_wigleaf_curation",
        "probe": "within-pool PAIRWISE comparative judging",
        "why": "absolute bank saturated (83% of responses 1.0, mean .899) and its "
               "K=50 battery inverted (pos .8798 < neg .9016, AUC .498) while "
               "coherent-vs-scrambled held at .993",
        "judge": "gpt-5.6-sol via codex exec (feedback_judge_checks_use_codex; "
                 "Sonnet+ frontier tier, feedback_judges_sonnet_or_better)",
        "criteria": rubrics,
        "n_real_pairs": len(pairs), "n_flip_replicates": min(N_FLIPPED, len(pairs)),
        "n_anchors": len(anchors),
        "match_composition": dict(match_kind),
        "piece_token_cap": PIECE_TOKEN_CAP,
        "position_randomisation": 'stable sha256("wigleaf-pairwise|"+row_id)%2, never a seeded shuffle',
        "label_blind": "labels never appear in a prompt; anchors render identically to real pairs",
        "items": items, "anchors": anchors, "meta": meta,
    }
    (OUT / "packet.json").write_text(json.dumps(packet, indent=1))
    print(f"[packet] {len(items)} items + {len(anchors)} anchors -> {OUT/'packet.json'}")
    print("BUILD_WIGLEAF_PAIRWISE_PACKET_DONE")


if __name__ == "__main__":
    main()
