#!/usr/bin/env python3
"""Wave 2 of the Wigleaf PAIRWISE probe: +400 pairs to reach ~600 total, so the
pairwise-native Layer-1 (design P2) has enough units to fit.

Wave 1 (200 pairs) established the instrument: anchors .9963, order consistency
.8406, composite AUC .610 [.5425, .6775], same-magazine stratum strongest (.6231).
This wave is powered for a FITTED combination rather than a majority vote.

RULES CARRIED FROM THE COORDINATOR'S RULING
  * NO PAIR REUSE. A ledger of every (pos_id, neg_id) combination used in wave 1
    is loaded and excluded. Item reuse is unavoidable -- there are only 404
    positives for 600 pairs -- so positives are reused at most PER_POS_CAP times
    and the LEDGER is over COMBINATIONS, which is what "no pair reuse" means here.
  * SAME-MAGAZINE-HEAVY. Same-magazine was the strongest stratum (.6231 vs .5857),
    so wave 2 takes same-magazine pairs first and only falls back to same-year.
  * ANCHOR DISCIPLINE at the same rate: scrambled anchors at ~11% of the wave,
    provenance-matched (the scrambled side is built from a piece drawn from the
    SAME pool, so fetch provenance cannot separate anchor from real).
  * Position randomisation by stable hash; a proportional block of flipped
    replicates keeps order-consistency measurable on the new wave too.

  python datasets/creative-writing/build_wigleaf_pairwise_wave2.py
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
PW = CW / "wigleaf/pairwise"
SEED = 20260812
N_NEW = 400
N_FLIPPED = 40
ANCHOR_RATE = 0.11
PER_POS_CAP = 2
PIECE_TOKEN_CAP = 1200

sys.path.insert(0, str(REPO / "datasets/va_gemma_banks"))


def h(s: str) -> int:
    return int(hashlib.sha256(f"wigleaf-pairwise-w2|{s}".encode()).hexdigest(), 16)


def main():
    import score_cw_expert_banks as SC
    tok = SC._tokenizer()

    def cut(t):
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) <= PIECE_TOKEN_CAP:
            return t.strip()
        return (tok.decode(ids[:int(PIECE_TOKEN_CAP * .6)], skip_special_tokens=True)
                + SC.TRUNC_MARKER
                + tok.decode(ids[-int(PIECE_TOKEN_CAP * .4):], skip_special_tokens=True))

    w1 = json.loads((PW / "packet.json").read_text())
    used = {(m["pos_id"], m["neg_id"]) for m in w1["meta"].values()
            if m.get("kind") in ("real", "flip_replicate")}
    pos_uses = defaultdict(int)
    for m in w1["meta"].values():
        if m.get("kind") == "real":
            pos_uses[m["pos_id"]] += 1
    print(f"[ledger] wave-1 combinations excluded: {len(used)}; "
          f"positives already used: {len(pos_uses)}")

    df = pd.read_csv(POP)
    df["text"] = df["text"].astype(str)
    pos = df[df.judgement == 1].to_dict("records")
    neg = df[df.judgement == 0].to_dict("records")
    neg_by_mag = defaultdict(list)
    neg_by_year = defaultdict(list)
    for r in neg:
        neg_by_mag[r["magazine"]].append(r)
        neg_by_year[r["year"]].append(r)

    pairs, kinds = [], defaultdict(int)
    # positives in stable order, cycling; same-magazine first
    order = sorted(pos, key=lambda r: h(r["row_id"]))
    for rnd in range(PER_POS_CAP):
        for p in order:
            if len(pairs) >= N_NEW:
                break
            if pos_uses[p["row_id"]] > rnd:
                continue
            picked, kind = None, None
            for cands, k in ((neg_by_mag.get(p["magazine"], []), "same_magazine"),
                             (neg_by_year.get(p["year"], []), "same_year"),
                             (neg, "unmatched")):
                avail = [c for c in cands
                         if (p["row_id"], c["row_id"]) not in used]
                if avail:
                    picked = min(avail, key=lambda r: h(p["row_id"] + "|" + r["row_id"]))
                    kind = k
                    break
            if picked is None:
                continue
            used.add((p["row_id"], picked["row_id"]))
            pos_uses[p["row_id"]] += 1
            kinds[kind] += 1
            pairs.append((p, picked, kind))
        if len(pairs) >= N_NEW:
            break
    print(f"[match] wave-2 {len(pairs)} pairs: {dict(kinds)}")
    print(f"[reuse] max uses of any positive across both waves: "
          f"{max(pos_uses.values())} (cap {PER_POS_CAP})")

    rubrics = [json.loads(l) for l in open(BANK) if l.strip()]
    items, meta = [], {}
    for i, (p, n, kind) in enumerate(pairs):
        flip = bool(h(p["row_id"] + "|" + n["row_id"] + "|pos") % 2)
        A, B = (n, p) if flip else (p, n)
        pid = f"w2p{i:04d}"
        items.append({"pair_id": pid, "A": cut(A["text"]), "B": cut(B["text"])})
        meta[pid] = {"pos_side": "B" if flip else "A", "match": kind,
                     "pos_id": p["row_id"], "neg_id": n["row_id"],
                     "magazine_pos": p["magazine"], "magazine_neg": n["magazine"],
                     "year_pos": int(p["year"]), "year_neg": int(n["year"]),
                     "kind": "real", "wave": 2}
    for i, (p, n, kind) in enumerate(pairs[:N_FLIPPED]):
        base = f"w2p{i:04d}"
        flip = not (meta[base]["pos_side"] == "B")
        A, B = (n, p) if flip else (p, n)
        pid = f"w2pflip{i:04d}"
        items.append({"pair_id": pid, "A": cut(A["text"]), "B": cut(B["text"])})
        meta[pid] = {"pos_side": "B" if flip else "A", "match": kind,
                     "pos_id": p["row_id"], "neg_id": n["row_id"],
                     "kind": "flip_replicate", "replicate_of": base, "wave": 2}

    n_anchor = max(1, round(ANCHOR_RATE * len(pairs)))
    rng = random.Random(SEED)
    anchors = []
    src = sorted(df.to_dict("records"), key=lambda r: h(r["row_id"] + "|anc-w2"))[:n_anchor]
    for i, r in enumerate(src):
        scr = SC.S.scramble([r["text"][:4000]], rng, n_words=220)
        flip = bool(h(r["row_id"] + "|anc2-w2") % 2)
        A, B = (scr, r["text"]) if flip else (r["text"], scr)
        pid = f"w2panc{i:03d}"
        anchors.append({"pair_id": pid, "A": cut(A), "B": cut(B)})
        meta[pid] = {"real_side": "B" if flip else "A", "kind": "anchor",
                     "source_id": r["row_id"], "wave": 2}

    packet = {
        "cell": "cw_wigleaf_curation", "wave": 2,
        "probe": "within-pool PAIRWISE comparative judging (wave 2, powered for P2)",
        "judge": "gpt-5.6-sol via codex exec",
        "criteria": rubrics,
        "n_real_pairs": len(pairs), "n_flip_replicates": min(N_FLIPPED, len(pairs)),
        "n_anchors": len(anchors), "anchor_rate": round(len(anchors) / max(len(pairs), 1), 3),
        "match_composition": dict(kinds),
        "per_positive_cap": PER_POS_CAP,
        "no_pair_reuse": "every (pos_id, neg_id) combination from wave 1 excluded; "
                         "ledger is over COMBINATIONS (item reuse is forced by a "
                         "404-positive pool at 600 pairs)",
        "piece_token_cap": PIECE_TOKEN_CAP,
        "items": items, "anchors": anchors, "meta": meta,
    }
    (PW / "packet_wave2.json").write_text(json.dumps(packet, indent=1))
    print(f"[packet] {len(items)} items + {len(anchors)} anchors "
          f"(anchor rate {packet['anchor_rate']}) -> {PW/'packet_wave2.json'}")
    print("BUILD_WIGLEAF_PAIRWISE_WAVE2_DONE")


if __name__ == "__main__":
    main()
