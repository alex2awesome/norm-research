"""Sample a stratified calibration set of rubric pairs for tuning the
same/different judge. Pairs are drawn evenly across cosine bands so the set
covers easy positives, easy negatives, and the hard decision zone.

Output: judge_calib.jsonl  (the pairs, with a stable calib_id)
Prints all pairs for hand-labelling -> judge_calib_gold.jsonl is written by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("/Users/spangher/Projects/stanford-research/norm-research/outputs/analyses")
BANDS = [(0.97, 1.01), (0.94, 0.97), (0.91, 0.94), (0.88, 0.91),
         (0.85, 0.88), (0.78, 0.85), (0.0, 0.78)]
PER_BAND = 10
RNG = np.random.default_rng(7)


def main():
    pool = [json.loads(l) for l in (OUT / "judge_pool.jsonl").open()]
    picked = []
    for lo, hi in BANDS:
        cand = [p for p in pool if lo <= p["cos"] < hi]
        if not cand:
            continue
        idx = RNG.choice(len(cand), size=min(PER_BAND, len(cand)), replace=False)
        picked += [cand[k] for k in idx]
    RNG.shuffle(picked)
    for i, p in enumerate(picked):
        p["calib_id"] = i

    with (OUT / "judge_calib.jsonl").open("w") as f:
        for p in picked:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(picked)} calibration pairs -> {OUT/'judge_calib.jsonl'}\n")

    for p in picked:
        print(f"[{p['calib_id']:>2}]  cos={p['cos']:.3f}  ({p['bucket']}/{p['task']})")
        print(f"     A: {p['canonical_a']}")
        print(f"     B: {p['canonical_b']}")


if __name__ == "__main__":
    main()
