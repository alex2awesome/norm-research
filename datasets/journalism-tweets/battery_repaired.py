#!/usr/bin/env python3
"""V9 journalism-tweets: SUPPLEMENTARY anchor battery with a headline-appropriate
scrambled control.

WHY THIS EXISTS. The shipped battery (run by score_tweets_bank.py via
score_scaleupC_banks.run_battery) uses the frozen shared helper
`score_va_gemma_banks.scramble`, which builds the nonsense control by pooling
tokens from the pos+neg anchor pair, shuffling, and reversing ALTERNATE words.
On the cell types that helper was written for -- a 200-word StackOverflow
answer, a joke, a piece of creative writing -- that destroys meaning. On a
14-word HEADLINE it does not, and the per-shard anchor check failed all four
attempts on shard 0 in a diagnostic way:

    attempt 0: pos 0.636 / neg 0.577 / scram 0.778
    attempt 1: pos 0.773 / neg 0.444 / scram 0.750
    attempt 2: pos 0.625 / neg 0.357 / scram 0.500
    attempt 3: pos 0.300 / neg 0.700 / scram 0.333

Inspecting the actual scrambled strings shows why -- intact proper nouns
survive, because only alternate words are reversed and the token pool is two
headlines wide:

    "more tuoba pathetic dewener Gaza snoitidnoC than NU says ni Hegseth's ..."
    "in noitibma to xilfteN Panama gnimoC Canal ekater Best lirpA Shows dna The VT"

A judge scoring "Elite political actor is a central subject" against a string
still containing "Hegseth's" or "Trump" answers 1.0 -- correctly. And because
headline-only judging leaves ~35% of criteria at NA, the row's `nanmean` is
taken over few surviving criteria, so one or two 1.0s dominate it. The scram leg
is therefore measuring entity survival, not coherence.

THE REPAIR, minimal and documented: keep the frozen transformation (shuffle +
reverse alternates) but widen the TOKEN POOL to `POOL_HEADLINES` randomly drawn
headlines instead of two, so no coherent story, topic or entity cluster
survives. Nothing else changes -- same rubrics, same system prompt, same judge,
same K, same independent pos/neg channel (homepage PLACEMENT).

The second, separate weakness the shipped battery exposes is NOT repaired here
because it is real rather than a bug: a single-row pos-vs-neg comparison on the
homepage-placement channel is underpowered. That bank separates placement at
about .60 AUC over a full population, so a 1-vs-1 draw fails roughly 40% of the
time by construction. K=50 per class is what fixes that, which is exactly why
the charge requires K>=50 -- and it is why the per-shard 3-row result should be
read as a smoke alarm, not as the certification.

Both batteries are reported. This one is written to
`anchor_battery_repaired.json` and never overwrites the shipped one.

  CUDA_VISIBLE_DEVICES=N python3 datasets/journalism-tweets/battery_repaired.py --battery 50
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import multiprocessing as _mp  # noqa: E402

try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

_HERE = Path(__file__).resolve()
REPO_GUESS = _HERE.parents[2]
sys.path.insert(0, str(REPO_GUESS / "datasets/va_gemma_banks"))
sys.path.insert(0, str(_HERE.parent))
import score_va_gemma_banks as S  # noqa: E402
import score_scaleupC_banks as C  # noqa: E402
import score_tweets_bank as TW  # noqa: E402

REPO = S.REPO
OUT = Path(os.environ.get("VA_OUT_TWEETS",
                          str(REPO / "outputs/va_gemma_banks_journalism_tweets")))
POOL_HEADLINES = 40
SEED = 20260808


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--util", type=float, default=0.60)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--battery", type=int, default=50)
    a = ap.parse_args()

    import pandas as pd
    b = TW.build_journalism_tweets()

    # rebuild the anchor pools exactly as the shipped bank does
    vf = S.load_module(TW.HP_DIR / "v_features.py", "vf_homepage")
    hp = pd.read_csv(TW.HP_DIR / "population.csv.gz")
    hp_pos = [vf.headline_of(t) for t, j in zip(hp["text"], hp["judgement"]) if j == 1]
    hp_neg = [vf.headline_of(t) for t, j in zip(hp["text"], hp["judgement"]) if j == 0]
    hp_pos = [h for h in hp_pos if len(h.split()) >= 4]
    hp_neg = [h for h in hp_neg if len(h.split()) >= 4]
    allh = hp_pos + hp_neg

    def anchors_repaired(shard):
        rng = random.Random(SEED + 607 * shard)
        pos = {"id": "", "group": "__anchor", "headline": rng.choice(hp_pos)}
        neg = {"id": "", "group": "__anchor", "headline": rng.choice(hp_neg)}
        # THE REPAIR: pool tokens across POOL_HEADLINES unrelated headlines so no
        # coherent entity cluster survives the shuffle.
        pool = [rng.choice(allh) for _ in range(POOL_HEADLINES)]
        scr = {"id": "", "group": "__anchor",
               "headline": S.scramble(pool, rng, n_words=14)}
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg),
                       ("anchor_scram", scr)):
            rr = dict(r)
            rr["anchor_tag"] = tag
            rr["id"] = f"__RANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    b = dict(b)
    b["anchors"] = anchors_repaired
    b["name"] = "journalism_tweets_repaired"

    print("[repaired battery] sample scrambled controls:")
    for s in range(3):
        print("   ", anchors_repaired(900_000 + s)[2]["headline"])

    from vllm import LLM, SamplingParams
    llm = LLM(model=S.GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              trust_remote_code=True, max_num_seqs=512)
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    res = C.run_battery(llm, sp, b, a.battery, OUT)

    # run_battery writes into anchor_battery.json keyed by bank name; move the
    # repaired result into its own file so the shipped battery is never clobbered.
    shipped = OUT / "anchor_battery.json"
    payload = json.loads(shipped.read_text()) if shipped.exists() else {}
    rep = payload.pop("journalism_tweets_repaired", res)
    shipped.write_text(json.dumps(payload, indent=1))
    (OUT / "anchor_battery_repaired.json").write_text(json.dumps(
        {"journalism_tweets_repaired": rep,
         "pool_headlines": POOL_HEADLINES,
         "note": "Supplementary battery: identical to the shipped one except the "
                 "scrambled control pools tokens across 40 unrelated headlines "
                 "instead of the pos+neg pair, so no coherent entity survives. "
                 "See the module docstring for the diagnosis."}, indent=1))
    print("REPAIRED_BATTERY_DONE", flush=True)


if __name__ == "__main__":
    main()
