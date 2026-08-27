#!/usr/bin/env python3
"""BBC most-read: diagnostic re-run of the K=50 anchor battery with BBC-ONLY
anchor rows.

WHY. The shipped battery for this cell failed its responsiveness leg:

    pos .4656 / neg .4778 / scram .5417 | pos-vs-neg AUC .481

i.e. at chance, where the instrument-matched V9 cell scored .647 on what is
nominally the same anchor pool. That is a certification failure and cannot be
waved through. But it also cannot be the whole story, because on this cell's own
y the same bank is strongly discriminative -- A_lin .6879, group-bootstrap CI
[.6833, .6921]. A bank that separates most-read at .69 is not a broken bank.

THE HYPOTHESIS THIS TESTS. The two cells' anchor draws come from the same
homepage-placement pool (all 8 outlets), but the two cells' SYSTEM PROMPTS
differ: V9 says the headline "appeared on a major outlet's home page", while this
cell says it "appeared on the BBC News home page". The anchor rows are mostly
NOT BBC -- they are nytimes / guardian / wsj / latimes / cnn / reuters /
washingtonpost headlines. So on anchor rows only, this cell's prompt asserts a
provenance that contradicts the item, while on the 50,761 real rows the prompt is
true. That would selectively damage the battery and leave the A matrix intact --
exactly the pattern observed.

THE TEST. Redraw both anchor classes from the BBC rows of the homepage
population (1,701 rows carrying the placement label), changing NOTHING else --
same bank, same prompt, same judge, same K, same scramble repair. If pos-vs-neg
recovers, the failure was the prompt/provenance mismatch on anchor rows and the
cell's A matrix is unaffected. If it stays at chance, the honest reading is that
the bank does not separate homepage placement on BBC specifically, and the
battery simply cannot certify this cell against that channel.

Either outcome is reportable. Neither changes the scored A matrix, which was
produced in a separate pass before any anchor row was seen.

  CUDA_VISIBLE_DEVICES=N python3 datasets/bbc-mostread/battery_bbc_anchors.py --battery 50
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
import score_mostread_bank as MB  # noqa: E402

REPO = S.REPO
OUT = Path(os.environ.get("VA_OUT_BBC",
                          str(REPO / "outputs/va_gemma_banks_bbc_mostread")))
SEED = 20260811
POOL_HEADLINES = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--util", type=float, default=0.60)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--battery", type=int, default=50)
    a = ap.parse_args()

    import pandas as pd
    b = dict(MB.build_bbc_mostread())
    vf = S.load_module(MB.HP_DIR / "v_features.py", "vf_homepage")
    hp = pd.read_csv(MB.HP_DIR / "population.csv.gz")
    bbc = hp[hp.outlet == "bbc"]
    pos = [vf.headline_of(t) for t, j in zip(bbc["text"], bbc["judgement"]) if j == 1]
    neg = [vf.headline_of(t) for t, j in zip(bbc["text"], bbc["judgement"]) if j == 0]
    pos = [h for h in pos if len(h.split()) >= 4]
    neg = [h for h in neg if len(h.split()) >= 4]
    allh = pos + neg
    print(f"[bbc anchors] pool pos={len(pos)} neg={len(neg)}", flush=True)
    if min(len(pos), len(neg)) < a.battery:
        print(f"WARN: pool smaller than K={a.battery}; draws will repeat", flush=True)

    def anchors_bbc(shard):
        rng = random.Random(SEED + 607 * shard)
        p = {"id": "", "group": "__anchor", "headline": rng.choice(pos)}
        n = {"id": "", "group": "__anchor", "headline": rng.choice(neg)}
        pool = [rng.choice(allh) for _ in range(POOL_HEADLINES)]
        s = {"id": "", "group": "__anchor",
             "headline": S.scramble(pool, rng, n_words=14)}
        out = []
        for tag, r in (("anchor_pos", p), ("anchor_neg", n), ("anchor_scram", s)):
            rr = dict(r)
            rr["anchor_tag"] = tag
            rr["id"] = f"__BBCANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    b["anchors"] = anchors_bbc
    b["name"] = "bbc_mostread_bbcanchors"

    from vllm import LLM, SamplingParams
    llm = LLM(model=S.GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              trust_remote_code=True, max_num_seqs=512)
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    res = C.run_battery(llm, sp, b, a.battery, OUT)

    shipped = OUT / "anchor_battery.json"
    payload = json.loads(shipped.read_text()) if shipped.exists() else {}
    rep = payload.pop("bbc_mostread_bbcanchors", res)
    shipped.write_text(json.dumps(payload, indent=1))
    (OUT / "anchor_battery_bbc_anchors.json").write_text(json.dumps(
        {"bbc_mostread_bbcanchors": rep,
         "pool_pos": len(pos), "pool_neg": len(neg),
         "note": "Diagnostic: identical to the shipped battery except both "
                 "anchor classes are drawn from the BBC rows of the homepage "
                 "placement population, testing whether the shipped battery's "
                 "chance-level pos-vs-neg (.481) came from asserting BBC "
                 "provenance over non-BBC anchor headlines."}, indent=1))
    print("BBC_ANCHOR_BATTERY_DONE", flush=True)


if __name__ == "__main__":
    main()
