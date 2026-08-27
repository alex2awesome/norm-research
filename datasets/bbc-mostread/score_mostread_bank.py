#!/usr/bin/env python3
"""BBC most-read: score the articulated-criteria (A) bank with the local
Gemma-4-31B judge, offline-batch vLLM, one token per (item, criterion).

REUSE, and deliberately IDENTICAL to the V9 tweets cell wherever it can be. The
whole point of this cell is a controlled comparison -- same field, same item
type (a news headline), same crowd question, different crowd and different
action (BBC readers clicking BBC vs Twitter users amplifying links). So every
instrument is held fixed and only the label changes:

  * bank: datasets/news-homepages/va/rubrics.jsonl -- the SAME 14 GEPA-revised
    news-values criteria V9 reused. Zero new criteria, zero re-GEPA.
  * V features: datasets/news-homepages/va/v_features.py -- the same 23.
  * system prompt: the V9 prompt, with only the platform-specific "do not
    predict" list adjusted (V9 forbids predicting likes/retweets; here it
    forbids predicting clicks/most-read-list membership).
  * scoring loop / shard checkpointing / NA parsing / anchor machinery:
    imported verbatim from datasets/va_gemma_banks/score_va_gemma_banks.py and
    score_scaleupC_banks.py.
  * ANCHORS: the SAME independent channel and the SAME pool as V9 -- homepage
    PLACEMENT from datasets/news-homepages/va/population.csv.gz. Using an
    identical anchor battery across the two journalism community cells means
    the two judge certifications are directly comparable, which they would not
    be if each cell anchored on its own channel.

SCRAMBLED CONTROL: this cell ships with the V9 REPAIR already applied --
the nonsense control pools tokens across 40 unrelated headlines rather than the
pos+neg pair. V9's build note (notes/2026-08-08__v9_journalism_community_build.md
S4.1) documents why: on a ~12-word headline the frozen `scramble` helper reverses
only alternate words drawn from two headlines, so intact proper nouns survive and
the judge correctly scores them 1.0. Note the deeper V9 finding still applies and
is NOT fixed by the repair -- a row score is `nanmean` over 14 criteria, and a
scramble that answers 1-3 of them is not commensurable with a headline that
answers 9. The coherence signal to read is the all-NA rate, not the mean.

TRUNCATION IS IN TOKENS, NOT CHARS. BBC headlines are short (mean 45 chars) so
nothing truncates in practice, but the budget is expressed and asserted in
tokenizer tokens so the limit means the same thing it does to the model.

GPU: one GPU only (CUDA_VISIBLE_DEVICES set by the caller).

  CUDA_VISIBLE_DEVICES=N python3 datasets/bbc-mostread/score_mostread_bank.py --smoke 24
  CUDA_VISIBLE_DEVICES=N python3 datasets/bbc-mostread/score_mostread_bank.py --battery 50
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
import score_va_gemma_banks as S  # noqa: E402
import score_scaleupC_banks as C  # noqa: E402

REPO = S.REPO
OUT = Path(os.environ.get("VA_OUT_BBC",
                          str(REPO / "outputs/va_gemma_banks_bbc_mostread")))
SEED = 20260810
POOL_HEADLINES = 40
MAX_HEADLINE_TOKENS = 128

BBC_DIR = REPO / "datasets/bbc-mostread/va"
HP_DIR = REPO / "datasets/news-homepages/va"
BANK = HP_DIR / "rubrics.jsonl"          # 100% REUSED, same as V9

SYS_BBC = (
    "You are an experienced news editor performing a measurement task. You are "
    "given the HEADLINE of a news article as it appeared on the BBC News home "
    "page, and ONE criterion. Decide how strongly the article, on the evidence "
    "of the headline text alone, satisfies that criterion. Answer with EXACTLY "
    "ONE token:\n"
    "  1.0 = clearly satisfies the criterion\n"
    "  0.5 = partly, weakly, inconsistently, or borderline\n"
    "  0.0 = the relevant element is present but fails or cuts against the criterion\n"
    "  NA = the headline gives no evidence bearing on this criterion\n"
    "Judge this headline on its own text. Do not consider or imagine the other "
    "stories published that day, and do not predict clicks, readership, "
    "most-read-list membership, home-page placement, or dataset membership. "
    "Output only the token."
)


def build_bbc_mostread():
    import pandas as pd
    vf = S.load_module(HP_DIR / "v_features.py", "vf_homepage")
    df = pd.read_csv(BBC_DIR / "population.csv.gz")

    items = [{"id": str(r.row_id), "group": str(r.group),
              "headline": str(r.raw_headline),
              "y_mostread": int(r.judgement),
              "rank": (None if r.rank != r.rank else int(r.rank))}
             for r in df.itertuples()]

    rubrics = [json.loads(l) for l in open(BANK) if l.strip()]
    blocks = [f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n\n"
              "Answer with one token:" for m in rubrics]

    # token-budget assertion (tokens, not chars)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(S.GEMMA4)
        lens = [len(tok.encode(r["headline"], add_special_tokens=False))
                for r in items[:4000]]
        over = sum(1 for x in lens if x > MAX_HEADLINE_TOKENS)
        print(f"[token budget] sampled {len(lens)} headlines: max={max(lens)} "
              f"p99={int(np.percentile(lens,99))} over_{MAX_HEADLINE_TOKENS}={over}",
              flush=True)
    except Exception as e:  # tokenizer unavailable -> record, do not silently skip
        print(f"[token budget] SKIPPED ({type(e).__name__}: {str(e)[:80]})", flush=True)

    def ctx(r):
        return f"HEADLINE: {r['headline']}"

    def vvec(r):
        return vf.vector(r["headline"])

    hp = pd.read_csv(HP_DIR / "population.csv.gz")
    hp_pos = [vf.headline_of(t) for t, j in zip(hp["text"], hp["judgement"]) if j == 1]
    hp_neg = [vf.headline_of(t) for t, j in zip(hp["text"], hp["judgement"]) if j == 0]
    hp_pos = [h for h in hp_pos if len(h.split()) >= 4]
    hp_neg = [h for h in hp_neg if len(h.split()) >= 4]
    allh = hp_pos + hp_neg

    def anchors(shard):
        rng = random.Random(SEED + 607 * shard)
        pos = {"id": "", "group": "__anchor", "headline": rng.choice(hp_pos)}
        neg = {"id": "", "group": "__anchor", "headline": rng.choice(hp_neg)}
        pool = [rng.choice(allh) for _ in range(POOL_HEADLINES)]   # V9 repair
        scr = {"id": "", "group": "__anchor",
               "headline": S.scramble(pool, rng, n_words=14)}
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg),
                       ("anchor_scram", scr)):
            rr = dict(r)
            rr["anchor_tag"] = tag
            rr["id"] = f"__ANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    ys = {"mostread": np.array([r["y_mostread"] for r in items], dtype=float)}
    return dict(name="bbc_mostread", items=items, rubrics=rubrics, blocks=blocks,
                sys=SYS_BBC, ctx=ctx, vvec=vvec, vnames=list(vf.V_NAMES),
                anchors=anchors, ys=ys, n_shards=8,
                meta={"population": "datasets/bbc-mostread/va/population.csv.gz",
                      "group_column": "capture_day",
                      "n_groups": int(df["group"].nunique()),
                      "bank_source": "datasets/news-homepages/va/rubrics.jsonl "
                                     "(100% REUSED; identical to the V9 tweets "
                                     "cell so the two journalism community cells "
                                     "are instrument-matched)",
                      "anchor_label_source": "homepage PLACEMENT, same pool as V9 "
                                             "(independent of the most-read y)",
                      "scramble": f"V9 repair applied: pool of {POOL_HEADLINES} "
                                  "unrelated headlines",
                      "token_budget": MAX_HEADLINE_TOKENS,
                      "y_definition": "1 = in the BBC home page ranked MOST READ "
                                      "top-10 on that capture; 0 = elsewhere on "
                                      "the same capture"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--util", type=float, default=0.60)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--battery", type=int, default=50)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    b = build_bbc_mostread()
    print(f"[build] bbc_mostread: {len(b['items'])} items, {len(b['blocks'])} "
          f"criteria, {len(set(str(r['group']) for r in b['items']))} groups",
          flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=S.GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              trust_remote_code=True, max_num_seqs=512)
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    if a.smoke:
        rows = b["items"][:a.smoke]
        convs = [[{"role": "user",
                   "content": f"{b['sys']}\n\n{b['ctx'](r)}\n\n{blk}"}]
                 for r in rows for blk in b["blocks"]]
        outs = llm.chat(convs, sp)
        X = np.array([S.parse_tok(o.outputs[0].text) for o in outs],
                     dtype=float).reshape(len(rows), len(b["blocks"]))
        print(f"[smoke] n={len(rows)} NA={np.isnan(X).mean():.3f} "
              f"mean={np.nanmean(X):.3f}", flush=True)
        collapsed = []
        for ci, nm in enumerate([m["name"] for m in b["rubrics"]]):
            col = X[:, ci]
            fin = col[np.isfinite(col)]
            vals, cnts = np.unique(fin, return_counts=True)
            modal = float(cnts.max() / max(len(col), 1)) if len(cnts) else 1.0
            na = float(np.isnan(col).mean())
            if na == 1.0 or modal >= 0.95:
                collapsed.append(nm)
            print(f"  {ci:02d} {nm[:52]:54s} mean={np.nanmean(col):.3f} "
                  f"na={na:.2f} modal={modal:.2f}", flush=True)
        # ENFORCED COLLAPSE GATE
        if collapsed:
            print(f"COLLAPSE_GATE_FAIL: {collapsed}", flush=True)
            sys.exit(3)
        print("COLLAPSE_GATE_PASS", flush=True)
        print("SMOKE_DONE", flush=True)
        return

    S.score_bank(llm, sp, b, OUT)
    if a.battery:
        C.run_battery(llm, sp, b, a.battery, OUT)
    print("BBC_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
