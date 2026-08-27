#!/usr/bin/env python3
"""V9 journalism-tweets: score the articulated-criteria (A) bank with the local
Gemma-4-31B judge, offline-batch vLLM, one token per (item, criterion).

REUSE: the scoring loop, shard checkpointing, per-shard 3-row blinded anchors,
NA parsing, prefix caching, temperature 0 / max_tokens 6, and the extended
anchor battery are ALL imported verbatim from
`datasets/va_gemma_banks/score_va_gemma_banks.py` and
`datasets/va_gemma_banks/score_scaleupC_banks.py`. Only the bank builder below
is new.

THE BANK IS 100% REUSED -- datasets/news-homepages/va/rubrics.jsonl, the 14
GEPA-revised news-values criteria authored for the homepage CURATION cell. Zero
new criteria are authored here and zero re-GEPA is run, mirroring the V8
N&C co-signing build ("A-bank 100% REUSED, zero new judging"). The reuse is
population-exact, not merely plausible: that bank was written to be scored on a
homepage headline from these same outlets, and its criterion texts are
explicitly headline-scoped ("...is one of the story's central subjects. Score
1.0 when such a figure is a principal actor or subject of the headline").
The V9 item IS such a headline. Two consequences of reusing rather than
authoring, recorded as inherited limitations:
  * the homepage bank carries no Track A / Track B field, so V9 cannot split
    A_real from A_surface the way V6 SO-votes can. That is a regression
    relative to V6 and is noted in the build note rather than papered over.
  * the bank was distilled against homepage PLACEMENT, so if it underperforms
    on engagement that is an interpretable finding (the articulated news-values
    vocabulary is an editor's vocabulary), not an instrument bug.

CONTEXT IS THE HEADLINE ALONE. Deliberate, and different from the homepage
sibling, which appends a CONTEXT block of the other headlines in the same
snapshot. Two reasons: (a) the V9 group IS the outlet-day, so a
sibling-headline block would be group-CONSTANT and therefore carries exactly
zero within-group rank information while multiplying prompt cost; (b) article
bodies exist for only part of the population (latimes/cnn/guardian ~92-100%,
nytimes 20%, wapo 6%, reuters 9% -- paywalls), so splicing bodies in would make
the evidence base differ systematically by outlet. Headline-only keeps V, A and
the dense arm on BYTE-IDENTICAL input, which is what the program's
apples-to-apples rule requires.

ANCHOR LABEL SOURCE = homepage PLACEMENT (the curation cell's `judgement`),
NOT this cell's engagement y. Anchor rows are drawn from
datasets/news-homepages/va/population.csv.gz, whose label is "link rendered in
the TOP half of the homepage capture's top-30% zone" -- an editorial prominence
channel that is independent of Twitter engagement. Anchoring on the engagement
y itself would make the battery partly circular with the quantity under test.
(Same discipline as V6 SO-votes, which anchored on y_accepted while testing
y_vote.)

GPU: one GPU only (CUDA_VISIBLE_DEVICES set by the caller).

  CUDA_VISIBLE_DEVICES=N python3 datasets/journalism-tweets/score_tweets_bank.py \
      --smoke 24                # inspect NA/modal before scale
  CUDA_VISIBLE_DEVICES=N python3 datasets/journalism-tweets/score_tweets_bank.py \
      --battery 50
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
OUT = Path(os.environ.get("VA_OUT_TWEETS",
                          str(REPO / "outputs/va_gemma_banks_journalism_tweets")))
SEED = 20260808

TW_DIR = REPO / "datasets/journalism-tweets/va"
HP_DIR = REPO / "datasets/news-homepages/va"
BANK = HP_DIR / "rubrics.jsonl"          # 100% REUSED

SYS_TW = (
    "You are an experienced news editor performing a measurement task. You are "
    "given the HEADLINE of a news article as it appeared on a major outlet's "
    "home page, and ONE criterion. Decide how strongly the article, on the "
    "evidence of the headline text alone, satisfies that criterion. Answer with "
    "EXACTLY ONE token:\n"
    "  1.0 = clearly satisfies the criterion\n"
    "  0.5 = partly, weakly, inconsistently, or borderline\n"
    "  0.0 = the relevant element is present but fails or cuts against the criterion\n"
    "  NA = the headline gives no evidence bearing on this criterion\n"
    "Judge this headline on its own text. Do not consider or imagine the other "
    "stories published that day, and do not predict clicks, shares, likes, "
    "retweets, traffic, home-page placement, outlet identity, or dataset "
    "membership. Some headlines begin with a run-on section or byline label "
    "(for example 'ArizonaOfficials investigating...'); read past it. Output "
    "only the token."
)


def build_journalism_tweets():
    import pandas as pd
    vf = S.load_module(HP_DIR / "v_features.py", "vf_homepage")
    df = pd.read_csv(TW_DIR / "population.csv.gz")

    items = []
    for r in df.itertuples():
        items.append({"id": str(r.row_id), "group": str(r.group),
                      "headline": str(r.raw_headline),
                      "y_engagement": int(r.judgement),
                      "y_maxlikes": int(r.y_maxlikes),
                      "y_quartile": int(r.y_quartile)})

    rubrics = [json.loads(l) for l in open(BANK) if l.strip()]
    blocks = [f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n\n"
              "Answer with one token:" for m in rubrics]

    def ctx(r):
        return f"HEADLINE: {r['headline'][:600]}"

    def vvec(r):
        return vf.vector(r["headline"])

    # ---- anchors from the INDEPENDENT homepage-placement channel -----------
    hp = pd.read_csv(HP_DIR / "population.csv.gz")
    hp_pos = [vf.headline_of(t) for t, j in zip(hp["text"], hp["judgement"]) if j == 1]
    hp_neg = [vf.headline_of(t) for t, j in zip(hp["text"], hp["judgement"]) if j == 0]
    hp_pos = [h for h in hp_pos if len(h.split()) >= 4]
    hp_neg = [h for h in hp_neg if len(h.split()) >= 4]

    def anchors(shard):
        rng = random.Random(SEED + 607 * shard)
        pos = {"id": "", "group": "__anchor", "headline": rng.choice(hp_pos)}
        neg = {"id": "", "group": "__anchor", "headline": rng.choice(hp_neg)}
        scr = dict(neg)
        scr["headline"] = S.scramble([pos["headline"], neg["headline"]],
                                     rng, n_words=14)
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg),
                       ("anchor_scram", scr)):
            rr = dict(r)
            rr["anchor_tag"] = tag
            rr["id"] = f"__ANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    ys = {"engagement": np.array([r["y_engagement"] for r in items], dtype=float),
          "maxlikes": np.array([np.nan if r["y_maxlikes"] < 0 else r["y_maxlikes"]
                                for r in items], dtype=float),
          "quartile": np.array([np.nan if r["y_quartile"] < 0 else r["y_quartile"]
                                for r in items], dtype=float)}

    return dict(name="journalism_tweets", items=items, rubrics=rubrics,
                blocks=blocks, sys=SYS_TW, ctx=ctx, vvec=vvec,
                vnames=list(vf.V_NAMES), anchors=anchors, ys=ys, n_shards=7,
                meta={"population": "datasets/journalism-tweets/va/population.csv.gz",
                      "group_column": "outlet_day",
                      "n_groups": int(df["group"].nunique()),
                      "bank_source": "datasets/news-homepages/va/rubrics.jsonl "
                                     "(100% REUSED from the homepage curation "
                                     "cell; 14 criteria; no Track A/B field)",
                      "anchor_label_source": "homepage PLACEMENT judgement from "
                                             "datasets/news-homepages/va/"
                                             "population.csv.gz (independent of "
                                             "the engagement y under test)",
                      "context": "headline ONLY -- sibling headlines would be "
                                 "group-constant, and article bodies exist for "
                                 "only part of the population (outlet-dependent "
                                 "paywalls), so headline-only keeps V/A/dense on "
                                 "identical input",
                      "y_definition": "1 = sum_likes strictly above the median "
                                      "sum_likes of its own (outlet, day) "
                                      "homepage group; 0 = strictly below; "
                                      "ties dropped"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--util", type=float, default=0.60)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--battery", type=int, default=50)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    b = build_journalism_tweets()
    print(f"[build] journalism_tweets: {len(b['items'])} items, "
          f"{len(b['blocks'])} criteria, "
          f"{len(set(str(r['group']) for r in b['items']))} groups", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=S.GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              trust_remote_code=True, max_num_seqs=512)
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    if a.smoke:
        rows = b["items"][:a.smoke]
        convs = []
        for r in rows:
            c = b["ctx"](r)
            for blk in b["blocks"]:
                convs.append([{"role": "user",
                               "content": f"{b['sys']}\n\n{c}\n\n{blk}"}])
        outs = llm.chat(convs, sp)
        X = np.array([S.parse_tok(o.outputs[0].text) for o in outs],
                     dtype=float).reshape(len(rows), len(b["blocks"]))
        print(f"[smoke] n={len(rows)} NA={np.isnan(X).mean():.3f} "
              f"mean={np.nanmean(X):.3f}", flush=True)
        for ci, nm in enumerate([m["name"] for m in b["rubrics"]]):
            col = X[:, ci]
            vals, cnts = np.unique(col[np.isfinite(col)], return_counts=True)
            modal = float(cnts.max() / max(len(col), 1)) if len(cnts) else 1.0
            print(f"  {ci:02d} {nm[:56]:58s} mean={np.nanmean(col):.3f} "
                  f"na={np.isnan(col).mean():.2f} modal={modal:.2f}", flush=True)
        print("SMOKE_DONE", flush=True)
        return

    S.score_bank(llm, sp, b, OUT)
    if a.battery:
        C.run_battery(llm, sp, b, a.battery, OUT)
    print("TWEETS_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
