#!/usr/bin/env python3
"""V6 SO-votes: score the articulated-criteria (A) bank with the local
Gemma-4-31B judge, offline-batch vLLM, one token per (item, criterion).

REUSE: the scoring loop, shard checkpointing, per-shard 3-row blinded anchors,
NA parsing, prefix caching, temperature 0 / max_tokens 6, and the extended
anchor battery are ALL imported verbatim from
`datasets/va_gemma_banks/score_va_gemma_banks.py` and
`datasets/va_gemma_banks/score_scaleupC_banks.py`. Only the bank builder below
is new -- exactly the pattern those files establish for a new cell.

Anchor discipline (binding, every judging batch): every shard carries 3 blinded
anchor rows (known positive / known negative / scrambled nonsense) AND the run
is followed by the extended battery at K >= 50 per class (`--battery 50`).

ANCHOR LABEL SOURCE = `y_accepted`, NOT this cell's `y_vote`. The accept verdict
is an independent quality channel on the same rows, so the battery certifies the
judge against a signal it is not being asked to reproduce. Using y_vote would
make the battery partly circular with the quantity under test. (The math.SE
sibling did the same: its multi-y bank anchored on y_accepted.)

GPU: one GPU only (CUDA_VISIBLE_DEVICES set by the caller).

  CUDA_VISIBLE_DEVICES=N python3 datasets/stackoverflow-votes/score_so_votes_bank.py \
      --smoke 24                # inspect before scale (validate-before-scaling)
  CUDA_VISIBLE_DEVICES=N python3 datasets/stackoverflow-votes/score_so_votes_bank.py \
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
OUT = Path(os.environ.get("VA_OUT_SO",
                          str(REPO / "outputs/va_gemma_banks_so_votes")))
SEED = 20260808

SO_DIR = REPO / "datasets/stackoverflow-votes/va"
SO_BANK = SO_DIR / "rubrics.jsonl"

SYS_SO = (
    "You are an expert Python engineer performing a measurement task. You are "
    "given the title of a question asked on a programming question-and-answer "
    "site, ONE answer posted to it, and ONE quality criterion. Decide how "
    "strongly the answer, on the evidence of the supplied text alone, satisfies "
    "that criterion. Answer with EXACTLY ONE token:\n"
    "  1.0 = clearly satisfies the criterion\n"
    "  0.5 = partly, weakly, inconsistently, or borderline\n"
    "  0.0 = the relevant attempt is present but fails or cuts against the criterion\n"
    "  NA = the answer gives no evidence bearing on this criterion\n"
    "Judge this answer on its own text. Do not consider or imagine other answers "
    "to the same question, and do not predict acceptance, votes, scores, "
    "reputation, authorship, posting order, or dataset membership. The answer is "
    "Markdown: fenced blocks and indented blocks are code. Long answers may have "
    "a deterministically omitted middle; judge what is shown. Output only the token."
)

TRUNC_SRC, TRUNC_HEAD, TRUNC_TAIL = 5000, 3000, 2000
TRUNC_MARK = "\n\n[... DETERMINISTIC MIDDLE OMISSION ...]\n\n"


def _trunc(s, src=TRUNC_SRC, head=TRUNC_HEAD, tail=TRUNC_TAIL):
    s = (s or "").strip()
    return s if len(s) <= src else s[:head] + TRUNC_MARK + s[-tail:]


def build_so_votes():
    import pandas as pd
    vf = S.load_module(SO_DIR / "v_features.py", "vf_so_votes")
    df = pd.read_csv(SO_DIR / "population.csv.gz")
    items = []
    for r in df.itertuples():
        items.append({"id": str(r.row_id), "group": str(r.group),
                      "question": str(r.q_title), "q_body": str(r.q_body),
                      "tags": str(r.tags), "answer": str(r.body),
                      "y_accepted": int(r.y_accepted),
                      "y_vote": (None if r.y_vote != r.y_vote else int(r.y_vote))})

    rubrics = [json.loads(l) for l in open(SO_BANK) if l.strip()]
    blocks = [f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n\n"
              "Answer with one token:" for m in rubrics]

    def ctx(r):
        # THE QUESTION BODY IS LOAD-BEARING HERE, unlike the math.SE sibling.
        # A title-only context (the math.SE recipe, copied verbatim in the first
        # pass) drove the smoke NA rate to 51.8%, with THREE criteria at 100% NA
        # -- "Engages the asker's actual code", "Edge cases in the asker's data
        # addressed", "Corrects a misconception in the question". Those are not
        # bad criteria; the judge was answering NA correctly, because a large
        # share of this bank is RELATIONAL to the asker's posted code and a SO
        # question carries its code in the body, not the title. math.SE's
        # criteria mostly judge an answer's internal argument, so a title
        # sufficed there. Tags are included because two criteria turn on the
        # version/library the question indicates.
        # Group-constant by construction (every answer to a question sees the
        # same question text), so this cannot manufacture within-question rank.
        return (f"QUESTION TITLE: {r['question'][:400]}\n"
                f"QUESTION TAGS: {r['tags'][:200]}\n\n"
                f"QUESTION BODY:\n{_trunc(r['q_body'], 2400, 1600, 800)}\n\n"
                f"ANSWER:\n{_trunc(r['answer'])}")

    def vvec(r):
        return vf.vector(r["answer"])

    def anchors(shard):
        rng = random.Random(SEED + 607 * shard)
        pos = dict(rng.choice([r for r in items if r["y_accepted"] == 1]))
        neg = dict(rng.choice([r for r in items if r["y_accepted"] == 0]))
        scr = dict(neg)
        scr["answer"] = S.scramble([pos["answer"][:4000], neg["answer"][:4000]],
                                   rng, n_words=200)
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg),
                       ("anchor_scram", scr)):
            rr = dict(r)
            rr["anchor_tag"] = tag
            rr["id"] = f"__ANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    ys = {"vote_score": np.array([np.nan if r["y_vote"] is None else r["y_vote"]
                                  for r in items], dtype=float),
          "accepted_verdict": np.array([r["y_accepted"] for r in items])}
    return dict(name="so_votes", items=items, rubrics=rubrics, blocks=blocks,
                sys=SYS_SO, ctx=ctx, vvec=vvec, vnames=list(vf.V_NAMES),
                anchors=anchors, ys=ys, n_shards=7,
                meta={"population": "datasets/stackoverflow-votes/va/population.csv.gz",
                      "group_column": "question_id",
                      "n_groups": int(df["group"].nunique()),
                      "anchor_label_source": "y_accepted (independent of the "
                                             "vote y under test)",
                      "context": "question title + tags + truncated question "
                                 "BODY + answer (the body is required: a "
                                 "title-only context left 3 criteria at 100% NA)",
                      "truncation": {"source_chars": TRUNC_SRC, "head": TRUNC_HEAD,
                                     "tail": TRUNC_TAIL, "question_title_chars": 400,
                                     "question_body": [2400, 1600, 800]}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--battery", type=int, default=50)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    b = build_so_votes()
    print(f"[build] so_votes: {len(b['items'])} items, {len(b['blocks'])} criteria, "
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
        print(f"[smoke:so_votes] n={len(rows)} NA={np.isnan(X).mean():.3f} "
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
    print("SO_VOTES_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
