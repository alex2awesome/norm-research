#!/usr/bin/env python3
"""U3/U4 — A-bank scoring for the two BOUNTY (curated) cells, one Gemma load.

INSTRUMENT IDENTITY: every convention is IMPORTED from the sibling scorers, never
re-typed — mathse_bounty uses score_scaleupC_banks' SYS_MATHSE / _trunc / rubric
file / v_features verbatim; so_bounty uses score_so_votes_bank's SYS_SO / ctx
shape (title + tags + truncated question body + answer) / rubric file /
v_features verbatim.  Shard scoring and the blinded anchor battery are the
framework's own (S.score_bank, C.run_battery).

Anchor label = the cell's own y (manual bounty award): pos/neg/scrambled anchors
per shard, the closure-scorer convention.  Recorded in the meta.

Usage (via gpu_runner):
  python score_bounty_banks.py --tasks mathse_bounty,so_bounty --smoke 40   # gate
  python score_bounty_banks.py --tasks mathse_bounty,so_bounty             # full
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
REPO = _HERE.parents[2]
sys.path.insert(0, str(REPO / "datasets/va_gemma_banks"))
sys.path.insert(0, str(REPO / "datasets/stackoverflow-votes"))
import score_va_gemma_banks as S  # noqa: E402
import score_scaleupC_banks as C  # noqa: E402
import score_so_votes_bank as SOB  # noqa: E402

SEED = 20260816


def build_mathse_bounty():
    import pandas as pd
    vf = S.load_module(REPO / "datasets/math/stackexchange/va/v_features.py", "vf_mb")
    df = pd.read_csv(REPO / "datasets/math-se/mathse_bounty/population.csv.gz")
    items = []
    for r in df.itertuples():
        q, ans = str(r.text).split("\n\nANSWER:\n", 1)
        items.append({"id": str(r.row_id), "group": str(r.group),
                      "question": q.removeprefix("QUESTION: ").strip(),
                      "answer": ans, "y": int(r.judgement)})
    rubrics = [json.loads(l) for l in
               open(REPO / "datasets/math/stackexchange/va/rubrics.jsonl") if l.strip()]
    blocks = [f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n\n"
              "Answer with one token:" for m in rubrics]

    def ctx(r):
        return f"QUESTION TITLE: {r['question'][:400]}\n\nANSWER:\n{C._trunc(r['answer'])}"

    def vvec(r):
        return vf.vector(r["answer"])

    def anchors(shard):
        rng = random.Random(SEED + 503 * shard)
        pos = dict(rng.choice([r for r in items if r["y"] == 1]))
        neg = dict(rng.choice([r for r in items if r["y"] == 0]))
        scr = dict(neg)
        scr["answer"] = S.scramble([pos["answer"][:4000], neg["answer"][:4000]],
                                   rng, n_words=200)
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg), ("anchor_scram", scr)):
            rr = dict(r)
            rr["anchor_tag"] = tag
            rr["id"] = f"__ANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    ys = {"bounty": np.array([r["y"] for r in items])}
    return dict(name="mathse_bounty", items=items, rubrics=rubrics, blocks=blocks,
                sys=C.SYS_MATHSE, ctx=ctx, vvec=vvec, vnames=list(vf.V_NAMES),
                anchors=anchors, ys=ys, n_shards=8,
                meta={"population": "datasets/math-se/mathse_bounty/population.csv.gz",
                      "group_column": "question_id",
                      "n_groups": int(df["group"].nunique()),
                      "bank_source": "datasets/math/stackexchange/va/rubrics.jsonl "
                                     "(the mathse_multiy sibling bank, verbatim)",
                      "anchor_label_source": "own y (manual bounty award)",
                      "conventions": "SYS/trunc/ctx = score_scaleupC_banks "
                                     "build_mathse_multiy verbatim (imported)"})


def build_so_bounty():
    import pandas as pd
    vf = S.load_module(REPO / "datasets/stackoverflow-votes/va/v_features.py", "vf_sb")
    df = pd.read_csv(REPO / "datasets/stackoverflow-votes/so_bounty/population.csv.gz")
    import gzip
    aux = {}
    for l in gzip.open("/lfs/skampere3/0/alexspan/data/se_dumps/"
                       "so_bounty_manual_population.jsonl.gz", "rt"):
        r = json.loads(l)
        aux[str(r["aid"])] = r
    # tags pre-dumped to JSON (gemma4 env's pandas cannot load a usable parquet
    # engine even with pyarrow installed; dumped by ai_usage env)
    tag_of = json.load(open(REPO / "datasets/stackoverflow-votes/so_bounty/qid_tags.json"))
    items = []
    for r in df.itertuples():
        a = aux[str(r.row_id)]
        items.append({"id": str(r.row_id), "group": str(r.group),
                      "question": str(a["q_title"]), "q_body": str(a["q_body"]),
                      "tags": tag_of.get(str(r.group), ""),
                      "answer": str(a["answer_body"]).strip(),
                      "y": int(r.judgement)})
    rubrics = [json.loads(l) for l in open(SOB.SO_BANK) if l.strip()]
    blocks = [f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n\n"
              "Answer with one token:" for m in rubrics]

    def ctx(r):
        return (f"QUESTION TITLE: {r['question'][:400]}\n"
                f"QUESTION TAGS: {r['tags'][:200]}\n\n"
                f"QUESTION BODY:\n{SOB._trunc(r['q_body'], 2400, 1600, 800)}\n\n"
                f"ANSWER:\n{SOB._trunc(r['answer'])}")

    def vvec(r):
        return vf.vector(r["answer"])

    def anchors(shard):
        rng = random.Random(SEED + 607 * shard)
        pos = dict(rng.choice([r for r in items if r["y"] == 1]))
        neg = dict(rng.choice([r for r in items if r["y"] == 0]))
        scr = dict(neg)
        scr["answer"] = S.scramble([pos["answer"][:4000], neg["answer"][:4000]],
                                   rng, n_words=200)
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg), ("anchor_scram", scr)):
            rr = dict(r)
            rr["anchor_tag"] = tag
            rr["id"] = f"__ANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    ys = {"bounty": np.array([r["y"] for r in items])}
    return dict(name="so_bounty", items=items, rubrics=rubrics, blocks=blocks,
                sys=SOB.SYS_SO, ctx=ctx, vvec=vvec, vnames=list(vf.V_NAMES),
                anchors=anchors, ys=ys, n_shards=8,
                meta={"population": "datasets/stackoverflow-votes/so_bounty/population.csv.gz",
                      "group_column": "question_id",
                      "n_groups": int(df["group"].nunique()),
                      "bank_source": str(SOB.SO_BANK) + " (the so_votes sibling bank, verbatim)",
                      "anchor_label_source": "own y (manual bounty award)",
                      "conventions": "SYS/ctx/trunc = score_so_votes_bank verbatim "
                                     "(imported; question BODY is load-bearing there)"})


BUILDERS = {"mathse_bounty": build_mathse_bounty, "so_bounty": build_so_bounty}
OUTS = {"mathse_bounty": REPO / "outputs/va_gemma_banks_mathse_bounty",
        "so_bounty": REPO / "outputs/va_gemma_banks_so_bounty"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="mathse_bounty,so_bounty")
    ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--battery", type=int, default=50)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()

    banks = []
    for t in [x for x in a.tasks.split(",") if x]:
        b = BUILDERS[t]()
        print(f"[build] {t}: {len(b['items'])} items, {len(b['blocks'])} criteria, "
              f"{len(set(str(r['group']) for r in b['items']))} groups", flush=True)
        banks.append(b)

    from vllm import LLM, SamplingParams
    llm = LLM(model=S.GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              trust_remote_code=True, max_num_seqs=512)
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    for b in banks:
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
            print(f"[smoke:{b['name']}] n={len(rows)} NA={np.isnan(X).mean():.3f} "
                  f"mean={np.nanmean(X):.3f}", flush=True)
            for ci, nm in enumerate([m["name"] for m in b["rubrics"]]):
                col = X[:, ci]
                _, cnts = np.unique(col[np.isfinite(col)], return_counts=True)
                modal = float(cnts.max() / max(len(col), 1)) if len(cnts) else 1.0
                print(f"  {ci:02d} {nm[:56]:58s} mean={np.nanmean(col):.3f} "
                      f"na={np.isnan(col).mean():.2f} modal={modal:.2f}", flush=True)
            continue
        out = OUTS[b["name"]]
        out.mkdir(parents=True, exist_ok=True)
        S.score_bank(llm, sp, b, out)
        if a.battery:
            C.run_battery(llm, sp, b, a.battery, out)
    print("SMOKE_DONE" if a.smoke else "BOUNTY_BANKS_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
