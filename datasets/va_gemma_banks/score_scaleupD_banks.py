#!/usr/bin/env python3
"""Scale-up wave D (2026-08-19): two new VERDICT cells, scored with the newest
bank machinery (user rule 2026-08-19: prefer the most recent implementations).

  jokes_removal   r/Jokes moderator-removal verdict (U2). SAME bank + SYS as
                  jokes_community (score_scaleupC_banks) — only the population
                  differs (removal_cell/population.jsonl.gz, 20,543 rows,
                  judgement=1 means REMOVED).
  kindle_scout    Kindle Scout publisher accept/reject (U7, 726 rows). SAME CW
                  bank + SYS as creative_writing (score_va_gemma_banks) — FRAME
                  NOTE: ctx is a novel excerpt with no writing prompt, and long
                  excerpts get the deterministic middle omission.

Machinery imported verbatim: S.score_bank (shards, per-shard blinded anchors,
NA parse, prefix caching) + C.run_battery (K>=50 extended battery).
GPU: one GPU, CUDA_VISIBLE_DEVICES set by caller.
  python score_scaleupD_banks.py --tasks jokes_removal --util 0.85
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import multiprocessing as _mp  # noqa: E402

try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_va_gemma_banks as S  # noqa: E402
import score_scaleupC_banks as C  # noqa: E402

REPO = S.REPO
OUT = C.OUT  # same output dir => loadable via scaleupC_layer1.load_scaleupC_bank
SEED = 20260819

JOKES_DIR = REPO / "datasets/humor/reddit_jokes"
KS_POP = REPO / "datasets/creative-writing/kindle_scout_cell/ks_verdict_population.jsonl.gz"

TRUNC_MARK = "\n\n[... DETERMINISTIC MIDDLE OMISSION ...]\n\n"


def _trunc(s, src=9500, head=6000, tail=3000):
    s = (s or "").strip()
    return s if len(s) <= src else s[:head] + TRUNC_MARK + s[-tail:]


def _month(ts):
    try:
        t = float(ts)
        if t > 0:
            return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m")
    except (TypeError, ValueError):
        pass
    return "unknown"


# ============================== jokes_removal ================================
def build_jokes_removal(pop="removal_cell/population.jsonl.gz", name="jokes_removal"):
    vf = S.load_module(JOKES_DIR / "va/v_features.py", "vf_jokes_removal")
    rows = [json.loads(l) for l in gzip.open(JOKES_DIR / pop, "rt")]
    items = [{"id": r["row_id"], "group": _month(r.get("created_utc")),
              "text": str(r["text"])[:5000], "judgement": int(r["judgement"])}
             for r in rows]

    rubrics = [json.loads(l) for l in open(JOKES_DIR / "va/rubrics.jsonl") if l.strip()]
    blocks = [f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n\n"
              "Answer with one token:" for m in rubrics]

    def ctx(r):
        return f'JOKE:\n"{r["text"]}"'

    def vvec(r):
        return vf.vector(r["text"])

    def anchors(shard):
        rng = random.Random(SEED + 401 * shard)
        pos_pool = [r for r in items if r["judgement"] == 1]
        neg_pool = [r for r in items if r["judgement"] == 0]
        pos, neg = dict(rng.choice(pos_pool)), dict(rng.choice(neg_pool))
        scr = dict(neg)
        scr["text"] = S.scramble([pos["text"], neg["text"]], rng)
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg), ("anchor_scram", scr)):
            rr = dict(r)
            rr["anchor_tag"] = tag
            rr["id"] = f"__ANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    ys = {"removed": np.array([r["judgement"] for r in items])}
    return dict(
        name=name, items=items, rubrics=rubrics, blocks=blocks,
        sys=C.SYS_JOKES, ctx=ctx, vvec=vvec, vnames=list(vf.V_NAMES),
        anchors=anchors, ys=ys, n_shards=10,
        meta={"population": "datasets/humor/reddit_jokes/removal_cell/population.jsonl.gz",
              "y_semantics": "judgement=1 means REMOVED by moderators "
                             "(anchor_pos = a removed joke)",
              "bank": "SAME rubrics + SYS as jokes_community (scaleupC)",
              "normalization": "v2 = matched renderer (leak fix 2026-08-19)" if name.endswith("v2") else "v1 RAW (retracted)",
              "confounds_declared": ["created era (stratified controls)", "over_18",
                                     "removal-reason mix (reposts vs rules vs quality)"],
              "group_column": "created month (era stratum)"},
    )


# ============================== kindle_scout =================================
def build_kindle_scout():
    base = S.build_creative()  # exact CW bank: rubrics/blocks/sys/vf as the community cell
    rows = [json.loads(l) for l in gzip.open(KS_POP, "rt")]
    items = [{"id": r["row_id"],
              "group": (r["genres"][0] if r.get("genres") else "none"),
              "text": _trunc(str(r["text"])), "judgement": int(r["judgement"])}
             for r in rows]

    def ctx(r):
        return f"NOVEL EXCERPT (opening pages submitted to a publisher):\n{r['text']}"

    vf = S.load_module(REPO / "datasets/creative-writing/va_bank_v2/v_features.py",
                       "vf_cw_ks")

    def vvec(r):
        return vf.feature_vector(r["text"])

    def anchors(shard):
        rng = random.Random(SEED + 907 * shard)
        pos_pool = [r for r in items if r["judgement"] == 1]
        neg_pool = [r for r in items if r["judgement"] == 0]
        pos, neg = dict(rng.choice(pos_pool)), dict(rng.choice(neg_pool))
        scr = dict(neg)
        scr["text"] = S.scramble([pos["text"][:4000], neg["text"][:4000]], rng,
                                 n_words=200)
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg), ("anchor_scram", scr)):
            rr = dict(r)
            rr["anchor_tag"] = tag
            rr["id"] = f"__ANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    ys = {"publisher_accept": np.array([r["judgement"] for r in items])}
    return dict(
        name="kindle_scout", items=items, rubrics=base["rubrics"],
        blocks=base["blocks"], sys=base["sys"], ctx=ctx, vvec=vvec,
        vnames=list(vf.V_NAMES), anchors=anchors, ys=ys, n_shards=2,
        meta={"population": str(KS_POP.relative_to(REPO)),
              "bank": "SAME CW va_bank_v2 rubrics + SYS_CW as creative_writing cell",
              "frame_note": "ctx = novel excerpt, NO writing prompt (community cell "
                            "carries prompt+story); excerpts middle-omitted at "
                            "6000+3000 chars — instrument same, INPUT FRAME differs",
              "leak_note": "action_msg excluded upstream (states verdict verbatim)",
              "group_column": "primary genre"},
    )


BUILDERS = {"jokes_removal": build_jokes_removal, "kindle_scout": build_kindle_scout,
            "jokes_removal_v2": lambda: build_jokes_removal(
                "removal_cell/population_v2.jsonl.gz", "jokes_removal_v2")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="jokes_removal")
    ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--battery", type=int, default=50)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
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
            print("SMOKE_DONE", flush=True)
            continue
        S.score_bank(llm, sp, b, OUT)
        if a.battery:
            C.run_battery(llm, sp, b, a.battery, OUT)
    print("SCALEUPD_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
