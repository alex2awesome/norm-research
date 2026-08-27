#!/usr/bin/env python3
"""U1 RoyalRoad unified-X triple — GPU scoring for the two NEW cells (wave-D
pattern, 2026-08-22). VERDICT (rr_v1) is already scored + ledgered; these are:

  rr_community   COMMUNITY: reader engagement. X = fiction DESCRIPTION (deep-
                 page blurb), y = followers > within-(genre,year)-stratum
                 median (listing snapshot 2026-08-12). Population:
                 datasets/creative-writing/royalroad_community_cell/
                 rr_community_population.jsonl.gz (n=3,604). Probe gate
                 PASS_WITH_CONSTRUCT_NOTE (content+blurb-convention
                 separability, chandra precedent; probe_results.json).
  rr_magazine    CURATED: Community Magazine contest picks. X = entry chapter
                 text, y = ranked winner in the blog announcement (26 pos /
                 2,012 labeled rows, 8 editions; PILOT power DECLARED).
                 Unlabeled-edition rows (2022-01/06) are scored too (scores
                 are label-blind); they carry edition_labeled=False.

Bank: SAME CW va_bank_v2 GEPA bank (45 criteria) + Gemma-4-31B judge as every
CW cell — REUSED, not re-mined (feedback_reuse_before_rebuild). Machinery
imported verbatim from score_va_gemma_banks (S) + score_scaleupC_banks (C).
Output dir = C.OUT so scaleupC_layer1 loaders work unchanged.

GPU: ONE GPU, CUDA_VISIBLE_DEVICES set by the caller. On sk2 export
NR_REPO=/lfs/skampere2/0/alexspan/norm-research and GEMMA4_PATH to the sk2
shared-cache snapshot.
  python score_rr_unified_banks.py --tasks rr_community,rr_magazine --util 0.85
"""
from __future__ import annotations

import argparse
import gzip
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_va_gemma_banks as S  # noqa: E402
import score_scaleupC_banks as C  # noqa: E402

REPO = S.REPO
OUT = C.OUT
CW = REPO / "datasets/creative-writing"
SEED = 20260822

TRUNC_MARK = "\n\n[... DETERMINISTIC MIDDLE OMISSION ...]\n\n"


def _trunc(s, src=9500, head=6000, tail=3000):
    s = (s or "").strip()
    return s if len(s) <= src else s[:head] + TRUNC_MARK + s[-tail:]


def _cw_bank():
    rubrics = [json.loads(l) for l in open(CW / "va_bank_v2/rubrics_initial.jsonl")
               if l.strip()]
    blocks = [f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n"
              f"GUIDANCE: {m.get('guidance', '')}\n\nAnswer with one token:"
              for m in rubrics]
    vf = S.load_module(CW / "va_bank_v2/v_features.py", "vf_rr_unified")
    return rubrics, blocks, vf


SYS_RR_COMMUNITY = (
    "You are an expert fiction editor performing a measurement task. You are given "
    "the DESCRIPTION (blurb) an author wrote for their serialised web novel and ONE "
    "craft criterion. Decide how strongly the description, on the evidence of the "
    "supplied text alone, satisfies that criterion. Answer with EXACTLY ONE token:\n"
    "  1.0 = clearly satisfies the criterion\n"
    "  0.5 = partly, weakly, inconsistently, or borderline\n"
    "  0.0 = the relevant attempt is present but fails or cuts against the criterion\n"
    "  NA = the supplied text gives no evidence bearing on this criterion\n"
    "Judge the writing of the description itself. Do not predict or infer follower "
    "counts, ratings, views, popularity, genre popularity, update cadence, "
    "completion status, authorship, or dataset membership, and do not compare it "
    "with other stories. A blurb is short by design: judge the craft of what is "
    "shown. Output only the token."
)

SYS_RR_MAGAZINE = (
    "You are an expert fiction editor performing a measurement task. You are given "
    "the OPENING CHAPTER of a web novel submitted to a community writing contest "
    "and ONE craft criterion. Decide how strongly the chapter, on the evidence of "
    "the supplied text alone, satisfies that criterion. Answer with EXACTLY ONE "
    "token:\n"
    "  1.0 = clearly satisfies the criterion\n"
    "  0.5 = partly, weakly, inconsistently, or borderline\n"
    "  0.0 = the relevant attempt is present but fails or cuts against the criterion\n"
    "  NA = the supplied text gives no evidence bearing on this criterion\n"
    "Judge the writing on its own text. Do not predict or infer contest results, "
    "prizes, judge picks, rankings, magazine inclusion, follower counts, ratings, "
    "views, authorship, or dataset membership, and do not compare this chapter with "
    "other entries. A chapter is a partial work: judge the craft of what is shown, "
    "not whether the story is finished. Long chapters may have a deterministically "
    "omitted middle; judge what is shown. Output only the token."
)


def _mk_anchors(items, label_fn, salt, scr_chars=3000, n_words=None):
    def anchors(shard):
        rng = random.Random(SEED + salt * shard)
        pos_pool = [r for r in items if label_fn(r) == 1]
        neg_pool = [r for r in items if label_fn(r) == 0]
        pos, neg = dict(rng.choice(pos_pool)), dict(rng.choice(neg_pool))
        scr = dict(neg)
        scr["text"] = S.scramble([pos["text"][:scr_chars], neg["text"][:scr_chars]],
                                 rng, n_words=n_words)
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg), ("anchor_scram", scr)):
            rr = dict(r)
            rr["anchor_tag"] = tag
            rr["id"] = f"__ANCHOR_{shard}_{tag}"
            out.append(rr)
        return out
    return anchors


def build_rr_community():
    rubrics, blocks, vf = _cw_bank()
    pop = CW / "royalroad_community_cell/rr_community_population.jsonl.gz"
    rows = [json.loads(l) for l in gzip.open(pop, "rt")]
    items = [{"id": r["row_id"], "group": r["stratum"], "text": str(r["text"])[:6000],
              "judgement": int(r["judgement"])} for r in rows]

    def ctx(r):
        return f'FICTION DESCRIPTION (author blurb for a serialised web novel):\n"{r["text"]}"'

    ys = {"followers_above_stratum_median": np.array([r["judgement"] for r in items])}
    return dict(
        name="rr_community", items=items, rubrics=rubrics, blocks=blocks,
        sys=SYS_RR_COMMUNITY, ctx=ctx, vvec=lambda r: vf.feature_vector(r["text"]),
        vnames=list(vf.V_NAMES),
        anchors=_mk_anchors(items, lambda r: r["judgement"], 613, scr_chars=2000),
        ys=ys, n_shards=6,
        meta={"population": str(pop.relative_to(REPO)),
              "y_semantics": "1 = followers above within-(genre,year)-stratum median",
              "bank": "SAME CW va_bank_v2 rubrics; SYS_RR_COMMUNITY (blurb frame, "
                      "community-channel ban)",
              "frame_note": "X = author DESCRIPTION, not story text (plan of record "
                            "2026-08-16: unified-X U1 community leg)",
              "probe_gate": "PASS_WITH_CONSTRUCT_NOTE (char .6367 / word .6384 "
                            "grouped-OOF; content+blurb-convention separability)",
              "group_column": "stratum (genre::year)"},
    )


def build_rr_magazine():
    rubrics, blocks, vf = _cw_bank()
    pop = CW / "royalroad_magazine_cell/rr_magazine_population_v3.jsonl.gz"
    rows = [json.loads(l) for l in gzip.open(pop, "rt")]
    items = [{"id": r["row_id"], "group": r["edition"], "text": _trunc(str(r["text"])),
              "judgement": int(r["judgement"]),
              "edition_labeled": bool(r["edition_labeled"])} for r in rows]
    labeled = [r for r in items if r["edition_labeled"]]

    def ctx(r):
        return f"CONTEST ENTRY (opening chapter):\n{r['text']}"

    def label_fn(r):
        # anchors only ever drawn from labeled editions
        return r["judgement"] if r["edition_labeled"] else -1

    ys = {"magazine_winner": np.array([r["judgement"] for r in items])}
    return dict(
        name="rr_magazine", items=items, rubrics=rubrics, blocks=blocks,
        sys=SYS_RR_MAGAZINE, ctx=ctx, vvec=lambda r: vf.feature_vector(r["text"]),
        vnames=list(vf.V_NAMES),
        anchors=_mk_anchors(labeled, lambda r: r["judgement"], 811,
                            scr_chars=4000, n_words=200),
        ys=ys, n_shards=4,
        meta={"population": str(pop.relative_to(REPO)),
              "y_semantics": "1 = ranked winner in the edition's blog announcement "
                             "(judge panel picks, 3-5 per edition)",
              "power_caveat": "26 positives / 2,012 labeled rows — PILOT flag "
                              "mandatory on every readout",
              "unlabeled_editions": "2022-01, 2022-06 scored but edition_labeled="
                                    "False (no parsable winner announcement)",
              "probe_gate": "BOUNDARY (char .5969 / word .5038 grouped-OOF by "
                            "edition; 26-pos noise floor declared)",
              "group_column": "edition"},
    )


BUILDERS = {"rr_community": build_rr_community, "rr_magazine": build_rr_magazine}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="rr_community,rr_magazine")
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
            with np.errstate(invalid="ignore"):
                print(f"[smoke:{b['name']}] finite {np.isfinite(X).mean():.3f} "
                      f"mean {np.nanmean(X):.3f} NA {np.isnan(X).mean():.3f}")
            continue
        S.score_bank(llm, sp, b, OUT)
        C.run_battery(llm, sp, b, a.battery, OUT)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
