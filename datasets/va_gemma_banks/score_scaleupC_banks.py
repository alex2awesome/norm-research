#!/usr/bin/env python3
"""Score the scale-up-wave (task D7) articulated-criteria (A) banks with the local
Gemma-4-31B judge, offline-batch vLLM, one token per (item, criterion).

Cells built here (each has its own bank authored elsewhere; this file ONLY scores):
  jokes_community  datasets/humor/reddit_jokes/va/rubrics.jsonl
  mathse_accepted  datasets/math-stackexchange/va/rubrics.jsonl   (multi-y)
  aops_curation    datasets/math/aops/va/rubrics.jsonl
  homepage_curation datasets/journalism/homepage/va/rubrics.jsonl

Machinery (scoring loop, shard checkpointing, per-shard 3-row blinded anchors,
NA parsing, prefix caching, temperature 0 / max_tokens 6) is imported verbatim
from score_va_gemma_banks.py -- only the bank builders are new.

Anchor discipline: every shard carries 3 blinded anchor rows (known positive /
known negative / scrambled nonsense) AND the run is followed by the extended
battery at K >= 50 per class (`--battery K`), per the CW-community finding that
K = 12 is too noisy to certify the positive/negative contrast.

GPU: one GPU only (CUDA_VISIBLE_DEVICES set by the caller).
"""
from __future__ import annotations

import argparse
import hashlib
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

REPO = S.REPO
OUT = Path(os.environ.get("VA_OUT_C", str(REPO / "outputs/va_gemma_banks_scaleupC")))
SEED = 20260808


# ============================== jokes_community ==============================
SYS_JOKES = (
    "You are an expert comedy writer performing a measurement task. You are given "
    "ONE joke posted to a public joke-sharing forum and ONE quality criterion. "
    "Decide how strongly the joke, on the evidence of the supplied text alone, "
    "satisfies that criterion. Answer with EXACTLY ONE token:\n"
    "  1.0 = clearly satisfies the criterion\n"
    "  0.5 = partly, weakly, inconsistently, or borderline\n"
    "  0.0 = the relevant attempt is present but fails or cuts against the criterion\n"
    "  NA = the joke attempts nothing that bears on this criterion\n"
    "Judge the joke on its text alone. Do not predict votes, popularity, labels, "
    "authorship, whether you have seen it before, or dataset membership, and do not "
    "compare it with other jokes. Output only the token."
)

JOKES_DIR = REPO / "datasets/humor/reddit_jokes"


def build_jokes_community():
    import pandas as pd
    vf = S.load_module(JOKES_DIR / "va/v_features.py", "vf_jokes")
    df = pd.read_csv(JOKES_DIR / "va/population.csv.gz")
    items = [{"id": r.row_id, "group": r.group, "text": r.text,
              "judgement": int(r.judgement)} for r in df.itertuples()]

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

    ys = {"crowd_top_quartile": np.array([r["judgement"] for r in items])}
    return dict(
        name="jokes_community", items=items, rubrics=rubrics, blocks=blocks,
        sys=SYS_JOKES, ctx=ctx, vvec=vvec, vnames=list(vf.V_NAMES), anchors=anchors,
        ys=ys, n_shards=8,
        meta={"source": "datasets/humor/reddit_humor_with_topics.csv.gz",
              "population": "datasets/humor/reddit_jokes/va/population.csv.gz",
              "sampling": 'stable hash sha256("jokes-va-v1|" + sha1(text)[:20]), first 16000',
              "group_column": "LDA topic (50)",
              "n_groups": int(df["group"].nunique())},
    )


# ============================== mathse (multi-y) =============================
SYS_MATHSE = (
    "You are an expert mathematician performing a measurement task. You are given "
    "the title of a question asked on a mathematics question-and-answer site, ONE "
    "answer posted to it, and ONE quality criterion. Decide how strongly the answer, "
    "on the evidence of the supplied text alone, satisfies that criterion. Answer "
    "with EXACTLY ONE token:\n"
    "  1.0 = clearly satisfies the criterion\n"
    "  0.5 = partly, weakly, inconsistently, or borderline\n"
    "  0.0 = the relevant attempt is present but fails or cuts against the criterion\n"
    "  NA = the answer gives no evidence bearing on this criterion\n"
    "Judge this answer on its own text. Do not consider or imagine other answers to "
    "the same question, and do not predict acceptance, votes, scores, reputation, "
    "authorship, or dataset membership. Long answers may have a deterministically "
    "omitted middle; judge what is shown. Output only the token."
)

MATHSE_DIR = REPO / "datasets/math-stackexchange"
MATHSE_BANK = REPO / "datasets/math/stackexchange/va/rubrics.jsonl"
TRUNC_SRC, TRUNC_HEAD, TRUNC_TAIL = 5000, 3000, 2000
TRUNC_MARK = "\n\n[... DETERMINISTIC MIDDLE OMISSION ...]\n\n"


def _trunc(s, src=TRUNC_SRC, head=TRUNC_HEAD, tail=TRUNC_TAIL):
    s = (s or "").strip()
    return s if len(s) <= src else s[:head] + TRUNC_MARK + s[-tail:]


def build_mathse_multiy():
    import pandas as pd
    vf = S.load_module(REPO / "datasets/math/stackexchange/va/v_features.py", "vf_mathse")
    df = pd.read_csv(MATHSE_DIR / "v2_va/population.csv.gz")
    items = []
    for r in df.itertuples():
        q, ans = str(r.text).split("\n\nANSWER:\n", 1) if "\n\nANSWER:\n" in str(r.text) \
            else ("", str(r.text))
        items.append({"id": r.row_id, "group": str(r.group),
                      "question": q.removeprefix("QUESTION: ").strip(), "answer": ans,
                      "y_accepted": int(r.y_accepted),
                      "y_vote": (None if r.y_vote != r.y_vote else int(r.y_vote))})

    rubrics = [json.loads(l) for l in open(MATHSE_BANK) if l.strip()]
    blocks = [f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n\n"
              "Answer with one token:" for m in rubrics]

    def ctx(r):
        return f"QUESTION TITLE: {r['question'][:400]}\n\nANSWER:\n{_trunc(r['answer'])}"

    def vvec(r):
        return vf.vector(r["answer"])

    def anchors(shard):
        rng = random.Random(SEED + 503 * shard)
        pos = dict(rng.choice([r for r in items if r["y_accepted"] == 1]))
        neg = dict(rng.choice([r for r in items if r["y_accepted"] == 0]))
        scr = dict(neg)
        scr["answer"] = S.scramble([pos["answer"][:4000], neg["answer"][:4000]],
                                   rng, n_words=200)
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg), ("anchor_scram", scr)):
            rr = dict(r); rr["anchor_tag"] = tag
            rr["id"] = f"__ANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    ys = {"accepted_verdict": np.array([r["y_accepted"] for r in items]),
          "vote_score": np.array([np.nan if r["y_vote"] is None else r["y_vote"]
                                  for r in items], dtype=float)}
    return dict(name="mathse_multiy", items=items, rubrics=rubrics, blocks=blocks,
                sys=SYS_MATHSE, ctx=ctx, vvec=vvec, vnames=list(vf.V_NAMES),
                anchors=anchors, ys=ys, n_shards=6,
                meta={"population": "datasets/math-stackexchange/v2_va/population.csv.gz",
                      "group_column": "question_id",
                      "n_groups": int(df["group"].nunique()),
                      "truncation": {"source_chars": TRUNC_SRC, "head": TRUNC_HEAD,
                                     "tail": TRUNC_TAIL, "question_chars": 400}})


# ================================ aops curation ==============================
SYS_AOPS = (
    "You are an experienced olympiad problem-solving coach performing a measurement "
    "task. You are given a competition problem statement, ONE solution posted to a "
    "public problem-solving forum, and ONE quality criterion. Decide how strongly the "
    "posted solution, on the evidence of the supplied text alone, satisfies that "
    "criterion. Answer with EXACTLY ONE token:\n"
    "  1.0 = clearly satisfies the criterion\n"
    "  0.5 = partly, weakly, inconsistently, or borderline\n"
    "  0.0 = the relevant attempt is present but fails or cuts against the criterion\n"
    "  NA = the post gives no evidence bearing on this criterion\n"
    "Judge the post on its own text. Do not imagine or compare against any official, "
    "editorial, or textbook solution, and do not predict thanks, replies, authorship, "
    "or dataset membership. `<imath>...</imath>` and `$...$` both delimit mathematics. "
    "Long posts may have a deterministically omitted middle; judge what is shown. "
    "Output only the token."
)

AOPS_DIR = REPO / "datasets/math/aops"


def build_aops_curation():
    import pandas as pd
    vf = S.load_module(AOPS_DIR / "va/v_features.py", "vf_aops")
    df = pd.read_csv(AOPS_DIR / "va/population.csv.gz")
    df["statement"] = df["statement"].fillna("")
    df["body"] = df["body"].fillna("")
    items = [{"id": r.row_id, "group": str(r.group), "statement": r.statement,
              "body": r.body, "judgement": int(r.judgement)} for r in df.itertuples()]

    rubrics = [json.loads(l) for l in open(AOPS_DIR / "va/rubrics.jsonl") if l.strip()]
    blocks = [f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n\n"
              "Answer with one token:" for m in rubrics]

    def ctx(r):
        return (f"PROBLEM: {r['statement'][:1500]}\n\n"
                f"FORUM SOLUTION:\n{_trunc(r['body'])}")

    def vvec(r):
        return vf.vector(r["body"])

    def anchors(shard):
        rng = random.Random(SEED + 601 * shard)
        pos = dict(rng.choice([r for r in items if r["judgement"] == 1]))
        neg = dict(rng.choice([r for r in items if r["judgement"] == 0]))
        scr = dict(neg)
        scr["body"] = S.scramble([pos["body"][:4000], neg["body"][:4000]], rng, n_words=200)
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg), ("anchor_scram", scr)):
            rr = dict(r); rr["anchor_tag"] = tag
            rr["id"] = f"__ANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    ys = {"same_approach": np.array([r["judgement"] for r in items])}
    return dict(name="aops_curation", items=items, rubrics=rubrics, blocks=blocks,
                sys=SYS_AOPS, ctx=ctx, vvec=vvec, vnames=list(vf.V_NAMES),
                anchors=anchors, ys=ys, n_shards=6,
                meta={"population": "datasets/math/aops/va/population.csv.gz",
                      "group_column": "problem",
                      "n_groups": int(df["group"].nunique()),
                      "truncation": {"source_chars": TRUNC_SRC, "head": TRUNC_HEAD,
                                     "tail": TRUNC_TAIL, "statement_chars": 1500}})


# ============================= homepage curation =============================
SYS_HOMEPAGE = (
    "You are a senior news editor performing a measurement task. You are given ONE "
    "headline from a news organisation's home page, the other headlines that appeared "
    "on the same capture as shared context, and ONE news-value criterion. Decide how "
    "strongly the FOCAL headline satisfies that criterion. Answer with EXACTLY ONE "
    "token:\n"
    "  1.0 = clearly satisfies the criterion\n"
    "  0.5 = partly, weakly, or borderline\n"
    "  0.0 = the relevant dimension is present but the headline fails or cuts against it\n"
    "  NA = the headline gives no evidence bearing on this criterion\n"
    "Judge ONLY the focal headline; the context list is background about the day and "
    "must never be scored in its place. Do not infer or predict where the story was "
    "placed on the page, which outlet published it, how popular it was, or dataset "
    "membership. Output only the token."
)

HOMEPAGE_DIR = REPO / "datasets/news-homepages"


def build_homepage_curation():
    import pandas as pd
    vf = S.load_module(HOMEPAGE_DIR / "va/v_features.py", "vf_homepage")
    df = pd.read_csv(HOMEPAGE_DIR / "va/population.csv.gz")
    items = []
    for r in df.itertuples():
        t = str(r.text)
        head = vf.headline_of(t)
        ctxs = t.split("\n\nCONTEXT: ", 1)[1] if "\n\nCONTEXT: " in t else ""
        items.append({"id": r.row_id, "group": str(r.group), "outlet": r.outlet,
                      "snapshot_id": str(r.snapshot_id), "headline": head,
                      "context": ctxs, "judgement": int(r.judgement)})

    rubrics = [json.loads(l) for l in open(HOMEPAGE_DIR / "va/rubrics.jsonl") if l.strip()]
    blocks = [f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n\n"
              "Answer with one token:" for m in rubrics]

    def ctx(r):
        return (f"FOCAL HEADLINE: {r['headline']}\n\n"
                f"OTHER HEADLINES ON THE SAME PAGE (context only): {r['context'][:1800]}")

    def vvec(r):
        return vf.vector(r["headline"])

    def anchors(shard):
        rng = random.Random(SEED + 701 * shard)
        pos = dict(rng.choice([r for r in items if r["judgement"] == 1]))
        neg = dict(rng.choice([r for r in items if r["judgement"] == 0]))
        scr = dict(neg)
        scr["headline"] = S.scramble([pos["headline"], neg["headline"]], rng)
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg), ("anchor_scram", scr)):
            rr = dict(r); rr["anchor_tag"] = tag
            rr["id"] = f"__ANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    ys = {"top_half_placement": np.array([r["judgement"] for r in items])}
    return dict(name="homepage_curation", items=items, rubrics=rubrics, blocks=blocks,
                sys=SYS_HOMEPAGE, ctx=ctx, vvec=vvec, vnames=list(vf.V_NAMES),
                anchors=anchors, ys=ys, n_shards=6,
                extra_cols={"outlet": np.array([r["outlet"] for r in items], dtype=object),
                            "snapshot_id": np.array([r["snapshot_id"] for r in items],
                                                    dtype=object)},
                meta={"population": "datasets/news-homepages/va/population.csv.gz",
                      "group_column": "outlet (OUTLET-HELD-OUT)",
                      "secondary_group_column": "snapshot_id",
                      "n_groups": int(df["group"].nunique()),
                      "outlets": sorted(df["outlet"].unique().tolist()),
                      "snapshot_ids": df["snapshot_id"].astype(str).tolist(),
                      "outlet_of_item": df["outlet"].tolist(),
                      "weak_instrument_flag":
                          "y is homepage spatial placement, jointly determined with "
                          "layout/ad/image constraints; outlet identity alone predicts "
                          "it strongly. Every number from this cell carries this flag."})


BUILDERS = {"jokes_community": build_jokes_community,
            "mathse_multiy": build_mathse_multiy,
            "aops_curation": build_aops_curation,
            "homepage_curation": build_homepage_curation}


# ================================ battery ====================================
def run_battery(llm, sp, bank, k, outdir):
    from sklearn.metrics import roc_auc_score
    rows, tags = [], []
    for j in range(k):
        for r in bank["anchors"](900_000 + j):
            rows.append(r)
            tags.append(r["anchor_tag"])
    convs = []
    for r in rows:
        c = bank["ctx"](r)
        for blk in bank["blocks"]:
            convs.append([{"role": "user", "content": f"{bank['sys']}\n\n{c}\n\n{blk}"}])
    print(f"[battery:{bank['name']}] {len(rows)} anchors x {len(bank['blocks'])} "
          f"= {len(convs)} prompts", flush=True)
    outs = llm.chat(convs, sp)
    X = np.array([S.parse_tok(o.outputs[0].text) for o in outs],
                 dtype=float).reshape(len(rows), len(bank["blocks"]))
    with np.errstate(invalid="ignore"):
        item_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=1)
    tags = np.array(tags)
    # An anchor row can legitimately draw NA on EVERY criterion (a scrambled
    # headline bears on none of 14 news-value tests). Those rows carry no rank
    # information; drop them from the battery statistics and report how many.
    ok = np.isfinite(item_mean)
    n_all_na = int((~ok).sum())
    item_mean, tags = item_mean[ok], tags[ok]
    res = {"k_per_class": k, "n_anchor_rows_all_NA_dropped": n_all_na,
           "n_anchor_rows_used": int(ok.sum())}
    for t in ("anchor_pos", "anchor_neg", "anchor_scram"):
        v = item_mean[tags == t]
        res[t] = {"n": int(len(v)),
                  "mean": (float(np.mean(v)) if len(v) else float("nan")),
                  "sd": (float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")),
                  "values": [round(float(x), 4) for x in v]}
    pv, nv = item_mean[tags == "anchor_pos"], item_mean[tags == "anchor_neg"]
    sv = item_mean[tags == "anchor_scram"]
    res["pos_vs_neg_auc"] = (float(roc_auc_score(
        [1] * len(pv) + [0] * len(nv), np.concatenate([pv, nv])))
        if len(pv) and len(nv) else float("nan"))
    res["coherent_vs_scrambled_auc"] = (float(roc_auc_score(
        [1] * (len(pv) + len(nv)) + [0] * len(sv), np.concatenate([pv, nv, sv])))
        if (len(pv) + len(nv)) and len(sv) else float("nan"))
    res["ordering_holds_on_means"] = bool(
        res["anchor_pos"]["mean"] > res["anchor_neg"]["mean"] > res["anchor_scram"]["mean"])
    p = outdir / "anchor_battery.json"
    payload = json.loads(p.read_text()) if p.exists() else {}
    payload[bank["name"]] = res
    p.write_text(json.dumps(payload, indent=1))
    print(f"[battery:{bank['name']}] pos {res['anchor_pos']['mean']:.4f} / "
          f"neg {res['anchor_neg']['mean']:.4f} / scram {res['anchor_scram']['mean']:.4f} "
          f"| pos-vs-neg AUC {res['pos_vs_neg_auc']:.3f} "
          f"| coherent-vs-scram AUC {res['coherent_vs_scrambled_auc']:.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="jokes_community")
    ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--battery", type=int, default=50)
    ap.add_argument("--smoke", type=int, default=0,
                    help="if >0, score only this many items and write nothing")
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
            for ci, nm in enumerate([m["name"] for m in b["rubrics"]]):
                col = X[:, ci]
                vals, cnts = np.unique(col[np.isfinite(col)], return_counts=True)
                modal = float(cnts.max() / max(len(col), 1)) if len(cnts) else 1.0
                print(f"  {ci:02d} {nm[:56]:58s} mean={np.nanmean(col):.3f} "
                      f"na={np.isnan(col).mean():.2f} modal={modal:.2f}", flush=True)
            print("SMOKE_DONE", flush=True)
            continue
        S.score_bank(llm, sp, b, OUT)
        if a.battery:
            run_battery(llm, sp, b, a.battery, OUT)
    print("SCALEUPC_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
