#!/usr/bin/env python3
"""Rescore the code-review articulated-criteria (A) bank on the ENRICHED v3 text
with a local Gemma-4-31B judge (offline batch vLLM, one token per (row, criterion)).

Why this exists: the published code cell (V .576 / V+A .592) measured A with the
DETERMINISTIC CODED backend (methods/existing_metrics_runner/coded/) on a bare
diff. The v3 corpus replaces the bare diff with Title + Description + inline
review comments + diff. Same-input rule => the bank has to be rescored on v3
text before any residual claim. The judge is Gemma-4-31B (the A-bank standard),
so this is an INSTRUMENT CHANGE as well as an input change; both are flagged in
the readout.

Rows scored: v3 eval (5,822) + test (5,630) only -- never the 47,659 train rows,
because the stack is fit grouped-OOF WITHIN eval and WITHIN test.

Protocol ported from datasets/va_gemma_banks/score_va_gemma_banks.py:
temperature 0, max_tokens 6, prefix caching, thousands of prompts per generate,
label-blind (judgement never enters a prompt), blinded anchor battery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import multiprocessing as _mp  # noqa: E402

try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
V3 = REPO / "datasets/code-review/dense_standard_v3"
OUTDIR = Path(os.environ.get("VA_OUT", str(V3 / "abank_rescore")))
GEMMA4 = os.environ.get(
    "GEMMA4_PATH",
    "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/"
    "3548789868c5356dbf307c98e6f609007b82b3eb",
)
SEED = 20260806
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")

SYS = (
    "You are an experienced open-source maintainer performing a measurement task on a "
    "GitHub pull request. You are given ONE pull request (its title, description, the "
    "inline review comments it received, and its code diff) and ONE code-review "
    "criterion. Decide how strongly the pull request, on the evidence of the supplied "
    "text alone, satisfies that criterion. Answer with EXACTLY ONE token:\n"
    "  1.0 = clearly satisfies the criterion\n"
    "  0.5 = partially, weakly, inconsistently, or borderline\n"
    "  0.0 = the relevant aspect is present but fails or cuts against the criterion\n"
    "  NA = the supplied text gives no evidence bearing on this criterion\n"
    "Do not predict whether the pull request was merged, closed, approved or rejected; "
    "do not infer the author's or repository's identity or reputation. Long pull "
    "requests may be truncated; judge what is shown. Output only the token."
)


def parse_tok(t: str) -> float:
    t = (t or "").strip().lower()
    if t.startswith("na") or "n/a" in t:
        return np.nan
    if "0.5" in t or t.startswith(".5"):
        return 0.5
    if re.match(r"^1(\.0+)?\b", t) or t.startswith("1"):
        return 1.0
    if re.match(r"^0(\.0+)?\b", t) or t.startswith("0"):
        return 0.0
    return np.nan


def sha(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def scramble(texts, rng, n_words=400):
    toks = []
    for t in texts:
        toks += WORD_RE.findall(t)
    rng.shuffle(toks)
    chosen = toks[:n_words]
    chosen[1::2] = [w[::-1] for w in chosen[1::2]]
    return " ".join(chosen)


# ------------------------------------------------------------------ data ----
def load_rows(tokenizer, max_text_tokens):
    import pandas as pd
    frames = []
    for sp in ("eval", "test"):
        d = pd.read_csv(V3 / "split" / f"{sp}.csv")
        d["split"] = sp
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["row_id"] = d["repo"].astype(str) + "/" + d["pr_number"].astype(str)
    texts = d["text"].astype(str).tolist()
    enc = tokenizer(texts, add_special_tokens=False)["input_ids"]
    n_trunc = 0
    cut = []
    for t, ids in zip(texts, enc):
        if len(ids) > max_text_tokens:
            n_trunc += 1
            cut.append(tokenizer.decode(ids[:max_text_tokens]))
        else:
            cut.append(t)
    d["judge_text"] = cut
    d["tok_len"] = [len(i) for i in enc]
    return d, n_trunc


def load_criteria(which):
    crit = [json.loads(l) for l in open(OUTDIR / "criteria_code_abank.jsonl") if l.strip()]
    crit = [c for c in crit if c["portable"]]
    if which == "cov5":
        crit = [c for c in crit if c["coded_cov_ge5pct"]]
    crit.sort(key=lambda c: int(re.sub(r"\D", "", c["aspect_id"])))
    return crit


def blocks_for(crit):
    return [f"CRITERION: {c['name']}\nDESCRIPTION: {c['description']}\n\nAnswer with one token:"
            for c in crit]


def ctx(text):
    return f"PULL REQUEST:\n{text}"


# --------------------------------------------------------------- anchors ----
def build_anchors(K, tokenizer, max_text_tokens):
    """Blinded 3-tier battery drawn from the TRAIN split only (never eval/test).

    pos    = merged PRs, neg = closed-unmerged PRs, scram = word-salad from both.
    The pos>neg contrast is the OUTCOME contrast and is expected to be weak in
    this cell by construction (that is the finding under test); the binding gate
    is scram << pos and scram << neg, i.e. the judge is reading content.
    """
    import pandas as pd
    tr = pd.read_csv(V3 / "split" / "train.csv", usecols=["text", "judgement", "repo", "pr_number"])
    rng = random.Random(SEED)
    pos_pool = tr.index[tr["judgement"] == 1].tolist()
    neg_pool = tr.index[tr["judgement"] == 0].tolist()
    rng.shuffle(pos_pool)
    rng.shuffle(neg_pool)
    pos = tr.loc[pos_pool[:K], "text"].astype(str).tolist()
    neg = tr.loc[neg_pool[:K], "text"].astype(str).tolist()
    scram = [scramble([pos[i][:12000], neg[i][:12000]], random.Random(SEED + i)) for i in range(K)]

    def cutall(xs):
        enc = tokenizer(xs, add_special_tokens=False)["input_ids"]
        return [tokenizer.decode(e[:max_text_tokens]) if len(e) > max_text_tokens else x
                for x, e in zip(xs, enc)]

    return {"pos": cutall(pos), "neg": cutall(neg), "scram": cutall(scram)}


# --------------------------------------------------------------- scoring ----
def run_batch(llm, sp, texts, blocks):
    convs = []
    for t in texts:
        c = ctx(t)
        for b in blocks:
            convs.append([{"role": "user", "content": f"{SYS}\n\n{c}\n\n{b}"}])
    outs = llm.chat(convs, sp, use_tqdm=False)
    return np.array([parse_tok(o.outputs[0].text) for o in outs],
                    dtype=float).reshape(len(texts), len(blocks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--max-text-tokens", type=int, default=11600)
    ap.add_argument("--criteria", default="portable", choices=["portable", "cov5"])
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--anchor-k", type=int, default=50)
    ap.add_argument("--smoke", type=int, default=0, help="if >0, score only N rows and exit")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    a = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(GEMMA4)

    crit = load_criteria(a.criteria)
    blocks = blocks_for(crit)
    d, n_trunc = load_rows(tk, a.max_text_tokens)
    print(f"[build] rows={len(d)} criteria={len(crit)} truncated={n_trunc} "
          f"({n_trunc/len(d):.4f}) median_tok={int(d['tok_len'].median())}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              trust_remote_code=True, max_num_seqs=a.max_num_seqs)
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    if a.smoke:
        sub = d.head(a.smoke)
        t0 = time.time()
        X = run_batch(llm, sp, sub["judge_text"].tolist(), blocks)
        dt = time.time() - t0
        npr = a.smoke * len(blocks)
        print(f"SMOKE {a.smoke} rows x {len(blocks)} crit = {npr} prompts in {dt:.1f}s "
              f"({npr/dt:.1f} prompt/s) -> full {len(d)} rows ETA "
              f"{len(d)*len(blocks)/(npr/dt)/3600:.2f} h", flush=True)
        print("SMOKE NA rate", float(np.isnan(X).mean()),
              "value dist", {v: int((X == v).sum()) for v in (0.0, 0.5, 1.0)}, flush=True)
        print("SMOKE_DONE", flush=True)
        return

    # ---- anchors (once, not per shard: K=50 x 3 tiers is its own battery) ----
    apath = OUTDIR / "anchors.npz"
    if not apath.exists():
        anc = build_anchors(a.anchor_k, tk, a.max_text_tokens)
        rep = {}
        AX = {}
        for tier in ("pos", "neg", "scram"):
            t0 = time.time()
            AX[tier] = run_batch(llm, sp, anc[tier], blocks)
            # NA ("no evidence bearing on this criterion") is itself a judgement:
            # report the mean over answered cells, the abstention rate, and the
            # mean with NA scored as 0 (no evidence => criterion not evidenced).
            rep[tier] = float(np.nanmean(AX[tier]))
            rep[f"{tier}_na_rate"] = float(np.isnan(AX[tier]).mean())
            rep[f"{tier}_mean_na0"] = float(np.nan_to_num(AX[tier], nan=0.0).mean())
            print(f"[anchor] {tier} K={a.anchor_k} mean={rep[tier]:.4f} "
                  f"na={rep[f'{tier}_na_rate']:.3f} mean_na0={rep[f'{tier}_mean_na0']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        rep["gate_scram_below_pos"] = bool(rep["scram_mean_na0"] < rep["pos_mean_na0"])
        rep["gate_scram_below_neg"] = bool(rep["scram_mean_na0"] < rep["neg_mean_na0"])
        rep["pos_gt_neg"] = bool(rep["pos_mean_na0"] > rep["neg_mean_na0"])
        rep["K"] = a.anchor_k
        np.savez_compressed(apath, pos=AX["pos"], neg=AX["neg"], scram=AX["scram"],
                            report=np.array(json.dumps(rep), dtype=object),
                            a_ids=np.array([c["aspect_id"] for c in crit], dtype=object))
        print("[anchor] report", json.dumps(rep), flush=True)

    # ---- main rows, sharded for resumability -------------------------------
    d["_shard"] = [int(hashlib.sha1(r.encode()).hexdigest(), 16) % a.shards for r in d["row_id"]]
    for si in range(a.shards):
        outp = OUTDIR / f"scores_shard{si:02d}.npz"
        if outp.exists():
            print(f"[shard {si}] exists, skip", flush=True)
            continue
        sub = d[d["_shard"] == si]
        t0 = time.time()
        X = run_batch(llm, sp, sub["judge_text"].tolist(), blocks)
        dt = time.time() - t0
        na = float(np.isnan(X).mean())
        np.savez_compressed(
            outp, X=X,
            row_ids=np.array(sub["row_id"].tolist(), dtype=object),
            repos=np.array(sub["repo"].astype(str).tolist(), dtype=object),
            splits=np.array(sub["split"].tolist(), dtype=object),
            y=np.array(sub["judgement"].astype(int).tolist()),
            tok_len=np.array(sub["tok_len"].tolist()),
            a_ids=np.array([c["aspect_id"] for c in crit], dtype=object),
            a_names=np.array([c["name"] for c in crit], dtype=object),
            na_rate=na,
        )
        print(f"[shard {si}] {len(sub)} rows x {len(blocks)} = {len(sub)*len(blocks)} prompts "
              f"in {dt/60:.1f} min, NA {na:.4f} -> {outp.name}", flush=True)

    meta = {
        "judge": "gemma-4-31b-it (bfloat16, vLLM offline batch)",
        "judge_path": GEMMA4,
        "instrument_change": True,
        "published_A_backend": "deterministic coded metrics (methods/existing_metrics_runner/coded/)",
        "n_rows": int(len(d)), "n_criteria": len(crit),
        "criteria_set": a.criteria,
        "aspect_ids": [c["aspect_id"] for c in crit],
        "max_model_len": a.max_model_len, "max_text_tokens": a.max_text_tokens,
        "n_truncated": int(n_trunc), "frac_truncated": float(n_trunc / len(d)),
        "median_text_tokens": int(d["tok_len"].median()),
        "temperature": 0.0, "max_tokens": 6, "seed": SEED,
        "text_source": str(V3 / "split/{eval,test}.csv"),
    }
    (OUTDIR / "score_meta.json").write_text(json.dumps(meta, indent=1))
    print("SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
