#!/usr/bin/env python3
"""Score ONE closure round's selected criteria on the full code_v3 population with the
Gemma-4-31B judge (offline batch vLLM on sk3 -- never an HTTP server).

Ported from `abank_rescore/score_code_abank_v3.py`, which produced the incoming bank, so
the instrument is identical to the one the round-0 A matrix was measured with: same
model, same system framing, same one-token 1.0/0.5/0.0/NA readout, same truncation
budget, same anchor construction, same sharding. The only difference is where the
criteria come from: a round's `<tag>_species.json` `selected` list (name + instruction)
instead of `criteria_code_abank.jsonl`.

Per-batch discipline required by the freeze:
  * blinded 3-tier anchor battery (K >= 50 per class) EVERY round, drawn from the TRAIN
    split only -- never eval/test;
  * per-criterion collapse check written into the score report before any use;
  * label-blind: the merge outcome never enters a prompt.

Usage on sk3:
  SEED_TAG=code_v3_r1 python score_round_code.py --tag code_v3_r1 --util .90
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
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

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
V3 = REPO / "datasets/code-review/dense_standard_v3"
GEMMA4 = os.environ.get(
    "GEMMA4_PATH",
    "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/"
    "3548789868c5356dbf307c98e6f609007b82b3eb",
)
SEED = 20260807
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")

# byte-identical to abank_rescore/score_code_abank_v3.py
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


def parse_tok(t):
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


def scramble(texts, rng, n_words=400):
    toks = []
    for t in texts:
        toks += WORD_RE.findall(t)
    rng.shuffle(toks)
    chosen = toks[:n_words]
    chosen[1::2] = [w[::-1] for w in chosen[1::2]]
    return " ".join(chosen)


def load_rows(tokenizer, max_text_tokens):
    import pandas as pd
    frames = []
    for sp in ("eval", "test"):
        d = pd.read_csv(V3 / "split" / f"{sp}.csv")
        d["split"] = sp
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["row_id"] = d["repo"].astype(str) + "/" + d["pr_number"].astype(str)
    enc = tokenizer(d["text"].astype(str).tolist(), add_special_tokens=False)["input_ids"]
    cut, n_trunc = [], 0
    for t, ids in zip(d["text"].astype(str), enc):
        if len(ids) > max_text_tokens:
            n_trunc += 1
            cut.append(tokenizer.decode(ids[:max_text_tokens]))
        else:
            cut.append(t)
    d["judge_text"] = cut
    d["tok_len"] = [len(i) for i in enc]
    return d, n_trunc


def build_anchors(K, tokenizer, max_text_tokens):
    import pandas as pd
    tr = pd.read_csv(V3 / "split" / "train.csv", usecols=["text", "judgement"])
    rng = random.Random(SEED)
    pos_pool = tr.index[tr["judgement"] == 1].tolist()
    neg_pool = tr.index[tr["judgement"] == 0].tolist()
    rng.shuffle(pos_pool)
    rng.shuffle(neg_pool)
    pos = tr.loc[pos_pool[:K], "text"].astype(str).tolist()
    neg = tr.loc[neg_pool[:K], "text"].astype(str).tolist()
    scram = [scramble([pos[i][:12000], neg[i][:12000]], random.Random(SEED + i))
             for i in range(K)]

    def cutall(xs):
        enc = tokenizer(xs, add_special_tokens=False)["input_ids"]
        return [tokenizer.decode(e[:max_text_tokens]) if len(e) > max_text_tokens else x
                for x, e in zip(xs, enc)]

    return {"pos": cutall(pos), "neg": cutall(neg), "scram": cutall(scram)}


def run_batch(llm, sp, texts, blocks):
    convs = []
    for t in texts:
        c = f"PULL REQUEST:\n{t}"
        for b in blocks:
            convs.append([{"role": "user", "content": f"{SYS}\n\n{c}\n\n{b}"}])
    outs = llm.chat(convs, sp, use_tqdm=False)
    return np.array([parse_tok(o.outputs[0].text) for o in outs],
                    dtype=float).reshape(len(texts), len(blocks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="e.g. code_v3_r1 or code_v3_rd")
    ap.add_argument("--util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--max-text-tokens", type=int, default=11600)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--anchor-k", type=int, default=50)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    a = ap.parse_args()

    out = HERE
    spec = json.loads((out / f"{a.tag}_species.json").read_text())["selected"]
    crit = [{"id": c["blind_id"], "name": c["name"], "instruction": c["instruction"]}
            for c in spec]
    blocks = [f"CRITERION: {c['name']}\nDESCRIPTION: {c['instruction']}\n\n"
              f"Answer with one token:" for c in crit]

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(GEMMA4)
    d, n_trunc = load_rows(tk, a.max_text_tokens)
    print(f"[build] tag={a.tag} rows={len(d)} criteria={len(crit)} truncated={n_trunc}",
          flush=True)

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
        print(f"SMOKE {npr} prompts in {dt:.1f}s ({npr/dt:.1f}/s) -> full ETA "
              f"{len(d)*len(blocks)/(npr/dt)/3600:.2f} h  NA={np.isnan(X).mean():.4f} "
              f"dist={{v: int((X==v).sum()) for v in (0.,0.5,1.)}}", flush=True)
        print("SMOKE_DONE", flush=True)
        return

    apath = out / f"{a.tag}_anchors.npz"
    arep = {}
    if not apath.exists():
        anc = build_anchors(a.anchor_k, tk, a.max_text_tokens)
        AX = {}
        for tier in ("pos", "neg", "scram"):
            AX[tier] = run_batch(llm, sp, anc[tier], blocks)
            arep[tier] = float(np.nanmean(AX[tier]))
            arep[f"{tier}_na_rate"] = float(np.isnan(AX[tier]).mean())
            arep[f"{tier}_mean_na0"] = float(np.nan_to_num(AX[tier], nan=0.0).mean())
            print(f"[anchor] {tier} mean={arep[tier]:.4f} na={arep[f'{tier}_na_rate']:.3f} "
                  f"mean_na0={arep[f'{tier}_mean_na0']:.4f}", flush=True)
        arep["gate_scram_below_pos"] = bool(arep["scram_mean_na0"] < arep["pos_mean_na0"])
        arep["gate_scram_below_neg"] = bool(arep["scram_mean_na0"] < arep["neg_mean_na0"])
        arep["pos_gt_neg"] = bool(arep["pos_mean_na0"] > arep["neg_mean_na0"])
        arep["K"] = a.anchor_k
        np.savez_compressed(apath, pos=AX["pos"], neg=AX["neg"], scram=AX["scram"],
                            report=np.array(json.dumps(arep), dtype=object),
                            a_ids=np.array([c["id"] for c in crit], dtype=object))
        print("[anchor] " + json.dumps(arep), flush=True)

    d["_shard"] = [int(hashlib.sha1(r.encode()).hexdigest(), 16) % a.shards
                   for r in d["row_id"]]
    for si in range(a.shards):
        outp = out / f"{a.tag}_scores_shard{si:02d}.npz"
        if outp.exists():
            print(f"[shard {si}] exists, skip", flush=True)
            continue
        sub = d[d["_shard"] == si]
        t0 = time.time()
        X = run_batch(llm, sp, sub["judge_text"].tolist(), blocks)
        np.savez_compressed(
            outp, X=X, row_ids=np.array(sub["row_id"].tolist(), dtype=object),
            a_ids=np.array([c["id"] for c in crit], dtype=object),
            a_names=np.array([c["name"] for c in crit], dtype=object))
        print(f"[shard {si}] {len(sub)}x{len(blocks)} in {(time.time()-t0)/60:.1f} min "
              f"NA {np.isnan(X).mean():.4f}", flush=True)

    # ---- assemble + collapse gate ------------------------------------------
    Xs, ids = [], []
    for si in range(a.shards):
        z = np.load(out / f"{a.tag}_scores_shard{si:02d}.npz", allow_pickle=True)
        Xs.append(z["X"])
        ids += [str(x) for x in z["row_ids"]]
    X = np.vstack(Xs)
    coll = []
    for j, c in enumerate(crit):
        col = X[:, j]
        nn = col[~np.isnan(col)]
        vals, cnt = (np.unique(nn, return_counts=True) if len(nn)
                     else (np.array([]), np.array([])))
        modal = float(cnt.max() / len(col)) if len(nn) else 1.0
        coll.append({"blind_id": c["id"], "name": c["name"],
                     "na_rate": float(np.isnan(col).mean()), "modal_frac": modal,
                     "collapsed": bool(modal > 0.98 or len(nn) == 0)})
    np.savez_compressed(out / f"{a.tag}_scores.npz", X=X,
                        row_ids=np.array(ids, dtype=object),
                        a_ids=np.array([c["id"] for c in crit], dtype=object),
                        a_names=np.array([c["name"] for c in crit], dtype=object))
    rep = {"tag": a.tag, "judge": "gemma-4-31b-it (bfloat16, vLLM offline batch)",
           "n_rows": int(X.shape[0]), "n_criteria": int(X.shape[1]),
           "n_prompts": int(X.size), "pooled_na_rate": float(np.isnan(X).mean()),
           "n_collapsed": int(sum(c["collapsed"] for c in coll)),
           "collapse": coll, "anchors": arep or "already on disk",
           "max_text_tokens": a.max_text_tokens, "n_truncated": int(n_trunc)}
    (out / f"{a.tag}_scores.report.json").write_text(json.dumps(rep, indent=1))
    print(f"SCORE_DONE {a.tag}: {X.shape} NA {rep['pooled_na_rate']:.4f} "
          f"collapsed {rep['n_collapsed']}/{X.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
