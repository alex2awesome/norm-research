#!/usr/bin/env python3
"""ADDENDUM G1, peer_verdict — score the 1,244 E-frame abstracts on parallel
forms b (blind paraphrase) and c (name-only) of the top-24 unique criteria.
Judge frame byte-matches the certified peer scorer (closure/score_round1_gemma.py):
0-10 integer + NA scale, expert-peer-reviewer SYS, ABSTRACT context.

Run on sk3 (small: ~67k calls):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 GPU_MEM_UTIL=0.30 \
    $HOME/envs/gemma4/bin/python g1_score_forms_peer.py
"""
from __future__ import annotations

import json
import os
import random
import re
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
HERE = REPO / "methods/taste_decomposition/rung2"
GEMMA4 = ("/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/"
          "snapshots/3548789868c5356dbf307c98e6f609007b82b3eb")

# byte-identical to closure/score_round1_gemma.py
SYS = (
    "You are an expert academic peer reviewer. You are given a paper's ABSTRACT and ONE "
    "criterion. Decide how strongly the abstract, on its own evidence, exhibits that "
    "criterion. Answer with EXACTLY ONE token:\n"
    "  an integer from 0 to 10, where 0 = not at all and 10 = to the fullest degree\n"
    "  NA = the abstract gives no evidence bearing on this criterion\n"
    "Judge the criterion as literally described, not whether the paper will be accepted. "
    "Output only the token."
)
NUM = re.compile(r"\d+")
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
SEED = 20260826


def parse_tok(t):
    t = (t or "").strip()
    if t.lower().startswith("na") or "n/a" in t.lower():
        return np.nan
    m = NUM.search(t)
    if m:
        v = int(m.group())
        return float(min(max(v, 0), 10))
    return np.nan


def scramble(texts, rng, n_words=180):
    toks = []
    for t in texts:
        toks += WORD_RE.findall(t)
    rng.shuffle(toks)
    chosen = toks[:n_words]
    chosen[1::2] = [w[::-1] for w in chosen[1::2]]
    return " ".join(chosen)


def main():
    # E-frame rows rebuilt positionally, exactly as fusion/t0_build_rows.do_peer_verdict
    import sys
    import pandas as pd
    CLOSURE = REPO / "methods/taste_decomposition/closure"
    sys.path.insert(0, str(CLOSURE))
    import importlib.util
    spec = importlib.util.spec_from_file_location("cl_lib_g1", CLOSURE / "closure_lib.py")
    L = importlib.util.module_from_spec(spec)
    sys.modules["cl_lib_g1"] = L
    spec.loader.exec_module(L)
    pop = L.load_population()
    dp = pd.read_csv(CLOSURE / "peer_verdict_dense_preds.csv")
    assert (dp["judgement"].astype(int).values == pop["y"]).all()
    E = np.isin(dp["dense_split"].values, ["eval", "test"])
    texts = [pop["texts"][i] for i in np.flatnonzero(E)]
    ids = [str(s) for s in np.asarray(pop["ntitle"])[E]]
    y = pop["y"][E]
    zt = np.load(REPO / "methods/taste_decomposition/fusion/t0_rows/peer_verdict.npz",
                 allow_pickle=True)
    assert [str(s) for s in zt["ids"]] == ids, "E-frame ids drifted from t0_rows"
    print(f"[G1-peer] E rows={len(ids)}", flush=True)

    forms = json.load(open(HERE / "g1_forms_peer.json"))
    blocks, tags = [], []
    for f in forms:
        for fk in ("form_b_paraphrase", "form_c_minimal"):
            blocks.append(f[fk])
            tags.append((f["name"], fk[5]))
    k = len(blocks)

    rng = random.Random(SEED)
    pos = [t for t, yy in zip(texts, y) if yy == 1]
    neg = [t for t, yy in zip(texts, y) if yy == 0]
    arows, atags = [], []
    for _ in range(40):
        p, n = rng.choice(pos), rng.choice(neg)
        for tag, tx in (("anchor_pos", p), ("anchor_neg", n),
                        ("anchor_scram", scramble([p, n], rng))):
            arows.append(tx)
            atags.append(tag)

    convs = [[{"role": "user", "content": f"{SYS}\n\nABSTRACT:\n{t}\n\n{b}"}]
             for t in texts for b in blocks]
    aconvs = [[{"role": "user", "content": f"{SYS}\n\nABSTRACT:\n{t}\n\n{b}"}]
              for t in arows for b in blocks]

    ckpt = HERE / "g1_forms_peer_parts"
    ckpt.mkdir(exist_ok=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model=GEMMA4, dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.30")),
              max_model_len=4096, enable_prefix_caching=True,
              trust_remote_code=True, max_num_seqs=256,
              tensor_parallel_size=int(os.environ.get("TP", "1")),
              limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=0.0, max_tokens=6)
    CH = 27000

    def run(cs, tag):
        vals = []
        for i in range(0, len(cs), CH):
            f = ckpt / f"{tag}_{i}.npy"
            if f.exists():
                vals.append(np.load(f))
                continue
            outs = llm.chat(cs[i:i + CH], sp)
            v = np.array([parse_tok(o.outputs[0].text) for o in outs], float)
            np.save(f, v)
            vals.append(v)
            print(f"[{tag}] {min(i+CH,len(cs))}/{len(cs)}", flush=True)
        return np.concatenate(vals)

    X = run(convs, "main").reshape(len(texts), k)
    aX = run(aconvs, "anchors").reshape(len(arows), k)

    from sklearn.metrics import roc_auc_score
    t = np.array(atags)
    im = np.nanmean(aX, axis=1)
    pv, nv, sv = im[t == "anchor_pos"], im[t == "anchor_neg"], im[t == "anchor_scram"]
    battery = {
        "pos_vs_neg_auc": float(roc_auc_score([1]*len(pv)+[0]*len(nv),
                                              np.concatenate([pv, nv]))),
        "coherent_vs_scrambled_auc": float(roc_auc_score(
            [1]*(len(pv)+len(nv))+[0]*len(sv), np.concatenate([pv, nv, sv]))),
        "ordering_holds": bool(np.mean(pv) > np.mean(nv) > np.mean(sv)),
    }
    np.savez_compressed(HERE / "g1_form_scores_peer.npz",
                        X=X, anchor_X=aX, ids=np.array(ids, dtype=object),
                        y=np.asarray(y).astype(int),
                        groups=np.array(ids, dtype=object),
                        form_names=np.array([f"{n}::{fm}" for n, fm in tags],
                                            dtype=object),
                        anchor_tags=t.astype(object))
    rep = {"n_rows": len(ids), "n_form_blocks": k,
           "na_rate": float(np.isnan(X).mean()), "anchor_battery": battery,
           "design": "ADDENDUM G1 peer_verdict"}
    (HERE / "g1_form_scores_peer.report.json").write_text(json.dumps(rep, indent=1))
    print("G1PEER_REPORT " + json.dumps(rep), flush=True)
    print("G1PEER_DONE", flush=True)


if __name__ == "__main__":
    main()
