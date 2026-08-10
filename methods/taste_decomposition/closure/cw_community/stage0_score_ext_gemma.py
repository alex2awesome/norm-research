#!/usr/bin/env python3
"""STAGE 0 scoring: the UNCHANGED 45-criterion CW A-bank on the extension rows.

Population extension only -- criteria, system prompt, truncation rule, token
vocabulary and sampling params are byte-identical to
datasets/va_gemma_banks/score_va_gemma_banks.py::build_creative()/score_bank().

Raised per the confirmatory freeze: anchor battery K>=50 per class
(pos / neg / scrambled), plus a programmatic per-criterion collapse check.

Run on sk3, ONE GPU:
  export HOME=/lfs/skampere3/0/alexspan
  CUDA_VISIBLE_DEVICES=6 $HOME/envs/gemma4/bin/python stage0_score_ext_gemma.py
"""
from __future__ import annotations

import argparse
import hashlib
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
HERE = REPO / "methods/taste_decomposition/closure/cw_community"
GEMMA4 = os.environ.get(
    "GEMMA4_PATH",
    "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/"
    "3548789868c5356dbf307c98e6f609007b82b3eb",
)
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
SEED = 20260728

SYS_CW = (
    "You are an expert fiction editor performing a measurement task. You are given a "
    "writing prompt, ONE story written to that prompt, and ONE craft criterion. "
    "Decide how strongly the story, on the evidence of the supplied text alone, "
    "satisfies that criterion. Answer with EXACTLY ONE token:\n"
    "  1.0 = clearly satisfies the criterion\n  0.5 = partly, weakly, inconsistently, "
    "or borderline\n"
    "  0.0 = clearly fails or cuts against the criterion\n"
    "  NA = the supplied text gives no evidence bearing on the criterion\n"
    "Do not predict votes, popularity, labels, authorship, or dataset membership. "
    "Long stories may have a deterministically omitted middle; judge what is shown. "
    "Output only the token."
)
TRUNCATE_SOURCE_CHARS = 6000
TRUNCATE_HEAD_CHARS = 3600
TRUNCATE_TAIL_CHARS = 2400
TRUNCATION_MARKER = "\n\n[... DETERMINISTIC MIDDLE OMISSION ...]\n\n"
CW_PROMPT_CHARS = 1200


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


def _trunc(story: str) -> str:
    story = str(story).strip()
    if len(story) <= TRUNCATE_SOURCE_CHARS:
        return story
    return story[:TRUNCATE_HEAD_CHARS] + TRUNCATION_MARKER + story[-TRUNCATE_TAIL_CHARS:]


def ctx(prompt, story):
    return (f"WRITING PROMPT: {str(prompt)[:CW_PROMPT_CHARS]}\n\n"
            f"STORY:\n{_trunc(story)}")


def scramble(texts, rng, n_words=220):
    toks = []
    for t in texts:
        toks += WORD_RE.findall(t)
    rng.shuffle(toks)
    k = n_words or max(8, min(14, len(toks)))
    chosen = toks[:k]
    chosen[1::2] = [w[::-1] for w in chosen[1::2]]
    return " ".join(chosen)


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(HERE / "cw_ext_to_score.csv"))
    ap.add_argument("--out", default=str(HERE / "cw_ext_scores.npz"))
    ap.add_argument("--anchor-k", type=int, default=50)
    ap.add_argument("--util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--chunk", type=int, default=27000)
    a = ap.parse_args()

    df = pd.read_csv(a.rows).fillna({"prompt": "", "story": ""})
    rubrics = [json.loads(l) for l in
               open(REPO / "datasets/creative-writing/va_bank_v2/rubrics_initial.jsonl")
               if l.strip()]
    blocks = [f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n"
              f"GUIDANCE: {m.get('guidance','')}\n\nAnswer with one token:"
              for m in rubrics]
    k = len(blocks)
    print(f"[stage0] rows={len(df)} criteria={k} prompts={len(df)*k}", flush=True)

    # -------- anchors: K per class, drawn from THIS population, blinded -------
    rng = random.Random(SEED + 4242)
    pos_pool = df[df.judgement == 1].to_dict("records")
    neg_pool = df[df.judgement == 0].to_dict("records")
    arows, atags = [], []
    for j in range(a.anchor_k):
        p, n = rng.choice(pos_pool), rng.choice(neg_pool)
        s = dict(n)
        s["story"] = scramble([str(p["story"])[:4000], str(n["story"])[:4000]], rng)
        for tag, r in (("anchor_pos", p), ("anchor_neg", n), ("anchor_scram", s)):
            arows.append(r)
            atags.append(tag)
    print(f"[stage0] anchors: {len(arows)} rows ({a.anchor_k}/class) x {k} = "
          f"{len(arows)*k} prompts", flush=True)

    convs = []
    for r in df.itertuples():
        c = ctx(r.prompt, r.story)
        for b in blocks:
            convs.append([{"role": "user", "content": f"{SYS_CW}\n\n{c}\n\n{b}"}])
    aconvs = []
    for r in arows:
        c = ctx(r["prompt"], r["story"])
        for b in blocks:
            aconvs.append([{"role": "user", "content": f"{SYS_CW}\n\n{c}\n\n{b}"}])

    # ---- chunk checkpointing: a SIGTERM'd run resumes, never rescores ---------
    ckpt = Path(a.out).with_suffix("")
    ckpt = ckpt.parent / f"{ckpt.name}_parts"
    ckpt.mkdir(parents=True, exist_ok=True)

    def done_parts(tag, n):
        return all((ckpt / f"{tag}_{i}.npy").exists()
                   for i in range(0, n, a.chunk))

    llm = None
    if not (done_parts("main", len(convs)) and done_parts("anchors", len(aconvs))):
        from vllm import LLM, SamplingParams
        llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.util,
                  max_model_len=a.max_model_len, enable_prefix_caching=True,
                  trust_remote_code=True, max_num_seqs=256)
        sp = SamplingParams(temperature=0.0, max_tokens=6)

    def run(cs, tag):
        vals = []
        for i in range(0, len(cs), a.chunk):
            f = ckpt / f"{tag}_{i}.npy"
            if f.exists():
                vals.append(np.load(f))
                print(f"[{tag}] {min(i+a.chunk,len(cs))}/{len(cs)} (cached)", flush=True)
                continue
            outs = llm.chat(cs[i:i + a.chunk], sp)
            v = np.array([parse_tok(o.outputs[0].text) for o in outs], dtype=float)
            np.save(f, v)
            vals.append(v)
            print(f"[{tag}] {min(i+a.chunk,len(cs))}/{len(cs)}", flush=True)
        return np.concatenate(vals)

    X = run(convs, "main").reshape(len(df), k)
    aX = run(aconvs, "anchors").reshape(len(arows), k)

    # -------------------------------- V features -----------------------------
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vf_cw", str(REPO / "datasets/creative-writing/va_bank_v2/v_features.py"))
    vf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vf)
    V = np.array([vf.feature_vector(str(s)) for s in df.story], dtype=float)

    # -------------------------- anchor + collapse gates ----------------------
    atags_a = np.array(atags)
    item_mean = np.nanmean(aX, axis=1)
    from sklearn.metrics import roc_auc_score
    pv = item_mean[atags_a == "anchor_pos"]
    nv = item_mean[atags_a == "anchor_neg"]
    sv = item_mean[atags_a == "anchor_scram"]
    battery = {
        "k_per_class": a.anchor_k,
        "anchor_pos": {"mean": float(np.mean(pv)), "sd": float(np.std(pv, ddof=1))},
        "anchor_neg": {"mean": float(np.mean(nv)), "sd": float(np.std(nv, ddof=1))},
        "anchor_scram": {"mean": float(np.mean(sv)), "sd": float(np.std(sv, ddof=1))},
        "pos_vs_neg_auc": float(roc_auc_score([1] * len(pv) + [0] * len(nv),
                                              np.concatenate([pv, nv]))),
        "coherent_vs_scrambled_auc": float(roc_auc_score(
            [1] * (len(pv) + len(nv)) + [0] * len(sv),
            np.concatenate([pv, nv, sv]))),
        "ordering_holds_on_means": bool(np.mean(pv) > np.mean(nv) > np.mean(sv)),
    }
    collapse = []
    for j, m in enumerate(rubrics):
        col = X[:, j]
        fin = np.isfinite(col)
        vals, cnts = np.unique(col[fin], return_counts=True) if fin.any() else (
            np.array([]), np.array([]))
        modal = float(cnts.max() / max(fin.sum(), 1)) if len(cnts) else 1.0
        collapse.append({"criterion": m["name"], "na_rate": float((~fin).mean()),
                         "modal_share": modal, "n_distinct": int(len(vals)),
                         "mean": float(np.nanmean(col)) if fin.any() else float("nan"),
                         "COLLAPSED": bool(modal > 0.98 or len(vals) < 2)})
    n_collapsed = sum(c["COLLAPSED"] for c in collapse)

    np.savez_compressed(
        a.out, X=X, V=V, anchor_X=aX,
        ids=df.id.values.astype(object),
        groups=df.prompt_id.values.astype(str).astype(object),
        y=df.judgement.values.astype(int),
        dense_split=df.dense_split.values.astype(object),
        anchor_tags=atags_a.astype(object),
        a_names=np.array([m["name"] for m in rubrics], dtype=object),
        v_names=np.array(list(vf.V_NAMES), dtype=object),
    )
    rep = {"n_rows": int(len(df)), "n_criteria": k, "na_rate": float(np.isnan(X).mean()),
           "anchor_battery": battery, "n_collapsed": int(n_collapsed),
           "collapse": collapse,
           "prompt_hash": hashlib.sha256(
               (SYS_CW + "||" + "||".join(blocks)).encode()).hexdigest()[:16]}
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(rep, indent=1))
    print("STAGE0_SCORE_REPORT " + json.dumps(
        {kk: vv for kk, vv in rep.items() if kk != "collapse"}), flush=True)
    print("STAGE0_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
