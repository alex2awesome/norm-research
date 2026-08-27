#!/usr/bin/env python3
"""RUNG 2, stage S — score generated candidates on the FROZEN enriched bank
(cw_community). Design: notes/2026-08-21__rung12_design_gap_consequences.md §2.4.1.

Byte-matches the certified instruments:
- system prompt, truncation rule, token vocabulary, sampling and parse =
  closure/cw_community/stage0_score_ext_gemma.py (base 45) and
  score_round_gemma.py (mined criteria block format);
- criteria set + column order = round7_state.npz bank_names (the F2 (a)-arm
  matrix): 15 programmatic V features + 45 base rubrics + 84 mined A-routed;
- anchor battery: K real pos / K real neg / K scrambled drawn from the REAL
  honest population, blinded into the same batches;
- per-criterion collapse check; chunk checkpointing (SIGTERM-safe resume).

Run on sk3, ONE allowed GPU:
  export HOME=/lfs/skampere3/0/alexspan
  setsid nohup env CUDA_VISIBLE_DEVICES=0 HOME=$HOME \
    $HOME/envs/gemma4/bin/python rung2_score_bank_cw.py --util 0.49 \
    > rung2_score_bank_cw.log 2>&1 < /dev/null &
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
CWD_ = REPO / "methods/taste_decomposition/closure/cw_community"
HERE = REPO / "methods/taste_decomposition/rung2"
GEMMA4 = os.environ.get(
    "GEMMA4_PATH",
    "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/"
    "3548789868c5356dbf307c98e6f609007b82b3eb",
)
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
SEED = 20260821

# ---- byte-identical scoring constants (stage0_score_ext_gemma.py) -----------
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
TRUNCATE_SOURCE_CHARS, TRUNCATE_HEAD_CHARS, TRUNCATE_TAIL_CHARS = 6000, 3600, 2400
TRUNCATION_MARKER = "\n\n[... DETERMINISTIC MIDDLE OMISSION ...]\n\n"
CW_PROMPT_CHARS = 1200


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


def _trunc(story):
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
    chosen = toks[:n_words]
    chosen[1::2] = [w[::-1] for w in chosen[1::2]]
    return " ".join(chosen)


def build_blocks():
    """Scoring block per judge-criterion, in round7 bank_names order.
    Prefers the pre-built manifest (the closure npz lives on the mac only)."""
    man = HERE / "rung2_bank_manifest_cw.json"
    if man.exists():
        d = json.load(open(man))
        return d["bank_names"], d["v_names"], d["judge_names"], d["blocks"]
    z7 = np.load(CWD_ / "round7_state.npz", allow_pickle=True)
    bank_names = [str(s) for s in z7["bank_names"]]
    rub = {}
    for l in open(REPO / "datasets/creative-writing/va_bank_v2/rubrics_initial.jsonl"):
        if l.strip():
            m = json.loads(l)
            rub[m["name"]] = (f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n"
                              f"GUIDANCE: {m.get('guidance','')}\n\nAnswer with one token:")
    mined = {}
    for r in range(1, 9):
        p = CWD_ / f"round{r}_criteria.json"
        if p.exists():
            for c in json.load(open(p)):
                mined.setdefault(c["name"], f"CRITERION: {c['name']}\n"
                                 f"DESCRIPTION: {c['instruction']}\n\n"
                                 f"Answer with one token:")
    judge_names = [n for n in bank_names if not n.startswith("v_")]
    blocks = []
    for n in judge_names:
        b = rub.get(n) or mined.get(n)
        assert b, f"no scoring definition for bank criterion: {n}"
        blocks.append(b)
    v_names = [n for n in bank_names if n.startswith("v_")]
    return bank_names, v_names, judge_names, blocks


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default=str(HERE / "rung2_candidates_cw_community_full.csv"))
    ap.add_argument("--out", default=str(HERE / "rung2_bank_scores_cw.npz"))
    ap.add_argument("--anchor-k", type=int, default=50)
    ap.add_argument("--util", type=float, default=0.49)
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--chunk", type=int, default=27000)
    a = ap.parse_args()

    df = pd.read_csv(a.cands).fillna({"prompt": "", "story": ""})
    bank_names, v_names, judge_names, blocks = build_blocks()
    k = len(blocks)
    print(f"[rung2-S] candidates={len(df)} judge-criteria={k} "
          f"prompts={len(df)*k}", flush=True)

    # anchors from the REAL population (blinded, same batches)
    pop = pd.read_csv(CWD_ / "cw_honest_population.csv")
    rng = random.Random(SEED + 4242)
    pos_pool = pop[pop.judgement == 1].to_dict("records")
    neg_pool = pop[pop.judgement == 0].to_dict("records")
    arows, atags = [], []
    for _ in range(a.anchor_k):
        p, n = rng.choice(pos_pool), rng.choice(neg_pool)
        s = dict(n)
        s["story"] = scramble([str(p["story"])[:4000], str(n["story"])[:4000]], rng)
        for tag, r in (("anchor_pos", p), ("anchor_neg", n), ("anchor_scram", s)):
            arows.append(r)
            atags.append(tag)

    convs = [[{"role": "user", "content": f"{SYS_CW}\n\n{ctx(r.prompt, r.story)}\n\n{b}"}]
             for r in df.itertuples() for b in blocks]
    aconvs = [[{"role": "user", "content": f"{SYS_CW}\n\n{ctx(r['prompt'], r['story'])}\n\n{b}"}]
              for r in arows for b in blocks]

    ckpt = Path(a.out).parent / (Path(a.out).stem + "_parts")
    ckpt.mkdir(parents=True, exist_ok=True)

    def done_parts(tag, n):
        return all((ckpt / f"{tag}_{i}.npy").exists() for i in range(0, n, a.chunk))

    llm = None
    if not (done_parts("main", len(convs)) and done_parts("anchors", len(aconvs))):
        from vllm import LLM, SamplingParams
        llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.util,
                  max_model_len=a.max_model_len, enable_prefix_caching=True,
                  trust_remote_code=True, max_num_seqs=256,
                  tensor_parallel_size=int(os.environ.get("TP", "1")),
                  limit_mm_per_prompt={"image": 0, "video": 0})
        sp = SamplingParams(temperature=0.0, max_tokens=6)

    def run(cs, tag):
        vals = []
        for i in range(0, len(cs), a.chunk):
            f = ckpt / f"{tag}_{i}.npy"
            if f.exists():
                vals.append(np.load(f))
                continue
            outs = llm.chat(cs[i:i + a.chunk], sp)
            v = np.array([parse_tok(o.outputs[0].text) for o in outs], dtype=float)
            np.save(f, v)
            vals.append(v)
            print(f"[{tag}] {min(i+a.chunk,len(cs))}/{len(cs)}", flush=True)
        return np.concatenate(vals)

    X = run(convs, "main").reshape(len(df), k)
    aX = run(aconvs, "anchors").reshape(len(arows), k)

    # V features — same module as the certified population build
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vf_cw", str(REPO / "datasets/creative-writing/va_bank_v2/v_features.py"))
    vf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vf)
    V = np.array([vf.feature_vector(str(s)) for s in df.story], dtype=float)
    assert list(vf.V_NAMES) == v_names, "V feature order drifted from bank_names"

    # anchor battery + collapse gates (same thresholds as stage0 ext)
    from sklearn.metrics import roc_auc_score
    t = np.array(atags)
    im = np.nanmean(aX, axis=1)
    pv, nv, sv = im[t == "anchor_pos"], im[t == "anchor_neg"], im[t == "anchor_scram"]
    battery = {
        "k_per_class": a.anchor_k,
        "pos_vs_neg_auc": float(roc_auc_score([1]*len(pv)+[0]*len(nv),
                                              np.concatenate([pv, nv]))),
        "coherent_vs_scrambled_auc": float(roc_auc_score(
            [1]*(len(pv)+len(nv))+[0]*len(sv), np.concatenate([pv, nv, sv]))),
        "ordering_holds_on_means": bool(np.mean(pv) > np.mean(nv) > np.mean(sv)),
    }
    collapse = []
    for j, nme in enumerate(judge_names):
        col = X[:, j]
        fin = np.isfinite(col)
        vals, cnts = (np.unique(col[fin], return_counts=True) if fin.any()
                      else (np.array([]), np.array([])))
        modal = float(cnts.max() / max(fin.sum(), 1)) if len(cnts) else 1.0
        collapse.append({"criterion": nme, "na_rate": float((~fin).mean()),
                         "modal_share": modal,
                         "COLLAPSED": bool(modal > 0.98 or len(vals) < 2)})

    np.savez_compressed(
        a.out, X=X, V=V, anchor_X=aX,
        cand_ids=df.cand_id.values.astype(object),
        prompt_ids=df.prompt_id.values.astype(str).astype(object),
        anchor_tags=t.astype(object),
        judge_names=np.array(judge_names, dtype=object),
        v_names=np.array(v_names, dtype=object),
        bank_names=np.array(bank_names, dtype=object),
    )
    rep = {"n_candidates": int(len(df)), "n_criteria": k,
           "na_rate": float(np.isnan(X).mean()),
           "anchor_battery": battery,
           "n_collapsed": int(sum(c["COLLAPSED"] for c in collapse)),
           "collapse": collapse,
           "prompt_hash": hashlib.sha256(
               (SYS_CW + "||" + "||".join(blocks)).encode()).hexdigest()[:16]}
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(rep, indent=1))
    print("RUNG2_SCORE_REPORT " + json.dumps(
        {kk: vv for kk, vv in rep.items() if kk != "collapse"}), flush=True)
    print("RUNG2_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
