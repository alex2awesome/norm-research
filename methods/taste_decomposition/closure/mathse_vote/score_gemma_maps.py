#!/usr/bin/env python3
"""Corpus-wide Gemma-4-31B scoring of one round's 25 criteria, for any number of
map-focused cells in ONE vLLM process (the model is loaded once).

Same machinery as closure/score_round1_gemma.py: offline-batch `llm.chat` (never an
HTTP server), temperature 0, single-token 0-10 / NA readout, 5,000-char truncation,
prefix caching on, per-criterion collapse check, blinded anchor battery in the SAME
batch.  Two freeze changes from the pilot:
  * anchors K >= 50 per class (pilot used 12);
  * the item noun and the judge persona are per-cell.

Run on sk3, one GPU, via gpu_runner.sh:
  ./gpu_runner.sh maps_r1 <log> $HOME/envs/gemma4/bin/python score_gemma_maps.py \
      --jobs cap_crowd_r1,cap_finalist_r1,nc_outcome_r1
"""
import argparse
import csv
import json
import os
import random
import re
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import multiprocessing as _mp  # noqa: E402

try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
GEMMA4 = ("/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/"
          "snapshots/3548789868c5356dbf307c98e6f609007b82b3eb")

PERSONA = {
    "mathse_vote": ("an expert mathematician performing a measurement task", "ANSWER"),
}

# TRUNCATION MATCHED TO THE INCOMING BANK.  The A bank was scored with a
# deterministic HEAD 3000 + TAIL 2000 middle omission at 5,000 source chars
# (datasets/va_gemma_banks/score_scaleupC_banks.py::_trunc), and the population
# text already carries the 400-char question title.  Mined criteria are scored on
# the IDENTICAL view so a round's criteria and the bank's criteria are answering
# about the same text.  Head-only truncation would have shown the mined criteria
# a different document.
TRUNC_SRC, TRUNC_HEAD, TRUNC_TAIL = 5000, 3000, 2000
TRUNC_MARK = "\n\n[... DETERMINISTIC MIDDLE OMISSION ...]\n\n"


def bank_trunc(s):
    s = (s or "").strip()
    return s if len(s) <= TRUNC_SRC else s[:TRUNC_HEAD] + TRUNC_MARK + s[-TRUNC_TAIL:]

NUM = re.compile(r"\d+")

# LANDMINE (math.SE, 2026-08-08): the bank-matched 5,000-char item view overflows a
# 4,096-token context on this corpus. LaTeX tokenises far denser than prose (~2 chars
# per token against ~4 for the press/peer cells), so the longest math answers render at
# >4,097 tokens and vLLM raises VLLMValidationError mid-render. Shortening the text was
# NOT an option -- the truncation is matched to the incoming A bank on purpose -- so the
# context is raised instead via --max-model-len. Use 8192 on this cell.
K_ANCHOR = 50
SEED = 20260806
MAXCHARS = 5000


def sys_prompt(persona, noun):
    return (f"You are {persona}. You are given one {noun} and ONE criterion. Decide how "
            f"strongly the {noun.lower()}, on its own evidence, exhibits that criterion. "
            "Answer with EXACTLY ONE token:\n"
            "  an integer from 0 to 10, where 0 = not at all and 10 = to the fullest degree\n"
            f"  NA = the {noun.lower()} gives no evidence bearing on this criterion\n"
            "Judge the criterion as literally described, not whether the item is good "
            "overall. Output only the token.")


def parse_tok(t):
    t = (t or "").strip()
    low = t.lower()
    if low.startswith("na") or "n/a" in low:
        return np.nan
    m = NUM.search(t)
    if not m:
        return np.nan
    v = float(m.group(0))
    return v if 0.0 <= v <= 10.0 else np.nan


def scramble(texts, rng):
    words = " ".join(texts).split()
    rng.shuffle(words)
    return " ".join(words[:220])


def build_job(tag):
    cell = tag.rsplit("_r", 1)[0]
    persona, noun = PERSONA[cell]
    sel = json.loads((HERE / f"{tag}_species.json").read_text())["selected"]
    crits = [{"id": c["blind_id"], "name": c["name"], "instruction": c["instruction"]}
             for c in sel]
    rows = []
    with open(HERE / f"{cell}_population.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return dict(tag=tag, cell=cell, persona=persona, noun=noun, crits=crits, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams

    jobs = []
    for t in a.jobs.split(","):
        t = t.strip()
        if not (HERE / f"{t}_species.json").exists():
            print(f"[maps] {t}: no species file yet, SKIPPING (rerun later)", flush=True)
            continue
        if (HERE / f"{t}_scores.npz").exists():
            print(f"[maps] {t}: already scored, skipping", flush=True)
            continue
        try:
            jobs.append(build_job(t))
        except Exception as e:  # noqa: BLE001
            print(f"[maps] {t}: build_job failed ({type(e).__name__}: {e}), SKIPPING", flush=True)
    if not jobs:
        print("ALL_SCORE_DONE (nothing to do)", flush=True)
        return
    total = sum(len(j["rows"]) * len(j["crits"]) for j in jobs)
    print(f"[maps] {len(jobs)} jobs, {total} population prompts", flush=True)

    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              trust_remote_code=True,
              max_num_seqs=256)
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    for j in jobs:
        tag, crits, rows = j["tag"], j["crits"], j["rows"]
        if (HERE / f"{tag}_scores.npz").exists():
            print(f"[{tag}] already scored, skip", flush=True)
            continue
        SYS = sys_prompt(j["persona"], j["noun"])
        blocks = [f"CRITERION: {c['name']}\nINSTRUCTION: {c['instruction']}\n\n"
                  "Answer with one token:" for c in crits]
        cids = [c["id"] for c in crits]
        texts = [r["text"] for r in rows]

        rng = random.Random(SEED)
        pos = [r["text"] for r in rows if str(r["judgement"]) == "1"]
        neg = [r["text"] for r in rows if str(r["judgement"]) == "0"]
        a_texts, a_tags = [], []
        for _ in range(K_ANCHOR):
            p, n = rng.choice(pos), rng.choice(neg)
            a_texts += [p, n, scramble([p, n], rng)]
            a_tags += ["anchor_pos", "anchor_neg", "anchor_scram"]

        all_texts = texts + a_texts
        convs = []
        for t in all_texts:
            f = bank_trunc(t)
            for b in blocks:
                convs.append([{"role": "user",
                               "content": f"{SYS}\n\n{j['noun']}:\n{f}\n\n{b}"}])
        print(f"[{tag}] {len(rows)} rows + {len(a_texts)} anchors x {len(crits)} crit "
              f"= {len(convs)} prompts", flush=True)
        outs = llm.chat(convs, sp)
        raw = [o.outputs[0].text for o in outs]
        X = np.array([parse_tok(t) for t in raw], dtype=float).reshape(
            len(all_texts), len(crits))
        Xpop, Xanc = X[:len(texts)], X[len(texts):]
        np.savez_compressed(
            HERE / f"{tag}_scores.npz", X=Xpop,
            crit_ids=np.array(cids, dtype=object),
            crit_names=np.array([c["name"] for c in crits], dtype=object),
            i=np.array([int(r["i"]) for r in rows]),
            row_id=np.array([r["id"] for r in rows], dtype=object),
            Xanchor=Xanc, anchor_tags=np.array(a_tags, dtype=object), scale="0-10")

        rep = {"tag": tag, "n_rows": len(texts), "n_criteria": len(crits),
               "per_criterion": {}}
        for k, cid in enumerate(cids):
            col = Xpop[:, k]
            ok = col[~np.isnan(col)]
            vals, counts = (np.unique(ok, return_counts=True) if len(ok)
                            else (np.array([]), np.array([])))
            rep["per_criterion"][cid] = {
                "name": crits[k]["name"], "na_rate": float(np.isnan(col).mean()),
                "mean": float(np.mean(ok)) if len(ok) else None,
                "std": float(np.std(ok)) if len(ok) else None,
                "n_distinct": int(len(vals)),
                "modal_frac": float(counts.max() / len(ok)) if len(ok) else None,
                "value_counts": {str(v): int(c) for v, c in zip(vals, counts)},
                "collapsed": bool(len(ok) == 0 or len(vals) <= 1
                                  or counts.max() / len(ok) > 0.98)}
        from sklearn.metrics import roc_auc_score
        tags = np.array(a_tags)
        item = np.nanmean(Xanc, axis=1)
        anc = {"k_per_class": K_ANCHOR}
        for t in ("anchor_pos", "anchor_neg", "anchor_scram"):
            v = item[tags == t]
            anc[t] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1))}
        pv, nv, sv = (item[tags == "anchor_pos"], item[tags == "anchor_neg"],
                      item[tags == "anchor_scram"])
        anc["pos_vs_neg_auc"] = float(roc_auc_score(
            [1] * len(pv) + [0] * len(nv), np.concatenate([pv, nv])))
        anc["coherent_vs_scrambled_auc"] = float(roc_auc_score(
            [1] * (len(pv) + len(nv)) + [0] * len(sv), np.concatenate([pv, nv, sv])))
        anc["pass_scrambled"] = bool(anc["coherent_vs_scrambled_auc"] >= 0.70)
        rep["anchors"] = anc
        rep["n_collapsed"] = int(sum(v["collapsed"] for v in rep["per_criterion"].values()))
        rep["overall_na_rate"] = float(np.isnan(Xpop).mean())
        # INTERRUPTED-GENERATION GATE (added 2026-08-08).  A row that is NA on EVERY
        # criterion is not a judging outcome, it is a request that never generated --
        # the signature of an engine that died mid-batch.  The collapse gate is
        # per-criterion and cannot see it.
        rep["n_rows_all_NA"] = int(np.isnan(Xpop).all(axis=1).sum())
        rep["rows_all_NA_rate"] = float(np.isnan(Xpop).all(axis=1).mean())
        rep["anchor_rows_all_NA"] = int(np.isnan(Xanc).all(axis=1).sum())
        rep["INTERRUPTED_GENERATION_SUSPECTED"] = bool(
            rep["rows_all_NA_rate"] > 0.02 or rep["anchor_rows_all_NA"] > 0)
        if rep["INTERRUPTED_GENERATION_SUSPECTED"]:
            print(f"[{tag}] !! INTERRUPTED GENERATION SUSPECTED: "
                  f"{rep['n_rows_all_NA']} population rows and "
                  f"{rep['anchor_rows_all_NA']} anchor rows are NA on every criterion",
                  flush=True)
        (HERE / f"{tag}_score_report.json").write_text(json.dumps(rep, indent=2))
        print(f"[{tag}] ANCHORS " + json.dumps(anc), flush=True)
        print(f"[{tag}] COLLAPSED {rep['n_collapsed']} NA {rep['overall_na_rate']:.3f}",
              flush=True)
        print(f"SCORE_DONE {tag}", flush=True)
    print("ALL_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
