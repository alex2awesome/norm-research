#!/usr/bin/env python3
"""Corpus-wide Gemma-4-31B scoring of one round's criteria for CAP_FINALIST,
offline batch vLLM (never an HTTP server; the model is loaded once).

Inherited from jokes_community/score_gemma_maps.py.  THREE cell-specific changes, each
recorded rather than silent:

1. ITEM VIEW MATCHED TO THE INCOMING BANK (this campaign's round-3 fix).  The A bank
   showed the judge `CARTOON: <description>\\n\\nCAPTION: "<text>"`
   (datasets/humor/caption_multiy/score_va_gemma_captions.py:190).  maps_batch1 scored
   rounds 1-2 on the caption ALONE.  A caption is close to ungradeable without the
   drawing it captions, so mined criteria were being measured on a strictly weaker view
   than the bank they were joining.  `item_block` now reproduces the bank's framing
   exactly, including its fallback to caption-only for the 2 contests with no
   description.

2. ANCHOR BATTERY, V9 REPAIRS (registry: the V9 tweets cell found two defects in this
   battery and this cell -- short, high-NA items -- is exactly the regime they live in).
     (a) `scramble` on two ~8-word captions leaves a 16-word blob in which proper nouns
         and short function words survive intact, so a "scrambled" anchor can still read
         as a caption.  The scrambled anchors are therefore DUMPED to the score report
         (`anchor_scram_texts`) for manual inspection before any certification.
     (b) Coherence is scored TWICE: on the item's mean judged score (the historic
         readout) and on its NON-NA COUNT (the V9 recommendation).  On a short,
         high-NA corpus the mean is computed over whichever criteria happened to answer,
         which selects for scrambles the judge could still score; the non-NA count
         cannot be gamed that way.  Both are reported; PASS requires the non-NA-count
         reading, with the mean reading printed beside it.
     (c) An all-NA anchor row is NOT silently dropped -- it is counted, and a scrambled
         anchor that is all-NA is the strongest possible evidence of incoherence, so it
         is scored as the minimum rather than discarded.
   The scrambled anchor is shown WITH the pos caption's cartoon, so it is scrambled text
   against a real cartoon -- the same view every population row gets.

3. Anchors are drawn K >= 50 per class from the population, in the SAME batch.

Run on sk3, one GPU, via gpu_lane_runner.sh:
  ./gpu_lane_runner.sh cap_finalist_r3 <log> 5 100000 $HOME/envs/gemma4/bin/python \
      score_gemma_maps.py --jobs cap_finalist_r3
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

# persona matched to the A bank's own system prompt
# (datasets/humor/caption_multiy/score_va_gemma_captions.py::SYS)
PERSONA = ("an expert comedy editor judging entries in a cartoon caption contest",
           "CAPTION")

NUM = re.compile(r"\d+")
K_ANCHOR = 50
SEED = 20260809
TRUNC = 4000


def item_block(desc, text):
    """The A bank's exact framing."""
    t = (text or "").strip()[:TRUNC]
    d = (desc or "").strip()
    return f'CARTOON: {d}\n\nCAPTION: "{t}"' if d else f'CAPTION: "{t}"'


def sys_prompt():
    persona, noun = PERSONA
    return (f"You are {persona}. You are given a description of the cartoon, ONE "
            f"submitted {noun.lower()}, and ONE criterion. Decide how strongly the "
            f"{noun.lower()}, on its own evidence, exhibits that criterion. "
            "Answer with EXACTLY ONE token:\n"
            "  an integer from 0 to 10, where 0 = not at all and 10 = to the fullest degree\n"
            f"  NA = the {noun.lower()} gives no evidence bearing on this criterion\n"
            "Judge the criterion as literally described, not whether the caption is good "
            "overall and not whether it won. Output only the token.")


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
    sel = json.loads((HERE / f"{tag}_species.json").read_text())["selected"]
    crits = [{"id": c["blind_id"], "name": c["name"], "instruction": c["instruction"]}
             for c in sel]
    with open(HERE / "cap_finalist_population.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return dict(tag=tag, crits=crits, rows=rows)


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
            print(f"[cap] {t}: no species file yet, SKIPPING", flush=True)
            continue
        if (HERE / f"{t}_scores.npz").exists():
            print(f"[cap] {t}: already scored, skipping", flush=True)
            continue
        jobs.append(build_job(t))
    if not jobs:
        print("ALL_SCORE_DONE (nothing to do)", flush=True)
        return
    total = sum(len(j["rows"]) * len(j["crits"]) for j in jobs)
    print(f"[cap] {len(jobs)} jobs, {total} population prompts", flush=True)

    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              trust_remote_code=True, max_num_seqs=256)
    sp = SamplingParams(temperature=0.0, max_tokens=6)
    SYS = sys_prompt()

    for j in jobs:
        tag, crits, rows = j["tag"], j["crits"], j["rows"]
        blocks = [f"CRITERION: {c['name']}\nINSTRUCTION: {c['instruction']}\n\n"
                  "Answer with one token:" for c in crits]
        cids = [c["id"] for c in crits]
        views = [(r["desc"], r["text"]) for r in rows]

        rng = random.Random(SEED)
        pos = [(r["desc"], r["text"]) for r in rows if str(r["judgement"]) == "1"]
        neg = [(r["desc"], r["text"]) for r in rows if str(r["judgement"]) == "0"]
        a_views, a_tags, scram_dump = [], [], []
        for _ in range(K_ANCHOR):
            p, n = rng.choice(pos), rng.choice(neg)
            s = scramble([p[1], n[1]], rng)
            a_views += [p, n, (p[0], s)]
            a_tags += ["anchor_pos", "anchor_neg", "anchor_scram"]
            scram_dump.append(s)

        all_views = views + a_views
        convs = []
        for desc, text in all_views:
            f = item_block(desc, text)
            for b in blocks:
                convs.append([{"role": "user", "content": f"{SYS}\n\n{f}\n\n{b}"}])
        print(f"[{tag}] {len(rows)} rows + {len(a_views)} anchors x {len(crits)} crit "
              f"= {len(convs)} prompts", flush=True)
        outs = llm.chat(convs, sp)
        raw = [o.outputs[0].text for o in outs]
        X = np.array([parse_tok(t) for t in raw], dtype=float).reshape(
            len(all_views), len(crits))
        Xpop, Xanc = X[:len(views)], X[len(views):]
        np.savez_compressed(
            HERE / f"{tag}_scores.npz", X=Xpop,
            crit_ids=np.array(cids, dtype=object),
            crit_names=np.array([c["name"] for c in crits], dtype=object),
            i=np.array([int(r["i"]) for r in rows]),
            row_id=np.array([r["id"] for r in rows], dtype=object),
            Xanchor=Xanc, anchor_tags=np.array(a_tags, dtype=object), scale="0-10",
            item_view="CARTOON+CAPTION (matched to the A bank)")

        rep = {"tag": tag, "n_rows": len(views), "n_criteria": len(crits),
               "item_view": "CARTOON: <desc>\\n\\nCAPTION: \"<text>\" -- matched to the A bank",
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
        n_crit = Xanc.shape[1]
        # V9 REPAIR (b): two coherence readouts.  `item_mean` is the historic one and is
        # computed only over criteria that answered; `nonna_count` cannot be selected by
        # a scramble the judge refused to score.
        item_mean = np.where(np.isnan(Xanc).all(axis=1), np.nan, np.nanmean(Xanc, axis=1))
        nonna = (~np.isnan(Xanc)).sum(axis=1).astype(float)
        # V9 REPAIR (c): an all-NA scramble is maximal incoherence, not a missing row.
        item_mean_filled = np.where(np.isnan(item_mean), 0.0, item_mean)

        anc = {"k_per_class": K_ANCHOR, "n_criteria": int(n_crit)}
        for t in ("anchor_pos", "anchor_neg", "anchor_scram"):
            m = tags == t
            anc[t] = {
                "mean_score": float(np.nanmean(item_mean[m])) if np.isfinite(item_mean[m]).any() else None,
                "sd_score": float(np.nanstd(item_mean[m], ddof=1)) if np.isfinite(item_mean[m]).sum() > 1 else None,
                "mean_nonNA_count": float(nonna[m].mean()),
                "n_all_NA": int(np.isnan(Xanc[m]).all(axis=1).sum()),
            }
        pm, nm, sm = tags == "anchor_pos", tags == "anchor_neg", tags == "anchor_scram"
        coh_y = [1] * int(pm.sum() + nm.sum()) + [0] * int(sm.sum())
        anc["pos_vs_neg_auc"] = float(roc_auc_score(
            [1] * int(pm.sum()) + [0] * int(nm.sum()),
            np.concatenate([item_mean_filled[pm], item_mean_filled[nm]])))
        anc["coherent_vs_scrambled_auc_itemmean"] = float(roc_auc_score(
            coh_y, np.concatenate([item_mean_filled[pm], item_mean_filled[nm],
                                   item_mean_filled[sm]])))
        anc["coherent_vs_scrambled_auc_nonNAcount"] = float(roc_auc_score(
            coh_y, np.concatenate([nonna[pm], nonna[nm], nonna[sm]])))
        anc["PASS_RULE"] = ("V9: PASS requires coherent_vs_scrambled_auc_nonNAcount >= .70; "
                            "the item-mean reading is reported beside it, never in its place")
        anc["pass_scrambled"] = bool(anc["coherent_vs_scrambled_auc_nonNAcount"] >= 0.70)
        anc["pass_scrambled_itemmean_reading"] = bool(
            anc["coherent_vs_scrambled_auc_itemmean"] >= 0.70)
        # V9 REPAIR (a): dump every scrambled anchor for manual inspection.
        anc["anchor_scram_texts"] = scram_dump
        rep["anchors"] = anc

        rep["n_collapsed"] = int(sum(v["collapsed"] for v in rep["per_criterion"].values()))
        rep["overall_na_rate"] = float(np.isnan(Xpop).mean())
        rep["n_rows_all_NA"] = int(np.isnan(Xpop).all(axis=1).sum())
        rep["rows_all_NA_rate"] = float(np.isnan(Xpop).all(axis=1).mean())
        rep["anchor_rows_all_NA"] = int(np.isnan(Xanc).all(axis=1).sum())
        rep["anchor_rows_all_NA_by_class"] = {
            t: int(np.isnan(Xanc[tags == t]).all(axis=1).sum())
            for t in ("anchor_pos", "anchor_neg", "anchor_scram")}
        # An all-NA SCRAMBLE row is expected and healthy; an all-NA pos/neg row is the
        # interrupted-generation signature.  The gate reads only the latter.
        rep["INTERRUPTED_GENERATION_SUSPECTED"] = bool(
            rep["rows_all_NA_rate"] > 0.02
            or rep["anchor_rows_all_NA_by_class"]["anchor_pos"] > 0
            or rep["anchor_rows_all_NA_by_class"]["anchor_neg"] > 0)
        if rep["INTERRUPTED_GENERATION_SUSPECTED"]:
            print(f"[{tag}] !! INTERRUPTED GENERATION SUSPECTED", flush=True)
        (HERE / f"{tag}_score_report.json").write_text(json.dumps(rep, indent=2))
        print(f"[{tag}] ANCHORS " + json.dumps(
            {k: v for k, v in anc.items() if k != "anchor_scram_texts"}), flush=True)
        print(f"[{tag}] COLLAPSED {rep['n_collapsed']} NA {rep['overall_na_rate']:.3f}",
              flush=True)
        print(f"SCORE_DONE {tag}", flush=True)
    print("ALL_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
