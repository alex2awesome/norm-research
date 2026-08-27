#!/usr/bin/env python3
"""Style Invitational v2 A bank: score with the local Gemma-4-31B judge.

REUSE: the scoring loop, shard checkpointing, per-shard 3-row blinded anchors,
NA parsing, prefix caching, temperature 0 / max_tokens 6, and the extended K>=50
anchor battery are imported verbatim from
`datasets/va_gemma_banks/score_va_gemma_banks.py` and `score_scaleupC_banks.py`.
Only the bank builder is new.

POPULATION: the PARSE-ARTIFACT-FREE population only
(`va_v2/population.csv.gz`, is_fragment == False, n = 8,063). The v1 bank was
scored over 9,637 rows of which 1,574 carried no joke text; see
build_si_clean_population.py.

ITEM VIEW: the judge context IS the population's `text` column verbatim --
'CONTEST PROMPT: {prompt}\\n\\nENTRY: "{entry}"' -- which is byte-identical to the
dense arm's training text. Item-view consistency therefore holds exactly, with
no sensitivity arm required.

TRUNCATION IS IN TOKENS (ruling), applied with the judge's own tokenizer rather
than by character count. It is a guard only: measured over the clean population
the longest item is ~596 tokens against a 1024-token cap, so the truncation
fires on ZERO rows. The count that actually fired is recorded in the meta.

ANCHORS: 3 blinded rows per shard plus the extended battery at K>=50 per class.
Anchor labels are `winner` vs `honorable_mention` -- the sharpest available
contrast. NOTE this is only PARTLY independent of the cell's y (top_tier =
winner + runnerup), because winners are a subset of the positive class. SI has
no second editorial channel to anchor on, so the battery should be read as
certifying that the judge discriminates quality at all, NOT as an independent
validation of the y.

  CUDA_VISIBLE_DEVICES=N python3 datasets/humor/style_invitational/score_si_v2_bank.py --smoke 24
  CUDA_VISIBLE_DEVICES=N python3 datasets/humor/style_invitational/score_si_v2_bank.py --battery 50
"""
from __future__ import annotations

import argparse
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

_HERE = Path(__file__).resolve()
REPO_GUESS = _HERE.parents[3]
sys.path.insert(0, str(REPO_GUESS / "datasets/va_gemma_banks"))
import score_va_gemma_banks as S  # noqa: E402
import score_scaleupC_banks as C  # noqa: E402

REPO = S.REPO
OUT = Path(os.environ.get("VA_OUT_SI2",
                          str(REPO / "outputs/va_gemma_banks_si_v2")))
SEED = 20260810
SI_DIR = REPO / "datasets/humor/style_invitational"
V2 = SI_DIR / "va_v2"
MAX_ITEM_TOKENS = 1024

SYS_SI2 = (
    "You are an experienced humor-contest judge performing a measurement task. "
    "You are given a weekly humor-contest prompt, ONE entry submitted to it, and "
    "ONE criterion. Decide how strongly the entry, on the evidence of the "
    "supplied text alone, satisfies that criterion. Answer with EXACTLY ONE "
    "token:\n"
    "  1.0 = clearly satisfies the criterion\n"
    "  0.5 = partly, weakly, inconsistently, or borderline\n"
    "  0.0 = the relevant attempt is present but fails or cuts against the criterion\n"
    "  NA = the entry gives no evidence bearing on this criterion\n"
    "EVERY entry you see was published, so do not ask whether it is good enough "
    "to print -- discriminate among entries that are all already competent. Some "
    "criteria are written so that 1.0 marks a FLAW; score what the criterion "
    "literally asks, not whether the entry is good. The trailing attribution in "
    "parentheses, such as \"(Jane Smith, Bethesda)\", is archive metadata and is "
    "NOT part of the entry -- never score it and never treat it as the entry's "
    "ending. Do not consider or imagine other entries to the same contest, and do "
    "not predict rank, prizes, the author, or dataset membership. Output only the "
    "token."
)


def build_si_v2(tok=None):
    import pandas as pd
    vf = S.load_module(SI_DIR / "va/v_features.py", "vf_si2")
    df = pd.read_csv(V2 / "population.csv.gz")
    df = df[~df.is_fragment].copy()

    n_trunc = 0
    items = []
    for r in df.itertuples():
        text = str(r.text)
        if tok is not None:
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) > MAX_ITEM_TOKENS:
                text = tok.decode(ids[:MAX_ITEM_TOKENS])
                n_trunc += 1
        items.append({"id": str(r.row_id), "group": str(r.group), "text": text,
                      "entry_text": str(r.entry_text),
                      "contest_prompt": str(r.contest_prompt),
                      "tier": str(r.tier),
                      "y_top_tier": int(r.y_top_tier),
                      "y_winner": int(r.y_winner)})

    rubrics = [json.loads(l) for l in open(V2 / "rubrics.jsonl") if l.strip()]
    blocks = [f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n\n"
              "Answer with one token:" for m in rubrics]

    def ctx(r):
        return r["text"]

    def vvec(r):
        return vf.vector(r["entry_text"], r["contest_prompt"])

    def anchors(shard):
        rng = random.Random(SEED + 311 * shard)
        wins = [r for r in items if r["tier"] == "winner"]
        hms = [r for r in items if r["tier"] == "honorable_mention"]
        pos, neg = dict(rng.choice(wins)), dict(rng.choice(hms))
        scr = dict(neg)
        scrambled = S.scramble([pos["entry_text"], neg["entry_text"]], rng)
        scr["text"] = (f"CONTEST PROMPT: {neg['contest_prompt']}\n\n"
                       f'ENTRY: "{scrambled}"')
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg),
                       ("anchor_scram", scr)):
            rr = dict(r)
            rr["anchor_tag"] = tag
            rr["id"] = f"__ANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    ys = {"top_tier": np.array([r["y_top_tier"] for r in items]),
          "winner_vs_rest": np.array([r["y_winner"] for r in items])}
    return dict(name="si_v2", items=items, rubrics=rubrics, blocks=blocks,
                sys=SYS_SI2, ctx=ctx, vvec=vvec, vnames=list(vf.V_NAMES),
                anchors=anchors, ys=ys, n_shards=7,
                meta={"population": "datasets/humor/style_invitational/va_v2/"
                                    "population.csv.gz (is_fragment == False)",
                      "group_column": "week_id",
                      "n_groups": int(df["group"].nunique()),
                      "item_view": "population.text verbatim == dense arm text "
                                   "(item-view consistency exact)",
                      "truncation": {"unit": "TOKENS", "max_item_tokens":
                                     MAX_ITEM_TOKENS, "n_items_truncated": n_trunc},
                      "anchor_label_source": "winner vs honorable_mention "
                                             "(PARTLY overlapping the top_tier y; "
                                             "SI has no independent channel)"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--util", type=float, default=0.80)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--battery", type=int, default=50)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S.GEMMA4)
    b = build_si_v2(tok)
    print(f"[build] si_v2: {len(b['items'])} items, {len(b['blocks'])} criteria, "
          f"{len(set(str(r['group']) for r in b['items']))} weeks, "
          f"token-truncated {b['meta']['truncation']['n_items_truncated']}",
          flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=S.GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              trust_remote_code=True, max_num_seqs=512)
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    if a.smoke:
        # Spread the smoke ACROSS weeks. items[:n] would take one contest (the
        # population is week-sorted), which makes every form-conditional
        # criterion look falsely collapsed -- a single prompt has one register,
        # one form and usually no wordplay.
        step = max(1, len(b["items"]) // a.smoke)
        rows = b["items"][::step][:a.smoke]
        print(f"[smoke] {len(set(r['group'] for r in rows))} distinct weeks",
              flush=True)
        convs = []
        for r in rows:
            c = b["ctx"](r)
            for blk in b["blocks"]:
                convs.append([{"role": "user",
                               "content": f"{b['sys']}\n\n{c}\n\n{blk}"}])
        outs = llm.chat(convs, sp)
        X = np.array([S.parse_tok(o.outputs[0].text) for o in outs],
                     dtype=float).reshape(len(rows), len(b["blocks"]))
        print(f"[smoke:si_v2] n={len(rows)} NA={np.isnan(X).mean():.3f} "
              f"mean={np.nanmean(X):.3f}", flush=True)
        for ci, m in enumerate(b["rubrics"]):
            col = X[:, ci]
            vals, cnts = np.unique(col[np.isfinite(col)], return_counts=True)
            modal = float(cnts.max() / max(len(col), 1)) if len(cnts) else 1.0
            flag = "  <== COLLAPSE" if modal > 0.98 else ""
            print(f"  {m['rubric_id']} {m['name'][:44]:46s} "
                  f"mean={np.nanmean(col):.3f} na={np.isnan(col).mean():.2f} "
                  f"modal={modal:.2f}{flag}", flush=True)
        print("SMOKE_DONE", flush=True)
        return

    S.score_bank(llm, sp, b, OUT)
    if a.battery:
        C.run_battery(llm, sp, b, a.battery, OUT)
    print("SI_V2_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
