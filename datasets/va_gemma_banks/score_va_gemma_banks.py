#!/usr/bin/env python3
"""Score three already-authored articulated-criteria (A) banks with a local
Gemma-4-31B judge, offline-batch vLLM, one token per (item, criterion).

Banks (already written elsewhere; this file ONLY scores them):
  hashtagwars  datasets/humor/hashtagwars/va/rubrics.jsonl          (30 criteria)
  style_inv    datasets/humor/style_invitational/va/rubrics.jsonl   (32 criteria)
  creative     datasets/creative-writing/va_bank_v2/rubrics_initial.jsonl (45)

Protocol ported verbatim from datasets/humor/caption_multiy/score_va_gemma_captions.py:
temperature 0, max_tokens 6, prefix caching, thousands of prompts per generate,
3 blinded anchor rows in EVERY shard (known positive / random negative /
scrambled nonsense), label-blind judging (y is never in a prompt).

GPU: one GPU only (CUDA_VISIBLE_DEVICES set by the caller), gpu_memory_utilization
kept low because a training job is stacked on the same device.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# --- multiprocessing spawn BEFORE vllm import (sk3 fork-wedge rule) -----------
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import multiprocessing as _mp  # noqa: E402

try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
GEMMA4 = os.environ.get(
    "GEMMA4_PATH",
    "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/"
    "3548789868c5356dbf307c98e6f609007b82b3eb",
)
OUT = Path(os.environ.get("VA_OUT", str(REPO / "outputs/va_gemma_banks")))
TOKENS = {"1.0": 1.0, "0.5": 0.5, "0.0": 0.0, "NA": np.nan}
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")

HW_DIR = REPO / "datasets/humor/hashtagwars"
SI_DIR = REPO / "datasets/humor/style_invitational"
CW_DIR = REPO / "datasets/creative-writing"

SEED = 20260728

SYS_HW = (
    "You are an expert comedy editor judging entries in a Twitter hashtag joke "
    "contest (@midnight #HashtagWars). You are given the contest hashtag prompt, "
    "ONE submitted tweet, and ONE quality criterion. Decide how strongly the tweet, "
    "on its own evidence, satisfies that criterion. Answer with EXACTLY ONE token:\n"
    "  1.0 = clearly satisfies the criterion\n  0.5 = partially / weakly / borderline\n"
    "  0.0 = the relevant attempt is present but fails or cuts against the criterion\n"
    "  NA = the tweet gives no evidence bearing on this criterion\n"
    "Judge the tweet's comedic quality on the text alone, not whether it won, who "
    "wrote it, or how popular it was. Output only the token."
)

SYS_SI = (
    "You are an expert humor editor judging entries in the Washington Post Style "
    "Invitational, a weekly humor contest. You are given the contest prompt, ONE "
    "submitted entry, and ONE quality criterion. Decide how strongly the entry, on "
    "its own evidence, satisfies that criterion. Answer with EXACTLY ONE token:\n"
    "  1.0 = clearly satisfies the criterion\n  0.5 = partially / weakly / borderline\n"
    "  0.0 = the relevant attempt is present but fails or cuts against the criterion\n"
    "  NA = the mechanism genuinely does not apply, or the evidence is unavailable\n"
    "Parenthetical author names and hometowns are archive bylines; ignore them when "
    "assessing humor. Do not infer editorial tier, placement, or popularity, and do "
    "not compare entries. Output only the token."
)

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

# CW deterministic truncation (matches va_bank_v2/score_va.py constants)
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


def sha(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def load_module(path: Path, alias: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def scramble(texts, rng, n_words=None):
    toks = []
    for t in texts:
        toks += WORD_RE.findall(t)
    rng.shuffle(toks)
    k = n_words or max(8, min(14, len(toks)))
    chosen = toks[:k]
    chosen[1::2] = [w[::-1] for w in chosen[1::2]]  # reversed alternates: word salad
    return " ".join(chosen)


# ============================== bank builders ================================
def build_hashtagwars():
    vf = load_module(HW_DIR / "va/v_features.py", "vf_hw")

    SALT = "hashtagwars-va-v1:"
    SPLIT_DIRS = ("train_data", "trial_data", "gold_labels")
    rows, labels = [], {}
    for sd in SPLIT_DIRS:
        for path in sorted((HW_DIR / sd).glob("*.tsv")):
            hashtag = path.stem
            with path.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    tid, text = parts[0], parts[1]
                    rid = hashlib.sha1(f"{hashtag}\0{tid}\0{text}".encode()).hexdigest()
                    rows.append({"id": rid, "group": hashtag, "text": text})
                    if len(parts) >= 3 and parts[2] in {"0", "1", "2"}:
                        labels[rid] = int(parts[2])
    groups = sorted({r["group"] for r in rows}, key=lambda h: sha(SALT + h))[:40]
    keep = set(groups)
    sample = [r for r in rows if r["group"] in keep]
    # dedup by row_id (same tweet can appear in >1 split dir)
    seen, dedup = set(), []
    for r in sample:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        dedup.append(r)
    sample = dedup

    rubrics = [json.loads(l) for l in open(HW_DIR / "va/rubrics.jsonl") if l.strip()]
    blocks = [
        f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n\nAnswer with one token:"
        for m in rubrics
    ]

    def ctx(r):
        return f"CONTEST HASHTAG: #{r['group']}\n\nTWEET: \"{r['text']}\""

    def vvec(r):
        return vf.vector(r["text"], r["group"])

    def anchors(shard):
        rng = random.Random(SEED + 101 * shard)
        wins = [r for r in rows if labels.get(r["id"]) == 2]
        negs = [r for r in rows if labels.get(r["id"]) == 0]
        pos, neg = dict(rng.choice(wins)), dict(rng.choice(negs))
        scr = dict(neg)
        scr["text"] = scramble([pos["text"], neg["text"]], rng)
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg), ("anchor_scram", scr)):
            rr = dict(r)
            rr["anchor_tag"] = tag
            rr["id"] = f"__ANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    ys = {
        "top10_or_winner": np.array([1 if labels.get(r["id"], 0) in (1, 2) else 0 for r in sample])
    }
    return dict(
        name="hashtagwars", items=sample, rubrics=rubrics, blocks=blocks, sys=SYS_HW,
        ctx=ctx, vvec=vvec, vnames=list(vf.V_NAMES), anchors=anchors, ys=ys,
        n_shards=8, meta={"selected_groups": groups, "n_total_rows": len(rows)},
    )


def build_style_invitational():
    vf = load_module(SI_DIR / "va/v_features.py", "vf_si")

    SALT = "style-va-v1"
    rows = [json.loads(l) for l in open(SI_DIR / "style_invitational.jsonl") if l.strip()]
    by_week = defaultdict(list)
    for i, r in enumerate(rows):
        r["group"] = str(r["week_id"])
        r["id"] = hashlib.sha1(
            f"{r['week_id']}\0{i}\0{r['entry_text']}".encode()).hexdigest()[:20]
        by_week[r["group"]].append(r)
    week_order = sorted(by_week, key=lambda w: sha(f"{SALT}|{w}"))
    precommit_110 = set(week_order[:110])
    sample = [r for w in week_order for r in by_week[w]]  # ALL weeks

    rubrics = [json.loads(l) for l in open(SI_DIR / "va/rubrics.jsonl") if l.strip()]
    blocks = [
        f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n\nAnswer with one token:"
        for m in rubrics
    ]

    def ctx(r):
        return f"CONTEST PROMPT: {r['contest_prompt']}\n\nENTRY: \"{r['entry_text']}\""

    def vvec(r):
        return vf.vector(r["entry_text"], r["contest_prompt"])

    def anchors(shard):
        rng = random.Random(SEED + 211 * shard)
        wins = [r for r in rows if r["tier"] == "winner"]
        hms = [r for r in rows if r["tier"] == "honorable_mention"]
        pos, neg = dict(rng.choice(wins)), dict(rng.choice(hms))
        scr = dict(neg)
        scr["entry_text"] = scramble([pos["entry_text"], neg["entry_text"]], rng)
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg), ("anchor_scram", scr)):
            rr = dict(r)
            rr["anchor_tag"] = tag
            rr["id"] = f"__ANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    tiers = np.array([r["tier"] for r in sample], dtype=object)
    ys = {
        "top_tier": (np.isin(tiers, ["winner", "runnerup"])).astype(int),
        "winner_vs_rest": (tiers == "winner").astype(int),
    }
    return dict(
        name="style_invitational", items=sample, rubrics=rubrics, blocks=blocks,
        sys=SYS_SI, ctx=ctx, vvec=vvec, vnames=list(vf.V_NAMES), anchors=anchors, ys=ys,
        n_shards=8,
        meta={"n_weeks": len(by_week), "precommit_110_weeks": sorted(precommit_110)},
        extra_cols={"tier": tiers, "in_precommit_110":
                    np.array([r["group"] in precommit_110 for r in sample], dtype=bool)},
    )


def _cw_split(text):
    parts = text.split("\n\nSTORY: ", 1)
    if len(parts) == 2:
        return parts[0].removeprefix("PROMPT: ").strip(), parts[1]
    return "", text


def _cw_trunc(story):
    story = story.strip()
    if len(story) <= TRUNCATE_SOURCE_CHARS:
        return story
    return story[:TRUNCATE_HEAD_CHARS] + TRUNCATION_MARKER + story[-TRUNCATE_TAIL_CHARS:]


def build_creative():
    vf = load_module(CW_DIR / "va_bank_v2/v_features.py", "vf_cw")
    import pandas as pd

    SALT = "cw-va-v2-sample"
    canonical = CW_DIR / "writingprompts_modeling_clean.csv.gz"
    df = pd.read_csv(canonical)
    df["prompt_id"] = df["prompt_id"].astype(str)
    order = sorted(df["prompt_id"].unique(), key=lambda p: sha(f"{SALT}|{p}"))
    idx, n = [], 0
    for pid in order:
        rowsel = df.index[df["prompt_id"] == pid].tolist()
        idx += rowsel
        n += len(rowsel)
        if n >= 2000:
            break
    sub = df.loc[idx].reset_index(drop=True)
    items = []
    for i, r in sub.iterrows():
        p, s = _cw_split(str(r["text"]))
        items.append({
            "id": f"{r['prompt_id']}_{hashlib.sha1(str(r['text']).encode()).hexdigest()[:10]}",
            "group": str(r["prompt_id"]), "prompt": p, "story": s,
            "judgement": int(r["judgement"]),
        })

    rubrics = [json.loads(l) for l in open(CW_DIR / "va_bank_v2/rubrics_initial.jsonl")
               if l.strip()]
    blocks = [
        f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n"
        f"GUIDANCE: {m.get('guidance','')}\n\nAnswer with one token:"
        for m in rubrics
    ]

    def ctx(r):
        return (f"WRITING PROMPT: {r['prompt'][:CW_PROMPT_CHARS]}\n\n"
                f"STORY:\n{_cw_trunc(r['story'])}")

    def vvec(r):
        return vf.feature_vector(r["story"])

    def anchors(shard):
        rng = random.Random(SEED + 307 * shard)
        pos_pool = [r for r in items if r["judgement"] == 1]
        neg_pool = [r for r in items if r["judgement"] == 0]
        pos, neg = dict(rng.choice(pos_pool)), dict(rng.choice(neg_pool))
        scr = dict(neg)
        scr["story"] = scramble([pos["story"][:4000], neg["story"][:4000]], rng, n_words=220)
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg), ("anchor_scram", scr)):
            rr = dict(r)
            rr["anchor_tag"] = tag
            rr["id"] = f"__ANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    ys = {"score_ge10": np.array([r["judgement"] for r in items])}
    return dict(
        name="creative_writing", items=items, rubrics=rubrics, blocks=blocks, sys=SYS_CW,
        ctx=ctx, vvec=vvec, vnames=list(vf.V_NAMES), anchors=anchors, ys=ys, n_shards=4,
        meta={"source": str(canonical), "n_prompt_groups": int(sub['prompt_id'].nunique()),
              "truncation": {"source_chars": TRUNCATE_SOURCE_CHARS,
                             "head": TRUNCATE_HEAD_CHARS, "tail": TRUNCATE_TAIL_CHARS,
                             "prompt_chars": CW_PROMPT_CHARS}},
    )


BUILDERS = {"hashtagwars": build_hashtagwars,
            "style_invitational": build_style_invitational,
            "creative_writing": build_creative}


# ================================ scoring ====================================
def score_bank(llm, sp, bank, outdir, max_anchor_attempts=4):
    name = bank["name"]
    items, blocks = bank["items"], bank["blocks"]
    k = len(blocks)
    ns = bank["n_shards"]
    shards = [[] for _ in range(ns)]
    for r in items:
        shards[int(hashlib.sha1(r["id"].encode()).hexdigest(), 16) % ns].append(r)

    reports = []
    for si, srows in enumerate(shards):
        outp = outdir / f"{name}_shard{si}.npz"
        if outp.exists():
            print(f"[{name}] shard {si} exists, skip", flush=True)
            rep = json.loads(str(np.load(outp, allow_pickle=True)["anchor_json"].item()))
            reports.append(rep)
            continue
        # --- main rows -------------------------------------------------------
        convs = []
        for r in srows:
            c = bank["ctx"](r)
            for b in blocks:
                convs.append([{"role": "user", "content": f"{bank['sys']}\n\n{c}\n\n{b}"}])
        print(f"[{name}] shard {si}: {len(srows)} items x {k} = {len(convs)} prompts",
              flush=True)
        outs = llm.chat(convs, sp)
        X = np.array([parse_tok(o.outputs[0].text) for o in outs],
                     dtype=float).reshape(len(srows), k)

        # --- anchors: 3 blinded rows, rescored until ordering holds -----------
        history, arows, aX = [], None, None
        for attempt in range(max_anchor_attempts):
            arows = bank["anchors"](si * 1000 + attempt)
            aconvs = []
            for r in arows:
                c = bank["ctx"](r)
                for b in blocks:
                    aconvs.append([{"role": "user", "content": f"{bank['sys']}\n\n{c}\n\n{b}"}])
            aouts = llm.chat(aconvs, sp)
            aX = np.array([parse_tok(o.outputs[0].text) for o in aouts],
                          dtype=float).reshape(3, k)
            m = np.nanmean(aX, axis=1)
            ok = bool(m[0] > m[1] > m[2])
            history.append({"attempt": attempt, "pos": float(m[0]), "neg": float(m[1]),
                            "scram": float(m[2]), "valid": ok})
            print(f"[{name}] shard {si} anchor attempt {attempt}: "
                  f"pos {m[0]:.3f} / neg {m[1]:.3f} / scram {m[2]:.3f} valid={ok}", flush=True)
            if ok:
                break
        rep = {"shard": si, "n_items": len(srows), "valid": history[-1]["valid"],
               "attempts": len(history), "attempt_history": history,
               "pos": history[-1]["pos"], "neg": history[-1]["neg"],
               "scram": history[-1]["scram"]}
        reports.append(rep)

        V = np.array([bank["vvec"](r) for r in srows], dtype=float)
        na = float(np.isnan(X).mean())
        print(f"[{name}] shard {si} NA rate {na:.3f}", flush=True)
        np.savez_compressed(
            outp, X=X, V=V, anchor_X=aX,
            ids=np.array([r["id"] for r in srows], dtype=object),
            groups=np.array([str(r["group"]) for r in srows], dtype=object),
            a_names=np.array([m.get("name", "") for m in bank["rubrics"]], dtype=object),
            v_names=np.array(bank["vnames"], dtype=object),
            anchor_json=np.array(json.dumps(rep), dtype=object), na_rate=na,
        )
        print(f"[{name}] saved -> {outp}", flush=True)
    (outdir / f"{name}_meta.json").write_text(json.dumps(
        {"meta": bank["meta"], "anchor_reports": reports,
         "n_items": len(items), "n_criteria": k,
         "a_names": [m.get("name", "") for m in bank["rubrics"]],
         "v_names": bank["vnames"],
         "ys": {kk: vv.tolist() for kk, vv in bank["ys"].items()},
         "item_ids": [r["id"] for r in items],
         "item_groups": [str(r["group"]) for r in items]}, indent=1))
    print(f"[{name}] DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="hashtagwars,style_invitational,creative_writing")
    ap.add_argument("--util", type=float, default=0.55)
    ap.add_argument("--max-model-len", type=int, default=6144)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in a.tasks.split(",") if t]
    banks = []
    for t in tasks:
        b = BUILDERS[t]()
        print(f"[build] {t}: {len(b['items'])} items, {len(b['blocks'])} criteria, "
              f"{len(set(str(r['group']) for r in b['items']))} groups", flush=True)
        banks.append(b)

    from vllm import LLM, SamplingParams
    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              trust_remote_code=True, max_num_seqs=256)
    sp = SamplingParams(temperature=0.0, max_tokens=6)
    for b in banks:
        score_bank(llm, sp, b, OUT)
    print("SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
