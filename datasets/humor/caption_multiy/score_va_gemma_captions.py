#!/usr/bin/env python3
"""Caption-contest multi-y VAT scorer (Gemma-4-31B). Ports vat_3y/score_va_gemma_3y.py
(same token protocol / parse / npz layout) to New Yorker caption contest captions.

LABEL-INDEPENDENT: every caption in built/caption_contest_v2.jsonl is scored ONCE
against the 364-rubric standup A-bank (data/humor/standup_reddit/rubrics.jsonl,
baseline-A choice 2026-07-27); finalist-B and crowd-C y's are attached later by the
laptop aggregation. Cartoon description (canny+uncanny) is included as context —
constant within a contest, so it cannot separate captions within the grouping unit.

Leak hygiene (see build_caption_contest_v2.py audit): the judged text and ALL
V-features use the aggressively normalized v2 `text`, NEVER `raw_text` (typography
is a curation artifact). crowd_mean / crowd_votes stay metadata — never features.

Anchors: 3 blinded rows per shard (contest="__ANCHOR"): a #1-placed finalist,
a random low-crowd entry, and a scrambled-nonsense caption. Expect mean-A ordering
pos > mid > scram; aggregation drops them.
"""
import argparse, hashlib, json, os, random, re
from pathlib import Path
import numpy as np

BASE = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/caption_contest")
RUBRICS = Path("/lfs/skampere3/0/alexspan/data/humor/standup_reddit/rubrics.jsonl")
DESC_CSV = BASE / "built/newyorker_cartoon_descriptions.csv.gz"
GEMMA4 = "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"

SYS = ("You are an expert comedy editor judging entries in a cartoon caption contest. "
       "You are given a description of the cartoon, ONE submitted caption, and ONE "
       "quality criterion. Decide how strongly the caption, on its own evidence, "
       "satisfies that criterion. Answer with EXACTLY ONE token:\n"
       "  1.0 = clearly satisfies the criterion\n  0.5 = partially / weakly / borderline\n"
       "  0.0 = fails / cuts against the criterion\n"
       "  NA = the caption gives no evidence bearing on this criterion\n"
       "Judge the caption's comedic quality, not whether it won or how popular it was. "
       "Output only the token.")


def load_rubrics():
    ms = []
    for line in open(RUBRICS):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("name"):
                ms.append(r)
    return ms


def metric_block(m):
    return f"CRITERION: {m['name']}\nDESCRIPTION: {m.get('description','')}\n\nAnswer with one token:"


def parse_tok(t):
    t = (t or "").strip().lower()
    if t.startswith("na") or "n/a" in t or t == "na":
        return np.nan
    if "0.5" in t or t.startswith("0.5"):
        return 0.5
    if re.search(r"\b1(\.0)?\b", t) or t.startswith("1"):
        return 1.0
    if re.search(r"\b0(\.0)?\b", t) or t.startswith("0"):
        return 0.0
    return np.nan


# --------------------------- V-features (v2-normalized text ONLY) ---------------
FIRST = re.compile(r"\b(i|we|my|our|me|us|i'm|i've|i'd|i'll)\b")
SECOND = re.compile(r"\b(you|your|you're|you've|you'll)\b")
THIRD = re.compile(r"\b(he|she|they|his|her|their|him|them|it's|it)\b")
NEG = re.compile(r"\b(not|never|no|nothing|nobody|can't|don't|won't|isn't|aren't|didn't|doesn't)\b")
SPEECH = re.compile(r"\b(says?|said|asks?|asked|tells?|told|call(?:ed|s)?)\b")
DIGIT = re.compile(r"\d")


def v_features(text):
    t = text or ""
    words = t.split()
    nw = max(len(words), 1)
    feats = {
        "v_char_len": float(len(t)),
        "v_word_len": float(len(words)),
        "v_avg_word_len": float(sum(len(w) for w in words) / nw),
        "v_comma": float(t.count(",")),
        "v_question": float(t.count("?")),
        "v_exclaim": float(t.count("!")),
        "v_apostrophe": float(t.count("'")),
        "v_dash": float(t.count("-")),
        "v_digit": float(len(DIGIT.findall(t))),
        "v_first_person": float(len(FIRST.findall(t))),
        "v_second_person": float(len(SECOND.findall(t))),
        "v_third_person": float(len(THIRD.findall(t))),
        "v_definite_the": float(len(re.findall(r"\bthe\b", t))),
        "v_negation": float(len(NEG.findall(t))),
        "v_speech_verb": float(len(SPEECH.findall(t))),
        "v_ttr": float(len(set(words)) / nw),
    }
    return feats


V_NAMES = ["v_char_len", "v_word_len", "v_avg_word_len", "v_comma", "v_question",
           "v_exclaim", "v_apostrophe", "v_dash", "v_digit", "v_first_person",
           "v_second_person", "v_third_person", "v_definite_the", "v_negation",
           "v_speech_verb", "v_ttr"]


def doc_id(r):
    return f"{r['contest']}_{hashlib.sha1(r['text'].encode()).hexdigest()[:12]}"


def load_descriptions():
    import csv, gzip, io
    descs = {}
    with gzip.open(DESC_CSV, "rt") as fh:
        for row in csv.DictReader(fh):
            c = int(row["contest_number"])
            canny = (row.get("canny") or "").strip()
            uncanny = (row.get("uncanny") or "").strip()
            descs[c] = f"{canny} {uncanny}".strip()
    return descs


def build_anchor_rows(rows, rng):
    """3 blinded anchors: best finalist / low-crowd random neg / scrambled nonsense."""
    fin = [r for r in rows if r["role"] == "finalist" and r.get("placement") == 1
           and r.get("crowd_mean") is not None]
    fin.sort(key=lambda r: -r["crowd_mean"])
    pos = dict(fin[0])
    lows = [r for r in rows if r["role"] == "neg_random" and (r.get("crowd_votes") or 0) >= 100
            and r.get("crowd_mean") is not None]
    lows.sort(key=lambda r: r["crowd_mean"])
    mid = dict(lows[0])
    words = pos["text"].split() + mid["text"].split()
    rng.shuffle(words)
    scram = dict(mid)
    scram["text"] = " ".join(words[:9])
    out = []
    for tag, r in [("anchor_pos", pos), ("anchor_mid", mid), ("anchor_scram", scram)]:
        rr = dict(r)
        rr["anchor_tag"] = tag
        rr["orig_contest"] = r["contest"]
        out.append(rr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--shards", type=int, default=5)
    ap.add_argument("--outdir", default=str(BASE / "vat_multiy"))
    a = ap.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(exist_ok=True)

    rows = [json.loads(l) for l in open(BASE / "built/caption_contest_v2.jsonl") if l.strip()]
    descs = load_descriptions()
    covered = sum(1 for r in rows if r["contest"] in descs)
    print(f"[cap] {len(rows)} captions, {len({r['contest'] for r in rows})} contests; "
          f"description coverage {covered}/{len(rows)}", flush=True)

    metrics = load_rubrics()
    blocks = [metric_block(m) for m in metrics]
    a_names = [m["name"] for m in metrics]
    rng = random.Random(0)
    anchors = build_anchor_rows(rows, rng)

    # stable-hash shard assignment by doc_id (never seeded-shuffle)
    shard_rows = [[] for _ in range(a.shards)]
    for r in rows:
        h = int(hashlib.sha1(doc_id(r).encode()).hexdigest(), 16) % a.shards
        shard_rows[h].append(r)

    from vllm import LLM, SamplingParams
    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    for si, srows in enumerate(shard_rows):
        outp = outdir / f"cap_scores_shard{si}.npz"
        if outp.exists():
            print(f"[cap] shard {si} exists, skip", flush=True)
            continue
        batch = srows + anchors  # anchors re-scored EVERY shard (blinded pass rule)
        convs = []
        for r in batch:
            d = descs.get(r["contest"] if r["contest"] != "__ANCHOR" else r.get("orig_contest"),
                          descs.get(r.get("orig_contest", r["contest"]), ""))
            ctx = f"CARTOON: {d}\n\nCAPTION: \"{r['text']}\"" if d else f"CAPTION: \"{r['text']}\""
            for b in blocks:
                convs.append([{"role": "user", "content": f"{SYS}\n\n{ctx}\n\n{b}"}])
        print(f"[cap] shard {si}: {len(batch)} captions x {len(metrics)} = {len(convs)} prompts",
              flush=True)
        outs = llm.chat(convs, sp)
        vals = [parse_tok(o.outputs[0].text) for o in outs]
        X = np.array(vals, dtype=float).reshape(len(batch), len(metrics))
        Vf = np.array([[v_features(r["text"])[n] for n in V_NAMES] for r in batch], dtype=float)
        dids = np.array([("__ANCHOR_" + r["anchor_tag"]) if "anchor_tag" in r else doc_id(r)
                         for r in batch], dtype=object)
        contests = np.array([str(r.get("orig_contest", r["contest"])) for r in batch], dtype=object)
        roles = np.array([r.get("anchor_tag", r["role"]) for r in batch], dtype=object)
        na = float(np.isnan(X[:len(srows)]).mean())
        anc = X[len(srows):]
        anc_means = np.nanmean(anc, axis=1)
        print(f"[cap] shard {si} NA {na:.3f}; anchors pos/mid/scram = "
              + "/".join(f"{v:.3f}" for v in anc_means), flush=True)
        np.savez_compressed(outp, X=X, V=Vf, doc_id=dids, contest=contests, role=roles,
                            a_names=np.array(a_names, dtype=object),
                            v_names=np.array(V_NAMES, dtype=object), na_rate=na)
        print(f"[cap] saved -> {outp}", flush=True)

    print("SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
