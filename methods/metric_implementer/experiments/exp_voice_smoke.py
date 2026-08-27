"""EXP-VOICE-NOUS-1 — SMOKE STAGE ONLY (prereg: notes/2026-08-15__exp-voice-nous-prereg.md).

Leg A smoke: 2 authors x 2 arms (examples, definition_padded) x 1 receiver (llama8b),
2 x 2 x 40 = 160 YES/NO judgment prompts (<=500 call budget). Abort only for infrastructure
failure — NEVER for effect direction (preregistered decision rule).

Modes:
  build  (CPU, laptop) : seeded author pick + splits + prompt assembly -> <out>/smoke_prompts.jsonl
  run    (sk3, 1 GPU)  : offline batch vLLM P(YES) readout        -> <out>/smoke_scores.jsonl
  score  (CPU, laptop) : balanced acc per (author x arm)          -> outputs/exp_voice_nous/smoke_v1.json

Prereg-faithful build details (Leg A):
  - eligible slate: authors with >=30 long pieces (corpus pieces are all >=150 words);
    smoke authors = seeded random (seed 0) pick of 2 (full run: 12, stratified — NOT here).
  - splits per author, disjoint, seeded: exemplar pool 12; held-out positives 20; matched
    negatives 20 (other eligible authors; matched on length decile + publication-year band;
    negative AUTHORS disjoint from every exemplar set, i.e. from all exemplar-negative donors).
  - excerpts capped at 300 words (declared support cap).
  - arm `examples`: 6 positive + 6 negative exemplar excerpts, polarity from ground truth.
  - arm `definition_padded`: SMOKE ONLY — a fixed template description marked SMOKE_STUB
    substitutes for the authoring-model description (the authoring-model chain is part of the
    full run, per smoke charge), plus neutral padding to match the examples arm's prompt
    length (word-count proxy for token length at build time; declared smoke simplification).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys

CORPUS = "datasets/humor/mcsweeneys_archive/authorship_corpus_v1.jsonl.gz"
RECEIVER = ("llama8b", "meta-llama/Llama-3.1-8B-Instruct")   # same ckpt as osl_sweep EXECUTORS
SEED = 0
MIN_PIECES = 30
N_EXEMPLAR_POOL = 12
N_POS = 20
N_NEG = 20
N_EX_POS = 6
N_EX_NEG = 6
EXCERPT_WORDS = 300
YEAR_BAND = 2   # |year diff| <= 2 counts as same publication-year band

SMOKE_STUB_DESC = (
    "[SMOKE_STUB — fixed template description; the full run replaces this with an "
    "authoring-model description written from the same 6+6 exemplar set] "
    "This author writes short comic prose in a distinctive personal voice: a consistent "
    "first-person comic persona, a characteristic rhythm of setup and escalation, recurring "
    "registers and reference points, and a signature way of ending a piece. Judge whether the "
    "test text is written in this same personal comic voice."
)

PAD_SENTENCE = (
    "The following padding text is neutral filler included only to equalize prompt length "
    "between conditions and carries no information about the author. "
)

# Constant 33-word site boilerplate prefixed to EVERY corpus text (verified: shared word-prefix
# across the corpus is exactly this string; no shared suffix). Stripped before excerpting —
# label-symmetric, so hygiene only, not a design change.
BOILERPLATE = (
    "Join our Patreon for as little as $5 a month and get access to author interviews, "
    "content calls, discounts at our store, and more. Help support our writers and keep "
    "our site ad-free.")

QUESTION = (
    "Question: Is the TEST TEXT written by the same author (in the same personal comic voice) "
    "as described above? Answer with exactly one word: YES or NO."
)


def _rng(*parts) -> random.Random:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _excerpt(text: str, cap: int = EXCERPT_WORDS) -> str:
    text = text.strip()
    if text.startswith(BOILERPLATE):
        text = text[len(BOILERPLATE):]
    return " ".join(text.split()[:cap])


def _year(piece) -> int:
    d = piece.get("date") or ""
    try:
        return int(str(d)[:4])
    except ValueError:
        return -1


def _load_corpus(root):
    path = os.path.join(root, CORPUS)
    pieces = []
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            d["_id"] = f"{i}:{hashlib.sha256(d['url'].encode()).hexdigest()[:12]}"
            pieces.append(d)
    return pieces


def _decile_edges(word_counts):
    ws = sorted(word_counts)
    return [ws[int(len(ws) * k / 10)] for k in range(1, 10)]


def _decile(n, edges):
    d = 0
    for e in edges:
        if n >= e:
            d += 1
    return d


# ---------------------------------------------------------------- build ----------------------
def build(a):
    pieces = _load_corpus(a.root)
    by_author = {}
    for p in pieces:
        by_author.setdefault(p["author"], []).append(p)
    for v in by_author.values():
        v.sort(key=lambda p: p["_id"])                      # stable order before any seeding
    eligible = sorted(au for au, v in by_author.items() if len(v) >= MIN_PIECES)
    edges = _decile_edges([p["n_words"] for p in pieces])
    print(f"[build] {len(pieces)} pieces, {len(by_author)} authors, "
          f"{len(eligible)} eligible (>= {MIN_PIECES} long pieces)")

    smoke_authors = _rng("voice-smoke-authors", SEED).sample(eligible, 2)
    print(f"[build] smoke authors (seed {SEED}): {smoke_authors}")

    cells, prompts = [], []
    exemplar_neg_donors_all = set()                          # donors across BOTH exemplar sets
    plans = {}
    for author in smoke_authors:
        r = _rng("voice-smoke-splits", SEED, author)
        own = list(by_author[author])
        r.shuffle(own)
        exemplar_pool = own[:N_EXEMPLAR_POOL]
        positives = own[N_EXEMPLAR_POOL:N_EXEMPLAR_POOL + N_POS]
        if len(positives) < N_POS:
            sys.exit(f"[build] {author}: only {len(positives)} held-out positives")
        # exemplar negatives: 6 excerpts from 6 distinct donor authors (eligible, != smoke authors)
        donor_cands = [au for au in eligible if au not in smoke_authors]
        donors = r.sample(donor_cands, N_EX_NEG)
        exemplar_negs = [r.choice(by_author[d]) for d in donors]
        exemplar_neg_donors_all.update(donors)
        plans[author] = dict(exemplar_pool=exemplar_pool, positives=positives,
                             donors=donors, exemplar_negs=exemplar_negs, rng=r)

    # evaluation negatives: authors disjoint from every exemplar set (smoke authors + all donors)
    banned = set(smoke_authors) | exemplar_neg_donors_all
    neg_universe = [p for p in pieces if p["author"] in eligible and p["author"] not in banned]
    used_neg_ids = set()

    def match_negative(pos, r):
        pd, py = _decile(pos["n_words"], edges), _year(pos)
        tiers = [  # relax in declared order; record which tier matched
            ("decile+year", lambda p: _decile(p["n_words"], edges) == pd
                and abs(_year(p) - py) <= YEAR_BAND),
            ("decile-only", lambda p: _decile(p["n_words"], edges) == pd),
            ("adjacent-decile", lambda p: abs(_decile(p["n_words"], edges) - pd) <= 1),
        ]
        for tier, ok in tiers:
            cands = [p for p in neg_universe if p["_id"] not in used_neg_ids and ok(p)]
            if cands:
                pick = r.choice(sorted(cands, key=lambda p: p["_id"]))
                used_neg_ids.add(pick["_id"])
                return pick, tier
        sys.exit("[build] negative matching exhausted")

    for author in smoke_authors:
        pl = plans[author]
        r = pl["rng"]
        negatives, match_tiers = [], []
        for pos in pl["positives"]:
            neg, tier = match_negative(pos, r)
            negatives.append(neg)
            match_tiers.append(tier)

        ex_pos = pl["exemplar_pool"][:N_EX_POS]              # 6 of the 12-piece pool
        blocks = []
        for i, p in enumerate(ex_pos):
            blocks.append(f"[Excerpt {i+1} — by the same author]\n{_excerpt(p['text'])}")
        for i, p in enumerate(pl["exemplar_negs"]):
            blocks.append(f"[Excerpt {N_EX_POS+i+1} — by other authors]\n{_excerpt(p['text'])}")
        examples_ctx = (
            "You will judge authorship of a TEST TEXT. Below are labeled excerpts: some are by "
            "one specific author (a personal comic voice), the rest are by other authors.\n\n"
            + "\n\n".join(blocks))

        n_pad = max(0, len(examples_ctx.split()) - len(SMOKE_STUB_DESC.split()) - 30)
        pad = (PAD_SENTENCE * (n_pad // len(PAD_SENTENCE.split()) + 1)).split()[:n_pad]
        defn_ctx = (
            "You will judge authorship of a TEST TEXT. Below is a description of one specific "
            "author's personal comic voice.\n\nDescription of the voice:\n" + SMOKE_STUB_DESC
            + "\n\n[Neutral length-matching padding follows.]\n" + " ".join(pad))

        items = ([(p, 1) for p in pl["positives"]] + [(p, 0) for p in negatives])
        for arm, ctx in (("examples", examples_ctx), ("definition_padded", defn_ctx)):
            for k, (p, label) in enumerate(items):
                prompts.append({
                    "author": author, "arm": arm, "item": k, "label": label,
                    "piece_id": p["_id"], "piece_author": p["author"],
                    "prompt": f"{ctx}\n\nTEST TEXT:\n{_excerpt(p['text'])}\n\n{QUESTION}",
                })
        cells.append({
            "author": author,
            "exemplar_pool_ids": [p["_id"] for p in pl["exemplar_pool"]],
            "exemplar_neg_donors": pl["donors"],
            "pos_ids": [p["_id"] for p in pl["positives"]],
            "neg_ids": [p["_id"] for p in negatives],
            "neg_match_tiers": match_tiers,
            "ctx_words": {"examples": len(examples_ctx.split()),
                          "definition_padded": len(defn_ctx.split())},
        })

    os.makedirs(a.out, exist_ok=True)
    pf = os.path.join(a.out, "smoke_prompts.jsonl")
    with open(pf, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    mf = os.path.join(a.out, "smoke_manifest.json")
    json.dump({"seed": SEED, "smoke_authors": smoke_authors, "receiver": RECEIVER[0],
               "n_prompts": len(prompts), "eligible_authors": len(eligible),
               "smoke_stub": True, "length_match": "word-count proxy (smoke)",
               "boilerplate_stripped": BOILERPLATE,
               "cells": cells}, open(mf, "w"), indent=1)
    print(f"[build] {len(prompts)} prompts -> {pf}\n[build] manifest -> {mf}")
    assert len(prompts) == 160, len(prompts)


# ---------------------------------------------------------------- run (sk3) ------------------
def run(a):
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")   # sk3 fork-wedge rule
    from .. import config as cfgmod
    from ..vllm_backend import make_judge_backend
    rows = [json.loads(l) for l in open(os.path.join(a.out, "smoke_prompts.jsonl"))]
    assert len(rows) <= 500, "smoke call budget exceeded"
    cfg = cfgmod.ImplementerConfig()
    ex = make_judge_backend(RECEIVER[1], cfg, temperature=None)
    scores = ex.score_binary([r["prompt"] for r in rows], pos="YES", neg="NO")
    sf = os.path.join(a.out, "smoke_scores.jsonl")
    with open(sf, "w") as f:
        for r, s in zip(rows, scores):
            rec = {k: r[k] for k in ("author", "arm", "item", "label", "piece_id")}
            rec["p_yes"] = None if s != s else float(s)
            f.write(json.dumps(rec) + "\n")
    print(f"[run] {len(rows)} prompts scored -> {sf}")


# ---------------------------------------------------------------- score ----------------------
def score(a):
    rows = [json.loads(l) for l in open(os.path.join(a.out, "smoke_scores.jsonl"))]
    manifest = json.load(open(os.path.join(a.out, "smoke_manifest.json")))
    cells = {}
    for r in rows:
        cells.setdefault((r["author"], r["arm"]), []).append(r)
    out_cells, parse_fail = [], 0
    for (author, arm), rs in sorted(cells.items()):
        ok = [r for r in rs if r["p_yes"] is not None]
        parse_fail += len(rs) - len(ok)

        def bal(recs):
            pos = [r for r in recs if r["label"] == 1]
            neg = [r for r in recs if r["label"] == 0]
            if not pos or not neg:
                return None
            tpr = sum(r["p_yes"] >= 0.5 for r in pos) / len(pos)
            tnr = sum(r["p_yes"] < 0.5 for r in neg) / len(neg)
            return (tpr + tnr) / 2
        out_cells.append({
            "author": author, "arm": arm, "n": len(rs), "n_parsed": len(ok),
            "balanced_acc": bal(ok),
            "balanced_acc_nan_as_wrong": bal(
                [dict(r, p_yes=(r["p_yes"] if r["p_yes"] is not None
                                else (0.0 if r["label"] == 1 else 1.0))) for r in rs]),
            "mean_p_yes_pos": (sum(r["p_yes"] for r in ok if r["label"] == 1)
                               / max(1, sum(r["label"] == 1 for r in ok))),
            "mean_p_yes_neg": (sum(r["p_yes"] for r in ok if r["label"] == 0)
                               / max(1, sum(r["label"] == 0 for r in ok))),
        })
    art = {
        "experiment": "EXP-VOICE-NOUS-1",
        "stage": "smoke (Leg A; 2 authors x 2 arms x llama8b; prereg "
                 "notes/2026-08-15__exp-voice-nous-prereg.md)",
        "receiver": RECEIVER,
        "seed": SEED,
        "smoke_authors": manifest["smoke_authors"],
        "total_calls": len(rows),
        "call_budget": 500,
        "parse_failures": parse_fail,
        "cells": out_cells,
        "manifest": manifest,
        "notes": [
            "definition_padded uses a fixed SMOKE_STUB template description (authoring-model "
            "chain is part of the full run only).",
            "length matching for padding is a word-count proxy at build time (smoke).",
            "No direction interpretation: prereg forbids abort/continue on effect direction.",
        ],
    }
    dst = os.path.join(a.root, "outputs/exp_voice_nous/smoke_v1.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(art, open(dst, "w"), indent=1)
    print(f"[score] artifact -> {dst}")
    for c in out_cells:
        print(f"  {c['author']:>24s} | {c['arm']:>17s} | bal_acc="
              f"{c['balanced_acc'] if c['balanced_acc'] is not None else 'NA'} "
              f"(parsed {c['n_parsed']}/{c['n']})")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["build", "run", "score"])
    p.add_argument("--root", default=".", help="repo root (corpus + artifact paths)")
    p.add_argument("--out", required=True, help="working dir for prompts/scores")
    a = p.parse_args(argv)
    {"build": build, "run": run, "score": score}[a.mode](a)


if __name__ == "__main__":                                   # spawn-safety (sk3 vLLM rule)
    main()
