"""EXP-VOICE-NOUS-1 — FULL LEG A FLEET (prereg: notes/2026-08-15__exp-voice-nous-prereg.md).

12 slate authors (seed 0; 4 high-volume >=60 pieces / 8 mid-volume, per prereg) x 5 arms
(examples, definition, name, examples_donorswap, definition_padded; secondary examples_flip
NOT run) x 40 items x 9 receivers (llama 1/3/8/70B, qwen2.5 3/7/14/32/72B) = 2,400 judgments
per receiver. Authoring model: openai/gpt-oss-120b (present on sk3; never a receiver, so no
receiver cell exclusions under the never-receiver-in-same-cell rule).

Pipeline (modes):
  build1 (CPU)  : slate + splits + authoring prompts        -> <out>/fleet_state.json
  author (sk3)  : gpt-oss-120b writes <=150-word voice desc -> <out>/fleet_definitions.json
  build2 (CPU)  : assemble all judgment prompts             -> <out>/fleet_prompts.jsonl
  run    (sk3)  : one receiver, offline batch P(YES)        -> <out>/fleet_scores_<recv>.jsonl
  score  (CPU)  : per-cell balanced acc + gate readouts     -> outputs/exp_voice_nous/legA_fleet_v1.json

Declared instantiation details (documented, seeded):
  - slate draw: seeded shuffle within each stratum; authors with <32 pieces are skipped during
    the walk (12 exemplar + 20 held-out disjoint pieces are arithmetically impossible at <32)
    and recorded in the manifest.
  - exemplar negatives come from a SHARED seeded donor pool of 12 non-slate eligible authors
    (keeps the negative-author universe large; prereg constraint is only that evaluation
    negative authors be disjoint from every exemplar set, enforced here as slate+donor ban).
  - donorswap derangement: seeded shuffle of the slate, then rotate by one.
  - Patreon boilerplate (constant 33-word prefix) stripped before excerpting (smoke finding).
  - padding length-match uses the word-count proxy (as in smoke).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .exp_voice_smoke import (BOILERPLATE, EXCERPT_WORDS, N_EX_NEG, N_EX_POS, N_EXEMPLAR_POOL,
                              N_NEG, N_POS, YEAR_BAND, PAD_SENTENCE, _decile, _decile_edges,
                              _excerpt, _load_corpus, _rng, _year)

SEED = 0
MIN_PIECES = 30
MIN_SPLIT = N_EXEMPLAR_POOL + N_POS          # 32: arithmetic floor for disjoint splits
N_HIGH, N_MID = 4, 8
HIGH_CUT = 60
DONOR_POOL_SIZE = 12
ARMS = ("examples", "definition", "name", "examples_donorswap", "definition_padded")
AUTHOR_MODEL = ("gpt-oss-120b", "openai/gpt-oss-120b")

RECEIVERS = {  # short -> (hf id, ~GB needed incl. kv/headroom) ; same ckpts as osl_sweep
    "llama1b":   ("meta-llama/Llama-3.2-1B-Instruct", 10),
    "llama3b":   ("meta-llama/Llama-3.2-3B-Instruct", 14),
    "llama8b":   ("meta-llama/Llama-3.1-8B-Instruct", 26),
    "llama70b":  ("meta-llama/Llama-3.3-70B-Instruct", 150),
    "qwen25-3b": ("Qwen/Qwen2.5-3B-Instruct", 14),
    "qwen25-7b": ("Qwen/Qwen2.5-7B-Instruct", 24),
    "qwen25-14b": ("Qwen/Qwen2.5-14B-Instruct", 42),
    "qwen25-32b": ("Qwen/Qwen2.5-32B-Instruct", 78),
    "qwen25-72b": ("Qwen/Qwen2.5-72B-Instruct", 155),
    # EXP-VOICE-NOUS-2 frontier receiver (prereg b1d73d3bb607a6a1); FP8 MoE, TP=2,
    # needs VLLM_USE_FLASHINFER_MOE_FP8=0 (see reference_qwen35_vllm_sk3)
    "qwen35-122b": ("Qwen/Qwen3.5-122B-A10B-FP8", 140),
}

HDR_EX = ("You will judge authorship of a TEST TEXT. Below are labeled excerpts: some are by "
          "one specific author (a personal comic voice), the rest are by other authors.\n\n")
HDR_DEF = ("You will judge authorship of a TEST TEXT. Below is a description of one specific "
           "author's personal comic voice.\n\nDescription of the voice:\n")
Q_TAIL = "Answer with exactly one word: YES or NO."
QUESTIONS = {
    "examples": ("Question: Is the TEST TEXT written by the same author (in the same personal "
                 f"comic voice) as the excerpts labeled 'by the same author' above? {Q_TAIL}"),
    "examples_donorswap": ("Question: Is the TEST TEXT written by the same author (in the same "
                           "personal comic voice) as the excerpts labeled 'by the same author' "
                           f"above? {Q_TAIL}"),
    "definition": ("Question: Is the TEST TEXT written by the author whose voice is described "
                   f"above? {Q_TAIL}"),
    "definition_padded": ("Question: Is the TEST TEXT written by the author whose voice is "
                          f"described above? {Q_TAIL}"),
    "name": "",   # filled per author
}

AUTHORING_INSTRUCTION = (
    "\n\nTask: In at most 150 words, write a description of the distinctive personal comic "
    "voice of the author of the excerpts labeled 'by the same author' — specific enough that a "
    "careful reader could recognize NEW pieces by this author from the description alone. "
    "Describe voice, rhythm, stance, register, and structural habits; contrast with the 'by "
    "other authors' excerpts where useful. Do NOT try to name the author. Output ONLY the "
    "description text.")


# ---------------------------------------------------------------- build1 ---------------------
def build1(a):
    pieces = _load_corpus(a.root)
    by_author = {}
    for p in pieces:
        by_author.setdefault(p["author"], []).append(p)
    for v in by_author.values():
        v.sort(key=lambda p: p["_id"])
    eligible = sorted(au for au, v in by_author.items() if len(v) >= MIN_PIECES)
    edges = _decile_edges([p["n_words"] for p in pieces])

    high = [au for au in eligible if len(by_author[au]) >= HIGH_CUT]
    mid = [au for au in eligible if au not in high]
    skipped = []

    def draw(pool, k, tag):
        pool = list(pool)
        _rng("voice-fleet-slate", SEED, tag).shuffle(pool)
        got = []
        for au in pool:
            if len(got) == k:
                break
            if len(by_author[au]) < MIN_SPLIT:
                skipped.append(au)
                continue
            got.append(au)
        if len(got) < k:
            sys.exit(f"[build1] stratum {tag}: only {len(got)}/{k} drawable")
        return got

    slate_high = draw(high, N_HIGH, "high")
    slate_mid = draw(mid, N_MID, "mid")
    slate = slate_high + slate_mid
    print(f"[build1] slate high={slate_high}\n[build1] slate mid={slate_mid}\n"
          f"[build1] skipped(<{MIN_SPLIT} pieces)={skipped}")

    donor_pool = _rng("voice-fleet-donors", SEED).sample(
        sorted(au for au in eligible if au not in slate), DONOR_POOL_SIZE)
    banned = set(slate) | set(donor_pool)
    neg_universe = sorted((p for p in pieces if p["author"] in eligible
                           and p["author"] not in banned), key=lambda p: p["_id"])
    used_neg = set()

    def match_negative(pos, r):
        pd, py = _decile(pos["n_words"], edges), _year(pos)
        tiers = [("decile+year", lambda p: _decile(p["n_words"], edges) == pd
                  and abs(_year(p) - py) <= YEAR_BAND),
                 ("decile-only", lambda p: _decile(p["n_words"], edges) == pd),
                 ("adjacent-decile", lambda p: abs(_decile(p["n_words"], edges) - pd) <= 1)]
        for tier, ok in tiers:
            cands = [p for p in neg_universe if p["_id"] not in used_neg and ok(p)]
            if cands:
                pick = r.choice(cands)
                used_neg.add(pick["_id"])
                return pick, tier
        sys.exit("[build1] negative matching exhausted")

    cells, authoring = {}, []
    for author in slate:
        r = _rng("voice-fleet-splits", SEED, author)
        own = list(by_author[author])
        r.shuffle(own)
        exemplar_pool = own[:N_EXEMPLAR_POOL]
        positives = own[N_EXEMPLAR_POOL:N_EXEMPLAR_POOL + N_POS]
        donors = r.sample(donor_pool, N_EX_NEG)
        exemplar_negs = [r.choice(by_author[d]) for d in donors]
        negatives, tiers = [], []
        for pos in positives:
            neg, tier = match_negative(pos, r)
            negatives.append(neg)
            tiers.append(tier)
        ex_pos = exemplar_pool[:N_EX_POS]
        blocks = [f"[Excerpt {i+1} — by the same author]\n{_excerpt(p['text'])}"
                  for i, p in enumerate(ex_pos)]
        blocks += [f"[Excerpt {N_EX_POS+i+1} — by other authors]\n{_excerpt(p['text'])}"
                   for i, p in enumerate(exemplar_negs)]
        exemplar_block = "\n\n".join(blocks)
        cells[author] = {
            "stratum": "high" if author in slate_high else "mid",
            "n_pieces": len(by_author[author]),
            "exemplar_pool_ids": [p["_id"] for p in exemplar_pool],
            "exemplar_neg_donors": donors,
            "exemplar_block": exemplar_block,
            "pos": [{"id": p["_id"], "author": p["author"], "text": _excerpt(p["text"])}
                    for p in positives],
            "neg": [{"id": p["_id"], "author": p["author"], "text": _excerpt(p["text"])}
                    for p in negatives],
            "neg_match_tiers": tiers,
        }
        authoring.append({"author": author,
                          "prompt": HDR_EX + exemplar_block + AUTHORING_INSTRUCTION})

    ds = list(slate)
    _rng("voice-fleet-donorswap", SEED).shuffle(ds)
    donorswap = {ds[i]: ds[(i + 1) % len(ds)] for i in range(len(ds))}   # derangement

    os.makedirs(a.out, exist_ok=True)
    state = {"seed": SEED, "slate": slate, "slate_high": slate_high, "slate_mid": slate_mid,
             "skipped_lt32": skipped, "donor_pool": donor_pool, "donorswap": donorswap,
             "authoring_model": AUTHOR_MODEL, "boilerplate_stripped": BOILERPLATE,
             "excerpt_cap_words": EXCERPT_WORDS, "year_band": YEAR_BAND,
             "length_match": "word-count proxy", "arms": list(ARMS),
             "eligible_authors": len(eligible), "cells": cells}
    json.dump(state, open(os.path.join(a.out, "fleet_state.json"), "w"), indent=1)
    with open(os.path.join(a.out, "fleet_authoring_prompts.jsonl"), "w") as f:
        for row in authoring:
            f.write(json.dumps(row) + "\n")
    print(f"[build1] state -> {a.out}/fleet_state.json ; {len(authoring)} authoring prompts")


# ---------------------------------------------------------------- author (sk3) ---------------
def _harmony_final(text: str) -> str:
    """gpt-oss emits harmony channels. With skip_special_tokens the channel-switch tokens
    detokenize away and the final channel opens as the literal 'assistantfinal' (verified on
    sk3: raw output starts 'analysis...' then 'assistantfinal<description>'). Return the final
    channel ONLY; empty string if no final marker (callers must treat that as failure —
    returning the whole text would leak chain-of-thought into the definition arm)."""
    for marker in ("<|channel|>final<|message|>", "assistantfinal", "\nfinal\n"):
        if marker in text:
            return text.split(marker)[-1].split("<|")[0].strip()
    return ""


def author(a):
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from .. import config as cfgmod
    from ..vllm_backend import make_judge_backend
    rows = [json.loads(l) for l in
            open(os.path.join(a.out, "fleet_authoring_prompts.jsonl"))]
    cfg = cfgmod.ImplementerConfig()
    ex = make_judge_backend(AUTHOR_MODEL[1], cfg, temperature=0.2)

    def valid(t):
        return len(_harmony_final(t).split()) >= 30
    outs = ex.generate_batch([r["prompt"] for r in rows], max_tokens=3072, validate=valid)
    defs, raws = {}, {}
    for r, o in zip(rows, outs):
        d = " ".join(_harmony_final(o).split()[:150])         # enforce <=150 words
        if len(d.split()) < 30:
            sys.exit(f"[author] degenerate/no-final-channel description for "
                     f"{r['author']!r}: {o[:200]!r}")
        defs[r["author"]] = d
        raws[r["author"]] = o
    json.dump({"model": AUTHOR_MODEL, "definitions": defs, "raw_outputs": raws},
              open(os.path.join(a.out, "fleet_definitions.json"), "w"), indent=1)
    print(f"[author] {len(defs)} definitions -> {a.out}/fleet_definitions.json")


# ---------------------------------------------------------------- build2 ---------------------
def build2(a):
    state = json.load(open(os.path.join(a.out, "fleet_state.json")))
    defs = json.load(open(os.path.join(a.out, "fleet_definitions.json")))["definitions"]
    prompts = []
    for author_name in state["slate"]:
        c = state["cells"][author_name]
        ex_ctx = HDR_EX + c["exemplar_block"]
        defn = defs[author_name]
        def_ctx = HDR_DEF + defn
        n_pad = max(0, len(ex_ctx.split()) - len(def_ctx.split()) - 10)
        pad = (PAD_SENTENCE * (n_pad // len(PAD_SENTENCE.split()) + 1)).split()[:n_pad]
        defpad_ctx = (def_ctx + "\n\n[Neutral length-matching padding follows.]\n"
                      + " ".join(pad))
        swap_ctx = HDR_EX + state["cells"][state["donorswap"][author_name]]["exemplar_block"]
        name_ctx = ("You will judge authorship of a TEST TEXT. The author in question is "
                    f"{author_name}, a comic writer published on McSweeney's Internet Tendency.")
        name_q = (f"Question: Is the TEST TEXT written by {author_name} (in {author_name}'s "
                  f"personal comic voice)? {Q_TAIL}")
        arm_ctx = {"examples": (ex_ctx, QUESTIONS["examples"]),
                   "definition": (def_ctx, QUESTIONS["definition"]),
                   "definition_padded": (defpad_ctx, QUESTIONS["definition_padded"]),
                   "examples_donorswap": (swap_ctx, QUESTIONS["examples_donorswap"]),
                   "name": (name_ctx, name_q)}
        items = [(p, 1) for p in c["pos"]] + [(p, 0) for p in c["neg"]]
        for arm in ARMS:
            ctx, q = arm_ctx[arm]
            for k, (p, label) in enumerate(items):
                prompts.append({"author": author_name, "arm": arm, "item": k, "label": label,
                                "piece_id": p["id"], "piece_author": p["author"],
                                "prompt": f"{ctx}\n\nTEST TEXT:\n{p['text']}\n\n{q}"})
    pf = os.path.join(a.out, "fleet_prompts.jsonl")
    with open(pf, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    n = len(state["slate"]) * len(ARMS) * (N_POS + N_NEG)
    assert len(prompts) == n, (len(prompts), n)
    print(f"[build2] {len(prompts)} judgment prompts -> {pf}")


# ---------------------------------------------------------------- run (sk3) ------------------
def run(a):
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from .. import config as cfgmod
    from ..vllm_backend import make_judge_backend
    hf_id, _need = RECEIVERS[a.receiver]
    rows = [json.loads(l) for l in open(os.path.join(a.out, "fleet_prompts.jsonl"))]
    sf = os.path.join(a.out, f"fleet_scores_{a.receiver}.jsonl")
    done = sum(1 for _ in open(sf)) if os.path.exists(sf) else 0
    if done >= len(rows):
        print(f"[run] {a.receiver} already complete ({done})")
        return
    cfg = cfgmod.ImplementerConfig()
    # VOICE_TP=2: user-authorized 2-GPU BF16 exception for llama70b / qwen25-72b ONLY
    cfg.vllm_tp_size = int(os.environ.get("VOICE_TP", "1"))
    ex = make_judge_backend(hf_id, cfg, temperature=None)
    CH = 800                                             # chunked + resumable (never one mega call)
    with open(sf, "a") as f:
        for lo in range(done, len(rows), CH):
            chunk = rows[lo:lo + CH]
            scores = ex.score_binary([r["prompt"] for r in chunk], pos="YES", neg="NO")
            for r, s in zip(chunk, scores):
                rec = {k: r[k] for k in ("author", "arm", "item", "label", "piece_id")}
                rec["p_yes"] = None if s != s else float(s)
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[run] {a.receiver} {min(lo+CH, len(rows))}/{len(rows)}", flush=True)
    print(f"[run] {a.receiver} DONE -> {sf}")


# ---------------------------------------------------------------- score ----------------------
def _bal(recs):
    pos = [r for r in recs if r["label"] == 1]
    neg = [r for r in recs if r["label"] == 0]
    if not pos or not neg:
        return None
    tpr = sum(r["p_yes"] >= 0.5 for r in pos) / len(pos)
    tnr = sum(r["p_yes"] < 0.5 for r in neg) / len(neg)
    return (tpr + tnr) / 2


def score(a):
    state = json.load(open(os.path.join(a.out, "fleet_state.json")))
    defs = json.load(open(os.path.join(a.out, "fleet_definitions.json")))
    strat = {au: state["cells"][au]["stratum"] for au in state["slate"]}
    receivers_done, cells_out, gates = [], [], {"G2_donorswap": {}, "G4_name_vs_examples": {}}
    parse_fail = total_calls = 0
    for recv in sorted(RECEIVERS):
        sf = os.path.join(a.out, f"fleet_scores_{recv}.jsonl")
        if not os.path.exists(sf):
            continue
        rows = [json.loads(l) for l in open(sf)]
        total_calls += len(rows)
        receivers_done.append(recv)
        ok = [r for r in rows if r["p_yes"] is not None]
        parse_fail += len(rows) - len(ok)
        by_cell = {}
        for r in ok:
            by_cell.setdefault((r["author"], r["arm"]), []).append(r)
        for (au, arm), rs in sorted(by_cell.items()):
            yes = [r["p_yes"] >= 0.5 for r in rs]
            cells_out.append({"receiver": recv, "author": au, "stratum": strat[au],
                              "arm": arm, "n": len(rs), "balanced_acc": _bal(rs),
                              "yes_rate": sum(yes) / len(yes),
                              "degenerate_constant": len(set(yes)) == 1})
        sw = [r for r in ok if r["arm"] == "examples_donorswap"]
        gates["G2_donorswap"][recv] = {"pooled_balanced_acc": _bal(sw), "n": len(sw)}
        g4 = {}
        for st in ("high", "mid"):
            sub = {arm: [r for r in ok if r["arm"] == arm and strat[r["author"]] == st]
                   for arm in ("name", "examples")}
            g4[st] = {arm: {"pooled_balanced_acc": _bal(v), "n": len(v)}
                      for arm, v in sub.items()}
        gates["G4_name_vs_examples"][recv] = g4
    art = {"experiment": "EXP-VOICE-NOUS-1", "stage": "Leg A full fleet",
           "prereg": "notes/2026-08-15__exp-voice-nous-prereg.md",
           "seed": SEED, "arms": list(ARMS), "authoring_model": AUTHOR_MODEL,
           "receiver_exclusions": [],   # authoring model is not a receiver
           "receivers_done": receivers_done, "total_calls": total_calls,
           "parse_failures": parse_fail, "gates": gates, "cells": cells_out,
           "definitions": defs["definitions"],
           "manifest": {k: state[k] for k in
                        ("slate", "slate_high", "slate_mid", "skipped_lt32", "donor_pool",
                         "donorswap", "boilerplate_stripped", "excerpt_cap_words", "year_band",
                         "length_match", "eligible_authors")},
           "split_ids": {au: {k: state["cells"][au][k] for k in
                              ("exemplar_pool_ids", "exemplar_neg_donors", "neg_match_tiers")}
                         for au in state["slate"]},
           "notes": ["Numbers only; no decision-rule evaluation here (in-session with user).",
                     "examples_flip secondary arm NOT run (per instruction)."]}
    dst = os.path.join(a.root, "outputs/exp_voice_nous/legA_fleet_v1.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(art, open(dst, "w"), indent=1)
    print(f"[score] {len(receivers_done)} receivers, {total_calls} calls, "
          f"{parse_fail} parse failures -> {dst}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["build1", "author", "build2", "run", "score"])
    p.add_argument("--root", default=".")
    p.add_argument("--out", required=True)
    p.add_argument("--receiver", choices=sorted(RECEIVERS))
    a = p.parse_args(argv)
    if a.mode == "run" and not a.receiver:
        sys.exit("run mode needs --receiver")
    {"build1": build1, "author": author, "build2": build2, "run": run, "score": score}[a.mode](a)


if __name__ == "__main__":                                   # spawn-safety (sk3 vLLM rule)
    main()
