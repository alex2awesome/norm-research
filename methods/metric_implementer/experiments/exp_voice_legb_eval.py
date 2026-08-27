"""EXP-VOICE-NOUS-1 — LEG B EVALUATION (prereg: notes/2026-08-15__exp-voice-nous-prereg.md).

Stage 1 — describability gate: describer gpt-oss-120b (assistantfinal-only) reads 6 exemplar
excerpts per persona and writes a <=150-word voice description; gate receiver qwen25-32b
(strongest available local receiver — declared 'frontier receiver' implementation) scores two
probes per persona over 40 gate items (20 held-out own / 20 same-grade sibling, topic-matched,
disjoint from the 6 exemplars): (a) DESCRIPTION-alone, (b) 6-EXAMPLES passthrough.

Stage 2 — arms fleet, Leg A machinery: per persona, exemplar pool 12 (first 6 used) / pos 20 /
neg 20 same-grade sibling topic-matched; arms examples / definition (the Stage-1 description,
uniform across grades) / name (persona slug; D3 = seed author's real name) /
examples_donorswap (same-grade derangement) / definition_padded (length-matched). Receivers:
the same 7-receiver ladder, P(YES) logprob readout.

Split reuse across stages (declared): stage-1 gate items ARE the stage-2 pos/neg splits — pos
is disjoint from the whole 12-text exemplar pool, hence from the 6 exemplars; a single split
also keeps the definition arm's description artifact identical to the gate description.

Modes:
  build1   (CPU) : splits + describer prompts  -> <out>/legbe_state.json, legbe_desc_prompts.jsonl
  describe (sk3) : gpt-oss-120b descriptions   -> <out>/legbe_descriptions.json
  build2   (CPU) : gate + arm prompts          -> <out>/legbe_gate_prompts.jsonl, legbe_prompts.jsonl
  gate     (sk3) : qwen25-32b gate scoring     -> <out>/legbe_gate_scores.jsonl
  run      (sk3) : one stage-2 receiver        -> <out>/legbe_scores_<recv>.jsonl
  score    (CPU) : artifact -> outputs/exp_voice_nous/legB_eval_v1.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys

from .exp_voice_smoke import PAD_SENTENCE, _rng
from .exp_voice_fleet import RECEIVERS, HDR_EX, HDR_DEF, Q_TAIL, QUESTIONS
from .exp_voice_legb import GEN_MODEL, MATERIALS_SHA, _harmony_final, _load_materials

SEED = 0
TEXTS = "outputs/exp_voice_nous/legB_texts_v1.jsonl.gz"
GATE_RECEIVER = "qwen25-32b"
EXCERPT_WORDS = 300
N_POOL, N_USED, N_POS, N_NEG = 12, 6, 20, 20
D3_SEED_NAMES = {"D3-teddy-wayne": "Teddy Wayne", "D3-suzanne-yeagley": "Suzanne Yeagley",
                 "D3-ben-greenman": "Ben Greenman", "D3-kevin-dolgin": "Kevin Dolgin"}


def _exc(text, cap=EXCERPT_WORDS):
    return " ".join(text.split()[:cap])


def _key(r):
    return f"{r['persona']}|t{r['topic_idx']}|d{r['draft']}"


def _load_texts(root):
    rows = [json.loads(l) for l in gzip.open(os.path.join(root, TEXTS), "rt")]
    by = {}
    for r in rows:
        by.setdefault(r["persona"], []).append(r)
    for v in by.values():
        v.sort(key=_key)
    return by


# ---------------------------------------------------------------- build1 ---------------------
def build1(a):
    mat = _load_materials(a.root)
    by = _load_texts(a.root)
    personas = sorted(by, key=lambda p: (p.split("-")[0], p))
    grade_of = {p: p.split("-")[0] for p in personas}
    state = {"seed": SEED, "materials_sha": MATERIALS_SHA, "gate_receiver": GATE_RECEIVER,
             "gate_receiver_note": "strongest available local receiver; declared 'frontier "
                                   "receiver' implementation for the describability gate",
             "describer": GEN_MODEL, "excerpt_cap_words": EXCERPT_WORDS,
             "d3_name_map": D3_SEED_NAMES, "personas": {}}
    used_neg = {p: set() for p in personas}
    for p in personas:
        r = _rng("voice-legbe-splits", SEED, p)
        own = list(by[p])
        r.shuffle(own)
        pool, pos = own[:N_POOL], own[N_POOL:N_POOL + N_POS]
        sibs = [q for q in personas if grade_of[q] == grade_of[p] and q != p]
        negs = []
        for it in pos:                       # topic-matched sibling negative, no text reuse
            cands = [t for s in sibs for t in by[s]
                     if t["topic_idx"] == it["topic_idx"] and _key(t) not in used_neg[p]]
            if not cands:
                cands = [t for s in sibs for t in by[s] if _key(t) not in used_neg[p]]
            pick = r.choice(sorted(cands, key=_key))
            used_neg[p].add(_key(pick))
            negs.append(pick)
        state["personas"][p] = {
            "grade": grade_of[p], "siblings": sibs,
            "pool_keys": [_key(t) for t in pool], "pos_keys": [_key(t) for t in pos],
            "neg_keys": [_key(t) for t in negs],
            "neg_topic_matched": sum(n["topic_idx"] == q["topic_idx"]
                                     for n, q in zip(negs, pos))}
    ds = {}
    for g in ("D1", "D2", "D3"):                        # same-grade derangement (rotate by 1)
        grp = [p for p in personas if grade_of[p] == g]
        _rng("voice-legbe-donorswap", SEED, g).shuffle(grp)
        for i, p in enumerate(grp):
            ds[p] = grp[(i + 1) % len(grp)]
    state["donorswap"] = ds
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "legbe_desc_prompts.jsonl"), "w") as f:
        for p in personas:
            pool = {_key(t): t for t in by[p]}
            ex6 = [pool[k] for k in state["personas"][p]["pool_keys"][:N_USED]]
            exemplars = "\n\n".join(f"[Piece {i+1}]\n{_exc(t['text'])}"
                                    for i, t in enumerate(ex6))
            f.write(json.dumps({"persona": p, "prompt":
                    mat["describability_gate"]["describer_prompt"]
                    .format(exemplars=exemplars)}) + "\n")
    json.dump(state, open(os.path.join(a.out, "legbe_state.json"), "w"), indent=1)
    print(f"[build1] {len(personas)} personas -> {a.out}/legbe_state.json")


# ---------------------------------------------------------------- describe (sk3) -------------
def describe(a):
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from .. import config as cfgmod
    from ..vllm_backend import make_judge_backend
    rows = [json.loads(l) for l in open(os.path.join(a.out, "legbe_desc_prompts.jsonl"))]
    cfg = cfgmod.ImplementerConfig()
    ex = make_judge_backend(GEN_MODEL[1], cfg, temperature=0.2)

    def valid(t):
        return len(_harmony_final(t).split()) >= 30
    outs = ex.generate_batch([r["prompt"] for r in rows], max_tokens=3072, validate=valid)
    descs, raws = {}, {}
    for r, o in zip(rows, outs):
        d = " ".join(_harmony_final(o).split()[:150])
        if len(d.split()) < 30:
            sys.exit(f"[describe] degenerate for {r['persona']}: {o[:200]!r}")
        descs[r["persona"]] = d
        raws[r["persona"]] = o
    json.dump({"model": GEN_MODEL, "descriptions": descs, "raw_outputs": raws},
              open(os.path.join(a.out, "legbe_descriptions.json"), "w"), indent=1)
    print(f"[describe] {len(descs)} -> {a.out}/legbe_descriptions.json")


# ---------------------------------------------------------------- build2 ---------------------
def build2(a):
    mat = _load_materials(a.root)
    by = _load_texts(a.root)
    state = json.load(open(os.path.join(a.out, "legbe_state.json")))
    descs = json.load(open(os.path.join(a.out, "legbe_descriptions.json")))["descriptions"]
    allt = {_key(t): t for v in by.values() for t in v}
    gate_rows, arm_rows = [], []
    for p, st in state["personas"].items():
        pool6 = [allt[k] for k in st["pool_keys"][:N_USED]]
        pos = [allt[k] for k in st["pos_keys"]]
        negs = [allt[k] for k in st["neg_keys"]]
        items = [(t, 1) for t in pos] + [(t, 0) for t in negs]
        desc = descs[p]
        # ---- stage 1 gate probes -------------------------------------------------------
        ex_pass = ("Here are 6 pieces by one writer:\n\n" + "\n\n".join(
            f"[Piece {i+1}]\n{_exc(t['text'])}" for i, t in enumerate(pool6)) +
            "\n\nIs the following piece by that writer? Answer YES or NO.\n\n")
        for k, (t, label) in enumerate(items):
            gate_rows.append({"persona": p, "probe": "description", "item": k, "label": label,
                              "piece_key": _key(t), "prompt":
                              mat["describability_gate"]["receiver_from_description"]
                              .format(description=desc, item=_exc(t["text"]))})
            gate_rows.append({"persona": p, "probe": "examples_passthrough", "item": k,
                              "label": label, "piece_key": _key(t),
                              "prompt": ex_pass + _exc(t["text"])})
        # ---- stage 2 arm contexts ------------------------------------------------------
        sib_ex = []                          # 6 exemplar negatives: 2 per sibling, seeded,
        r = _rng("voice-legbe-exneg", SEED, p)          # never this persona's eval negatives
        for s in st["siblings"]:
            cands = [t for t in by[s] if _key(t) not in set(st["neg_keys"])]
            sib_ex += r.sample(sorted(cands, key=_key), 2)
        blocks = [f"[Excerpt {i+1} — by the same voice]\n{_exc(t['text'])}"
                  for i, t in enumerate(pool6)]
        blocks += [f"[Excerpt {N_USED+i+1} — by other writers]\n{_exc(t['text'])}"
                   for i, t in enumerate(sib_ex)]
        ex_ctx = HDR_EX + "\n\n".join(blocks)
        def_ctx = HDR_DEF + desc
        n_pad = max(0, len(ex_ctx.split()) - len(def_ctx.split()) - 10)
        pad = (PAD_SENTENCE * (n_pad // len(PAD_SENTENCE.split()) + 1)).split()[:n_pad]
        defpad_ctx = def_ctx + "\n\n[Neutral length-matching padding follows.]\n" + " ".join(pad)
        nm = D3_SEED_NAMES.get(p, p)
        name_ctx = ("You will judge authorship of a TEST TEXT. The voice in question is "
                    + (f"the comic writer {nm}." if p in D3_SEED_NAMES
                       else f"a comedic voice known as '{nm}'."))
        name_q = (f"Question: Is the TEST TEXT written in the voice of "
                  + (nm if p in D3_SEED_NAMES else f"'{nm}'") + f"? {Q_TAIL}")
        state["personas"][p]["exemplar_neg_keys"] = [_key(t) for t in sib_ex]
        state["personas"][p]["ctx_words"] = {"examples": len(ex_ctx.split()),
                                             "definition_padded": len(defpad_ctx.split())}
        arm_ctx = {"examples": (ex_ctx, QUESTIONS["examples"]),
                   "definition": (def_ctx, QUESTIONS["definition"]),
                   "definition_padded": (defpad_ctx, QUESTIONS["definition_padded"]),
                   "name": (name_ctx, name_q)}
        for arm, (ctx, q) in arm_ctx.items():
            for k, (t, label) in enumerate(items):
                arm_rows.append({"persona": p, "arm": arm, "item": k, "label": label,
                                 "piece_key": _key(t),
                                 "prompt": f"{ctx}\n\nTEST TEXT:\n{_exc(t['text'])}\n\n{q}"})
    for p, st in state["personas"].items():             # donorswap needs all ex_ctx built
        donor = state["donorswap"][p]
        dpool6 = [allt[k] for k in state["personas"][donor]["pool_keys"][:N_USED]]
        dsib = [allt[k] for k in state["personas"][donor]["exemplar_neg_keys"]]
        blocks = [f"[Excerpt {i+1} — by the same voice]\n{_exc(t['text'])}"
                  for i, t in enumerate(dpool6)]
        blocks += [f"[Excerpt {N_USED+i+1} — by other writers]\n{_exc(t['text'])}"
                   for i, t in enumerate(dsib)]
        ctx = HDR_EX + "\n\n".join(blocks)
        items = ([(allt[k], 1) for k in st["pos_keys"]]
                 + [(allt[k], 0) for k in st["neg_keys"]])
        for k, (t, label) in enumerate(items):
            arm_rows.append({"persona": p, "arm": "examples_donorswap", "item": k,
                             "label": label, "piece_key": _key(t),
                             "prompt": f"{ctx}\n\nTEST TEXT:\n{_exc(t['text'])}\n\n"
                                       f"{QUESTIONS['examples_donorswap']}"})
    json.dump(state, open(os.path.join(a.out, "legbe_state.json"), "w"), indent=1)
    for fn, rows in (("legbe_gate_prompts.jsonl", gate_rows),
                     ("legbe_prompts.jsonl", arm_rows)):
        with open(os.path.join(a.out, fn), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"[build2] gate={len(gate_rows)} arm={len(arm_rows)} prompts -> {a.out}")


# ---------------------------------------------------------------- gate / run (sk3) -----------
def _score_file(a, src, dst, hf_id):
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from .. import config as cfgmod
    from ..vllm_backend import make_judge_backend
    rows = [json.loads(l) for l in open(os.path.join(a.out, src))]
    sf = os.path.join(a.out, dst)
    done = sum(1 for _ in open(sf)) if os.path.exists(sf) else 0
    if done >= len(rows):
        print(f"[score-run] {dst} already complete")
        return
    cfg = cfgmod.ImplementerConfig()
    ex = make_judge_backend(hf_id, cfg, temperature=None)
    CH = 800
    with open(sf, "a") as f:
        for lo in range(done, len(rows), CH):
            chunk = rows[lo:lo + CH]
            scores = ex.score_binary([r["prompt"] for r in chunk], pos="YES", neg="NO")
            for r, s in zip(chunk, scores):
                rec = {k: v for k, v in r.items() if k != "prompt"}
                rec["p_yes"] = None if s != s else float(s)
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[score-run] {dst} {min(lo+CH, len(rows))}/{len(rows)}", flush=True)
    print(f"[score-run] DONE -> {sf}")


def gate(a):
    _score_file(a, "legbe_gate_prompts.jsonl", "legbe_gate_scores.jsonl",
                RECEIVERS[GATE_RECEIVER][0])


def run(a):
    _score_file(a, "legbe_prompts.jsonl", f"legbe_scores_{a.receiver}.jsonl",
                RECEIVERS[a.receiver][0])


# ---------------------------------------------------------------- score ----------------------
def _bal(recs):
    pos = [r for r in recs if r["label"] == 1]
    neg = [r for r in recs if r["label"] == 0]
    if not pos or not neg:
        return None
    return (sum(r["p_yes"] >= 0.5 for r in pos) / len(pos)
            + sum(r["p_yes"] < 0.5 for r in neg) / len(neg)) / 2


def score(a):
    state = json.load(open(os.path.join(a.out, "legbe_state.json")))
    descs = json.load(open(os.path.join(a.out, "legbe_descriptions.json")))
    grade = {p: st["grade"] for p, st in state["personas"].items()}
    parse_fail = total = 0
    # stage 1
    grows = [json.loads(l) for l in open(os.path.join(a.out, "legbe_gate_scores.jsonl"))]
    total += len(grows)
    gok = [r for r in grows if r["p_yes"] is not None]
    parse_fail += len(grows) - len(gok)
    stage1 = []
    for p in sorted(state["personas"]):
        row = {"persona": p, "grade": grade[p]}
        for probe in ("description", "examples_passthrough"):
            rs = [r for r in gok if r["persona"] == p and r["probe"] == probe]
            row[probe] = {"balanced_acc": _bal(rs), "n": len(rs)}
        stage1.append(row)
    # stage 2
    cells, pooled, g2, receivers_done = [], {}, {}, []
    for recv in sorted(RECEIVERS):
        sf = os.path.join(a.out, f"legbe_scores_{recv}.jsonl")
        if not os.path.exists(sf):
            continue
        rows = [json.loads(l) for l in open(sf)]
        total += len(rows)
        receivers_done.append(recv)
        ok = [r for r in rows if r["p_yes"] is not None]
        parse_fail += len(rows) - len(ok)
        by_cell = {}
        for r in ok:
            by_cell.setdefault((r["persona"], r["arm"]), []).append(r)
        for (p, arm), rs in sorted(by_cell.items()):
            yes = [r["p_yes"] >= 0.5 for r in rs]
            cells.append({"receiver": recv, "persona": p, "grade": grade[p], "arm": arm,
                          "n": len(rs), "balanced_acc": _bal(rs),
                          "yes_rate": sum(yes) / len(yes),
                          "degenerate_constant": len(set(yes)) == 1})
        for g in ("D1", "D2", "D3"):
            for arm in set(r["arm"] for r in ok):
                rs = [r for r in ok if r["arm"] == arm and grade[r["persona"]] == g]
                pooled.setdefault(recv, {}).setdefault(g, {})[arm] = {
                    "pooled_balanced_acc": _bal(rs), "n": len(rs)}
        sw = [r for r in ok if r["arm"] == "examples_donorswap"]
        g2[recv] = {"pooled_balanced_acc": _bal(sw), "n": len(sw)}
    art = {"experiment": "EXP-VOICE-NOUS-1", "stage": "Leg B evaluation",
           "prereg": "notes/2026-08-15__exp-voice-nous-prereg.md",
           "materials_sha": MATERIALS_SHA, "seed": SEED,
           "describer": GEN_MODEL, "gate_receiver": GATE_RECEIVER,
           "gate_receiver_note": state["gate_receiver_note"],
           "receivers_done": receivers_done, "total_calls": total,
           "parse_failures": parse_fail,
           "stage1_gate": stage1,
           "stage2_pooled_grade_arm_receiver": pooled,
           "G2_donorswap": g2, "cells": cells,
           "descriptions": descs["descriptions"],
           "manifest": {"donorswap": state["donorswap"], "d3_name_map": D3_SEED_NAMES,
                        "excerpt_cap_words": EXCERPT_WORDS,
                        "splits": {p: {k: st[k] for k in
                                       ("grade", "pool_keys", "pos_keys", "neg_keys",
                                        "exemplar_neg_keys", "neg_topic_matched")}
                                   for p, st in state["personas"].items()}},
           "notes": ["Numbers only; gate rule and decision rule evaluated in-session.",
                     "Stage-1 gate items = stage-2 pos/neg splits (declared reuse)."]}
    dst = os.path.join(a.root, "outputs/exp_voice_nous/legB_eval_v1.json")
    json.dump(art, open(dst, "w"), indent=1)
    print(f"[score] {len(receivers_done)} receivers, {total} calls, {parse_fail} parse "
          f"failures -> {dst}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["build1", "describe", "build2", "gate", "run", "score"])
    p.add_argument("--root", default=".")
    p.add_argument("--out", required=True)
    p.add_argument("--receiver", choices=sorted(RECEIVERS))
    a = p.parse_args(argv)
    if a.mode == "run" and not a.receiver:
        sys.exit("run mode needs --receiver")
    {"build1": build1, "describe": describe, "build2": build2, "gate": gate,
     "run": run, "score": score}[a.mode](a)


if __name__ == "__main__":                                   # spawn-safety (sk3 vLLM rule)
    main()
