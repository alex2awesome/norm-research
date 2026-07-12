#!/usr/bin/env python3
"""GEPA prompt-mutation loop for the L0 clustering prompt, tuned FOR GLM-4.7.

Why: the Opus-tuned L0 rule ("Do NOT over-merge; when uncertain keep separate") makes GLM-4.7
over-split (pilot: 88% singletons, only 15% recall on v6-same pairs, 100% precision). GLM-4.7 needs
the opposite bias. This loop rewrites the RULE TEXT itself (genuine GEPA prompt mutation, not the
one-directional self-refine), scoring each variant's recall/precision on a fixed dev set of kNN
batches, keeping the winner.

GLM-sparing: ~N_DEV_BATCHES * (1 + N_ROUNDS) clusterer calls + N_ROUNDS mutator calls + a few test.
Dev forms (seed=7) and test forms (seed=99) are disjoint by construction.

Usage:
  GLMCLUSTER_KEY=~/.z-ai-api-key-alexander-spangher.txt \
  python -m methods.metric_implementer.experiments.gepa_cluster_prompt --task peer-review
"""
from __future__ import annotations
import argparse, json, os, sys, time
from itertools import combinations
from statistics import mean

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from methods.metric_implementer.experiments.glm_cluster import (  # reuse validated internals
    glm_call, load_forms, build_knn, build_batches, parse_groups, _VERDICTS,
)

# The baseline rule (the one that over-splits on GLM-4.7).
BASELINE_RULE = (
    'Group by the L0 rule: two statements go in the SAME group ONLY IF they are the SAME '
    'criterion restated in different words (identical evaluation, just rephrased). Statements '
    'that are merely RELATED but are DIFFERENT criteria, or are unrelated, MUST be in different '
    'groups. Do NOT over-merge — when uncertain, keep them separate.'
)


def make_prompt(rule_text, task, items):
    listing = "\n".join(f"{i}. {t}" for i, t in enumerate(items))
    return (f"Below are {len(items)} evaluative rubric statements about {task}. {rule_text}\n\n{listing}\n\n"
            'Reply with ONLY JSON: {"groups": [[indices 0-based], ...]}; every index in exactly one group.')


def load_pairs(keys, task):
    """One-time: v6 pairs among these keys -> {frozenset({ka,kb}): score}."""
    kset = set(keys)
    pairs = {}
    with open(_VERDICTS) as f:
        for line in f:
            o = json.loads(line)
            if o.get("task") != task:
                continue
            ka, kb, sc = o.get("key_a"), o.get("key_b"), o.get("score")
            if ka in kset and kb in kset and sc in (0, 2):
                pairs[frozenset((ka, kb))] = sc
    return pairs


def score_prompt(rule_text, task, batches, texts, keys, pairs, key2text, model, max_mistakes=8):
    """Per-batch recall (score-2 kept together) + precision (score-0 kept apart). Returns dict +
    list of split true-same mistakes (with text) for the mutator."""
    recs, precs = [], []
    split = []  # (text_a, text_b)
    n_s2 = n_s0 = 0
    for mem in batches:
        items = [texts[i] for i in mem]
        try:
            g = parse_groups(glm_call(model, make_prompt(rule_text, task, items)), len(items))
        except Exception:
            g = None
        if g is None:
            g = [[j] for j in range(len(items))]
        g_of = {}
        for gi, grp in enumerate(g):
            for j in grp:
                g_of[j] = gi
        s2 = s0 = tog2 = sep0 = 0
        for a, b in combinations(range(len(mem)), 2):
            sc = pairs.get(frozenset((keys[mem[a]], keys[mem[b]])))
            if sc is None:
                continue
            same = g_of.get(a) == g_of.get(b)
            if sc == 2:
                s2 += 1; tog2 += same; n_s2 += 1
                if (not same) and len(split) < max_mistakes:
                    split.append((key2text[keys[mem[a]]], key2text[keys[mem[b]]]))
            elif sc == 0:
                s0 += 1; sep0 += (not same); n_s0 += 1
        if s2:
            recs.append(tog2 / s2)
        if s0:
            precs.append(sep0 / s0)
    return {"recall": mean(recs) if recs else 0.0, "precision": mean(precs) if precs else 1.0,
            "n_s2": n_s2, "n_s0": n_s0, "split_mistakes": split}


def mutate(rule_text, sc, model):
    mistakes = "\n".join(f'- "{a}"  <SHOULD MERGE WITH>  "{b}"'
                         for a, b in sc["split_mistakes"][:6]) or "(none captured this round)"
    prompt = (
        'You are optimizing a clustering INSTRUCTION used by an LLM to group rubric statements.\n'
        'Current instruction:\n"""\n' + rule_text + '\n"""\n\n'
        f'On held-out batches it gets recall={sc["recall"]:.2f} (true-paraphrase pairs correctly '
        f'merged) and precision={sc["precision"]:.2f} (truly-different pairs correctly kept apart).\n\n'
        'These TRUE-SAME paraphrase pairs were WRONGLY kept SEPARATE (the model was too conservative):\n'
        f'{mistakes}\n\n'
        'Revise the INSTRUCTION so the model merges more genuine paraphrases (same criterion, just '
        'rephrased) while STILL keeping truly-different criteria apart. Keep it 2-4 sentences. Be '
        'specific about what counts as "same criterion restated". Output ONLY the new instruction text, '
        'no quotes, no preamble.')
    out = glm_call(model, prompt, max_tokens=450, temp=0.5)
    out = out.strip().strip('"').strip()
    # strip a leading "New instruction:" style preamble if present
    if ":" in out.split("\n")[0] and len(out.split("\n")[0]) < 40:
        out = "\n".join(out.split("\n")[1:]).strip()
    return out


def build_dev_batches(keys, texts, task, n_batches, batch, seed):
    """Exactly n_batches kNN-anchored batches of `batch` forms each — predictable, GLM-sparing (no
    singleton sweep). Anchors spread evenly so batches sample different regions of the form space."""
    knn = build_knn(texts, keys, task, "tfidf", batch)
    n = len(keys)
    anchors = np.linspace(0, n - 1, n_batches).astype(int)
    return [[int(x) for x in knn[a][:batch]] for a in anchors]


def prep(task, n_forms, seed, n_batches):
    keys, texts = load_forms(task, n_forms, biased=True, seed=seed)
    batches = build_dev_batches(keys, texts, task, n_batches, 30, seed)
    pairs = load_pairs(keys, task)
    key2text = {k: t for k, t in zip(keys, texts)}
    print(f"[{task}] seed={seed}: {len(keys)} forms, {len(batches)} batches x30, {len(pairs)} v6 pairs",
          flush=True)
    return batches, texts, keys, pairs, key2text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer-review")
    ap.add_argument("--model", default="glm-4.7")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--dev-forms", type=int, default=240)
    ap.add_argument("--dev-batches", type=int, default=5)
    ap.add_argument("--test-forms", type=int, default=150)
    ap.add_argument("--test-batches", type=int, default=4)
    ap.add_argument("--prec-floor", type=float, default=0.90)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or f"outputs/analyses/gepa_prompt_{a.task}.json"

    dev = prep(a.task, a.dev_forms, seed=7, n_batches=a.dev_batches)
    test = prep(a.task, a.test_forms, seed=99, n_batches=a.test_batches)

    rule = BASELINE_RULE
    t0 = time.time()
    best = score_prompt(rule, a.task, *dev, a.model)
    print(f"[r0 baseline] recall={best['recall']:.3f} prec={best['precision']:.3f} "
          f"(n_s2={best['n_s2']} n_s0={best['n_s0']})  [{time.time()-t0:.0f}s]")
    history = [{"round": 0, "rule": rule, **{k: best[k] for k in ("recall", "precision")}}]

    for r in range(1, a.rounds + 1):
        new_rule = mutate(rule, best, a.model)
        sc = score_prompt(new_rule, a.task, *dev, a.model)
        kept = sc["recall"] > best["recall"] and sc["precision"] >= a.prec_floor
        print(f"[r{r}] recall={sc['recall']:.3f} prec={sc['precision']:.3f} "
              f"{'KEEP' if kept else 'revert'} (prev best recall={best['recall']:.3f})  "
              f"[{time.time()-t0:.0f}s]")
        history.append({"round": r, "rule": new_rule, "kept": kept,
                        **{k: sc[k] for k in ("recall", "precision")}})
        if kept:
            rule, best = new_rule, sc

    # held-out test eval of the winning rule
    tst = score_prompt(rule, a.task, *test, a.model)
    base_tst = score_prompt(BASELINE_RULE, a.task, *test, a.model)
    print(f"\n[TEST seed=99] winning recall={tst['recall']:.3f} prec={tst['precision']:.3f}  "
          f"vs baseline recall={base_tst['recall']:.3f} prec={base_tst['precision']:.3f}")

    json.dump({"task": a.task, "model": a.model, "baseline_rule": BASELINE_RULE,
               "winning_rule": rule, "history": history,
               "test": {"winning": {k: tst[k] for k in ("recall", "precision")},
                        "baseline": {k: base_tst[k] for k in ("recall", "precision")}},
               "elapsed_s": round(time.time() - t0, 1)}, open(out, "w"), indent=2)
    print(f"wrote {out}")
    print(f"\nWINNING RULE:\n{rule}")


if __name__ == "__main__":
    main()
