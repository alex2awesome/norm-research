#!/usr/bin/env python3
"""GEPA loop that tunes GLM-4.7's PAIRWISE same/different prompt toward the GLM-5.2 arbiter labels
(the 500 arbiter_pairs/labels). This is Stage-1(B) calibration: distill the arbiter's pairwise
judgment into GLM-4.7's prompt so GLM-4.7 can judge pairs at scale (free) and feed correlation clustering.

Scorer = F1/recall/precision of GLM-4.7's labels vs GLM-5.2 gold on dev; mutate on mistakes; held-out test.
"""
import json, os, sys, re, time, random
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from methods.metric_implementer.experiments.glm_cluster import glm_call

PAIRS = "outputs/analyses/arbiter_pairs.jsonl"
LABELS = "outputs/analyses/arbiter_labels.jsonl"
MODEL = "glm-4.7"
BATCH = 12
DEV_N = 300

BASELINE = ("For each pair of peer-review rubric statements, decide if they express the SAME evaluation "
            "criterion (one is merely a rephrasing of the other — identical judgment in different words) or "
            "DIFFERENT criteria. Label: 2 = SAME (rephrased), 1 = related but genuinely different / "
            "borderline, 0 = DIFFERENT. Be conservative: only 2 if truly the same judgment.")


def judge_prompt(rule, batch):
    lines = [f"[{p['pid']}] A: \"{p['text_a']}\" | B: \"{p['text_b']}\"" for p in batch]
    return (rule + "\n\nJudge each pair independently. Return ONLY a JSON array, one entry per pair in "
            'order: [{"pid":0,"label":2},...]\n\nPairs:\n' + "\n".join(lines))


def parse(txt):
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        return {}
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return {}
    return {int(e["pid"]): int(e["label"]) for e in arr if "pid" in e and "label" in e}


def score(rule, pairset, gold):
    pred = {}
    for bi in range(0, len(pairset), BATCH):
        batch = pairset[bi:bi + BATCH]
        for attempt in range(3):
            try:
                pred.update(parse(glm_call(MODEL, judge_prompt(rule, batch), max_tokens=600,
                                           temp=0.0 if attempt == 0 else 0.3)))
                if all(p["pid"] in pred for p in batch):
                    break
            except Exception:
                pass
    tp = fp = fn = 0; mistakes = []; correct = 0; n = 0
    for p in pairset:
        g = gold.get(p["pid"]); pr = pred.get(p["pid"])
        if g is None or pr is None:
            continue
        n += 1; correct += (g == pr)
        if g == 2 and pr == 2:
            tp += 1
        elif pr == 2 and g != 2:
            fp += 1; mistakes.append((p["text_a"], p["text_b"], g, pr))
        elif g == 2 and pr != 2:
            fn += 1; mistakes.append((p["text_a"], p["text_b"], g, pr))
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"f1": f1, "recall": rec, "precision": prec, "acc": correct / n if n else 0,
            "n": n, "mistakes": mistakes}


def mutate(rule, sc, model):
    ex = "\n".join(f'- "{a}" vs "{b}": arbiter_says={g}, you_said={pr}'
                   for a, b, g, pr in sc["mistakes"][:6]) or "(none)"
    prompt = ("You are optimizing a pairwise same/different instruction for rubric statements.\n"
              "Current instruction:\n\"\"\"\n" + rule + "\n\"\"\"\n"
              f'F1={sc["f1"]:.2f} recall={sc["recall"]:.2f} precision={sc["precision"]:.2f}.\n'
              "These pairs were judged wrong (arbiter = ground truth):\n" + ex + "\n\n"
              "Revise the instruction (2-4 sentences) so the model matches the arbiter better — merging "
              "more genuine paraphrases (fix low recall) without wrongly merging different criteria. "
              "Output ONLY the new instruction text, no quotes, no preamble.")
    out = glm_call(model, prompt, max_tokens=450, temp=0.5).strip().strip('"')
    if ":" in out.split("\n")[0] and len(out.split("\n")[0]) < 40:
        out = "\n".join(out.split("\n")[1:]).strip()
    return out


def main():
    pairs = {json.loads(l)["pid"]: json.loads(l) for l in open(PAIRS)}
    gold = {json.loads(l)["pid"]: json.loads(l)["label"] for l in open(LABELS)}
    pids = [p for p in pairs if p in gold]
    random.Random(7).shuffle(pids)
    dev = [pairs[i] for i in pids[:DEV_N]]
    test = [pairs[i] for i in pids[DEV_N:]]
    print(f"dev={len(dev)} test={len(test)} | gold dist dev: "
          f"same={sum(gold[p['pid']]==2 for p in dev)} diff={sum(gold[p['pid']]==0 for p in dev)}", flush=True)

    rule = BASELINE
    t0 = time.time()
    best = score(rule, dev, gold)
    print(f"[r0 baseline] f1={best['f1']:.3f} recall={best['recall']:.3f} prec={best['precision']:.3f} "
          f"acc={best['acc']:.3f}  [{time.time()-t0:.0f}s]", flush=True)
    hist = [{"round": 0, "f1": best["f1"], "recall": best["recall"], "precision": best["precision"]}]
    for r in range(1, 5):
        new = mutate(rule, best, "glm-5")          # mutator = GLM-5.2 (few calls, smarter)
        sc = score(new, dev, gold)
        keep = sc["f1"] > best["f1"]
        print(f"[r{r}] f1={sc['f1']:.3f} recall={sc['recall']:.3f} prec={sc['precision']:.3f} "
              f"{'KEEP' if keep else 'revert'}  [{time.time()-t0:.0f}s]", flush=True)
        hist.append({"round": r, "f1": sc["f1"], "recall": sc["recall"], "precision": sc["precision"], "kept": keep})
        if keep:
            rule, best = new, sc
    tst = score(rule, test, gold)
    base_tst = score(BASELINE, test, gold)
    print(f"\n[TEST] winning f1={tst['f1']:.3f} recall={tst['recall']:.3f} prec={tst['precision']:.3f} | "
          f"baseline f1={base_tst['f1']:.3f} recall={base_tst['recall']:.3f}", flush=True)
    json.dump({"model": MODEL, "baseline": BASELINE, "winning": rule, "history": hist,
               "test": {"winning": {k: tst[k] for k in ('f1', 'recall', 'precision', 'acc')},
                        "baseline": {k: base_tst[k] for k in ('f1', 'recall', 'precision', 'acc')}}},
              open("outputs/analyses/pairwise_gepa_peer-review.json", "w"), indent=2)
    print(f"\nWINNING PAIRWISE PROMPT:\n{rule}", flush=True)


if __name__ == "__main__":
    main()
