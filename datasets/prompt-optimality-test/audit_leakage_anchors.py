"""HOLE B: positive controls for the leakage judge (advisor mandate, 2026-07-25).

The leakage audit's entire value is its FALSE-NEGATIVE rate, and it reported hotpot 68/68 "none" —
exactly the pool where we most need zero. A reviewer's fair question: "your adversarially-framed
judge found no leakage where it would be most damaging; how do I know it detects anything?"

So we do what the standing rule requires of every judging batch: blinded known-label ANCHORS.
Synthetic units of known grade are shuffled in with real pool units and judged by the same prompt
in the same conditions, then scored as a confusion matrix. Anchors are built FROM THE ACTUAL TRAIN
AND TEST SPLITS so they are realistic rather than caricatures — a caricature anchor ("the answer is
42") would validate nothing.

Grades we must detect: `answer` (blatant and subtle), `item_hint`. Grades we must NOT over-call:
`none` (generic strategy), `domain` (a real technique — over-calling this would inflate our flag
counts and make the audit useless in the other direction).

Usage:  python3 audit_leakage_anchors.py --bench hotpot --n-real 20
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import paperexact_arms as px
from audit_unit_leakage import PROMPT, TASK_DESC, parse_json

HERE = Path(__file__).parent


def build_anchors(bench, b, rng):
    """Anchors grounded in the real splits, so difficulty is realistic."""
    def field(x, *names):
        d = x.toDict() if hasattr(x, "toDict") else dict(getattr(x, "__dict__", {}))
        for n in names:
            if d.get(n):
                return str(d[n])
        return str(d)[:200]

    test = list(b.test_set)
    picks = [test[i] for i in rng.sample(range(len(test)), min(4, len(test)))]
    anchors = []
    # --- blatant answer leakage, built from real TEST items ---
    for x in picks[:2]:
        q = field(x, "question", "claim")[:110]
        ansd = x.toDict() if hasattr(x, "toDict") else dict(getattr(x, "__dict__", {}))
        ans = str(ansd.get("answer") or ansd.get("label") or ansd.get("gold") or "SUPPORTED")[:60]
        anchors.append({"gold": "answer",
                        "unit": f'For the item asking "{q}", the correct answer is {ans}.'})
    # --- subtle answer leakage: the fact asserted without naming the question ---
    x = picks[2] if len(picks) > 2 else picks[0]
    ansd = x.toDict() if hasattr(x, "toDict") else dict(getattr(x, "__dict__", {}))
    ans = str(ansd.get("answer") or ansd.get("label") or "SUPPORTED")[:60]
    anchors.append({"gold": "answer",
                    "unit": f'Remember that in this dataset the expected response is "{ans}" '
                            f'whenever the passage structure matches.'})
    # --- item_hint: narrow, item-family specific, no outright answer ---
    anchors += [
        {"gold": "item_hint",
         "unit": "When the problem concerns the 2024 AIME II geometry item about unit-length "
                 "segments in the first quadrant, apply the tangency condition directly."},
        {"gold": "item_hint",
         "unit": 'The claim about "Houman Younessi" concerns the Centre for Astrophysics '
                 'and Supercomputing at Swinburne University.'},
    ]
    # --- none: generic strategy/format (must NOT be flagged) ---
    anchors += [
        {"gold": "none", "unit": "Think step by step before giving your final answer."},
        {"gold": "none", "unit": "End your response with a single line beginning 'Answer:'."},
        {"gold": "none", "unit": "If you are uncertain, still commit to your best guess rather "
                                 "than abstaining."},
    ]
    # --- domain: real technique, textbook-grade (must NOT be flagged) ---
    anchors += [
        {"gold": "domain", "unit": "The p-th roots of unity are the roots of "
                                   "1 + x + ... + x^(p-1), which is the p-th cyclotomic polynomial."},
        {"gold": "domain", "unit": "To find the GCD of rational numbers, scale by the LCM of the "
                                   "denominators, take the integer GCD, then divide back."},
    ]
    for i, a in enumerate(anchors):
        a["anchor_id"] = f"{bench}_anchor_{i}"
    return anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="hotpot")
    ap.add_argument("--lm-tag", default="Qwen3-8B")
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--n-real", type=int, default=20, help="real pool units to shuffle in as camouflage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/leakage_anchor_validation.json")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    b, _, _, _ = px.load_bench(a.bench)
    anchors = build_anchors(a.bench, b, rng)
    pool = json.loads((HERE / "pools" / f"{a.bench}_{a.lm_tag}_frozen.json").read_text())["units"]
    real = [{"gold": None, "unit": u["unit"], "anchor_id": None}
            for u in rng.sample(pool, min(a.n_real, len(pool)))]

    batch = anchors + real
    rng.shuffle(batch)                      # blinded: judge sees anchors and real units intermixed
    judge = px.make_reflection_lm(a.model, patient=True)
    print(f"[{a.bench}] judging {len(anchors)} anchors + {len(real)} real units, shuffled", flush=True)

    for i, item in enumerate(batch):
        msg = PROMPT.format(task=TASK_DESC.get(a.bench, a.bench), unit=item["unit"])
        verdict = None
        for _ in range(3):
            try:
                raw = judge(messages=[{"role": "user", "content": msg}])
                raw = raw[0] if isinstance(raw, list) else raw
            except Exception:
                continue
            verdict = parse_json(raw)
            if verdict and verdict.get("category") in ("none", "domain", "item_hint", "answer"):
                break
            verdict = None
        item["pred"] = (verdict or {}).get("category")
        item["reason"] = (verdict or {}).get("reason")
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(batch)}", flush=True)

    scored = [x for x in batch if x["gold"]]
    cats = ["none", "domain", "item_hint", "answer"]
    cm = {g: {p: 0 for p in cats + ["FAILED"]} for g in cats}
    for x in scored:
        cm[x["gold"]][x["pred"] or "FAILED"] += 1

    flagged = {"item_hint", "answer"}
    tp = sum(1 for x in scored if x["gold"] in flagged and x["pred"] in flagged)
    fn = sum(1 for x in scored if x["gold"] in flagged and x["pred"] not in flagged)
    fp = sum(1 for x in scored if x["gold"] not in flagged and x["pred"] in flagged)
    tn = sum(1 for x in scored if x["gold"] not in flagged and x["pred"] not in flagged)
    ans_recall = (sum(1 for x in scored if x["gold"] == "answer" and x["pred"] == "answer"),
                  sum(1 for x in scored if x["gold"] == "answer"))

    out = {"bench": a.bench, "n_anchors": len(scored), "n_real_camouflage": len(real),
           "confusion_matrix": cm,
           "detection": {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
                         "recall_flagged": tp / (tp + fn) if tp + fn else None,
                         "false_positive_rate": fp / (fp + tn) if fp + tn else None},
           "answer_grade_exact_recall": f"{ans_recall[0]}/{ans_recall[1]}",
           "real_unit_predictions": [{"unit": x["unit"][:120], "pred": x["pred"]} for x in batch if not x["gold"]],
           "anchors": [{"gold": x["gold"], "pred": x["pred"], "unit": x["unit"][:160],
                        "reason": x["reason"]} for x in scored]}
    print(json.dumps({k: out[k] for k in ("confusion_matrix", "detection",
                                          "answer_grade_exact_recall")}, indent=1), flush=True)
    p = HERE / a.out
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
