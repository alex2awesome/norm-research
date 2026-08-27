"""Calibration runner for the rubric-pair sameness judge.

Runs the judge prompt (judge_prompt.py) on the hand-labelled calibration set
via OpenRouter Llama-3.3-70B, then compares the model scores to the gold
labels: exact agreement, +/-1 agreement, the confusion matrix, and every
disagreement printed for manual inspection.

Usage: python scripts/calib_judge.py [--tag r1] [--model meta-llama/llama-3.3-70b-instruct]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
OUT = ROOT / "outputs" / "analyses"
sys.path.insert(0, str(ROOT / "scripts"))
from judge_prompt import SYSTEM, build_user, salvage, JUDGE_VERSION

KEY = (Path.home() / ".openrouter-api-key.txt").read_text().strip()


async def judge_one(client, sem, model, row):
    user = build_user(row["task"], row["canonical_a"], row["canonical_b"])
    async with sem:
        for attempt in range(6):
            try:
                resp = await client.chat.completions.create(
                    model=model, temperature=0.0, max_tokens=300,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": user}])
            except Exception as e:
                await asyncio.sleep(2 ** attempt)
                continue
            p = salvage(resp.choices[0].message.content or "")
            if p is not None and "score" in p:
                try:
                    return {**row, "score": int(p["score"]),
                            "judge_reasoning": p.get("reasoning", "")}
                except Exception:
                    pass
        return {**row, "score": None}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct")
    args = ap.parse_args()

    pairs = [json.loads(l) for l in (OUT / "judge_calib.jsonl").open()]
    gold = {json.loads(l)["calib_id"]: json.loads(l)["gold"]
            for l in (OUT / "judge_calib_gold.jsonl").open()}
    print(f"calibrating judge {JUDGE_VERSION} on {len(pairs)} pairs | model={args.model}")

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=KEY)
    sem = asyncio.Semaphore(15)
    t0 = time.perf_counter()
    res = await asyncio.gather(*(judge_one(client, sem, args.model, p) for p in pairs))
    print(f"done in {time.perf_counter()-t0:.0f}s\n")

    res.sort(key=lambda r: r["calib_id"])
    with (OUT / f"calib_judge_{args.tag}.jsonl").open("w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")

    exact = pm1 = ok = 0
    # binary "mergeable" view: top level vs not
    b_tp = b_fp = b_tn = b_fn = 0
    conf = Counter()
    disagree = []
    for r in res:
        # collapse 4-level hand-gold -> 3-level: only "3 = same criterion"
        # maps to mergeable(2); "2 = nearly-same w/ a real difference" -> 1.
        g = {0: 0, 1: 1, 2: 1, 3: 2}[gold[r["calib_id"]]]
        s = r["score"]
        if s is None:
            continue
        ok += 1
        conf[(g, s)] += 1
        if g == s:
            exact += 1
        if abs(g - s) <= 1:
            pm1 += 1
        gb, sb = g >= 2, s >= 2
        b_tp += gb and sb
        b_fp += (not gb) and sb
        b_tn += (not gb) and (not sb)
        b_fn += gb and (not sb)
        if g != s:
            disagree.append((r, g))

    print(f"=== judge {JUDGE_VERSION} vs gold ({ok} scored) ===")
    print(f"  exact agreement : {exact}/{ok} = {exact/ok*100:.0f}%")
    print(f"  within +/-1     : {pm1}/{ok} = {pm1/ok*100:.0f}%")
    print(f"  mergeable (score==2):  TP={b_tp} FP={b_fp} TN={b_tn} FN={b_fn}")
    print(f"\n  confusion (gold row, judge col):")
    print("       j0   j1   j2")
    for g in (0, 1, 2):
        print(f"   g{g} " + " ".join(f"{conf.get((g,s),0):>4}" for s in (0, 1, 2)))

    print(f"\n=== {len(disagree)} disagreements (for manual review) ===")
    for r, g in sorted(disagree, key=lambda x: -abs(x[0]['score'] - x[1])):
        print(f"\n  [{r['calib_id']}] gold={g} judge={r['score']}  cos={r['cos']:.3f}")
        print(f"    A: {r['canonical_a'][:100]}")
        print(f"    B: {r['canonical_b'][:100]}")
        print(f"    judge: {r.get('judge_reasoning','')[:160]}")


if __name__ == "__main__":
    asyncio.run(main())
