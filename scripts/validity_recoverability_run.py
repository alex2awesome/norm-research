"""Recoverability test: given an obfuscated code, can an LLM identify which
rubric it scores? Multiple-choice across all 218 R2 aspects.

For each obfuscated code (1087 codes = 218 R2 × 5 paraphrases):
  - Show the code
  - Show ALL R2 aspect names (with brief descriptions)
  - Ask LLM: which aspect does this code score?
  - Score top-1 accuracy + top-5 accuracy

Run via OpenRouter (cheap, fast). Default model: Qwen3-coder or Claude.

Output:
  runs/validity_full/<run>/recovery_prompts/<key>.txt
  runs/validity_full/<run>/recovery_responses/<key>.json
  runs/validity_full/<run>/recovery_results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from openrouter import chat, make_client


RECOVERY_SYSTEM = """You are given a Python function `score(text: str) -> float` that evaluates peer-review paper texts on a single evaluation rubric. The function has been obfuscated — variable names and comments are stripped — but the LOGIC and any keyword strings are intact.

Your task: identify which of the listed rubrics this code is scoring.

Below is a numbered list of candidate rubrics (one per line). Pick the SINGLE rubric whose meaning best matches what the code actually checks.

Output VALID JSON ONLY:
{"choice": <integer index>, "reasoning": "<one-sentence justification>"}"""


def build_user(code: str, candidates: list[dict]) -> str:
    lines = ["CODE:", "```python", code, "```", "",
             "CANDIDATE RUBRICS:"]
    for i, c in enumerate(candidates):
        lines.append(f"{i}: {c['name']} — {c.get('description', '')[:120]}")
    lines.append("")
    lines.append("Output the JSON now.")
    return "\n".join(lines)


async def recover_one(client, sem, model, key, code, candidates):
    async with sem:
        try:
            user = build_user(code, candidates)
            raw = await chat(client, model, RECOVERY_SYSTEM, user,
                              max_tokens=300)
            # Parse JSON from response
            m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
            if m: raw = m.group(1).strip()
            obj = None
            try: obj = json.loads(raw)
            except json.JSONDecodeError:
                s, e = raw.find("{"), raw.rfind("}")
                if s >= 0 and e > s:
                    obj = json.loads(raw[s:e + 1])
            if obj is None:
                return key, None, "parse fail"
            return key, obj.get("choice"), None
        except Exception as e:
            return key, None, str(e)[:200]


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="full_v1")
    ap.add_argument("--src-subdir", default="codegen_responses_qwen_r2_obfuscated")
    ap.add_argument("--out-subdir", default="recovery_responses_qwen")
    ap.add_argument("--model", default="qwen/qwen3-coder")
    ap.add_argument("--concurrency", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0,
                    help="Only run first N codes (for pilot). 0 = all.")
    args = ap.parse_args()

    base = Path(f"runs/validity_full/{args.run_name}")
    src = base / args.src_subdir
    out_dir = base / args.out_subdir
    out_dir.mkdir(exist_ok=True, parents=True)

    # Load candidates (R2 aspects)
    aspects = json.loads((base / "r2_aspects.json").read_text())
    aspect_by_id = {a["aspect_id"]: a for a in aspects}
    aspect_idx = {a["aspect_id"]: i for i, a in enumerate(aspects)}

    # All obfuscated codes
    files = sorted(src.glob("*.py"))
    if args.limit > 0:
        files = files[:args.limit]
    print(f"recovering {len(files)} obfuscated codes against "
          f"{len(aspects)} candidate aspects")

    # Build jobs
    jobs = []
    for fp in files:
        key = fp.stem  # e.g., "a5__p0"
        code = fp.read_text()
        jobs.append((key, code))

    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)
    coros = [recover_one(client, sem, args.model, k, c, aspects)
             for k, c in jobs]

    import time
    t0 = time.time()
    results = []
    for i, coro in enumerate(asyncio.as_completed(coros)):
        key, choice, err = await coro
        results.append({"key": key, "predicted_choice": choice, "error": err})
        # Save individual response
        (out_dir / f"{key}.json").write_text(
            json.dumps({"predicted_choice": choice, "error": err}, indent=1))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(jobs)} done ({(i+1)/(time.time()-t0):.1f}/s)",
                  flush=True)

    # Score
    n_correct_top1 = 0
    n_predicted = 0
    n_err = 0
    per_aspect = {}  # aspect_id -> [correct_per_paraphrase]
    for r in results:
        if r["error"] or r["predicted_choice"] is None:
            n_err += 1; continue
        aspect_id = r["key"].split("__")[0]  # "a5" from "a5__p0"
        para_idx = int(r["key"].split("__p")[1])
        true_idx = aspect_idx[aspect_id]
        is_correct = (r["predicted_choice"] == true_idx)
        n_predicted += 1
        if is_correct: n_correct_top1 += 1
        per_aspect.setdefault(aspect_id, []).append(is_correct)

    top1_accuracy = n_correct_top1 / max(n_predicted, 1)
    # Per-aspect recoverability: fraction of paraphrases recovered correctly
    aspect_recover_rates = {aid: sum(rs)/len(rs)
                             for aid, rs in per_aspect.items()}
    fully_recovered = sum(1 for r in aspect_recover_rates.values() if r >= 0.6)
    never_recovered = sum(1 for r in aspect_recover_rates.values() if r == 0)

    summary = {
        "n_total": len(jobs),
        "n_predicted": n_predicted,
        "n_errors": n_err,
        "top1_accuracy": top1_accuracy,
        "aspects_with_data": len(aspect_recover_rates),
        "aspects_>=60%_recovered": fully_recovered,
        "aspects_never_recovered": never_recovered,
        "per_aspect_recovery": aspect_recover_rates,
    }
    (base / "recovery_results.json").write_text(json.dumps(summary, indent=1))
    print(f"\n=== RECOVERY RESULTS ===")
    print(f"  total codes: {len(jobs)}")
    print(f"  successful queries: {n_predicted}/{len(jobs)} ({n_err} errors)")
    print(f"  TOP-1 ACCURACY: {top1_accuracy*100:.1f}%")
    print(f"  aspects with ≥60% recovery rate (4/5 paraphrases): {fully_recovered}/{len(aspect_recover_rates)}")
    print(f"  aspects with 0% recovery: {never_recovered}")


if __name__ == "__main__":
    asyncio.run(amain())
