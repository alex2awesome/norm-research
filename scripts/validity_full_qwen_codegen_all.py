"""Run Qwen code-gen on ALL R1 prompts (replacing Llama as primary coder).

Reuses scripts/validity_full_codegen_prep.py prompt files.

Output:
  runs/validity_full/<run>/codegen_responses_qwen_all/<key>.py
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


CODEGEN_SYSTEM = """You write a single Python function `score(text: str) -> float` returning a value in [0.0, 1.0] indicating how well `text` satisfies a single evaluation rubric for peer-review papers.

Strict requirements:
- Return 1.0 if the rubric is fully satisfied, 0.0 if clearly violated, intermediate for partial. Return 0.5 if the rubric is not applicable to this text or you cannot tell.
- Use ONLY the Python standard library. Do NOT import third-party packages.
- The function must never raise. Wrap risky logic in try/except and return 0.5 on failure.
- Handle empty / very short text gracefully — return 0.5 if text is implausibly short.
- The function must run deterministically without network or filesystem access.

Output ONLY the Python code. No markdown fences, no commentary. Start with `def score` or with the `import` lines you need."""


def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


async def gen_one(client, sem, model, key, user_text, out_dir, retry_count=2):
    out_path = out_dir / f"{key}.py"
    if out_path.exists() and out_path.stat().st_size > 0:
        return key, "cached"
    async with sem:
        for attempt in range(retry_count):
            try:
                raw = await chat(client, model, CODEGEN_SYSTEM, user_text, max_tokens=2500)
                code = extract_code(raw)
                if code:
                    out_path.write_text(code)
                    return key, None
            except Exception as e:
                if attempt == retry_count - 1:
                    return key, str(e)[:200]
                await asyncio.sleep(2 * (attempt + 1))
        return key, "no code returned"


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="full_v1")
    ap.add_argument("--prompt-subdir", default="codegen_prompts")
    ap.add_argument("--out-subdir", default="codegen_responses_qwen_all")
    ap.add_argument("--model", default="qwen/qwen3-coder")
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()

    base = Path(f"runs/validity_full/{args.run_name}")
    prompt_dir = base / args.prompt_subdir
    out_dir = base / args.out_subdir
    out_dir.mkdir(exist_ok=True, parents=True)

    prompts = sorted(prompt_dir.glob("*.txt"))
    print(f"prompts: {len(prompts)}")

    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)

    jobs = []
    for pf in prompts:
        text = pf.read_text()
        if "\n=== USER ===\n" in text:
            _sys, user = text.split("\n=== USER ===\n", 1)
        else:
            user = text
        jobs.append(gen_one(client, sem, args.model, pf.stem, user.strip(), out_dir))

    n_ok = n_cached = n_err = 0
    errors = []
    import time
    t0 = time.time()
    for i, coro in enumerate(asyncio.as_completed(jobs)):
        key, err = await coro
        if err == "cached": n_cached += 1
        elif err is None: n_ok += 1
        else: n_err += 1; errors.append((key, err))
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{len(prompts)} done ({rate:.1f}/s, ok={n_ok}, "
                  f"cached={n_cached}, err={n_err})", flush=True)
    print(f"\nDONE: ok={n_ok}, cached={n_cached}, err={n_err}")
    for k, e in errors[:10]: print(f"  ERR {k}: {e[:80]}")


if __name__ == "__main__":
    asyncio.run(amain())
