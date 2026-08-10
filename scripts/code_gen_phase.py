"""Code-gen phase of the validity pilot.

For each rubric, ask K coding models to write `score(text: str) -> float` in
[0,1]. Save raw generated code (so we can audit later for obfuscation /
faithfulness checks). Then code_exec_phase.py runs each generated function
against the datapoints.

MTMM framing: the K coding models should be from DIFFERENT model families
(convergent validity only works if the methods are independent).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

# `scripts/` is the package dir
sys.path.insert(0, str(Path(__file__).parent))
from openrouter import chat, make_client

CODE_GEN_SYSTEM = """You write a single Python function `score(text: str) -> float` that returns a score in [0.0, 1.0] indicating how well `text` satisfies a single evaluation rubric.

Constraints:
- `text` is a string (a code snippet, a review comment, a press release, etc.).
- Return 1.0 if the rubric is fully satisfied, 0.0 if clearly violated, 0.5 if not applicable or genuinely uncertain.
- Use ONLY the Python standard library. Do NOT import third-party packages.
- The function must run deterministically without network or filesystem access.
- The function must handle empty / malformed input gracefully -- never raise; wrap risky logic in try/except and return 0.5 on failure.
- If the rubric is fundamentally not code-checkable (e.g., "the writing has voice"), DO NOT fake it -- write a function that returns 0.5 and a one-line comment explaining why.

Output ONLY the Python code, no markdown fences, no commentary. Start with `def score` or with imports."""

CODE_GEN_USER = """Rubric: "{name}"
Description: {description}

Write the score(text) function."""


# Three coding models from three different families for MTMM convergent validity.
DEFAULT_CODE_MODELS = [
    "anthropic/claude-sonnet-4.5",
    "deepseek/deepseek-chat-v3",
    "qwen/qwen-2.5-coder-32b-instruct",
]


def extract_code(text: str) -> str:
    """Strip markdown fences if the model put them in."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


async def gen_one(client, sem, model, rubric):
    async with sem:
        user = CODE_GEN_USER.format(
            name=rubric["name"],
            description=rubric.get("description", ""),
        )
        try:
            raw = await chat(client, model, CODE_GEN_SYSTEM, user,
                              max_tokens=1500)
            code = extract_code(raw)
            return {"rubric_id": rubric["family_id"],
                    "rubric_name": rubric["name"],
                    "model": model, "code": code, "raw": raw, "ok": True}
        except Exception as e:
            return {"rubric_id": rubric["family_id"],
                    "rubric_name": rubric["name"],
                    "model": model, "code": "", "raw": "",
                    "ok": False, "error": str(e)[:200]}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubrics", required=True,
                    help="JSON file from metric_test_pilot.py (rubrics.json)")
    ap.add_argument("--models", nargs="+", default=DEFAULT_CODE_MODELS)
    ap.add_argument("--output", required=True,
                    help="JSONL file to write {rubric_id, model, code, ...}")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N rubrics (0 = all)")
    args = ap.parse_args()

    rubrics = json.loads(Path(args.rubrics).read_text())
    if args.limit > 0:
        rubrics = rubrics[:args.limit]
    print(f"code-gen on {len(rubrics)} rubrics x {len(args.models)} models "
          f"= {len(rubrics) * len(args.models)} calls "
          f"(concurrency {args.concurrency})", flush=True)

    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)

    jobs = [(model, r) for model in args.models for r in rubrics]
    results = await asyncio.gather(*(gen_one(client, sem, m, r)
                                     for m, r in jobs))

    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_ok = sum(1 for r in results if r["ok"])
    print(f"done: {n_ok}/{len(results)} ok -> {args.output}", flush=True)
    if n_ok < len(results):
        print("failures:")
        for r in results:
            if not r["ok"]:
                print(f"  - {r['rubric_name'][:60]} / {r['model']}: "
                      f"{r.get('error', '')[:120]}")


if __name__ == "__main__":
    asyncio.run(main())
