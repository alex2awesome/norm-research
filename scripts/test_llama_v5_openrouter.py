"""
Test the v3 rubric-extraction prompt with Llama-3.3-70B via OpenRouter.

Differs from test_llama_v2_openrouter.py:
  - Uses v3 prompt + JSON_SCHEMA_V5 (reasoning field first).
  - DROPS response_format entirely — OpenRouter routes Llama-3.3 to several
    providers, not all of which honor json_schema/json_object (we got 4/6
    empty responses on v2 retry). The schema is described in the user prompt
    instead, and we salvage the JSON from whatever shape the model returns.
  - Adds a one-shot retry on empty/malformed response, this time pinning the
    provider order to known-good ones.
  - Includes a small JSON salvage pass (markdown fences, leading prose).

Usage: python test_llama_v3_openrouter.py [--max-chars 12000]
"""

from __future__ import annotations
import argparse, asyncio, json, re, sys, time
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
sys.path.insert(0, str(ROOT / "scripts"))

from extract_rubric_features_v5_prompt import build_prompt_for_task, JSON_SCHEMA_V5

OPENROUTER_KEY = (Path.home() / ".openrouter-api-key.txt").read_text().strip()
SAMPLES_PATH   = ROOT / "logs/llama_prompt_test/v1_samples.json"
OUT_DIR        = ROOT / "logs/llama_prompt_test/v5_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_HINT = """
You MUST respond with a single JSON object exactly matching this shape (NOTE: `reasoning` comes FIRST):

{
  "reasoning": "1-3 sentences walking through the work-test before extracting. What kind of page? What is the work-of-this-task? Which items are work-criteria (extract) vs. service/policy/transactional/meta-artifact items (skip)? Name any borderline items here.",
  "orientation": one of ["research_article","academic_page","how_to","formal_guideline","blog_post","dataset","tutorial","textbook_excerpt","professional_standard","contest_criteria","stylebook","course_syllabus","wiki","forum_post","news_article","error","other"],
  "intended_audience": "string",
  "subtask_short": "≤8 word label",
  "subtask_description": "1-2 sentences",
  "subtask_keywords": ["3", "to", "7", "lowercase_snake_case"],
  "subtask_breadth": one of ["very_narrow","narrow","moderate","broad","very_broad"],
  "error": null  OR  "short reason string",
  "rubrics_metrics": [
    {"name": "short label", "description": "verbatim or close paraphrase", "guidance": "surrounding explanation or empty string"},
    ...
  ]
}

Return ONLY the JSON object. No prose outside it. No markdown fences.
"""


def salvage_json(raw: str) -> dict | None:
    """Best-effort JSON extraction from a possibly-prose-wrapped response."""
    if not raw or not raw.strip():
        return None
    s = raw.strip()
    # Strip markdown fences
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # Quick try
    try:
        return json.loads(s)
    except Exception:
        pass
    # Find the first { ... } that parses (greedy but fallible)
    start = s.find("{")
    if start < 0:
        return None
    # Try progressively shorter substrings ending at the last } we find
    for end in range(len(s), start, -1):
        if s[end-1] == "}":
            try:
                return json.loads(s[start:end])
            except Exception:
                continue
    return None


async def call_llama(client, text: str, task: str, source_file: str, max_chars: int,
                     pin_provider: bool = False) -> dict:
    user_msg = (
        f"PARENT TASK CONTEXT: This page was collected for the broader task: {task}\n\n"
        f"SOURCE FILE: {source_file}\n\n"
        f"PAGE TEXT:\n{text[:max_chars]}\n\n"
        + SCHEMA_HINT
    )
    extra: dict = {}
    if pin_provider:
        # Pin to providers known to behave on Llama-3.3 70B
        extra["extra_body"] = {"provider": {"order": ["fireworks", "together", "deepinfra"],
                                            "allow_fallbacks": False}}
    t0 = time.perf_counter()
    sys_prompt = build_prompt_for_task(task)
    try:
        resp = await client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=4096,
            **extra,
        )
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "elapsed_s": time.perf_counter() - t0}
    elapsed = time.perf_counter() - t0
    raw = resp.choices[0].message.content or ""
    parsed = salvage_json(raw)
    if parsed is None:
        return {"ok": False, "error": "json_salvage_failed", "elapsed_s": elapsed, "raw": raw}
    return {"ok": True, "extracted": parsed, "elapsed_s": elapsed,
            "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
            "raw": raw,
            "provider_pinned": pin_provider}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=12_000)
    args = ap.parse_args()

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)

    samples = json.loads(SAMPLES_PATH.read_text())
    print(f"loaded {len(samples)} samples; max_chars={args.max_chars}")

    summary = []
    for s in samples:
        rel = s["rel"]; task = rel.split("/")[0]; sf = Path(rel).name
        print(f"\n=== {sf} ({s['len']:,} chars; task={task}) ===")
        print(f"    {s['desc']}")
        r = await call_llama(client, s["text"], task, sf, args.max_chars, pin_provider=False)
        if not r["ok"]:
            print(f"    1st pass failed ({r['error']}); retrying with provider pinned...")
            r = await call_llama(client, s["text"], task, sf, args.max_chars, pin_provider=True)
        out_path = OUT_DIR / f"{sf}.v5.json"
        out_path.write_text(json.dumps(r, indent=2))
        if r["ok"]:
            ex = r["extracted"]
            n_rubrics = len(ex.get("rubrics_metrics", []))
            orient = ex.get("orientation"); err = ex.get("error")
            print(f"    -> orientation={orient}  error={err}  n_rubrics={n_rubrics}  elapsed={r['elapsed_s']:.1f}s  pinned={r.get('provider_pinned',False)}")
            print(f"    REASONING: {ex.get('reasoning','')[:400]}")
            for i, m in enumerate(ex.get("rubrics_metrics", [])[:8]):
                print(f"       [{i}] {m['name'][:80]}")
                d = m.get("description", "")
                print(f"           {d[:120]}{'...' if len(d) > 120 else ''}")
            if n_rubrics > 8:
                print(f"       ... [{n_rubrics-8} more]")
            summary.append({"file": sf, "orient": orient, "n_rubrics": n_rubrics, "desc": s["desc"],
                            "reasoning": ex.get("reasoning", "")[:200]})
        else:
            print(f"    !! ERROR: {r['error']}")
            summary.append({"file": sf, "orient": "ERROR", "n_rubrics": -1, "desc": s["desc"], "error": r["error"]})

    print("\n\n========== SUMMARY (v5 prompt + task-specific few-shots, Llama-3.3-70B via OpenRouter) ==========")
    for s in summary:
        print(f"  n={s['n_rubrics']:>4}  orient={s['orient']:<22s}  {s['file']}")
        if "reasoning" in s:
            print(f"    R: {s['reasoning']}")


if __name__ == "__main__":
    asyncio.run(main())
