"""
A/B test: v5 (current) vs v5-verbose extractor on the same set of pages
via OpenRouter Llama-3.3-70B. Compares description/guidance length,
faithfulness to source, and rubric structure.
"""

from __future__ import annotations
import argparse, asyncio, json, re, sys, time
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
sys.path.insert(0, str(ROOT / "scripts"))

from extract_rubric_features_v5_prompt import build_prompt_for_task as build_v5
from extract_rubric_features_v5verbose_prompt import build_prompt_for_task as build_v5verbose
from extract_rubric_features import load_clean_text

OPENROUTER_KEY = (Path.home() / ".openrouter-api-key.txt").read_text().strip()
OUT_DIR = ROOT / "logs/llama_prompt_test/v5verbose_ab"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_HINT = """
You MUST respond with a single JSON object exactly matching this shape:

{
  "reasoning": "1-3 sentences walking through the work-test.",
  "orientation": one of ["research_article","academic_page","how_to","formal_guideline","blog_post","dataset","tutorial","textbook_excerpt","professional_standard","contest_criteria","stylebook","course_syllabus","wiki","forum_post","news_article","error","other"],
  "intended_audience": "string",
  "subtask_short": "≤8 word label",
  "subtask_description": "1-2 sentences",
  "subtask_keywords": ["3", "to", "7", "lowercase_snake_case"],
  "subtask_breadth": one of ["very_narrow","narrow","moderate","broad","very_broad"],
  "error": null  OR  "short reason string",
  "rubrics_metrics": [
    {"name": "...", "description": "...", "guidance": "...", "inputs": ["≥2 specific noun phrases"]},
    ...
  ]
}

Return ONLY the JSON object. No prose outside it. No markdown fences.
"""

# 4 pages spanning tasks (mix of dense legal, craft, regulatory)
TEST_PAGES = [
    ("patents", "patents/online-rubrics/raw/waveh5_mpep_chapter_2100.html", 12_000),
    ("creative-writing", "creative-writing/online-rubrics/raw/km_weiland_novel_checklist.md", 12_000),
    ("press-releases", "press-releases/online-rubrics/raw/sec_regulation_fd_ecfr.md", 12_000),
    ("peer-review", "peer-review/online-rubrics/raw/aaai_23_review_process.html", 12_000),
]


def salvage_json(raw: str):
    if not raw or not raw.strip(): return None
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try: return json.loads(s)
    except: pass
    start = s.find("{")
    if start < 0: return None
    for end in range(len(s), start, -1):
        if s[end-1] == "}":
            try: return json.loads(s[start:end])
            except: continue
    return None


async def call_one(client, sys_prompt: str, user_msg: str) -> dict:
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct",
        messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_msg}],
        temperature=0.0, max_tokens=4096, response_format={"type":"json_object"},
    )
    elapsed = time.perf_counter() - t0
    raw = resp.choices[0].message.content or ""
    parsed = salvage_json(raw)
    return {"parsed": parsed, "raw": raw, "elapsed": elapsed,
            "in_tok": resp.usage.prompt_tokens, "out_tok": resp.usage.completion_tokens}


async def main():
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)

    for task, rel, max_chars in TEST_PAGES:
        fp = ROOT / "datasets" / rel
        text, _ = load_clean_text(fp)
        user_msg = (f"PARENT TASK CONTEXT: {task}\nSOURCE FILE: {fp.name}\n\nPAGE TEXT:\n{text[:max_chars]}\n\n" + SCHEMA_HINT)

        print(f"\n=== {task}: {fp.name} ({len(text):,} chars; using first {max_chars:,}) ===")

        # v5 baseline
        sys_v5 = build_v5(task)
        r5 = await call_one(client, sys_v5, user_msg)
        # v5 verbose
        sys_v5v = build_v5verbose(task)
        r5v = await call_one(client, sys_v5v, user_msg)

        for label, r in [("v5", r5), ("v5-verbose", r5v)]:
            ex = r["parsed"]
            if ex is None:
                print(f"  [{label}] PARSE FAIL ({r['out_tok']} out_tok)")
                continue
            rubrics = ex.get("rubrics_metrics", [])
            desc_lens = [len(rm.get("description","")) for rm in rubrics]
            guid_lens = [len(rm.get("guidance","")) for rm in rubrics]
            inp_counts = [len(rm.get("inputs",[])) for rm in rubrics]
            print(f"  [{label}] orient={ex.get('orientation')}  n_rubrics={len(rubrics)}  "
                  f"avg(desc)={sum(desc_lens)//max(1,len(desc_lens))}c  "
                  f"avg(guid)={sum(guid_lens)//max(1,len(guid_lens))}c  "
                  f"avg(inputs)={sum(inp_counts)/max(1,len(inp_counts)):.1f}  "
                  f"out_tok={r['out_tok']}")
            # Show first rubric's description in full
            if rubrics:
                rm = rubrics[0]
                print(f"      first rubric: \"{rm.get('name','')}\"")
                print(f"        desc: {rm.get('description','')[:300]}{'…' if len(rm.get('description',''))>300 else ''}")
                print(f"        guid: {rm.get('guidance','')[:200]}{'…' if len(rm.get('guidance',''))>200 else ''}")
                print(f"        inputs: {rm.get('inputs',[])[:4]}")
        # Save outputs for further inspection
        (OUT_DIR / f"{fp.stem}.v5.json").write_text(json.dumps(r5["parsed"] or {"_raw": r5["raw"][:1000]}, indent=2))
        (OUT_DIR / f"{fp.stem}.v5verbose.json").write_text(json.dumps(r5v["parsed"] or {"_raw": r5v["raw"][:1000]}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
