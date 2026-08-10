"""
Run the verifiability test capturing FULL model output (reasoning, justification,
all 3 axis labels) and pretty-print every disagreement so we can spot the
pattern in tractability errors.
"""

from __future__ import annotations
import asyncio, json, re, sys
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
sys.path.insert(0, str(ROOT / "scripts"))

from classify_rubric_llama_prompt import build_prompt_for_task, SCHEMA_HINT
from openai import AsyncOpenAI

KEY = (Path.home() / ".openrouter-api-key.txt").read_text().strip()
TEST_RUBRICS_MOD = __import__("test_classifier_verifiability_openrouter")


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


async def main():
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=KEY)
    results = []
    for item in TEST_RUBRICS_MOD.TEST_RUBRICS:
        sys_prompt = build_prompt_for_task(item['task'])
        user_msg = (
            f"PAGE CONTEXT:\n  task: {item['task']}\n  subtask_short: {item['subtask_short']}\n\n"
            f"RUBRIC TO CLASSIFY:\n  name: {item['rubric_name']}\n"
            f"  description: {item['rubric_description']}\n  guidance: {item['rubric_guidance']}\n"
        )
        for attempt in range(2):  # retry once on parse fail
            try:
                resp = await client.chat.completions.create(
                    model="meta-llama/llama-3.3-70b-instruct",
                    messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_msg}],
                    temperature=0.0 if attempt == 0 else 0.2,
                    max_tokens=1024, response_format={"type":"json_object"},
                )
                d = salvage_json(resp.choices[0].message.content or "")
                if d: break
            except: pass
        results.append({"item": item, "out": d})

    # Pretty-print all 12, highlighting disagreements
    print("=" * 100)
    print("FULL OUTPUT — focusing on tractability disagreements")
    print("=" * 100)
    n_disagree = 0
    for r in results:
        item = r["item"]; out = r["out"]
        if not out:
            print(f"\n[NULL] {item['rubric_name']}")
            continue
        v_ok = (out.get("verifiability_type") == item["expected_verif"])
        t_ok = (out.get("tractability")       == item["expected_tract"])
        s_ok = (out.get("specificity")        == item["expected_spec"])
        if t_ok:
            continue
        n_disagree += 1
        print(f"\n--- {item['task']} :: {item['rubric_name']} ---")
        print(f"  RUBRIC desc: {item['rubric_description'][:160]}")
        print(f"")
        print(f"  HAND-LABEL:  verif={item['expected_verif']:<14s}  tract={item['expected_tract']:<20s}  spec={item['expected_spec']}")
        print(f"  MODEL:       verif={out.get('verifiability_type','?'):<14s}  tract={out.get('tractability','?'):<20s}  spec={out.get('specificity','?')}")
        print(f"  TRACT MATCH: {'✓' if t_ok else '✗'}")
        print(f"  reasoning: {out.get('reasoning','')[:300]}")
        print(f"  justification: {out.get('justification','')[:300]}")
    print(f"\n=== {n_disagree} tractability disagreements ===")


if __name__ == "__main__":
    asyncio.run(main())
