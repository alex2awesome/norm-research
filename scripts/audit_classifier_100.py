"""
Run the Llama classifier on 100 stratified rubrics, capture full output,
flag suspicious label combinations for manual review.
"""

from __future__ import annotations
import asyncio, json, re, sys, time
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
sys.path.insert(0, str(ROOT / "scripts"))

from classify_rubric_llama_prompt import build_prompt_for_task
from openai import AsyncOpenAI

KEY = (Path.home() / ".openrouter-api-key.txt").read_text().strip()
OUT_DIR = ROOT / "logs/rubric_labeling/audit_100"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def salvage_json(raw: str):
    if not raw: return None
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


async def call_one(client, sem, item):
    sys_prompt = build_prompt_for_task(item['task'])
    user_msg = (
        f"PAGE CONTEXT:\n  task: {item['task']}\n  page_id: {item['page_id']}\n  subtask_short: {item.get('subtask_short','') or ''}\n\n"
        f"RUBRIC TO CLASSIFY:\n  name: {item['rubric_name']}\n"
        f"  description: {item['rubric_description']}\n  guidance: {item.get('rubric_guidance','') or ''}\n"
    )
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model="meta-llama/llama-3.3-70b-instruct",
                    messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_msg}],
                    temperature=0.0 if attempt == 0 else 0.3,
                    max_tokens=1024, response_format={"type":"json_object"},
                )
                d = salvage_json(resp.choices[0].message.content or "")
                if d: return d
            except Exception as e:
                if attempt == 2: return {"_error": str(e)}
        return {"_error": "no_parse"}


async def main():
    import pandas as pd
    df = pd.read_parquet(ROOT / "notebooks/_explore_cache/rubrics.parquet")
    sample = df.groupby('task', group_keys=False).apply(
        lambda g: g.sample(min(9, len(g)), random_state=8888)  # fresh seed
    ).reset_index(drop=True)
    # 11 tasks × 9 = 99
    print(f"sampled {len(sample)} rubrics across {sample['task'].nunique()} tasks")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=KEY)
    sem = asyncio.Semaphore(8)
    t0 = time.perf_counter()

    items = []
    for r in sample.itertuples():
        items.append({
            "task": r.task, "page_id": r.page_id,
            "subtask_short": r.subtask_short or "",
            "rubric_idx": int(r.rubric_idx),
            "rubric_name": r.rubric_name,
            "rubric_description": r.rubric_description or "",
            "rubric_guidance": r.rubric_guidance or "",
        })

    results = await asyncio.gather(*(call_one(client, sem, it) for it in items))
    print(f"all calls done in {time.perf_counter()-t0:.1f}s")

    # Save full results
    bundle = [{"input": it, "output": r} for it, r in zip(items, results)]
    (OUT_DIR / "results.json").write_text(json.dumps(bundle, indent=2))

    # Print compact summary
    print(f"\n{'#':>3} {'task':<24s} {'name':<45s} {'target':<22s} {'action':<10s} {'verif':<13s} {'tract':<18s} {'lookup':>6s} {'spec':<14s} {'keep':<10s}")
    print("="*200)
    n_ok = 0
    for i, (it, r) in enumerate(zip(items, results)):
        if "_error" in r:
            print(f"{i:>3} {it['task'][:24]:<24s} {it['rubric_name'][:45]:<45s} !! ERROR: {r['_error'][:60]}")
            continue
        n_ok += 1
        print(f"{i:>3} {it['task'][:24]:<24s} {it['rubric_name'][:45]:<45s} {str(r.get('target','?'))[:22]:<22s} {str(r.get('action','?'))[:10]:<10s} {str(r.get('verifiability_type','?'))[:13]:<13s} {str(r.get('tractability','?'))[:18]:<18s} {str(r.get('requires_lookup','?'))[:6]:>6s} {str(r.get('specificity','?'))[:14]:<14s} {str(r.get('keep','?'))[:10]:<10s}")

    print(f"\nn_ok={n_ok}/{len(items)}")

    # Flag suspicious label combos
    print("\n\n=== SUSPICIOUS PATTERNS ===\n")
    flags = []
    for i, (it, r) in enumerate(zip(items, results)):
        if "_error" in r: continue
        v = r.get('verifiability_type'); t = r.get('tractability'); rl = r.get('requires_lookup'); spec = r.get('specificity')

        # Heuristic: computational verifiability should usually be programmatic_check
        if v == "computational" and t not in ("programmatic_check",):
            flags.append((i, "computational_not_programmatic", it, r))
        # Heuristic: factual verifiability should usually require lookup
        if v == "factual" and rl != True:
            flags.append((i, "factual_no_lookup", it, r))
        # Heuristic: normative + requires_lookup=True is unusual (normative shouldn't need external sources)
        if v == "normative" and rl == True and t != "expert_judgment":
            flags.append((i, "normative_with_lookup_but_not_expert", it, r))
        # Heuristic: hyper_specific should usually have programmatic_check or llm_judge tractability
        if spec == "hyper_specific" and t == "intractable":
            flags.append((i, "hyper_specific_intractable", it, r))
        # Heuristic: vague specificity + programmatic_check is contradictory
        if spec == "vague" and t == "programmatic_check":
            flags.append((i, "vague_programmatic", it, r))
        # Heuristic: pragmatic verifiability with normative tractability is odd
        if v == "pragmatic" and t == "intractable":
            flags.append((i, "pragmatic_intractable", it, r))
        # Inputs sanity check: empty list or single item
        inputs = r.get('inputs', [])
        if not isinstance(inputs, list) or len(inputs) < 2:
            flags.append((i, "inputs_too_short", it, r))
        # Generic inputs
        if isinstance(inputs, list):
            generics = {'input_text','the work','the manuscript','the article','evidence','conclusions','the process','the paper'}
            for inp in inputs:
                if isinstance(inp, str) and inp.lower().strip() in generics:
                    flags.append((i, f"generic_input:{inp}", it, r))
                    break

    for i, reason, it, r in flags:
        print(f"  [{i:>3}] {reason}")
        print(f"        task={it['task']}  rubric={it['rubric_name'][:60]}")
        print(f"        labels: target={r.get('target')} action={r.get('action')} verif={r.get('verifiability_type')} tract={r.get('tractability')} lookup={r.get('requires_lookup')} spec={r.get('specificity')}")
        print(f"        inputs: {r.get('inputs', [])}")
        print(f"        reasoning: {r.get('reasoning','')[:200]}")
        print()
    print(f"=== {len(flags)} flagged rows ===")


if __name__ == "__main__":
    asyncio.run(main())
