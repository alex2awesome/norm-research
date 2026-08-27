"""
Scale the v2 classifier on a stratified random sample of ~200 rubrics
(roughly 18 per task × 11 tasks ≈ 200) to validate at scale and produce
a labeled set for hand-validation.

Output: parquet with one row per rubric × classifier output columns.
Cost estimate: ~$0.10-0.20 with gpt-5-mini at this volume.
"""

from __future__ import annotations
import asyncio, json, sys, time
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
sys.path.insert(0, str(ROOT / "scripts"))

from classify_rubric_v2_prompt import SYSTEM_PROMPT_CLASSIFY_V2, JSON_SCHEMA_CLASSIFY_V2
from extract_rubric_features import _load_api_key

OUT_DIR = ROOT / "logs/rubric_labeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PARQUET = OUT_DIR / "classifier_v2_scaled_200.parquet"

N_PER_TASK = 18    # ≈ 200 total
CONCURRENCY = 12


async def call_classifier(client, sem, item: dict) -> dict:
    user_msg = (
        f"PAGE CONTEXT:\n"
        f"  task: {item['task']}\n"
        f"  page_id: {item['page_id']}\n"
        f"  subtask_short: {item['subtask_short']}\n\n"
        f"RUBRIC TO CLASSIFY:\n"
        f"  name: {item['rubric_name']}\n"
        f"  description: {item['rubric_description']}\n"
        f"  guidance: {item['rubric_guidance']}\n"
    )
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_CLASSIFY_V2},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_schema", "json_schema": JSON_SCHEMA_CLASSIFY_V2},
            )
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "elapsed_s": time.perf_counter() - t0}
    elapsed = time.perf_counter() - t0
    try:
        parsed = json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"ok": False, "error": f"json_parse: {e}", "elapsed_s": elapsed}
    return {"ok": True, "parsed": parsed, "elapsed_s": elapsed,
            "in_tok": resp.usage.prompt_tokens if resp.usage else 0,
            "out_tok": resp.usage.completion_tokens if resp.usage else 0}


async def main():
    import pandas as pd
    from openai import AsyncOpenAI

    df = pd.read_parquet(ROOT / "notebooks/_explore_cache/rubrics.parquet")
    # Stratified random sample N_PER_TASK per task
    sample = df.groupby('task', group_keys=False).apply(lambda g: g.sample(min(N_PER_TASK, len(g)), random_state=42))
    sample = sample.reset_index(drop=True)
    print(f"sampled {len(sample)} rubrics across {sample['task'].nunique()} tasks; per-task counts:")
    print(sample['task'].value_counts().to_string())

    client = AsyncOpenAI(api_key=_load_api_key())
    sem = asyncio.Semaphore(CONCURRENCY)

    items = sample.to_dict('records')

    async def run_one(idx, item):
        r = await call_classifier(client, sem, item)
        return idx, item, r

    t_total = time.perf_counter()
    results = await asyncio.gather(*(run_one(i, it) for i, it in enumerate(items)))
    print(f"\nALL DONE in {time.perf_counter()-t_total:.1f}s")

    rows = []
    n_ok = n_err = 0
    tot_in = tot_out = 0
    for idx, item, r in results:
        row = dict(item)
        if r["ok"]:
            ex = r["parsed"]
            row.update({
                "cls_target":   ex["target"],
                "cls_actor":    ex["actor"],
                "cls_action":   ex["action"],
                "cls_keep":     ex["keep"],
                "cls_reasoning": ex["reasoning"],
                "cls_justification": ex["justification"],
                "cls_ok":       True,
                "cls_error":    None,
                "elapsed_s":    r["elapsed_s"],
            })
            n_ok += 1
            tot_in += r["in_tok"]; tot_out += r["out_tok"]
        else:
            row.update({"cls_target":None,"cls_actor":None,"cls_action":None,"cls_keep":None,
                        "cls_reasoning":None,"cls_justification":None,
                        "cls_ok": False, "cls_error": r["error"], "elapsed_s": r.get("elapsed_s",0)})
            n_err += 1
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_parquet(OUT_PARQUET)
    cost = tot_in * 0.15 / 1e6 + tot_out * 0.60 / 1e6
    print(f"\nok={n_ok} err={n_err}  in={tot_in:,} out={tot_out:,} cost≈${cost:.3f}")
    print(f"\n=== KEEP/DROP rates (n={n_ok}) ===")
    print(out[out['cls_ok']]['cls_keep'].value_counts().to_string())
    print(f"\n=== TARGET distribution ===")
    print(out[out['cls_ok']]['cls_target'].value_counts().to_string())
    print(f"\n=== ACTION distribution ===")
    print(out[out['cls_ok']]['cls_action'].value_counts().to_string())
    print(f"\n=== KEEP rate by task ===")
    by_task = out[out['cls_ok']].groupby('task')['cls_keep'].value_counts().unstack(fill_value=0)
    print(by_task.to_string())
    print(f"\nsaved to {OUT_PARQUET}")


if __name__ == "__main__":
    asyncio.run(main())
