"""Generate 2 paraphrased variants of (expanded_definition + what_to_look_for)
per aspect. Keep exemplars, applicability_note, name verbatim.

Output: runs/validity_full/full_v2/judge_enrichment_paraphrased/
  <aspect_id>_p0.json  (original)
  <aspect_id>_p1.json  (paraphrase 1)
  <aspect_id>_p2.json  (paraphrase 2)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path


def make_openai_client():
    from openai import AsyncOpenAI
    # Source salt-lab key if not in env
    if "OPENAI_API_KEY" not in os.environ:
        key_path = Path.home() / ".openai-salt-lab-key.txt"
        if key_path.exists():
            os.environ["OPENAI_API_KEY"] = key_path.read_text().strip()
    return AsyncOpenAI()


async def chat_openai(client, model, system, user,
                       *, max_tokens=1500, retries=3):
    last = None
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_completion_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last = e
            await asyncio.sleep(2 * (attempt + 1))
    raise last


PARAPHRASE_SYSTEM = """You rewrite peer-review evaluation rubric definitions and checklists into PARAPHRASED versions that preserve the EXACT MEANING but use different wording, sentence structure, and emphasis.

You will be given:
  ORIGINAL DEFINITION: a paragraph defining what the rubric measures.
  ORIGINAL CHECKLIST: 3-5 bullet points listing concrete textual signals.

Your task: produce ONE paraphrased version that:
  - Says the SAME THINGS — same scope, same failure modes, same edge cases — using DIFFERENT words and sentence shape.
  - Keeps approximately the same length (±20%).
  - Keeps the checklist as 3-5 bullets covering the SAME signals, but rephrased.
  - Does NOT introduce new criteria or drop existing ones.

Output VALID JSON ONLY:
{
  "expanded_definition": "<paraphrased definition (3-5 sentences)>",
  "what_to_look_for": ["<paraphrased bullet 1>", "<paraphrased bullet 2>", ...]
}"""


async def paraphrase_one(client, sem, model, enr):
    async with sem:
        user = (f"ORIGINAL DEFINITION:\n{enr['expanded_definition']}\n\n"
                f"ORIGINAL CHECKLIST:\n" +
                "\n".join(f"- {b}" for b in enr["what_to_look_for"]) +
                "\n\nWrite the paraphrased JSON.")
        for attempt in range(3):
            try:
                raw = await chat_openai(client, model,
                                         PARAPHRASE_SYSTEM, user,
                                         max_tokens=1500)
                m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
                if m: raw = m.group(1).strip()
                try: obj = json.loads(raw)
                except json.JSONDecodeError:
                    s, e = raw.find("{"), raw.rfind("}")
                    obj = json.loads(raw[s:e + 1])
                if "expanded_definition" in obj and "what_to_look_for" in obj:
                    return obj
            except Exception as ex:
                if attempt == 2:
                    print(f"  PARSE FAIL {enr['aspect_id']}: {ex}",
                          flush=True)
                    return None
                await asyncio.sleep(2 * (attempt + 1))
        return None


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--concurrency", type=int, default=30)
    ap.add_argument("--n-paraphrases", type=int, default=2,
                    help="paraphrases to generate per aspect (p0 = original)")
    args = ap.parse_args()

    v2 = Path("runs/validity_full/full_v2")
    src_dir = v2 / "judge_enrichment"
    out_dir = v2 / "judge_enrichment_paraphrased"
    out_dir.mkdir(exist_ok=True, parents=True)

    enrs = []
    for sp in sorted(src_dir.glob("a*.json")):
        try:
            enrs.append(json.loads(sp.read_text()))
        except Exception as e:
            print(f"skip {sp.name}: {e}")

    print(f"loaded {len(enrs)} enrichments; "
          f"generating {args.n_paraphrases} paraphrases each")

    client = make_openai_client()
    sem = asyncio.Semaphore(args.concurrency)

    # Save originals as p0
    for e in enrs:
        (out_dir / f"{e['aspect_id']}_p0.json").write_text(
            json.dumps(e, indent=1))

    # Generate paraphrases p1..pN
    for p_idx in range(1, args.n_paraphrases + 1):
        # Skip already-done
        todo = [e for e in enrs
                if not (out_dir / f"{e['aspect_id']}_p{p_idx}.json").exists()]
        print(f"\n=== paraphrase pass {p_idx}: {len(todo)} todo ===")
        coros = [paraphrase_one(client, sem, args.model, e) for e in todo]
        import time
        t0 = time.time()
        n_done = n_fail = 0
        for i, coro in enumerate(asyncio.as_completed(coros)):
            obj = await coro
            e = todo[i]  # NOTE: this index may not align since as_completed
            # The above doesn't work for index mapping. Use a different approach:
            # we'll redo with a different pattern.
            break
        # Use a cleaner pattern: gather with order preserved
        results = await asyncio.gather(
            *[paraphrase_one(client, sem, args.model, e) for e in todo],
            return_exceptions=False)
        for e, obj in zip(todo, results):
            if obj is None:
                n_fail += 1
                continue
            # Build the paraphrased file: copy original, replace
            # expanded_definition and what_to_look_for
            new_enr = dict(e)
            new_enr["expanded_definition"] = obj["expanded_definition"]
            new_enr["what_to_look_for"] = obj["what_to_look_for"]
            new_enr["paraphrase_idx"] = p_idx
            (out_dir / f"{e['aspect_id']}_p{p_idx}.json").write_text(
                json.dumps(new_enr, indent=1))
            n_done += 1
        print(f"  pass {p_idx}: done={n_done} fail={n_fail} "
              f"in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    asyncio.run(amain())
