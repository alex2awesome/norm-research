"""Noun/verb thickness-chain extraction over R2 merged_groups.

Decomposes each evaluation criterion (a merged_group) into the chain of
operations a judge performs to apply it:

  input_noun -> verb -> intermediate_noun -> verb -> ... -> final_noun

Each element gets a thickness rating 1-4. See notes/2026-05-14__noun-verb-thickness.md.

This is a CALIBRATION-MODE script for prompt workshopping:
  python scripts/extract_noun_verb_chains.py --sample 20          # run on stratified 20
  python scripts/extract_noun_verb_chains.py --print              # pretty-print last run
  python scripts/extract_noun_verb_chains.py --sample 20 --tag v2 # versioned output

Output: outputs/analyses/chains_calib_<tag>.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
HIER = ROOT / "outputs" / "hierarchy"
OUT = ROOT / "outputs" / "analyses"
KEY_PATH = Path("/Users/spangher/.openai-salt-lab-key.txt")

MODEL = "gpt-5-mini"
SAMPLE_SEED = 14

TASKS = [
    "code-review", "creative-writing", "grant-funding", "humor",
    "legal-outcome-prediction", "math-stackexchange", "news-homepages",
    "notice-and-comment", "patents", "peer-review", "press-releases",
]

CHAIN_SCHEMA = {
    "name": "noun_verb_chain",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "chain": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {"type": "string", "enum": ["noun", "verb"]},
                        "content": {"type": "string"},
                        "thickness": {"type": "integer", "minimum": 1, "maximum": 4},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["type", "content", "thickness", "reasoning"],
                },
            },
            "overall_reasoning": {"type": "string"},
        },
        "required": ["chain", "overall_reasoning"],
    },
}

SYSTEM_PROMPT = """You decompose an evaluation criterion into the chain of operations a judge performs to apply it to a piece of work.

A criterion (e.g. "the methods section reports a power analysis") is applied via a chain of alternating NOUNS and VERBS:

  input_noun -> verb -> intermediate_noun -> verb -> ... -> final_noun

- NOUNS are things operated on: the input artifact, intermediate extracted/derived things, and the final judgment.
- VERBS are operations that transform one noun into the next.

Both nouns and verbs vary in THICKNESS — how hard the element is to pin down precisely. Thickness is NOT importance and NOT how much the element matters; it is how much latent interpretation the element requires.

NOUN thickness (1-4) — how richly must this thing be apprehended?
  1 = raw / typed data. A character or token stream, a number. Example: "the text as a token stream", "a word count".
  2 = a bounded local unit. One sentence, one heading, one citation, one figure.
  3 = a structured object with internal relations. A section, an argument, a methods description.
  4 = a holistic latent state. The whole work apprehended as an experience or gestalt. Example: "the piece as a full reading experience", "the narrative arc", "voice".

VERB thickness (1-4) — how hard is this operation to specify?
  1 = mechanical / procedural. Count, look up, regex-match, compare numbers.
  2 = shallow-semantic. Locate a section, classify by surface type, extract labeled spans.
  3 = reasoning / inference. Judge a relationship; evaluate whether an argument addresses a counter-claim; assess sufficiency.
  4 = irreducible holistic judgment. Taste; "does the joke land"; "does the piece have voice"; aesthetic gestalt.

Rules:
- The chain STARTS with an input_noun (what the judge first apprehends) and ENDS with a final_noun (the judgment/verdict).
- Strictly alternate: noun, verb, noun, verb, ..., noun. Each verb transforms the preceding noun into the following noun.
- Aim for 3-7 elements. Each verb should be a meaningfully distinct operation, not a micro-step.
- The chain is CONCEPTUAL — describe the latent operations a judge performs, not the grammar of the criterion's sentence.
- If a criterion bundles several sub-checks, pick the dominant / most representative path; do not branch.

WORKED EXAMPLES:

Criterion: "Use Oxford commas"
chain:
  noun  "the text as a token stream"                          thickness 1
  verb  "find every list of three or more items"              thickness 1
  noun  "the set of in-text lists"                            thickness 1
  verb  "check each list for a serial comma before the conjunction"  thickness 1
  noun  "compliance verdict"                                  thickness 1
overall: thin everywhere — fully mechanical, a program could do every step.

Criterion: "The methods section reports a power analysis"
chain:
  noun  "the full paper"                                      thickness 3
  verb  "locate the methods section"                          thickness 2
  noun  "the methods section text"                            thickness 2
  verb  "determine whether a power / sample-size analysis is described"  thickness 3
  noun  "presence verdict"                                    thickness 1
overall: thickness concentrated in the final reasoning verb; input is a structured doc, the key judgment needs domain reasoning but no taste.

Criterion: "The piece has a distinctive voice"
chain:
  noun  "the piece apprehended as a full reading experience"  thickness 4
  verb  "judge whether a distinctive, consistent authorial voice is present"  thickness 4
  noun  "voice verdict"                                       thickness 1
overall: short chain; all thickness in one irreducible holistic verb — no procedure decomposes it.

Output JSON: a "chain" array of {type, content, thickness, reasoning} elements, plus an "overall_reasoning" string describing where the thickness lives in the chain."""


def build_user_msg(mg: dict) -> str:
    leaves = mg.get("all_leaves", [])
    # 3 representative children: take first 3 leaf names (already size-ordered upstream)
    child_names = [l.get("name", "") for l in leaves[:3] if l.get("name")]
    parts = [
        f"EVALUATION CRITERION: {mg.get('merged_name', '')}",
        f"DESCRIPTION: {mg.get('merged_description', '')}",
    ]
    if child_names:
        parts.append("REPRESENTATIVE SUB-RUBRICS:")
        for c in child_names:
            parts.append(f"  - {c}")
    parts.append("")
    parts.append("Decompose this criterion into its noun/verb thickness chain.")
    return "\n".join(parts)


def sample_merged_groups(n: int) -> list[dict]:
    """Stratified random sample: n//11 per task (min 1)."""
    random.seed(SAMPLE_SEED)
    per_task = max(1, n // len(TASKS))
    out = []
    for task in TASKS:
        p = HIER / f"{task}_general_r2_expanded.json"
        if not p.exists():
            continue
        mg = json.loads(p.read_text()).get("merged_groups", [])
        if not mg:
            continue
        picks = random.sample(mg, min(per_task, len(mg)))
        for m in picks:
            out.append({"task": task, **m})
    return out


async def call_llm(client, user, sem, timeout_sec=120.0):
    async with sem:
        for attempt in range(3):
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user},
                        ],
                        response_format={"type": "json_schema", "json_schema": CHAIN_SCHEMA},
                        service_tier="flex",
                    ),
                    timeout=timeout_sec,
                )
                return json.loads(resp.choices[0].message.content or "{}")
            except asyncio.TimeoutError:
                if attempt == 2:
                    return {"_error": "timeout"}
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 2:
                    return {"_error": str(e)[:200]}
                await asyncio.sleep(2 ** attempt)
    return {}


async def run_sample(n: int, tag: str):
    sample = sample_merged_groups(n)
    print(f"sampled {len(sample)} merged_groups across {len(TASKS)} tasks")
    api_key = KEY_PATH.read_text().strip()
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(50)

    async def one(mg):
        res = await call_llm(client, build_user_msg(mg), sem)
        return {
            "task": mg["task"],
            "merged_name": mg.get("merged_name", ""),
            "merged_description": mg.get("merged_description", ""),
            "result": res,
        }

    t0 = time.perf_counter()
    results = await asyncio.gather(*[one(mg) for mg in sample])
    print(f"done in {time.perf_counter()-t0:.0f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"chains_calib_{tag}.jsonl"
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")
    return out_path


def pretty_print(tag: str):
    path = OUT / f"chains_calib_{tag}.jsonl"
    if not path.exists():
        print(f"no file {path}")
        return
    with path.open() as f:
        rows = [json.loads(line) for line in f]
    for i, r in enumerate(rows, 1):
        res = r.get("result", {})
        print(f"\n{'='*92}")
        print(f"[{i}] [{r['task']}] {r['merged_name']}")
        print(f"    desc: {r['merged_description'][:160]}")
        if "_error" in res:
            print(f"    ERROR: {res['_error']}")
            continue
        chain = res.get("chain", [])
        for el in chain:
            t = el.get("type", "?")
            arrow = "  " if t == "noun" else "->"
            marker = "N" if t == "noun" else "V"
            print(f"    {arrow} [{marker} th={el.get('thickness','?')}] {el.get('content','')}")
            print(f"          ({el.get('reasoning','')[:110]})")
        print(f"    OVERALL: {res.get('overall_reasoning','')[:240]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="run on N stratified merged_groups")
    ap.add_argument("--tag", default="v1", help="version tag for output file")
    ap.add_argument("--print", action="store_true", help="pretty-print results for the tag")
    args = ap.parse_args()

    if args.sample:
        asyncio.run(run_sample(args.sample, args.tag))
        pretty_print(args.tag)
    elif args.print:
        pretty_print(args.tag)
    else:
        print("specify --sample N or --print")


if __name__ == "__main__":
    main()
