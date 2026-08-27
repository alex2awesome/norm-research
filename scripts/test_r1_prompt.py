"""Test R1 (concept-family) prompt on a peer-review sample via OpenRouter.

Picks N peer-review clusters whose representative text mentions a chosen
keyword (so the batch contains likely family-mates), submits them to an LLM
with the R1 family-grouping prompt + few-shot examples, and prints the
resulting families.

Usage:  python scripts/test_r1_prompt.py [keyword] [--model X]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
FORMS = ROOT / "outputs" / "analyses" / "canon_all_real_forms.jsonl"
CL = ROOT / "outputs" / "analyses" / "structural_metrics" / "clusters_peer-review.json"
KEY = (Path.home() / ".openrouter-api-key.txt").read_text().strip()

SYSTEM = """You are organizing fine-grained rubric concepts into broader CRITERION FAMILIES.

A "rubric concept" is a deduplicated cluster of rubric statements with a representative text. Many concepts that look superficially different encode the same underlying RULE -- they prescribe the same target behaviour, only stated with different wording, examples, qualifications, or specificity. **Group all variants of one rule into one family.**

A FAMILY = every variant of one underlying rule. The TEST is: "would a competent reviewer say these are checking the SAME thing, even if the way they phrase the check differs?" If yes, same family.

Specificity matters for the SCOPE of a rule, not its wording. Two concepts about the SAME scope of behaviour belong together even if one says it broadly ("X should be evidence-based") and another adds a condition or example ("X should provide concrete evidence and specific references"). But two concepts at clearly DIFFERENT scopes stay apart: function-level vs module-level scope check different things, even though both invoke "responsibility"; indentation and line-length are both formatting rules but check different things.

Under-merging (many near-duplicate singleton families) is the more common failure mode here. When in doubt and the two concepts plausibly check the same thing -- merge.

OUTPUT VALID JSON ONLY, no commentary, no markdown fences:
{
  "families": [
    {"name": "<short noun phrase, 4-8 words>",
     "description": "<one sentence stating the underlying rule>",
     "members": ["<id>", "<id>", ...]}
  ]
}

Every input id must appear in exactly one family. Singleton families are allowed when a concept genuinely doesn't share a rule with any other input. Do not invent ids.
"""


FEWSHOT_USER = json.dumps([
    {"id": "C1", "rep": "The code should have consistent indentation",
     "alt": ["indent code uniformly across the file"]},
    {"id": "C2", "rep": "Use 4-space indentation",
     "alt": ["indent code with four spaces"]},
    {"id": "C3", "rep": "Do not mix tabs and spaces", "alt": []},
    {"id": "C4", "rep": "Lines should be under 80 characters", "alt": []},
    {"id": "C5", "rep": "The function should have a single responsibility", "alt": []},
    {"id": "C6", "rep": "Each module should have a single responsibility", "alt": []},
    {"id": "C7", "rep": "The function should not have too many return statements",
     "alt": []},
    {"id": "C8", "rep": "The claim should be supported by evidence", "alt": []},
    {"id": "C9", "rep": "The conclusion should not exceed what the evidence supports",
     "alt": []},
    {"id": "C10", "rep": "Strong evidence should be provided for the main claims, and unsupported statements should be moderated or removed",
     "alt": []},
    {"id": "C11", "rep": "The tone of the conclusions should be appropriate to the strength of the evidence",
     "alt": []},
    {"id": "C12", "rep": "The review should be supported with concrete evidence and specific references",
     "alt": []},
], indent=1)

FEWSHOT_ASSISTANT = json.dumps({"families": [
    {"name": "Consistent indentation",
     "description": "The code should be indented consistently (style, width, no mixed tabs/spaces).",
     "members": ["C1", "C2", "C3"]},
    {"name": "Maximum line length",
     "description": "Lines should not exceed a maximum length.",
     "members": ["C4"]},
    {"name": "Single-responsibility principle",
     "description": "A code unit (function or module) should have one responsibility -- function-level and module-level are the same rule applied to different scopes.",
     "members": ["C5", "C6"]},
    {"name": "Limit function return statements",
     "description": "A function should not have excessive or scattered return statements.",
     "members": ["C7"]},
    {"name": "Match claims to evidence strength",
     "description": "Claims, conclusions, and the language used to state them should be supported by, and not exceed, the available evidence.",
     "members": ["C8", "C9", "C10", "C11", "C12"]},
]}, indent=1)


def parse_json(text):
    """Extract the JSON object even if wrapped in markdown fences or prose."""
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


async def call(model, system, fewshots, user):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=KEY)
    messages = [{"role": "system", "content": system}]
    for u, a in fewshots:
        messages += [{"role": "user", "content": u},
                     {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": user})
    resp = await client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=4000, messages=messages)
    return resp.choices[0].message.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keyword", nargs="?", default="evidence")
    ap.add_argument("--model", default="anthropic/claude-sonnet-4.5")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    forms = [json.loads(l) for l in FORMS.open()
             if json.loads(l)["task"] == "peer-review"]
    cl = json.loads(CL.read_text())
    members = defaultdict(list)
    for r in forms:
        members[cl[r["key"]]].append(r["canonical"] or "")

    cands = []
    for cid, ms in members.items():
        if not ms:
            continue
        rep = Counter(ms).most_common(1)[0][0]
        if args.keyword.lower() in rep.lower():
            cands.append((cid, rep, ms))
    cands.sort(key=lambda x: -len(x[2]))  # biggest first
    cands = cands[:args.n]
    print(f"selected {len(cands)} peer-review clusters matching '{args.keyword}'")

    cand_list = []
    for cid, rep, ms in cands:
        alts = [m for m in dict.fromkeys(ms) if m != rep][:2]
        cand_list.append({"id": f"C{cid}", "rep": rep[:200],
                          "alt": [a[:160] for a in alts]})

    print("\nbatch:")
    for c in cand_list:
        print(f"  {c['id']:>7}  ({len(members[int(c['id'][1:])]):>2})  {c['rep'][:90]}")

    user_msg = json.dumps(cand_list, indent=1)
    out = asyncio.run(call(args.model, SYSTEM,
                           [(FEWSHOT_USER, FEWSHOT_ASSISTANT)], user_msg))

    print("\n--- raw LLM output ---")
    print(out)

    parsed = parse_json(out)
    if not parsed:
        print("\nFAILED to parse JSON")
        return
    print(f"\n--- {len(parsed['families'])} families ---")
    for f in parsed["families"]:
        print(f"\n  [{len(f['members'])}] {f['name']}")
        print(f"      {f.get('description','')}")
        for mid in f["members"]:
            try:
                cid = int(mid[1:])
                rep = Counter(members[cid]).most_common(1)[0][0]
                print(f"      - {rep[:96]}")
            except Exception:
                print(f"      - <unknown {mid}>")

    used = {mid for f in parsed["families"] for mid in f["members"]}
    given = {c["id"] for c in cand_list}
    miss = given - used
    extra = used - given
    print(f"\ncoverage: {len(used)}/{len(given)} used, {len(miss)} missing, "
          f"{len(extra)} hallucinated")


if __name__ == "__main__":
    main()
