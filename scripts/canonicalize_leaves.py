"""Canonicalize leaf-rubric names with an LLM, via OpenRouter (or Anthropic).

Each leaf rubric was written by a different author in their own words. The
clustering/dedup pipeline under-merged because same-concept rubrics phrased
differently never collided. This rewrites every leaf into a canonical sentence
that normalizes WORDING ONLY -- so the same concept written several ways
converges, while specificity and distinct concepts are preserved (we measure
how specific each task's rubrics are; canonicalization must not flatten that).

Calibration mode runs on the seeded stress sample and prints results grouped by
seed_group for manual inspection. Iterate the prompt across rounds (--tag).

Usage:
  python scripts/canonicalize_leaves.py --tag r3 --model meta-llama/llama-3.3-70b-instruct
  python scripts/canonicalize_leaves.py --tag r3 --model claude-sonnet-4-6 --concurrency 4
  python scripts/canonicalize_leaves.py --full --tag prod   # all 32K leaves
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
OUT = ROOT / "outputs" / "analyses"
SAMPLE = OUT / "canon_stress_sample.jsonl"
LEAF_FULL = OUT / "_sk3_leaf_input.jsonl"
try:  # sk3/offline (e.g. nohup w/o AFS token): OpenRouter key not needed for vLLM
    KEY = (Path.home() / ".openrouter-api-key.txt").read_text().strip()
except Exception:
    KEY = ""

PROMPT_VERSION = "v5"

# --- per-task background ----------------------------------------------------
TASK_BACKGROUND = {
    "code-review": "Reviewing a proposed code change (a pull request) before it is merged. Rubrics come from code-review checklists, language style guides, refactoring catalogs, and engineering standards; each states something a reviewer checks about the code, its tests, or its design.",
    "creative-writing": "Evaluating a piece of creative writing (fiction, a story, a manuscript). Rubrics come from craft books, style manuals, magazine submission guidelines, and writing-contest criteria; each states something that distinguishes strong creative writing -- craft, structure, prose, character, voice.",
    "grant-funding": "Evaluating a grant or research-funding proposal. Rubrics come from funder guidelines and grant-writing guides; each states something reviewers look for in a fundable proposal -- significance, approach, feasibility, budget, team.",
    "humor": "Evaluating humor -- a joke, comedy set, cartoon, or comic piece. Rubrics come from comedy-writing manuals and analyses of what makes things funny; each states something that distinguishes effective humor -- joke construction, timing, target, originality.",
    "legal-outcome-prediction": "Evaluating a legal argument or case for how it is likely to fare. Rubrics come from legal writing and advocacy guides; each states something about the strength of a legal argument -- its reasoning, use of precedent, framing, evidence.",
    "math-stackexchange": "Evaluating a mathematics question or answer on a Q&A site. Rubrics come from guides on mathematical writing and on good questions/answers; each states something that distinguishes a strong, clear, correct, well-posed math post.",
    "news-homepages": "Judging the newsworthiness of a story and whether it belongs on a news homepage. Rubrics come from journalism guides and newsroom practice; each states something that makes a story newsworthy or homepage-worthy -- timeliness, impact, audience relevance, prominence.",
    "notice-and-comment": "Evaluating a public comment submitted on a proposed government regulation (notice-and-comment rulemaking). Rubrics come from guides on writing effective regulatory comments; each states something that makes a comment persuasive or useful to the agency -- specificity, evidence, legal grounding, constructiveness.",
    "patents": "Evaluating a patent application or claim. Rubrics come from patent-drafting guides and patent-office standards; each states something that distinguishes a strong application -- claim clarity, scope, novelty support, specification quality.",
    "peer-review": "Evaluating an academic paper and the peer reviews of it. Rubrics come from journal/conference reviewer guidelines and peer-review guides; each states something reviewers assess -- novelty, soundness, clarity, evidence, reproducibility, related work.",
    "press-releases": "Evaluating a press release. Rubrics come from PR and communications writing guides; each states something that distinguishes an effective press release -- newsworthiness, structure, clarity, quotes, headline, audience targeting.",
}

# --- per-task few-shot examples ---------------------------------------------
# Each task: a convergence/one-word case, a KEEP-SPECIFIC case (do not broaden),
# and a heading/stance case.
TASK_FEWSHOTS = {
    "code-review": [
        {"input": "Extract Method", "reasoning": "A specific named refactoring whose content (splitting long routines) is well known; keep that content so it stays distinct from Extract Class/Superclass.", "canonical": "Extract a method: split a long routine into smaller named methods."},
        {"input": "Maintainability", "reasoning": "A single-word criterion name, not a section heading -- it asks whether the code is maintainable.", "canonical": "The code should be maintainable so others can understand and change it later."},
        {"input": "Keep functions under 50 lines", "reasoning": "The '50 lines' threshold is the criterion itself; keep it specific rather than broadening to a generic 'keep functions short'.", "canonical": "Functions should be kept under 50 lines."},
        {"input": "Function: Detect", "reasoning": "A bare section label from a framework's table of contents; no evaluative criterion is stated.", "canonical": None},
    ],
    "creative-writing": [
        {"input": "Showing vs telling (dramatic technique)", "reasoning": "A rephrasing of the general show-don't-tell principle; the parenthetical is incidental labelling.", "canonical": "Convey information through action, dialogue, and sensory detail rather than direct statement (show, don't tell)."},
        {"input": "Sanderson's Third Law: Expand, Don't Add", "reasoning": "A specific named principle whose content is stated; keep that content so it stays distinct from the other Sanderson laws.", "canonical": "Expand and develop existing story elements before introducing new ones."},
        {"input": "Open with a hook in the very first sentence", "reasoning": "'First sentence' is part of the criterion's specificity, so keep it; 'hook' is colloquial, so translate it into plain descriptive language.", "canonical": "The very first sentence should immediately engage the reader and draw them into the story."},
        {"input": "Myth: passive voice is a grammatical error", "reasoning": "This debunks a myth; preserve that stance rather than flipping it into a prohibition.", "canonical": "The passive voice is not a grammatical error and is acceptable when used appropriately."},
    ],
    "grant-funding": [
        {"input": "Significance", "reasoning": "A single-word criterion name -- whether the proposed work is significant.", "canonical": "The proposed work should address a significant problem."},
        {"input": "Include a detailed timeline covering years 1 through 3", "reasoning": "The 'years 1 through 3' scope is part of the criterion; keep it rather than broadening to 'include a timeline'.", "canonical": "The proposal should include a detailed timeline covering years 1 through 3."},
        {"input": "Feasibility of the proposed approach", "reasoning": "Criterion about whether the approach can realistically be carried out.", "canonical": "The proposed approach should be feasible and realistically achievable."},
    ],
    "humor": [
        {"input": "Timing", "reasoning": "A single-word criterion name -- comedic timing.", "canonical": "The humor should have effective comedic timing."},
        {"input": "Land the punchline on the final word of the sentence", "reasoning": "'Final word of the sentence' is the specific criterion; keep it rather than broadening to 'have a strong punchline'.", "canonical": "The punchline should land on the final word of the sentence."},
        {"input": "Avoid 'low-hanging fruit' easy targets", "reasoning": "Criterion about comedic target selection; the quoted phrase is the author's wording.", "canonical": "Avoid easy, obvious comedic targets."},
    ],
    "legal-outcome-prediction": [
        {"input": "Strength of evidence", "reasoning": "Criterion about how strong the supporting evidence is.", "canonical": "The argument should be supported by strong evidence."},
        {"input": "Cite binding precedent from the same jurisdiction", "reasoning": "'Binding' and 'same jurisdiction' narrow the criterion; keep them rather than broadening to 'cite precedent'.", "canonical": "The argument should cite binding precedent from the same jurisdiction."},
        {"input": "Procedural Posture", "reasoning": "A bare doctrinal section label; no evaluative criterion is stated.", "canonical": None},
    ],
    "math-stackexchange": [
        {"input": "Rigor", "reasoning": "A single-word criterion name -- mathematical rigor.", "canonical": "The argument should be mathematically rigorous."},
        {"input": "State the domain and codomain of every function introduced", "reasoning": "'Domain and codomain of every function' is the specific criterion; keep it rather than broadening to 'define your terms'.", "canonical": "State the domain and codomain of every function introduced."},
        {"input": "Show your own attempt", "reasoning": "Criterion that a question show the asker's own effort.", "canonical": "The question should show the asker's own attempt or effort."},
    ],
    "news-homepages": [
        {"input": "Timeliness", "reasoning": "A single-word criterion name -- whether the story is timely.", "canonical": "The story should be timely and current."},
        {"input": "Lead with the most recent development in the story", "reasoning": "'Most recent development' is the specific criterion; keep it rather than broadening to a generic 'be timely'.", "canonical": "The story should lead with its most recent development."},
        {"input": "Impact / consequence", "reasoning": "Two phrasings of one criterion: how much the story affects readers.", "canonical": "The story should have significant impact on or consequence for readers."},
    ],
    "notice-and-comment": [
        {"input": "Specificity of the comment", "reasoning": "Criterion that a regulatory comment be specific rather than vague.", "canonical": "The comment should be specific rather than vague or general."},
        {"input": "Identify the regulatory provision by its exact section number", "reasoning": "'Exact section number' is part of the criterion; keep it rather than broadening to 'reference the regulation'.", "canonical": "The comment should identify the regulatory provision by its exact section number."},
        {"input": "Summary of the proposed rule", "reasoning": "A descriptive section label, not an evaluative criterion of a comment.", "canonical": None},
    ],
    "patents": [
        {"input": "Claim clarity", "reasoning": "Criterion that the patent claims be clear and definite.", "canonical": "The patent claims should be clear and definite."},
        {"input": "Every independent claim should have at least one dependent claim", "reasoning": "The exact structural rule is the criterion; keep it rather than broadening to 'structure claims well'.", "canonical": "Every independent claim should have at least one dependent claim."},
        {"input": "Enablement", "reasoning": "A single-word criterion name; canonicalize minimally without inventing its full legal definition.", "canonical": "The patent specification should be enabling."},
    ],
    "peer-review": [
        {"input": "Reproducibility", "reasoning": "A single-word criterion name, not a heading -- whether the work can be reproduced.", "canonical": "The work should be reproducible by others."},
        {"input": "Every figure must be referenced in the main text", "reasoning": "The exact rule (every figure referenced in the main text) is the criterion; keep it rather than broadening to 'figures should be clear'.", "canonical": "Every figure should be referenced in the main text."},
        {"input": "Novelty / originality", "reasoning": "Two words for one criterion; normalize to a single statement.", "canonical": "The work should make a novel, original contribution."},
    ],
    "press-releases": [
        {"input": "Strong headline", "reasoning": "Criterion that the headline be compelling.", "canonical": "The headline should be strong and compelling."},
        {"input": "Put the key news in the first paragraph", "reasoning": "'First paragraph' is the specific criterion; keep it rather than broadening to 'lead with the news'.", "canonical": "The key news should appear in the first paragraph."},
        {"input": "Inverted pyramid structure", "reasoning": "A specific named structure: most important information first.", "canonical": "Structure the release as an inverted pyramid, with the most important information first."},
    ],
}

SYSTEM_TEMPLATE = """You normalize evaluation criteria ("rubrics") extracted from many different guidebooks, standards, and how-to documents. Different authors state the SAME criterion in different words; your job is to standardize the WORDING so same-criterion rubrics come out as nearly identical text and can be merged -- without changing what any rubric means or how specific it is.

TASK: [[TASKNAME]]
[[BACKGROUND]]

Rewrite the given rubric into ONE canonical sentence stating the underlying evaluative criterion in plain, directive terms.

Rules:
1. COLLAPSE ONLY PURE REPHRASINGS. Two rubrics that express the SAME criterion at the SAME level of detail must rewrite to nearly identical text -- the same criterion in different words must yield the same canonical sentence. That convergence is the only thing you merge.
2. BE DESCRIPTIVE; TRANSLATE COLLOQUIALISMS. State plainly what the criterion actually checks -- do not just echo the rubric's wording. When the rubric uses a colloquial, idiomatic, metaphorical, or jargon phrasing, translate it into plain descriptive language for what it asks for: "hook the reader" -> "engage the reader and draw them into the piece"; "trim the fat" -> "cut unnecessary words and elements"; "kill your darlings" -> "cut passages you are attached to when they do not serve the work". For a widely recognized named principle you may append the name in parentheses ("... (show, don't tell)"). If a rubric is already plain and explicit, state it faithfully and clearly; do not pad it.
3. PRESERVE SPECIFICITY. Do not change how specific or general the rubric is. Keep every concrete threshold, number, named sub-case, and scope condition -- these are part of the criterion, and we measure how specific each task's rubrics are, so lost specificity corrupts the data. A specific rubric must stay specific; never broaden it into a generic one. Drop only purely illustrative material (an "e.g." example offered to illustrate a general principle).
4. KEEP DISTINCT CONCEPTS DISTINCT. Do not merge different criteria, including same-topic rubrics that differ in specificity ("vary sentence length" and "do not begin three sentences in a row with the same word" are different criteria). If the rubric names a specific technique/principle and states its content, keep that distinguishing content.
5. ADD NOTHING BEYOND TRANSLATION. Use only what the rubric says or unambiguously implies. Unpacking a colloquialism or metaphor into its plain meaning (rule 2) is fine -- but never invent extra criteria, rationale, or detail, and never use outside knowledge to expand a name you do not recognize from the rubric's own words.
6. PRESERVE STANCE. If the rubric debunks a myth or says something is NOT a problem, keep that stance -- do not flip it into a prohibition.
7. HEADINGS / BARE REFERENCES vs ONE-WORD CRITERIA. If the input is a structural section label with no criterion ("Crates", "Roadmap", "Function: Detect"), or a bare reference to a named law/principle/technique with no content stated (a bare "Sanderson's Third Law"), set canonical to null -- never invent its content. But a single-word CRITERION NAME ("Reproducibility", "Clarity", "Timing") IS a real criterion -- canonicalize it minimally ("The work should be reproducible / clear / well-timed"); do NOT null it.
8. canonical is ONE sentence in directive form, or null.

OFF-TOPIC CHECK. Also judge whether the rubric genuinely belongs to the task domain described above. If it is clearly from a different domain (for example, science-journalism advice filed under a math Q&A task), set "off_topic" to true; otherwise false. Canonicalize the rubric either way.

First think briefly in "reasoning" (1-2 sentences: what is the underlying criterion; how specific is it and what specificity must be kept; is it colloquial/idiomatic and how should it be translated; is this a heading or a real criterion; does it belong to this task), then give "off_topic" and "canonical".

EXAMPLES (this task):
[[FEWSHOTS]]

Respond with ONE JSON object: {"reasoning": "...", "off_topic": false, "canonical": "..."} -- canonical may be null. No markdown."""


def build_system_prompt(task: str) -> str:
    fs = "\n\n".join(
        f'RUBRIC: {e["input"]}\n'
        + json.dumps({"reasoning": e["reasoning"], "off_topic": False,
                      "canonical": e["canonical"]})
        for e in TASK_FEWSHOTS.get(task, []))
    return (SYSTEM_TEMPLATE
            .replace("[[TASKNAME]]", task.replace("-", " "))
            .replace("[[BACKGROUND]]", TASK_BACKGROUND.get(task, ""))
            .replace("[[FEWSHOTS]]", fs))


def salvage_json(raw: str):
    if not raw or not raw.strip():
        return None
    s = re.sub(r"\s*```$", "", re.sub(r"^```(?:json)?\s*", "", raw.strip()))
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    if start < 0:
        return None
    for end in range(len(s), start, -1):
        if s[end - 1] == "}":
            try:
                return json.loads(s[start:end])
            except Exception:
                continue
    return None


async def _get_raw(client, anthropic, model, sys_p, user, pin):
    if anthropic:
        resp = await client.messages.create(
            model=model, max_tokens=1200, temperature=0.0,
            system=sys_p, messages=[{"role": "user", "content": user}])
        return resp.content[0].text if resp.content else ""
    extra = {}
    if pin:
        extra["extra_body"] = {"provider": {
            "order": ["fireworks", "together", "deepinfra"],
            "allow_fallbacks": False}}
    resp = await client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=1200,
        messages=[{"role": "system", "content": sys_p},
                  {"role": "user", "content": user}], **extra)
    return resp.choices[0].message.content or ""


async def canon_one(client, anthropic, sem, model, row):
    sys_p = build_system_prompt(row["task"])
    user = f'RUBRIC: {row["name"]}'
    raw, last_err = "", "unknown"
    async with sem:
        for attempt in range(7):
            try:
                raw = await _get_raw(client, anthropic, model, sys_p, user,
                                     pin=(attempt >= 2 and not anthropic))
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                msg = str(e).lower()
                wait = min(2 ** attempt, 40) if (
                    "429" in msg or "rate" in msg or "overload" in msg) else 2
                await asyncio.sleep(wait + 0.5 * attempt)
                continue
            parsed = salvage_json(raw)
            if parsed is not None and "canonical" in parsed:
                c = parsed["canonical"]
                return {**row, "ok": True,
                        "canonical": c.strip() if isinstance(c, str) else None,
                        "reasoning": parsed.get("reasoning", ""),
                        "off_topic": bool(parsed.get("off_topic", False)),
                        "raw": raw}
            last_err = "json_salvage_failed"
            await asyncio.sleep(1)
    return {**row, "ok": False, "error": last_err, "raw": raw}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r3")
    ap.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--input", default=None, help="override source jsonl")
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()

    src = (Path(args.input) if args.input
           else (LEAF_FULL if args.full else SAMPLE))
    rows = [json.loads(l) for l in src.open()]
    for r in rows:
        r.setdefault("seed_group", "full")
        r.setdefault("kind", "full")
    print(f"canonicalizing {len(rows)} leaves | model={args.model} | "
          f"prompt={PROMPT_VERSION} tag={args.tag}")

    is_anthropic = args.model.startswith("claude")
    if is_anthropic:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(
            api_key=(Path.home() / ".anthropic-usc-key.txt").read_text().strip())
    else:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=KEY)
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.perf_counter()
    results = await asyncio.gather(*(canon_one(client, is_anthropic, sem,
                                               args.model, r) for r in rows))
    ok = sum(r["ok"] for r in results)
    print(f"done in {time.perf_counter()-t0:.0f}s | {ok}/{len(results)} ok\n")

    outp = OUT / f"canon_{args.tag}.jsonl"
    with outp.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    if not args.full:
        for grp in sorted(set(r["seed_group"] for r in results)):
            sub = [r for r in results if r["seed_group"] == grp]
            print(f"\n{'='*78}\n{grp}  [{sub[0]['kind']}]\n{'='*78}")
            for r in sub:
                if r["ok"]:
                    ot = "  [OFF-TOPIC]" if r.get("off_topic") else ""
                    print(f"  IN : {r['name']}{ot}")
                    print(f"  RSN: {r.get('reasoning','')}")
                    print(f"  OUT: {r['canonical'] if r['canonical'] else '<null - bare heading>'}")
                else:
                    print(f"  IN : {r['name']}  -> FAILED ({r.get('error')})")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    asyncio.run(main())
