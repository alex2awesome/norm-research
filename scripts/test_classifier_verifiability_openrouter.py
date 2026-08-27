"""
Probe Llama-3.3-70B on the proposed verifiability/tractability/specificity
fields — see if it can label them reliably before we restart the sk3 run.

Uses an EXTENDED classifier prompt with 3 new fields appended to the current
classify_rubric_llama_prompt.SHARED_BASE. We test on 12 hand-picked rubrics
spanning the 9-type verifiability spectrum.
"""

from __future__ import annotations
import asyncio, json, re, sys
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
sys.path.insert(0, str(ROOT / "scripts"))

from classify_rubric_llama_prompt import build_prompt_for_task as base_build, SCHEMA_HINT
from openai import AsyncOpenAI

KEY = (Path.home() / ".openrouter-api-key.txt").read_text().strip()


# Extended schema hint with the new fields appended
EXTENDED_HINT = """
You MUST respond with a single JSON object. Generate the fields IN THIS ORDER — earlier decisions inform later ones:

{
  "reasoning": "1-2 sentences",
  "target": one of ["work","production_process","submission_form","evaluation_judgment","selection_criterion","meta_artifact","actor_attribute","service_or_logistics"],
  "actor":  one of ["producer","evaluator","gatekeeper","consumer","platform"],
  "action": one of ["produce","constrain","judge","select","transact","distribute","describe"],
  "inputs": ["≥2 specific noun phrases — the features/observations someone applying the rubric must attend to"],

  "verifiability_type": one of [
    "computational",   // a DETERMINISTIC program over the work's TEXT can check it. Output is a function of the work alone. Examples: word count, file format, regex match, type-checking, syntactic structure. NOT for process compliance or for "downstream effect" rules.
    "factual",         // check against an EXTERNAL AUTHORITATIVE SOURCE (database, registry, statute). Examples: citation exists in Semantic Scholar, statute §X actually says what's quoted, prior art exists in USPTO.
    "consistency",     // INTERNAL CROSS-CHECK within the work. No external source. Examples: claims in abstract match results section; PR description matches the diff; argument doesn't contradict itself.
    "procedural",      // DEFINED PROCESS was followed (compliance with rules ABOUT the process, not about the work content). Examples: PR template filled in, comment filed within period, IRB approval shown, pre-registration linked. The work's CONTENT is fine — what's checked is that procedural steps occurred.
    "statistical",     // QUANTITATIVE CLAIMS are supported by data + appropriate methods. Examples: p-values match test statistics, power calculation justified, sample size adequate for effect size. Distinct from computational because re-analysis is required, not just text-level checking.
    "causal",          // X-CAUSED-Y claims. But-for causation, proximate cause, mechanism. Counterfactuals. Distinct from factual because no ground-truth source exists.
    "completeness",    // DID THE OUTPUT ADDRESS EVERYTHING relevant? Examples: agency addressed every substantive comment, brief addressed all elements of the claim, paper responded to all reviewer concerns. Coverage is the criterion, not accuracy.
    "pragmatic",       // DID IT ACHIEVE ITS GOAL (downstream)? Examples: did the PR fix the bug (verified by regression test), did the policy reduce X by Y%, did the press release drive coverage. Verified by downstream outcome, not by inspecting the artifact.
    "normative"        // VALUE JUDGMENT — quality / novelty / significance / proportionality / aesthetic merit. The only category with no clean external referent. Examples: "compelling characters", "novel contribution", "elegant proof", "funny", "proportionate punishment".
  ],

  "tractability": one of [
    "programmatic_check",  // a code program with deterministic output exists or can be written
    "llm_judge",           // an LLM can plausibly judge it from the work alone (no special expertise required to judge)
    "expert_judgment",     // domain expert required (patent examiner, statistician, senior editor)
    "intractable"          // pure taste / no clean operationalization / faultless-disagreement expected
  ],

  "specificity": one of [
    "vague",          // generic platitude applicable to almost any work ("be clear", "write with style", "be original")
    "general",        // operative but broad ("avoid passive voice", "compelling characters", "rigorous methodology")
    "specific",       // names a domain-specific criterion or section/category (e.g., "must satisfy 35 U.S.C. §112 written description", "include a Statement of Facts", "Commitment to Nonpartisanship and Fairness")
    "hyper_specific"  // EXPLICIT THRESHOLDS or enumerable conditions ("≤5000 words", "12pt font, PDF format", "p < 0.05", "4 of the 4 statutory categories")
  ],

  "keep":   one of ["keep","drop","borderline"],
  "justification": "1 sentence"
}

DISAMBIGUATION HINTS for verifiability_type:
- procedural vs computational: if the rule is about COMPLIANCE WITH AN EXTERNAL PROCESS (PR has tests, comment filed by deadline), it's procedural. If it's about the OUTPUT'S OWN PROPERTIES (length, format, regex), it's computational.
- pragmatic vs computational: if the rule's verifier requires running the work / checking downstream effect (does the bug fix actually fix the bug; did readers engage), it's pragmatic. Computational is verification BY INSPECTION of the work text.
- statistical vs computational: statistical requires RE-ANALYSIS or quantitative reasoning (recalculate p-values, validate test choice). Computational is shallow text-level matching.
- pragmatic vs normative: if the goal is OBJECTIVE (bug-fix works, citation count > X), it's pragmatic. If the goal is SUBJECTIVE ("good", "compelling"), it's normative.
- factual vs computational: factual REQUIRES AN EXTERNAL SOURCE (lookup against a registry). Computational is fully self-contained.

DISAMBIGUATION HINTS for specificity:
- hyper_specific = THRESHOLDS PRESENT (numbers, exact lists, named formats)
- specific = DOMAIN-SPECIFIC NAMED CRITERION (statute, section, formal rule), no explicit threshold but operationalizable in that domain
- general = OPERATIVE BUT BROAD (could apply across many works in the task)
- vague = PLATITUDE (applies to almost any creative work, no operational handle)

Return ONLY the JSON object. No prose. No markdown fences.
"""

# Test rubrics spanning the verifiability spectrum + diverse tasks
TEST_RUBRICS = [
    # computational
    {"task":"peer-review","subtask_short":"AAAI submission requirements",
     "rubric_name":"Submissions must be ≤9 pages including references",
     "rubric_description":"All papers must be submitted in PDF format and contain no more than 9 pages of content including references, in single-column 11pt format.",
     "rubric_guidance":"","expected_verif":"computational","expected_tract":"programmatic_check","expected_spec":"hyper_specific"},
    # factual
    {"task":"peer-review","subtask_short":"citation validity",
     "rubric_name":"All cited works must exist and be retrievable",
     "rubric_description":"Authors must ensure all references can be located in standard databases (Semantic Scholar, DBLP, or DOI registry). Broken or fabricated citations are grounds for desk reject.",
     "rubric_guidance":"","expected_verif":"factual","expected_tract":"programmatic_check","expected_spec":"specific"},
    # consistency
    {"task":"peer-review","subtask_short":"internal consistency",
     "rubric_name":"Claims in the abstract must match the results",
     "rubric_description":"The numerical claims and headline contributions stated in the abstract must be supported by the experiments described in the body of the paper.",
     "rubric_guidance":"","expected_verif":"consistency","expected_tract":"llm_judge","expected_spec":"general"},
    # procedural
    {"task":"code-review","subtask_short":"PR template compliance",
     "rubric_name":"PR must include unit tests covering the changed lines",
     "rubric_description":"Every pull request that modifies executable code must include unit tests that cover the modified lines, with coverage ≥90% for new code.",
     "rubric_guidance":"","expected_verif":"procedural","expected_tract":"programmatic_check","expected_spec":"hyper_specific"},
    # statistical
    {"task":"peer-review","subtask_short":"statistical reporting",
     "rubric_name":"Reported p-values must be recalculable from test statistics",
     "rubric_description":"Authors must provide test statistics (t, F, chi-squared) and degrees of freedom such that reviewers can independently recalculate p-values. The statistical test used must be appropriate for the sample size and distribution.",
     "rubric_guidance":"","expected_verif":"statistical","expected_tract":"programmatic_check","expected_spec":"specific"},
    # causal
    {"task":"legal-outcome-prediction","subtask_short":"tort causation",
     "rubric_name":"Plaintiff must establish but-for causation",
     "rubric_description":"The plaintiff bears the burden of showing that the harm would not have occurred but for the defendant's conduct, AND that the conduct was a proximate cause of the harm.",
     "rubric_guidance":"","expected_verif":"causal","expected_tract":"expert_judgment","expected_spec":"specific"},
    # completeness
    {"task":"notice-and-comment","subtask_short":"agency response duty",
     "rubric_name":"Agency must respond to all substantive comments",
     "rubric_description":"The final rule's preamble must address every substantive comment received during the comment period, explaining why the agency did or did not adopt the suggestion.",
     "rubric_guidance":"","expected_verif":"completeness","expected_tract":"llm_judge","expected_spec":"specific"},
    # pragmatic
    {"task":"code-review","subtask_short":"bug fix verification",
     "rubric_name":"The PR must actually fix the bug it claims to fix",
     "rubric_description":"The PR description must reference a bug report (issue ID), and the PR must include a regression test that fails before the change and passes after.",
     "rubric_guidance":"","expected_verif":"pragmatic","expected_tract":"programmatic_check","expected_spec":"specific"},
    # normative — taste
    {"task":"creative-writing","subtask_short":"craft criteria",
     "rubric_name":"Compelling characters",
     "rubric_description":"Characters should be three-dimensional with internal conflicts, recognizable desires, and motivations that drive the plot. The protagonist's interiority should be evident through behavior and decision-making.",
     "rubric_guidance":"","expected_verif":"normative","expected_tract":"expert_judgment","expected_spec":"general"},
    # normative — humor
    {"task":"humor","subtask_short":"stand-up craft",
     "rubric_name":"The joke must land",
     "rubric_description":"The punchline should subvert expectation in a way that produces laughter from a representative audience.",
     "rubric_guidance":"","expected_verif":"normative","expected_tract":"intractable","expected_spec":"vague"},
    # vague
    {"task":"creative-writing","subtask_short":"general advice",
     "rubric_name":"Write with clarity",
     "rubric_description":"Strive for clear, accessible prose.",
     "rubric_guidance":"","expected_verif":"normative","expected_tract":"llm_judge","expected_spec":"vague"},
    # hyper-specific patent
    {"task":"patents","subtask_short":"35 U.S.C. §112",
     "rubric_name":"Written description requirement",
     "rubric_description":"The specification must convey to one of ordinary skill in the art that the inventor had possession of the claimed invention as of the filing date. Adequate written description requires more than recitation of a desired result; it requires describing the structure or steps that produce the result.",
     "rubric_guidance":"","expected_verif":"normative","expected_tract":"expert_judgment","expected_spec":"specific"},
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


async def main():
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=KEY)
    print(f"testing {len(TEST_RUBRICS)} rubrics with extended classifier (verifiability + tractability + specificity)\n")

    correct_verif = correct_tract = correct_spec = 0
    n = 0
    rows = []
    for item in TEST_RUBRICS:
        # Replace the default SCHEMA_HINT with the extended one in the system prompt
        sys_prompt = base_build(item['task']).replace(SCHEMA_HINT, EXTENDED_HINT)
        user_msg = (
            f"PAGE CONTEXT:\n  task: {item['task']}\n  subtask_short: {item['subtask_short']}\n\n"
            f"RUBRIC TO CLASSIFY:\n  name: {item['rubric_name']}\n"
            f"  description: {item['rubric_description']}\n  guidance: {item['rubric_guidance']}\n"
        )
        try:
            resp = await client.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct",
                messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_msg}],
                temperature=0.0, max_tokens=1024, response_format={"type":"json_object"},
            )
            d = salvage_json(resp.choices[0].message.content or "")
        except Exception as e:
            d = None
            print(f"  ERROR on {item['rubric_name']}: {e}")
        if d is None:
            print(f"\n[FAIL] {item['rubric_name']}")
            continue
        n += 1
        v = d.get("verifiability_type"); t = d.get("tractability"); s = d.get("specificity")
        ev = item["expected_verif"]; et = item["expected_tract"]; es = item["expected_spec"]
        v_ok = (v == ev); t_ok = (t == et); s_ok = (s == es)
        correct_verif += v_ok; correct_tract += t_ok; correct_spec += s_ok
        flag_v = "✓" if v_ok else "✗"
        flag_t = "✓" if t_ok else "✗"
        flag_s = "✓" if s_ok else "✗"
        print(f"\n[{item['task']}] {item['rubric_name'][:55]}")
        print(f"  verif:   {flag_v} got={v:<14s}  expected={ev}")
        print(f"  tract:   {flag_t} got={t:<20s}  expected={et}")
        print(f"  specif:  {flag_s} got={s:<16s}  expected={es}")
        rows.append({"rubric":item['rubric_name'], "v":v,"t":t,"s":s,"ev":ev,"et":et,"es":es})

    print(f"\n=== Agreement with hand-labels (n={n}) ===")
    if n:
        print(f"  verifiability_type: {correct_verif}/{n} ({100*correct_verif/n:.0f}%)")
        print(f"  tractability:       {correct_tract}/{n} ({100*correct_tract/n:.0f}%)")
        print(f"  specificity:        {correct_spec}/{n} ({100*correct_spec/n:.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
