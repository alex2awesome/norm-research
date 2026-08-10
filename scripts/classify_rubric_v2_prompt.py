"""
v2 GPT-5-mini classifier prompt (per-task).

This file used to contain a monolithic 21-shot prompt. It now imports the
shared per-task taxonomy from task_taxonomy.py and the per-task KEEP/DROP
exemplars from classify_rubric_llama_prompt.py — single source of truth
across both classifier models.

Differences from the Llama version:
  - SHARED_BASE is kept similar (same taxonomy + KEEP rule + disambiguation)
    but slightly tightened — gpt-5-mini follows instructions better than
    Llama-3.3 and doesn't need the same triple-reinforcement.
  - Uses strict response_format=json_schema (gpt-5-mini honors it reliably);
    no SCHEMA_HINT in the user prompt.
  - JSON_SCHEMA_CLASSIFY_V2 is preserved for backwards-compat with callers
    that import it directly.
"""

# Single source of truth for per-task content + exemplars
from task_taxonomy import TASK_INFO, as_semicolon_list
from classify_rubric_llama_prompt import TASK_EXEMPLARS

JSON_SCHEMA_CLASSIFY_V2 = {
    "name": "rubric_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning":     {"type": "string"},
            "target":        {"type": "string", "enum": ["work","production_process","submission_form","evaluation_judgment","selection_criterion","meta_artifact","actor_attribute","service_or_logistics"]},
            "actor":         {"type": "string", "enum": ["producer","evaluator","gatekeeper","consumer","platform"]},
            "action":        {"type": "string", "enum": ["produce","constrain","judge","select","transact","distribute","describe"]},
            "keep":          {"type": "string", "enum": ["keep","drop","borderline"]},
            "inputs":        {"type": "array", "items": {"type": "string"}, "minItems": 2, "description": "2-6 short noun phrases naming the features/observations the rubric requires considering. Must contain at least 2 items, never empty."},
            "justification": {"type": "string"},
        },
        "required": ["reasoning","target","actor","action","keep","inputs","justification"],
    },
}


SHARED_BASE = """You classify a single rubric extracted from a web page on three closed-vocabulary axes (target, actor, action) and emit a keep/drop decision.

# Context

The page was collected for a research study on rubrics across 11 tasks. The earlier extraction step over-extracted: it captured many service-policy / transactional / meta-artifact items alongside genuine work-criteria. Your job: label each rubric so the noise can be filtered out and so kept items can be analyzed along the (target, actor, action) axes.

# KEEP RULE

```
KEEP iff target ∈ {work, production_process, submission_form, evaluation_judgment, selection_criterion}
     AND action ∈ {produce, constrain, judge, select}
otherwise → DROP
```

Use `borderline` only when the rule mislabels a genuine work-criterion.

# TARGET — what the directive is about

- **work** — the actual creative product / invention / application / brief / comment being produced or judged. The directive describes a property the OUTPUT itself must have. (Substantive content, not formatting.)
- **production_process** — how the work is made: a step the producer DOES during creation (outline before drafting, validate with stakeholders).
- **submission_form** — a copy/format constraint that shapes the work itself (length, file type, anonymity, citation format INSIDE the work, ordering of sections).
- **evaluation_judgment** — the act/discipline of judging the work. Two sub-types: judgment-discipline ("be honest and impartial") and judgment-question ("ask whether X about the given work").
- **selection_criterion** — criteria for CHOOSING AMONG OPTIONS / SELECTING FROM A POOL, not for evaluating a single given work. Examples: "Vendor must have 5+ years experience to be considered" (vendor selection); "Top news is decided by subscriber engagement metrics" (news curation); "Admissions weighs research-experience over test scores" (applicant short-listing). Distinct from `evaluation_judgment` because selection is *comparative across candidates*. KEEP-side.
- **meta_artifact** — a derived object created BY THE EVALUATOR OR PLATFORM about the work AFTER it is produced — not the work itself. Bibliographies, citations, references INSIDE the work are part of the work, NOT meta-artifacts. (Task-specific meta-artifact examples in the task section below.)
- **actor_attribute** — a property of the people producing/evaluating, not the work (qualifications, biographies, hiring criteria, organizational eligibility).
- **service_or_logistics** — platform / transaction / scheduling / internal operations.

# ACTOR — who is being directed

- **producer** — the writer / inventor / applicant — the one creating the work
- **evaluator** — the reviewer / judge / examiner — the one assessing
- **gatekeeper** — editor / publisher / committee chair / clerk — institutional middleman
- **consumer** — customer / reader / audience
- **platform** — the service operator / website / agency itself

# ACTION — what kind of directive it is

- **produce** — DO X to make the work (active verb, producer's creative activity)
- **constrain** — the work MUST satisfy X (passive constraint on the output)
- **judge** — ASK / EVALUATE X about the work (active verb, evaluator's assessment of one work)
- **select** — CHOOSE / RANK / SHORT-LIST among multiple candidates using criterion X (comparative selection). Pairs naturally with target=selection_criterion. Example: "Pick the news stories with highest subscriber-engagement potential"; "Choose vendors with 5+ years experience".
- **transact** — pay, sign, register, schedule (operational, not creative)
- **distribute** — publish, share, archive a derived artifact
- **describe** — passive description of state, with no actor directive

# Disambiguation rules

**(1) work vs evaluation_judgment.** Same rule can read either way. Decide by audience:
- If page's `intended_audience` is producers (authors, applicants), prefer **work / constrain**.
- If audience is evaluators (examiners, reviewers, judges), prefer **evaluation_judgment / judge**.
- If the rubric is phrased as a QUESTION ("Does the X?"), prefer **evaluation_judgment / judge**.
- If unsure, default to **work / constrain**.

**(2) work vs submission_form.** Both are about the work-instance.
- **work** = SUBSTANTIVE content (what the output must contain or be true about).
- **submission_form** = FORMAT/PACKAGING (length, file type, font, anonymity).

**(3) work vs meta_artifact.** Distinguish original-output from derived-object.
- **work** = what the producer creates.
- **meta_artifact** = what gets created BY THE EVALUATOR or PLATFORM about the work afterward (review, score, certificate, decision letter).
- Test: if the directive disappeared, would the WORK be different (work) or would the REVIEW/SCORE/CERTIFICATE be different (meta_artifact)?
- Bibliographies and citations INSIDE the work are part of the work, not meta-artifacts.

**(4) describe vs transact.**
- **describe** = static fact about the platform/people ("our reviewers are X").
- **transact** = procedural step ("we assign two reviewers per submission").

**(5) actor_attribute vs service_or_logistics.** Both → DROP.
- **actor_attribute** = about the *people* (qualifications, biographies).
- **service_or_logistics** = about the *operations* (pricing, scheduling, mechanics).

# Republication-license boilerplate (cross-task — always DROP regardless of task)

Many crawled pages republish public-domain texts from sources like Project Gutenberg (historical literature, math, science, journalism, comedy, legal treatises). These pages carry standard license boilerplate that some extractors mistakenly catch:
- "Do not unlink or detach or remove the full Project Gutenberg License terms from this work"
- "Pay a royalty fee of 20% of gross profits to the trademark owner"
- "AS-IS with no other warranties, express or implied"
- "Disclaimer of warranties and limitation of liability"

These clauses appear regardless of the parent task. They are always:
  target = service_or_logistics, action = transact OR describe, keep = drop.

Same applies to Creative Commons attribution boilerplate, GFDL clauses, and similar republication-license terms.

# The `inputs` field

For each rubric, emit `inputs` — a list of 2-6 short noun phrases naming the SPECIFIC ASPECTS of the work (or its production / evaluation context) that someone applying this rubric must attend to. Inputs are the THICK PART of the rule.

## CRITICAL: inputs must be SPECIFIC, not generic

DO NOT write generic catch-alls like:
- ❌ "input_text", "the work", "the manuscript", "the article", "evidence", "conclusions", "the process"

These are useless — they apply to every rubric. Name the PARTICULAR feature/judgment the rule is asking for.

## The thick/thin frame

THIN rule = few simple inputs, mechanically observable. Example: "submissions must be ≤5000 words" → inputs = ["word count of the manuscript"].
THICK rule = many inputs, or inputs requiring expert judgment.

## Genuinely holistic rubrics (escape hatch)

Some rubrics are inherently holistic ("the story should be entertaining"; "the code should be maintainable"; "the proof should be elegant"). For these, it's OK to use "the work as a whole" or similar holistic phrases. BUT this should be a last resort, not the default. If the rubric mentions any specific feature, prefer naming THAT feature instead.

## NEVER emit empty inputs arrays

`inputs` is REQUIRED and must contain AT LEAST 2 items. NEVER emit `"inputs": []`. If you genuinely can't think of specific inputs, use the holistic placeholder — but that's still TWO concrete strings (e.g., `["the work as a whole", "overall reader engagement"]`), not an empty list.

Empty inputs arrays defeat the purpose of the field. Always populate with at least 2 noun phrases.

## For DROP rubrics

Even when DROPPING, emit `inputs` based on what the rubric literally requires (e.g., "Filing fee deadline" → inputs = ["the filing date", "the 1-month deadline window"]). Always present, never empty.

(Worked task-specific examples appear in the task-specific section below.)
"""


def build_prompt_for_task(task: str) -> str:
    """Assemble shared base + task-specific module + final reminder for a given task.

    Imports per-task content from task_taxonomy.TASK_INFO (single source of truth)
    and per-task KEEP/DROP exemplars from classify_rubric_llama_prompt.TASK_EXEMPLARS
    (shared across both classifier models).
    """
    info = TASK_INFO.get(task)
    exemp = TASK_EXEMPLARS.get(task)
    if info is None or exemp is None:
        task_section = (
            f"# Parent task\n\n"
            f"This rubric came from a page collected for parent task: **{task}**.\n"
            f"Identify the work-of-this-task from the page context.\n"
        )
    else:
        task_section = f"""# Parent task: {task}

## Work-of-this-task
**{info['work'][0]}**

## Task-specific meta_artifact patterns (these are derived objects → DROP under the KEEP rule)
{as_semicolon_list(info['meta_artifacts'])}

## Task-specific service/logistics noise (DROP)
{as_semicolon_list(info['service_or_logistics'])}

## Task-specific actor_attribute noise (DROP)
{as_semicolon_list(info['actor_attribute'])}

## Task-specific `inputs` examples (specific noun phrases, not generic)
{chr(10).join('  - ' + ex for ex in info['inputs_examples'])}

## Worked KEEP example for {task}
{exemp['keep_exemplar']}

## Worked DROP example for {task}
{exemp['drop_exemplar']}
"""
    closing = """
# Final reminders

- Identify the work-of-this-task FIRST, anchored to the parent task above.
- Apply the KEEP rule mechanically.
- Use `borderline` only when a genuine work-criterion would be mislabeled.
- Output strictly conforming JSON. No prose outside it.
"""
    return SHARED_BASE + "\n" + task_section + closing


# Backwards-compat default (creative-writing) — old callers used SYSTEM_PROMPT_CLASSIFY_V2
SYSTEM_PROMPT_CLASSIFY_V2 = build_prompt_for_task("creative-writing")
