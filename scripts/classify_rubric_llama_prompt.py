"""
Llama-3.3-70B classifier prompt (per-task).

Adapts the gpt-5-mini v2 classifier prompt to be Llama-friendly:
  - Per-task structure (build_prompt_for_task) — only the relevant task's
    exemplars + DROP-noise are in the prompt for each call. Mirrors the v5
    extractor architecture.
  - JSON_OBJECT mode (no strict schema) + JSON salvage in caller. OpenRouter's
    Llama providers don't reliably honor json_schema; json_object is more
    robust.
  - KEEP rule restated 3 times (top, middle, end) — Llama tends to drift on
    long prompts; reinforcement helps.
  - Audience-of-instruction rule explicit at the top.
  - Per-task: 1 positive (KEEP) exemplar + 1 task-shaped DROP exemplar drawn
    from real v1 corpus noise (deadlines, fees, applicant biographies, etc.).
"""

# ──────────────────────────────────────────────────────────────────────────
# JSON schema (kept for callers that DO want strict output, e.g. vLLM
# guided_json on sk3; OpenRouter callers should use json_object mode).
# ──────────────────────────────────────────────────────────────────────────
JSON_SCHEMA_LLAMA = {
    "name": "rubric_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning":         {"type": "string"},
            "target":            {"type": "string", "enum": ["work","production_process","submission_form","evaluation_judgment","selection_criterion","meta_artifact","actor_attribute","service_or_logistics"]},
            "actor":             {"type": "string", "enum": ["producer","evaluator","gatekeeper","consumer","platform"]},
            "action":            {"type": "string", "enum": ["produce","constrain","judge","select","transact","distribute","describe"]},
            "inputs":            {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "verifiability_type": {"type": "string", "enum": ["computational","factual","consistency","procedural","statistical","causal","completeness","pragmatic","normative"]},
            "tractability":      {"type": "string", "enum": ["programmatic_check","llm_judge","expert_judgment","intractable"]},
            "requires_lookup":   {"type": "boolean"},
            "specificity":       {"type": "string", "enum": ["vague","general","specific","hyper_specific"]},
            "keep":              {"type": "string", "enum": ["keep","drop","borderline"]},
            "justification":     {"type": "string"},
        },
        "required": ["reasoning","target","actor","action","inputs","verifiability_type","tractability","requires_lookup","specificity","keep","justification"],
    },
}


SCHEMA_HINT = """
You MUST respond with a single JSON object. Generate the fields IN THIS ORDER — earlier decisions inform later ones (especially: inputs informs verifiability_type informs tractability informs specificity):

{
  "reasoning": "1-2 sentences reasoning through the work-of-this-task.",
  "target":  one of ["work","production_process","submission_form","evaluation_judgment","selection_criterion","meta_artifact","actor_attribute","service_or_logistics"],
  "actor":   one of ["producer","evaluator","gatekeeper","consumer","platform"],
  "action":  one of ["produce","constrain","judge","select","transact","distribute","describe"],
  "inputs":  ["≥2 specific noun phrases — features/observations someone applying the rubric must attend to"],

  "verifiability_type": one of [
    "computational",   // a deterministic program over the work's own text/structure can verify it. {EX_computational}
    "factual",         // verification requires looking up an external authoritative source. {EX_factual}
    "consistency",     // internal cross-check within the work itself. {EX_consistency}
    "procedural",      // compliance with a defined process rule applied to the work. {EX_procedural}
    "statistical",     // quantitative re-analysis of the work's claims/data. {EX_statistical}
    "causal",          // an X-caused-Y claim that requires causal/counterfactual reasoning. {EX_causal}
    "completeness",    // did the work address every required element / cover everything? {EX_completeness}
    "pragmatic",       // verified by downstream effect / running the work / observing impact. {EX_pragmatic}
    "normative"        // pure value judgment; quality / aesthetic / significance. {EX_normative}
  ],

  "tractability": one of [
    "programmatic_check",   // can be implemented as deterministic code. {EX_t_programmatic_check}
    "llm_judge",            // an LLM can judge it from the work alone, no special expertise. {EX_t_llm_judge}
    "expert_judgment",      // domain training required. {EX_t_expert_judgment}
    "intractable"           // pure taste; faultless disagreement expected. {EX_t_intractable}
  ],

  "requires_lookup": boolean,  // does applying this rubric require consulting an EXTERNAL AUTHORITATIVE SOURCE that lives outside the work itself (statute, database, registry, regulation, doctrine, taxonomy, prior-art)? See per-task examples below.

  "specificity": one of [
    "vague",          // generic platitude applicable to nearly any work in this task. {EX_s_vague}
    "general",        // operative but broadly applicable. {EX_s_general}
    "specific",       // domain-specific named criterion with formal anchor. {EX_s_specific}
    "hyper_specific"  // explicit numeric thresholds or named formats. {EX_s_hyper_specific}
  ],

  "keep":           one of ["keep","drop","borderline"],
  "justification":  "1 sentence"
}

DISAMBIGUATION for verifiability_type (abstract definitions — task-specific worked examples live in the per-task section below; consult them, not these definitions, when in doubt).

**STRONG RULE for `factual`:** Use `factual` ONLY when verifying the rubric requires LOOKUP AGAINST AN EXTERNAL AUTHORITATIVE SOURCE that exists OUTSIDE the work itself. If `verifiability_type=factual` then `requires_lookup` MUST be true. If verification means "look at the work itself," do NOT use factual — pick procedural / consistency / completeness / normative instead.

- procedural vs computational: procedural = compliance with an EXTERNAL PROCESS RULE applied to the work (timing window, required template, mandated disclosure). computational = the work's OWN TEXT/STRUCTURE (length, format, regex match, type check).
- pragmatic vs computational: pragmatic = DOWNSTREAM EFFECT (run the work or observe its impact). computational = TEXT-LEVEL inspection of the work.
- statistical vs computational: statistical = QUANTITATIVE RE-ANALYSIS (recompute values, check methodological adequacy). computational = SHALLOW TEXT CHECK.
- consistency vs factual: consistency = internal cross-reference WITHIN the work (one part of the work matches another). factual = check against external authoritative source.
- completeness vs procedural: completeness = did the output COVER EVERYTHING relevant (did all required elements appear)? procedural = was a defined process followed?
- normative vs pragmatic: normative = SUBJECTIVE value judgment about quality. pragmatic = OBJECTIVE downstream outcome (measurable result).
- **normative is the LAST-RESORT label.** Before assigning normative, check whether the rubric is really about: a process (procedural), an external lookup (factual), an internal cross-check (consistency), a downstream goal (pragmatic), a coverage requirement (completeness). Only if none of those fit AND the rubric requires subjective quality judgment → normative.

DISAMBIGUATION for tractability — pick the LEAST-EXPENSIVE label that suffices:

Ladder, cheapest to most expensive: **programmatic_check < llm_judge < expert_judgment < intractable**. Always pick the cheapest label that genuinely fits.

- **programmatic_check** if you can imagine a deterministic Python function returning yes/no.
- **llm_judge** if a careful educated reader can apply the rubric to the work and reach a consistent answer. THIS IS THE DEFAULT FOR MOST NORMATIVE RUBRICS — assigning llm_judge does NOT require the rubric to be quantifiable, only that the work itself is readable by a non-specialist.
- **expert_judgment** ONLY when domain training is REQUIRED to even read the work or apply the criterion — either because the work itself is in a specialist register (legal brief, scientific paper in a technical subfield, patent claim language) OR because the rule cites specialist concepts a layperson wouldn't know how to apply (named legal doctrines, statutory standards, methodological adequacy criteria).
- **intractable** is RARE. Use it ONLY when domain experts themselves predictably disagree by design (cross-cultural humor, value-pluralistic ethics). If experts converge, it's expert_judgment — not intractable.

Common error: choosing expert_judgment when llm_judge would do. ASK YOURSELF: "would a smart undergraduate be able to apply this rubric to the work after a few minutes of reading?" If yes → llm_judge. If they'd need specialized training first → expert_judgment.

DISAMBIGUATION for specificity:
- hyper_specific = NUMBERS / THRESHOLDS / NAMED FORMATS present in the rule
- specific = DOMAIN-SPECIFIC NAMED CRITERION (statute, section, formal-rule anchor), no explicit threshold
- general = operative but BROADLY applicable across many works in the task
- vague = PLATITUDE with no operational handle

(Task-specific worked examples for each of these three axes appear in the per-task section below. Use those examples to anchor your labels for this task.)

Return ONLY the JSON object. No prose outside it. No markdown fences.
"""


# ──────────────────────────────────────────────────────────────────────────
# Shared base — universal taxonomy, KEEP rule, disambiguation
# ──────────────────────────────────────────────────────────────────────────
SHARED_BASE = """You classify a single rubric extracted from a web page on three closed-vocabulary axes (target, actor, action) and emit a keep/drop decision.

# Context

The page was collected for a research study on rubrics across 11 tasks. The earlier extraction step over-extracted: it captured many service-policy / transactional / meta-artifact items alongside genuine work-criteria. Your job: label each rubric so the noise can be filtered out and so kept items can be analyzed along the (target, actor, action) axes.

# KEEP RULE (memorize this)

```
KEEP iff target ∈ {work, production_process, submission_form, evaluation_judgment, selection_criterion}
     AND action ∈ {produce, constrain, judge, select}
otherwise → DROP
```

(Use `borderline` only when the rule mislabels a genuine work-criterion.)

# TARGET — what the directive is about

- **work** — the actual creative product / invention / application / brief / comment being produced or judged. The directive describes a property the OUTPUT itself must have. (Substantive content, not formatting.)
- **production_process** — how the work is made: a step the producer DOES during creation (outline before drafting, validate with stakeholders).
- **submission_form** — a copy/format constraint that shapes the work itself (length, file type, anonymity, citation format INSIDE the work, ordering of sections).
- **evaluation_judgment** — the act/discipline of judging the work. Two sub-types: judgment-discipline ("be honest and impartial") and judgment-question ("ask whether X about the given work").
- **selection_criterion** — criteria for CHOOSING AMONG OPTIONS / SELECTING FROM A POOL, not for evaluating a single given work. Examples: "Vendor must have 5+ years experience to be considered" (vendor selection); "Top news is decided by subscriber engagement metrics" (news curation); "Admissions committee weighs research-experience over test scores" (applicant short-listing). Distinct from `evaluation_judgment` because selection is *comparative across candidates* rather than evaluation of one work in isolation. KEEP-side.
- **meta_artifact** — a derived object created BY THE EVALUATOR OR PLATFORM about the work AFTER it is produced — not the work itself. Bibliographies, citations, references INSIDE the work are part of the work, NOT meta-artifacts. (Task-specific meta-artifact examples in the task section below.)
- **actor_attribute** — a property of the people producing/evaluating, not the work (qualifications, biographies, hiring criteria, organizational eligibility).
- **service_or_logistics** — platform / transaction / scheduling / internal operations. Pricing, deadlines, account management, internal staffing, queueing.

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
- "Commercial redistribution additional obligations / royalty / refund / defect handling"

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

THIN rule = few simple inputs, mechanically observable.
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


# ──────────────────────────────────────────────────────────────────────────
# Per-task EXEMPLARS. Shared per-task content (work, meta_artifacts,
# service_or_logistics, actor_attribute) is imported from task_taxonomy.py
# (single source of truth across all prompt files). This file only adds the
# CLASSIFIER-SPECIFIC fields:
#   - keep_exemplar: a real KEEP rubric for this task with full labels
#   - drop_exemplar: a real DROP rubric for this task with full labels
# ──────────────────────────────────────────────────────────────────────────
from task_taxonomy import TASK_INFO, as_semicolon_list

TASK_EXEMPLARS: dict = {
    "creative-writing": {
        "keep_exemplar": """RUBRIC: name="Honest and impartial evaluation" / desc="Kirkus Indie reviewers are experienced professionals who honestly and impartially evaluate the books they receive."
LABELS: target=evaluation_judgment, actor=evaluator, action=judge, keep=keep
WHY: Discipline criterion telling the reviewer how to judge the book.""",
        "drop_exemplar": """RUBRIC: name="Review lengths by option" / desc="Traditional Reviews: 250 words. Expanded Review: About 500 words. Picture Book: 200 words."
LABELS: target=meta_artifact, actor=platform, action=constrain, keep=drop
WHY: Constrains the *review* (a derived artifact about the book), not the book itself. The literal phrase "review lengths" can mislead — the constraint is on the deliverable Kirkus produces, NOT on the manuscript. (Compare with "submissions must be ≤5000 words" which constrains the work itself → KEEP.)

ADDITIONAL DROP example: name="Publication control / confidentiality" / desc="You may choose to publish your review on KirkusReviews.com... If you receive a negative review, you can choose NOT to publish it."
LABELS: target=meta_artifact, actor=consumer, action=distribute, keep=drop
WHY: Distribution policy for the review (a derived object), not the book itself.""",
    },

    "peer-review": {
        "keep_exemplar": """RUBRIC: name="First-page content and anonymity" / desc="The first page should include title, abstract, content areas, and ID number, but not names or affiliations of authors."
LABELS: target=submission_form, actor=producer, action=constrain, keep=keep
WHY: Format constraint on the submitted paper.""",
        "drop_exemplar": """RUBRIC: name="Two-phase review: Phase 2 allocation" / desc="Each paper will be allocated additional reviewers in Phase 2. These reviewers will not be given access to Phase 1 reviews until after submitting their own."
LABELS: target=service_or_logistics, actor=gatekeeper, action=transact, keep=drop
WHY: Internal reviewer-allocation procedure — no implicit constraint on the paper.""",
    },

    "math-stackexchange": {
        "keep_exemplar": """RUBRIC: name="Proof / rigorous justification requirement" / desc="When a problem states 'Find (with proof)' or 'Prove or disprove', the solver must provide a rigorous proof of the asserted statement or a valid counterexample together with justification."
LABELS: target=work, actor=producer, action=constrain, keep=keep
WHY: Substantive content constraint on the math solution.""",
        "drop_exemplar": """RUBRIC: name="No fee for access unless complying with commercial terms" / desc="Do not charge a fee for access to, viewing, displaying, performing, copying or distributing any Project Gutenberg works unless you comply with..."
LABELS: target=service_or_logistics, actor=platform, action=transact, keep=drop
WHY: Project Gutenberg license boilerplate appearing on a historical math text — not a math criterion.""",
    },

    "news-homepages": {
        "keep_exemplar": """RUBRIC: name="A commitment to Nonpartisanship and Fairness" / desc="Signatory organizations fact-check claims using the same standard for every fact check. They do not concentrate their fact-checking on any one side."
LABELS: target=evaluation_judgment, actor=evaluator, action=judge, keep=keep
WHY: Discipline criterion for the act of fact-checking.""",
        "drop_exemplar": """RUBRIC: name="Subscription-driving potential" / desc="Subscription-driving potential"
LABELS: target=service_or_logistics, actor=platform, action=describe, keep=drop
WHY: Business metric for the news outlet, not a journalistic quality criterion.""",
    },

    "press-releases": {
        "keep_exemplar": """RUBRIC: name='Definition of "intentional" disclosure' / desc="A selective disclosure is 'intentional' when the person making it either knows, or is reckless in not knowing, that the information is both material and nonpublic."
LABELS: target=evaluation_judgment, actor=evaluator, action=judge, keep=keep
WHY: Defines the legal criterion the regulator uses to judge whether a disclosure is intentional.""",
        "drop_exemplar": """RUBRIC: name="Timely public updates and documentation availability" / desc="The page provides news releases, operational updates, newsletters, documents, and a point of contact for press inquiries."
LABELS: target=service_or_logistics, actor=platform, action=distribute, keep=drop
WHY: Operational distribution requirement for the agency website, not a press-release quality criterion.""",
    },

    "code-review": {
        "keep_exemplar": """RUBRIC: name="Resources should represent a single API object" / desc="A Terraform resource should be a declarative representation of single component, usually with create, read, delete, and optionally update methods."
LABELS: target=work, actor=producer, action=constrain, keep=keep
WHY: Design constraint on the Terraform resource (the code).""",
        "drop_exemplar": """RUBRIC: name="Install our code-review IDE plugin" / desc="To get inline review suggestions, install our IDE plugin from the marketplace and authenticate with your GitHub account."
LABELS: target=service_or_logistics, actor=platform, action=transact, keep=drop
WHY: Tooling installation instructions — operational, not a code-quality criterion.""",
    },

    "grant-funding": {
        "keep_exemplar": """RUBRIC: name="Independence of Thought" / desc="We seek visionary thinkers who are curious, open-minded, analytical, eager for cross-cultural perspective, and genuinely excited to boldly and creatively address our world's important challenges."
LABELS: target=evaluation_judgment, actor=evaluator, action=judge, keep=keep
WHY: Judgment criterion the selection committee applies to the application.""",
        "drop_exemplar": """RUBRIC: name="Eligibility — fully independent researcher status" / desc="You are a fully independent researcher with access to your own lab space and with the ability to recruit and be registered as the primary supervisor of PhD students."
LABELS: target=actor_attribute, actor=producer, action=describe, keep=drop
WHY: Person-level eligibility (about the PI), not a constraint on the proposal.""",
    },

    "humor": {
        "keep_exemplar": """RUBRIC: name="Clear ironic persona" / desc="Is there a clear ironic persona? Mocks bigots by impersonating them. The persona is a character who inhabits the performer's body — views may not be shared by the real person."
LABELS: target=work, actor=producer, action=constrain, keep=keep
WHY: Content constraint on the comedy performance.""",
        "drop_exemplar": """RUBRIC: name="Application deadline" / desc="Application Deadline: July 15, 2024"
LABELS: target=service_or_logistics, actor=platform, action=transact, keep=drop
WHY: Festival/competition deadline — pure scheduling, not a comedy criterion.""",
    },

    "legal-outcome-prediction": {
        "keep_exemplar": """RUBRIC: name="Statement of the Case (FRAP 28(a)(6) requirement)" / desc="The statement of the case must include a narrative statement of all of the facts necessary for the Court to reach the conclusion which the brief desires with references to the specific pages in the appendix."
LABELS: target=submission_form, actor=producer, action=constrain, keep=keep
WHY: Content/format requirement on the brief.""",
        "drop_exemplar": """RUBRIC: name="Deadline for submitting comments on the information collection" / desc="Written comments must be submitted on or before August 1, 2011."
LABELS: target=service_or_logistics, actor=platform, action=transact, keep=drop
WHY: Administrative deadline.""",
    },

    "notice-and-comment": {
        "keep_exemplar": """RUBRIC: name="Full factual and legal basis (§1.31(b)(3))" / desc="Petitions must include a full statement of the factual and legal basis on which the petitioner relies for the action requested, including all relevant facts, views, argument, and data."
LABELS: target=work, actor=producer, action=constrain, keep=keep
WHY: Substantive content constraint on the petition.""",
        "drop_exemplar": """RUBRIC: name="Comments deadline" / desc="Comments must be received on or before February 27, 2026."
LABELS: target=service_or_logistics, actor=platform, action=transact, keep=drop
WHY: Administrative deadline.""",
    },

    "patents": {
        "keep_exemplar": """RUBRIC 1 (work + constrain): name="Statutory category requirement (Step 1)" / desc="A claim must fall within one or more of the four statutory categories enumerated in 35 U.S.C. 101: process, machine, manufacture, or composition of matter."
LABELS: target=work, actor=producer, action=constrain, keep=keep
WHY: Stated PASSIVELY ("a claim must..."). Substantive constraint on the patent claim.

RUBRIC 2 (evaluation_judgment + judge): name="Step 2B: Whether claim recites additional elements amounting to 'significantly more'" / desc="When claims are directed to a judicial exception, evaluate whether additional elements, individually or in ordered combination, provide 'significantly more' than the judicial exception itself."
LABELS: target=evaluation_judgment, actor=evaluator, action=judge, keep=keep
WHY: MPEP page audience = patent EXAMINERS. The rubric NAME starts with "Whether..." (interrogative) and the verb is "evaluate". Both signals point to (evaluation_judgment, judge), NOT (work, constrain). DO NOT default to work/constrain just because the rubric mentions the claim — the disambiguation rule says: question-phrased + examiner audience → evaluation_judgment.""",
        "drop_exemplar": """RUBRIC: name="Fees (payment, reduction, refund)" / desc="Requirements and procedures for payment of patent and official fees, rules for fee reduction and refund."
LABELS: target=service_or_logistics, actor=platform, action=transact, keep=drop
WHY: Fee schedule — operational, not a patent-quality criterion.
NOTE: a rule like 'application reference list must be in numerical order' would be KEEP (work / submission_form), since bibliographies INSIDE the application are part of the work.""",
    },
}


# ──────────────────────────────────────────────────────────────────────────
# Prompt assembly
# ──────────────────────────────────────────────────────────────────────────
def build_prompt_for_task(task: str) -> str:
    """Assemble shared base + task-specific module + final reminder for a given task.

    Pulls work / meta-artifact / service-or-logistics / actor-attribute content
    from task_taxonomy.TASK_INFO (single source of truth) and pairs it with the
    classifier-specific keep_exemplar + drop_exemplar from TASK_EXEMPLARS in
    this file.
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
        axis_block = ""
        if "axis_examples" in info:
            axis_block = "\n## Task-specific (verifiability_type, tractability, specificity) examples — anchor your labels to these\n" + \
                         "\n".join('  - ' + ex for ex in info['axis_examples']) + "\n"
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
{axis_block}
## Worked KEEP example for {task}
{exemp['keep_exemplar']}

## Worked DROP example for {task}
{exemp['drop_exemplar']}
"""
    # Fill SCHEMA_HINT slots with task-specific examples for each enum value
    # (verifiability_type, tractability, specificity). Generic fallback for
    # types that don't have a task-specific example.
    fallback = "(see per-task examples below)"
    filled_hint = SCHEMA_HINT
    verif_inline = (info or {}).get("verifiability_inline", {})
    for vt in ["computational","factual","consistency","procedural","statistical",
               "causal","completeness","pragmatic","normative"]:
        filled_hint = filled_hint.replace("{EX_" + vt + "}", verif_inline.get(vt, fallback))
    tract_inline = (info or {}).get("tractability_inline", {})
    for tt in ["programmatic_check","llm_judge","expert_judgment","intractable"]:
        filled_hint = filled_hint.replace("{EX_t_" + tt + "}", tract_inline.get(tt, fallback))
    spec_inline = (info or {}).get("specificity_inline", {})
    for sp in ["vague","general","specific","hyper_specific"]:
        filled_hint = filled_hint.replace("{EX_s_" + sp + "}", spec_inline.get(sp, fallback))

    closing = """
# Final reminders

- Identify the work-of-this-task FIRST, anchored to the parent task above.
- Apply the KEEP rule: target ∈ {work, production_process, submission_form, evaluation_judgment, selection_criterion} AND action ∈ {produce, constrain, judge, select} → KEEP. Otherwise DROP.
- Use `borderline` ONLY when a genuine work-criterion would be mislabeled by the rule.
- Output strictly conforming JSON. No prose outside it. No markdown fences.
""" + filled_hint
    return SHARED_BASE + "\n" + task_section + closing


# Backwards-compat default (creative-writing)
SYSTEM_PROMPT_LLAMA = build_prompt_for_task("creative-writing")
