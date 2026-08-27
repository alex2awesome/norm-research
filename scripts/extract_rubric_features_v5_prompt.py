"""
v5 (per-task) system prompt for rubric extraction.

This file is now structured as:
  - JSON_SCHEMA_V5: the structured-output schema (unchanged from earlier v5)
  - SHARED_BASE: universal definitions, three tests, KEEP rule, page-handling
    rules, and a corrected meta_artifact definition.
  - TASK_MODULES: dict[task -> per-task content] with the work-of-this-task
    description, task-specific meta_artifact / service_or_logistics /
    actor_attribute examples, the positive exemplar (drawn from real GPT-5-mini
    extractions), and the task-specific anti-pattern.
  - build_prompt_for_task(task): assembles SHARED_BASE + the right TASK_MODULE
    + closing instructions. Each call's prompt only contains content relevant
    to its task — avoids the cross-task contamination problem in the old
    monolithic v5 prompt.

Backwards compat: SYSTEM_PROMPT_V5 is preserved as a default that uses
build_prompt_for_task("creative-writing"); existing callers keep working but
should migrate to build_prompt_for_task(<actual task>).
"""

# ──────────────────────────────────────────────────────────────────────────
# JSON schema (reasoning field first so the model can think before structured
# output)
# ──────────────────────────────────────────────────────────────────────────
JSON_SCHEMA_V5 = {
    "name": "rubric_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "Walk through the work-test BEFORE extracting. 1-3 sentences. "
                    "What kind of page? What is the work-of-this-task? Walk through items, KEEP/DROP each."
                ),
            },
            "orientation": {
                "type": "string",
                "enum": [
                    "research_article", "academic_page", "how_to", "formal_guideline",
                    "blog_post", "dataset", "tutorial", "textbook_excerpt",
                    "professional_standard", "contest_criteria", "stylebook",
                    "course_syllabus", "wiki", "forum_post", "news_article",
                    "error", "other",
                ],
            },
            "intended_audience": {"type": "string"},
            "subtask_short": {"type": "string"},
            "subtask_description": {"type": "string"},
            "subtask_keywords": {"type": "array", "items": {"type": "string"}},
            "subtask_breadth": {
                "type": "string",
                "enum": ["very_narrow", "narrow", "moderate", "broad", "very_broad"],
            },
            "error": {"type": ["string", "null"]},
            "rubrics_metrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "guidance": {"type": "string"},
                        "inputs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "description": "2-6 short noun phrases naming the features/observations someone applying this rubric must consider. MUST contain at least 2 items, never empty. See `# inputs` section in the system prompt.",
                        },
                    },
                    "required": ["name", "description", "guidance", "inputs"],
                },
            },
            # Note: target/action enums for the per-rubric metadata are not in this
            # extractor schema — the extractor produces (name, description, guidance,
            # inputs) only. The classifier emits target/action downstream.
        },
        "required": [
            "reasoning",
            "orientation", "intended_audience",
            "subtask_short", "subtask_description", "subtask_keywords", "subtask_breadth",
            "error", "rubrics_metrics",
        ],
    },
}


# ──────────────────────────────────────────────────────────────────────────
# Shared base — universal definitions, three tests, KEEP rule, page-handling
# rules. Same content for every task call.
# ──────────────────────────────────────────────────────────────────────────
SHARED_BASE = """You analyze a single web page collected for a research study on rubrics across 11 tasks. Your job: extract every distinct *rubric* the page articulates for evaluating or producing the work-of-the-parent-task.

# Definition of a rubric

A **rubric** is a directive (imperative, interrogative, or normative-declarative) whose subject-matter is THE WORK ITSELF — not the service, platform, transaction, or people surrounding the work.

A rubric must satisfy ALL THREE tests:

1. **Target test** — the directive governs one of:
   - the work itself
   - the production process for the work
   - the evaluation/judgment of the work
   - a copy/submission constraint that shapes the work itself
   - a selection criterion for CHOOSING AMONG MULTIPLE CANDIDATES (e.g., "vendors must have 5+ years experience to be considered"; "top news is decided by subscriber engagement") — distinct from evaluation_judgment because it's comparative across candidates rather than judgment of one work in isolation

2. **Action test** — the directive is one of: PRODUCE, CONSTRAIN, JUDGE, or SELECT the work. (SELECT = choose/rank/short-list among multiple candidates.)

3. **Exclusion test** — the directive is NOT primarily about pricing/fees, service turnaround, distribution of meta-artifacts, account management, marketing copy, service-tier comparisons, platform mechanics, biographies of the producers/evaluators, or how the platform internally assigns staff. (Task-specific examples appear below.)

# What is — and is NOT — a meta-artifact

A **meta-artifact** is something CREATED ABOUT the work AFTER the work is produced — by an evaluator or by the platform. The work is what the producer makes; the meta-artifact is what gets created in response to (or alongside) the work.

CRITICAL distinction: bibliographies, citations, section labels, footnotes, references, author affiliations, and other elements **inside** the work are PART OF THE WORK ITSELF — not meta-artifacts. A rule that constrains the work's internal structure (its reference-list ordering, its section headings, its anonymized first page) is a `work` or `submission_form` rule. A rule that constrains a separate derived object (a review, a certificate, a score) is a `meta_artifact` rule.

(Task-specific meta-artifact examples appear in the task-specific section below.)

# Reasoning step (REQUIRED)

The schema starts with `reasoning`. Use it to:
- Identify the page genre and the work-of-this-task (anchored to the parent task).
- Walk through items and tag KEEP / DROP using the three tests.
- Name borderline items explicitly.

# Audience-of-instruction rule (work vs. evaluation_judgment)

When the same rule could be read either way ("the claim must be novel" vs. "the examiner should ask: is the claim novel?"):
- If the page's intended_audience is producers (authors, applicants, petitioners), phrase the rubric as a constraint on the work.
- If the audience is evaluators (examiners, reviewers, judges), phrase the rubric as an evaluation question or judgment-discipline directive.
This consistency helps downstream classification be accurate.

# Page-handling rules

**Index pages.** If the page is a pure INDEX/table-of-contents that lists external rule sets, forms, or sections by name without articulating their substance, do NOT extract a stub rubric per listed item. Return `rubrics_metrics=[]`, set `orientation="other"`, and note in `reasoning` "page is an index/listing".

**Page IS an instance of the work.** If the page is itself an instance of the work-of-this-task (rather than a rubric about the work), do NOT extract its substantive content as rubrics. Return `rubrics_metrics=[]`, set `orientation="other"`, and note in `reasoning` "page is an instance of the work, not a rubric about the work". (Task-specific examples of what counts as an instance-of-the-work appear in the task-specific section below.)

**Service-heavy pages.** If a page is mostly service description (pricing, turnaround, biographies) but DOES contain a few genuine work-criteria, extract those few and DROP the rest. Use `orientation="other"` (or whichever orientation best fits — "professional_standard" is often right). DO NOT use `orientation="error"` just because most of the page is service-policy noise — error is reserved for stub/paywall/interstitial pages with NO usable content.

**Stub / paywall / interstitial pages (only here use orientation="error").** If the page is essentially empty (subscription wall, captcha interstitial, 404, abstract-only landing, pure navigation chrome), set `orientation="error"` with a brief `error` reason. Examples: a page whose entire content is "Stonewalled by federal agencies? We want to document it." is too thin to extract from. A page whose content is "To access this content, you must purchase a subscription" is paywalled.

**Off-topic pages.** If the page wandered off-topic for its parent task, return `rubrics_metrics=[]`, `orientation="other"`. Do NOT invent rubrics by re-purposing the page's content into the wrong domain. (Task-specific off-topic examples appear in the task-specific section below.)

# KEEP / DROP discipline (lean toward extracting — downstream filter exists)

For every candidate item, run the three-question checklist as a SOFT GUIDE (not a strict gate):
1. Does this directive govern the WORK (or its production / judgment / submission-form / selection)?
2. Is the action one of PRODUCE / CONSTRAIN / JUDGE / SELECT?
3. Is it NOT clearly a service / transactional / biographical / distribution item?

If YES to all three → KEEP. If clearly NO to a test (the item is obvious noise from the exclusion list below) → DROP. **If unsure or borderline → EXTRACT IT.** A downstream classifier will re-evaluate every rubric and filter out the noise; extraction's job is to capture every plausible work-criterion the page articulates, not to perfectly partition them.

Be EXHAUSTIVE: when in doubt, include. Missing a legitimate work-criterion is worse than including a borderline item that the classifier will later drop.

# `inputs` field per rubric (REQUIRED)

For every rubric, populate `inputs` — a list of 2-6 short noun phrases naming the SPECIFIC ASPECTS of the work (or its production / evaluation context) that someone applying this rubric must attend to. Inputs are the THICK PART of the rule: once you've identified them, applying the rule itself may be mechanical.

## CRITICAL: inputs must be SPECIFIC, not generic

DO NOT write generic catch-alls like:
- ❌ "input_text"
- ❌ "the work"
- ❌ "the manuscript" / "the article" / "the paper"
- ❌ "evidence"
- ❌ "conclusions"
- ❌ "the process"

These are useless — they apply to every rubric. Name the PARTICULAR feature/observation/judgment the rule is asking the evaluator to make.

## The thick/thin frame

A THIN rule has few simple inputs that are mechanically observable:
- "submissions must be ≤5000 words" → inputs = ["word count of the manuscript"]
- "PDF format required" → inputs = ["file format"]

A THICK rule has many inputs and/or inputs that require expert judgment. Your inputs should make the thickness of the rule LEGIBLE — a reader of the inputs alone should be able to imagine what would have to be observed/judged to apply the rule.

## Genuinely holistic rubrics (escape hatch)

Some rubrics are inherently holistic and don't decompose into specific features:
- "The story should be entertaining"
- "The code should be maintainable"
- "The proof should be elegant"
- "The press release should engage readers"

For these, it's OK to use "the work as a whole" or "overall narrative impression" or similar holistic phrases. BUT this should be a last resort, not the default. If the rubric mentions any specific feature (a section, a beat, a quality dimension), prefer naming THAT specific feature in `inputs` instead of falling back to the holistic placeholder. Use the holistic fallback only when you've genuinely tried to decompose and failed.

## NEVER emit empty inputs arrays

`inputs` is REQUIRED for every rubric and must contain AT LEAST 2 items. NEVER emit `"inputs": []`. If you genuinely can't think of any specific inputs, fall back to the holistic placeholder above — but the holistic placeholder is still TWO concrete strings (e.g., `["the work as a whole", "overall reader engagement"]`), not an empty list.

Empty inputs arrays defeat the purpose of the field — they make downstream analysis impossible. Always populate with at least 2 specific (preferred) or holistic (fallback) noun phrases per rubric.

(Worked task-specific examples appear in the task-specific section below.)
"""


# ──────────────────────────────────────────────────────────────────────────
# Per-task modules. Shared per-task content (work definition, meta-artifact
# noise, service-or-logistics noise, actor-attribute noise) is imported from
# task_taxonomy.py — single source of truth across all prompt files. This
# file only adds the EXTRACTOR-SPECIFIC fields:
#   - positive_exemplar: a real high-quality v1 KEEP extraction for this task
#   - anti_pattern: a worked walk-through of the canonical over-extraction
#                    failure for this task
# ──────────────────────────────────────────────────────────────────────────
from task_taxonomy import TASK_INFO, as_bullets

# Extractor-specific exemplars per task. Keys MUST match TASK_INFO keys.
TASK_EXEMPLARS: dict = {
    "creative-writing": {
        "positive_exemplar": """PAGE: K.M. Weiland novel-writing checklist  |  SUBTASK: novel-writing checklist
Expected `reasoning`: "Craft-focused checklist page. Work-of-this-task = a novel manuscript. Each item is a quality criterion for the novel."
Selected expected rubrics (4 of 5):
  - {"name": "Three-Act Structure", "description": "Three Acts: Setup, Conflict, Resolution; Three Major Plot Points: First Plot Point (~25%), Midpoint (~50%), Third Plot Point (~75%); Two Pinch Points...", "guidance": "Weiland provides a concrete beat schema with approximate positions to use evaluatively."}
  - {"name": "Character Arc", "description": "The protagonist needs: A 'Lie They Believe' (false worldview rooted in past wound), A 'Truth' (the new worldview the story will lead them to), A 'Want' (externally driven), A 'Need' (internal truth)...", "guidance": ""}
  - {"name": "Theme", "description": "Theme should emanate from character actions rather than be stated; it must connect directly to plot...", "guidance": ""}
  - {"name": "Setting", "description": "Setting must matter to the plot as a catalyst rather than merely a backdrop...", "guidance": ""}""",
        "anti_pattern": """ANTI-PATTERN walk-through (Kirkus paid-review service description):
Page snippet: "Reviews focus on craft elements: characterization, plot, prose style, pacing. Reviews are 250-300 words. Reviewers honestly and impartially evaluate the books they receive. You may choose not to publish negative reviews. Standard turnaround: 7-9 weeks. Pricing: $425 / $575."

Expected `reasoning`: "Service-description page. Work = the book. 'craft elements: characterization, plot, prose style, pacing' → judgment-criterion → KEEP. 'Reviews are 250-300 words' → constrains the review (meta-artifact) → DROP. 'Honest and impartial' → judgment-discipline → KEEP. 'Choose not to publish' → meta-artifact distribution → DROP. 'Turnaround' → service → DROP. 'Pricing' → transaction → DROP."

Expected `orientation`: "other" (page IS readable; just service-heavy — NOT "error").
Expected extraction: 2 rubrics MUST be kept:
  - {"name": "Reviews focus on craft elements", "description": "characterization, plot, prose style, pacing", "guidance": "These are the dimensions a Kirkus Indie review must address about the book", "inputs": ["the book's characterization", "the book's plot", "prose style", "pacing"]}
  - {"name": "Honest and impartial evaluation", "description": "Reviewers honestly and impartially evaluate the books they receive", "guidance": "", "inputs": ["the reviewer's potential biases", "the book's quality independent of reviewer preferences"]}

CRITICAL: do NOT let the page's overall service-heavy framing cause you to skip the 2 substantive work-criteria. The craft-elements rubric in particular is the single most important rubric on any Kirkus-style page — it directly tells reviewers what to look at in the book.""",
    },

    "peer-review": {
        "positive_exemplar": """PAGE: Nature transparent peer review policy  |  SUBTASK: transparent peer review at Nature
Expected `reasoning`: "Journal-policy page. Work-of-this-task = the peer review report. Each rule constrains the contents/handling of the review file."
Selected expected rubrics (4 of 11):
  - {"name": "Reviewer consent and acceptance implies permission", "description": "Peer reviewers are informed of this initiative when invited to review and can decline... Acceptance to review is regarded as permission to release.", "guidance": ""}
  - {"name": "Reviewer anonymity and naming policy", "description": "Reviewer names are not published. Unless reviewers sign with their name, we will respect and maintain their full anonymity. Reviewers can request that their name be added...", "guidance": ""}
  - {"name": "Contents of the published peer review file", "description": "The peer review file will contain the full reviewer reports to authors and the author rebuttal letters.", "guidance": ""}
  - {"name": "Editing and redaction policy for peer review file", "description": "The peer review file will not be edited except to redact confidential information or third-party material. Authors may suggest redaction; the editor reviews.", "guidance": ""}""",
        "anti_pattern": """ANTI-PATTERN: pure process step → DROP
Rubric: "Phase 2 reviewer allocation: Each paper allocated additional reviewers in Phase 2; they cannot see Phase 1 reviews until submitting their own."
Reasoning: Internal reviewer-allocation procedure (service_or_logistics, gatekeeper, transact). DROP.""",
    },

    "math-stackexchange": {
        "positive_exemplar": """PAGE: Putnam Competition problems  |  SUBTASK: solving Putnam problems
Expected `reasoning`: "Competition-criteria page. Work = a math-competition solution. Each item is a constraint or expectation about the proof/answer format."
Selected expected rubrics (4 of 9):
  - {"name": "Proof / rigorous justification requirement", "description": "When a problem states 'Find (with proof)' or 'Prove or disprove', the solver must provide a rigorous proof of the asserted statement or a valid counterexample together with justification.", "guidance": ""}
  - {"name": "Answer-format conformity", "description": "If a problem asks to 'Express' or 'Find' the answer in a specific form... follow the exact formatting/representation requested.", "guidance": ""}
  - {"name": "Construction vs. existence requirement", "description": "When a problem asks for an object 'Find with proof a set ... for which this minimum k is achieved,' the solver must explicitly construct an example.", "guidance": ""}
  - {"name": "Combinatorial closed-form requirement", "description": "When a problem asks for the number of objects or for a formula, the expected solution is a closed-form, not asymptotics or bounds.", "guidance": ""}""",
        "anti_pattern": """ANTI-PATTERN: Project Gutenberg license clause appearing on a historical math textbook → DROP
Rubric: "Conditions for charging a fee — pay 20% royalty of gross profits to the trademark owner."
Reasoning: License/distribution boilerplate (service_or_logistics, transact). Not a math-quality criterion. DROP.""",
    },

    "news-homepages": {
        "positive_exemplar": """PAGE: IFCN fact-checking code of principles  |  SUBTASK: IFCN fact-checking code
Expected `reasoning`: "Professional-standards page. Work = a fact-check article. Each numbered commitment is a substantive standard the work must satisfy."
Selected expected rubrics (4 of 6):
  - {"name": "Commitment to Nonpartisanship and Fairness", "description": "Signatory organizations fact-check claims using the same standard for every fact check. They do not concentrate fact-checking on one side.", "guidance": ""}
  - {"name": "Commitment to Transparency of Sources", "description": "Signatories want their readers to be able to verify findings themselves. Signatories provide all sources in enough detail that readers can replicate their work...", "guidance": ""}
  - {"name": "Commitment to Transparency of Funding & Organization", "description": "Signatory organizations are transparent about their funding sources... ensure that funders have no influence over the conclusions...", "guidance": ""}
  - {"name": "Commitment to an Open & Honest Corrections Policy", "description": "Signatories publish their corrections policy and follow it scrupulously. They correct clearly and transparently...", "guidance": ""}""",
        "anti_pattern": """ANTI-PATTERN: 1-2 sentence stub page (e.g., AHCJ "Stonewalled by federal agencies? We want to document it.") → return rubrics_metrics=[] with orientation="error".
A stub with no substantive criteria should NOT be over-extracted into invented rubrics.""",
    },

    "press-releases": {
        "positive_exemplar": """PAGE: SEC Regulation FD  |  SUBTASK: Regulation FD compliance for issuers
Expected `reasoning`: "Federal-regulation page. Work = an issuer's selective/public disclosure. Each rule is a substantive constraint on the disclosure or its timing."
Selected expected rubrics (4 of 12):
  - {"name": "General rule regarding selective disclosure", "description": "Whenever an issuer discloses material nonpublic information to any person described in (b)(1), the issuer shall make public disclosure of that information...", "guidance": ""}
  - {"name": "Definition of \\"intentional\\" disclosure", "description": "A selective disclosure is 'intentional' when the person making it either knows, or is reckless in not knowing, that the information is both material and nonpublic.", "guidance": ""}
  - {"name": "Definition of \\"promptly\\" for non-intentional disclosures", "description": "'Promptly' means as soon as reasonably practicable (but in no event after the later of 24 hours or the next NYSE trading day) after a senior official learns of the selective disclosure.", "guidance": ""}
  - {"name": "Public disclosure methods (Form 8-K)", "description": "An issuer shall make the public disclosure required by §243.100(a) by furnishing a Form 8-K (17 CFR 249.308) or by another method reasonably designed for broad, non-exclusionary distribution.", "guidance": ""}""",
        "anti_pattern": """ANTI-PATTERN: PR-distribution wire pricing or "embargo expires at 9am ET Tuesday" → service_or_logistics → DROP.""",
    },

    "code-review": {
        "positive_exemplar": """PAGE: HashiCorp Terraform plugin best-practices  |  SUBTASK: terraform provider design principles
Expected `reasoning`: "Substantive technical-standard page. Work = code in a Terraform provider. Each design principle is a constraint or judgment-criterion."
Selected expected rubrics (4 of 8):
  - {"name": "Providers should focus on a single API or problem domain", "description": "A Terraform provider should manage a single collection of components based on the underlying API or SDK...", "guidance": ""}
  - {"name": "Resources should represent a single API object", "description": "A Terraform resource should be a declarative representation of single component, usually with create, read, delete, and optionally update methods...", "guidance": ""}
  - {"name": "Resource and attribute schema should closely match the underlying API", "description": "A Terraform resource and associated schema should follow the naming and structure of the API, unless it degrades the user experience...", "guidance": ""}
  - {"name": "Functions should be pure and offline", "description": "A provider-defined function should always produce the same result for the same arguments. Functions should avoid logic which is environment-based, time-based, or network-based.", "guidance": ""}""",
        "anti_pattern": """ANTI-PATTERN: "Install our IDE plugin to get inline review suggestions" → service_or_logistics → DROP.""",
    },

    "grant-funding": {
        "positive_exemplar": """PAGE: Knight-Hennessy Scholars criteria  |  SUBTASK: KHS application evaluation
Expected `reasoning`: "Formal selection-criteria page. Work = the Knight-Hennessy fellowship application. The 3 named criteria are the official judgment dimensions."
Expected rubrics (all 3):
  - {"name": "Independence of Thought", "description": "We seek visionary thinkers who are curious, open-minded, analytical, eager for cross-cultural perspective...", "guidance": ""}
  - {"name": "Purposeful Leadership", "description": "We seek courageous leaders who are ethical, decisive, resilient, driven to achieve meaningful results...", "guidance": ""}
  - {"name": "Civic Mindset", "description": "We seek collaborative community members who are humble, empathetic, trustworthy...", "guidance": ""}""",
        "anti_pattern": """ANTI-PATTERN: PI eligibility statements like "fully independent researcher with access to lab space and authority to recruit PhD students" → actor_attribute (about the PI as a person) → DROP. (KEEP only if the rule constrains the *application*, not the *applicant*.)""",
    },

    "humor": {
        "positive_exemplar": """PAGE: Sarah Silverman ironic-persona analysis  |  SUBTASK: ironic_persona_comedy
Expected `reasoning`: "Craft-analysis page about a specific comic technique. Work = an ironic-persona comedy performance. Each numbered item is a craft criterion or risk dimension."
Selected expected rubrics (4 of 10):
  - {"name": "Clear ironic persona", "description": "Is there a clear ironic persona? Mocks bigots by impersonating them. The persona is a character who inhabits the performer's body — views may not be shared by the real person.", "guidance": ""}
  - {"name": "Persona delivering bigoted views to mock them", "description": "Is the persona delivering bigoted views in order to mock those views? The technique involves a comic character endorsing bigoted views satirically/deadpan to subvert them.", "guidance": ""}
  - {"name": "Irony detectable to the intended audience", "description": "Is the irony detectable to the intended audience? Success requires that the audience 'see through' the persona...", "guidance": ""}
  - {"name": "Reveals something true about the targeted bigotry", "description": "Does the technique reveal something true about the targeted bigotry?", "guidance": ""}""",
        "anti_pattern": """ANTI-PATTERN: a fan-store landing page (e.g., "Conan The Barbarian Official Home and Store" with character bio, allies, skills, and merchandise) is OFF-TOPIC for the humor task. Do NOT extract "character skills" or "character backstory" as humor rubrics. Return rubrics_metrics=[].""",
    },

    "legal-outcome-prediction": {
        "positive_exemplar": """PAGE: Fourth Circuit briefing rules (SCOTUS Rule 24-style)  |  SUBTASK: appellate briefing rules
Expected `reasoning`: "Court-rules page. Work = an appellate brief. Each rule is a procedural/format constraint on the brief."
Selected expected rubrics (4 of 8):
  - {"name": "Citation of supplemental authorities (Rule 28(j) letters)", "description": "If pertinent and significant authorities come to a party's attention after the brief has been filed... Letters are limited to 350 words.", "guidance": ""}
  - {"name": "Consolidated cases — one brief per side and lead counsel selection", "description": "When appeals are consolidated, one brief shall be permitted per side... coordinate early to select lead counsel.", "guidance": ""}
  - {"name": "Statement of the Case (FRAP 28(a)(6) requirement)", "description": "The statement of the case must include a narrative statement of all of the facts necessary for the Court to reach the conclusion which the brief desires...", "guidance": ""}
  - {"name": "Citations to the appendix (formatting rule)", "description": "Citations in the brief to a joint or supplemental appendix must be in the format required by the Fourth Circuit Appendix Pagination & Brief Citation Guide.", "guidance": ""}""",
        "anti_pattern": """ANTI-PATTERN: "Hearing Motions; Submission on Briefs (Rule 78)" — court procedure for hearing motions → service_or_logistics → DROP.""",
    },

    "notice-and-comment": {
        "positive_exemplar": """PAGE: FTC procedures for petitions for rulemaking  |  SUBTASK: FTC petitions
Expected `reasoning`: "Federal-procedure page. Work = a petition for rulemaking. Each subsection is a substantive content-or-format requirement on the petition."
Selected expected rubrics (4 of 12):
  - {"name": "Petitioner's identity and interest (§1.31(b)(1))", "description": "Petitions must include the petitioner's full name, address, telephone number, and email address, along with an explanation of how the petitioner's interests would be affected by the requested action.", "guidance": ""}
  - {"name": "Full statement of action requested (§1.31(b)(2))", "description": "Petitions must include a full statement of the action requested, including the text and substance of the proposed rule or amendment...", "guidance": ""}
  - {"name": "Full factual and legal basis (§1.31(b)(3))", "description": "Petitions must include a full statement of the factual and legal basis on which the petitioner relies...", "guidance": ""}
  - {"name": "Supporting data standards (§1.31(c))", "description": "If an original research report is used to support a petition, the information should be presented in a form acceptable for publication in a peer-reviewed scientific or technical journal.", "guidance": ""}""",
        "anti_pattern": """ANTI-PATTERN 1: "Comments deadline: comments must be received on or before February 27, 2026" → service_or_logistics, transact → DROP.
ANTI-PATTERN 2: page IS itself a comment letter (e.g., "ABA letter to CFPB on advisory opinions proposal"). The page is an INSTANCE of the work, not a rubric ABOUT comment letters. Return rubrics_metrics=[].""",
    },

    "patents": {
        "positive_exemplar": """PAGE: MPEP first-action final rejection rules  |  SUBTASK: first-action final rejection rules
Expected `reasoning`: "USPTO procedural-standards page. Work = a continuing/RCE patent application + the examiner's first Office action on it. Each rule is a substantive condition for what the examiner can/cannot do."
Selected expected rubrics (4 of 11):
  - {"name": "Final rejection on first Office action for new continuing or substitute applications", "description": "For a new application, claims may be finally rejected in the first Office action when (A) the new application is a continuing/substitute application, and (B) all claims would have been properly finally rejected on the grounds and art of record...", "guidance": ""}
  - {"name": "Prohibition on using RCE to switch inventions", "description": "Applicants cannot use an RCE to obtain continued examination on the basis of claims that are independent and distinct from, or lack unity of invention with, the claims previously claimed and examined...", "guidance": ""}
  - {"name": "Improper to make final a first Office action when previously-denied amendments raised new issues or new matter", "description": "It would not be proper to make final a first Office action where that application contains material that was presented in the earlier application after final rejection but was denied entry...", "guidance": ""}
  - {"name": "Continuation-in-part (CIP) rule: new subject matter prevents first-action final", "description": "It would not be proper to make final a first Office action in a continuation-in-part application where any claim includes subject matter not present in the earlier application.", "guidance": ""}""",
        "anti_pattern": """ANTI-PATTERN: "Filing fee payment deadline: 1 month from filing the patent application" → service_or_logistics, transact → DROP.
NOTE on citation/bibliography rules: a rule like "the application must list cited prior art in numerical order" constrains the *application's internal structure* (work / submission_form), NOT a meta-artifact. KEEP.""",
    },
}


# ──────────────────────────────────────────────────────────────────────────
# Prompt assembly
# ──────────────────────────────────────────────────────────────────────────
def build_prompt_for_task(task: str) -> str:
    """Assemble shared base + task-specific module + closing for a given task.

    Pulls work / meta-artifact / service-or-logistics / actor-attribute content
    from task_taxonomy.TASK_INFO (single source of truth across all prompt
    files) and pairs it with the extractor-specific positive_exemplar +
    anti_pattern from TASK_EXEMPLARS in this file.
    """
    info = TASK_INFO.get(task)
    exemp = TASK_EXEMPLARS.get(task)
    if info is None or exemp is None:
        # Fall back to a generic prompt if task is unknown to taxonomy
        task_section = (
            f"# Parent task\n\n"
            f"This page was collected for the parent task: **{task}**. "
            f"Identify the work-of-this-task from the page content and apply the three tests.\n"
        )
    else:
        task_section = f"""# Parent task: {task}

## Work-of-this-task

For this page, the work-of-this-task is: **{info['work'][0]}**

## Task-specific meta-artifacts (DROP — these are derived objects, not the work)

{as_bullets(info['meta_artifacts'])}

## Task-specific service/logistics noise (DROP)

{as_bullets(info['service_or_logistics'])}

## Task-specific actor-attribute noise (DROP — describes the people, not the work)

{as_bullets(info['actor_attribute'])}

## Task-specific `inputs` examples (specific noun phrases, not generic)

{as_bullets(info['inputs_examples'])}

## Positive exemplar (this is what good extraction looks like for {task})

{exemp['positive_exemplar']}

## Anti-pattern (the most common over-extraction failure for {task})

{exemp['anti_pattern']}
"""
    closing = """
# Final reminders

- Use `reasoning` to think first.
- Be EXHAUSTIVE about substantive work-criteria; be RESTRICTIVE about the task-specific noise listed above.
- An empty rubrics_metrics list is the correct answer when the page has no work-criteria.
- DO NOT mark as "error" just because a page is dense.
- DO mark as "error" only for stub / paywall / interstitial / 404 / pure-navigation pages.
- Output strictly conforming JSON. No prose outside the JSON. No markdown fences.
"""
    return SHARED_BASE + "\n" + task_section + closing


# ──────────────────────────────────────────────────────────────────────────
# Backwards-compat: SYSTEM_PROMPT_V5 is now the assembled prompt for a
# default task (creative-writing). Existing callers should migrate to
# build_prompt_for_task(<actual task>).
# ──────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT_V5 = build_prompt_for_task("creative-writing")
