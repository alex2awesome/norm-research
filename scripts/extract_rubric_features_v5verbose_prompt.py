"""
v5-verbose extractor prompt variant.

Identical to v5 except for an added "Verbosity & fidelity" section that pushes
Llama to produce more verbose, source-grounded `name`/`description`/`guidance`
fields. The goal: a reader of `description + guidance` alone should recover
everything the source page said about that rubric, with NO inventions.

Used for A/B comparison vs. base v5. Imports the v5 module and patches
SYSTEM_PROMPT_V5 build to inject the extra section.
"""

from __future__ import annotations
import extract_rubric_features_v5_prompt as v5
from extract_rubric_features_v5_prompt import (
    JSON_SCHEMA_V5,
    SHARED_BASE as _BASE,
    TASK_EXEMPLARS,
)
from task_taxonomy import TASK_INFO, as_bullets

VERBOSITY_SECTION = """
# Verbosity & fidelity for `name` / `description` / `guidance`

For every rubric you extract, the `name`, `description`, and `guidance` fields should be as VERBOSE and SPECIFIC as the source page supports, while NEVER inventing content beyond what the page actually says.

- **name**: a short label (5-15 words) summarizing the rubric. Use the page's own terminology where possible.
- **description**: **VERBATIM or near-verbatim** from the source page. Quote the rule in full. Include specific numbers, references, citation IDs (e.g., "§1.31(b)(3)", "FRAP 28(a)(6)", "MPEP 2106"), examples explicitly given on the page, and the sub-clauses/conditions the page enumerates. Do NOT paraphrase down to a 1-sentence summary if the source has 4 sentences of substantive detail — give all 4.
- **guidance**: surrounding context the page provides — examples, anti-patterns, scoring notes, sub-rules, illustrative cases, exceptions. If the page shows 3 worked examples for a rule, include all 3. Empty string ONLY when the page genuinely says nothing more.

CRITICAL: never embellish. If the source is terse, the description should be terse. If the source is verbose with examples and sub-clauses, the description should capture them too. The test: a reader who only sees your `description + guidance` should be able to recover everything the original page said about this rubric.

DO NOT:
- Generalize away source-specific details (numbers, statutes, code references, examples)
- Invent supporting reasoning the page doesn't state
- Drop sub-clauses or enumerated conditions
- Smooth idiosyncratic wording into generic phrasing

DO:
- Quote distinctive phrases verbatim
- Preserve citations and section numbers
- Include any worked examples / anti-patterns the page shows
- Reproduce all numbered sub-clauses
"""

# Splice the verbosity section into the SHARED_BASE just before the inputs section
SHARED_BASE_VERBOSE = _BASE.replace(
    "# `inputs` field per rubric (REQUIRED)",
    VERBOSITY_SECTION + "\n# `inputs` field per rubric (REQUIRED)"
)


def build_prompt_for_task(task: str) -> str:
    """Same per-task assembly as v5, but with the verbose-fidelity section spliced in."""
    info = TASK_INFO.get(task)
    exemp = TASK_EXEMPLARS.get(task)
    if info is None or exemp is None:
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
- Per the Verbosity & fidelity section, keep `description`/`guidance` close to the source text.
- An empty rubrics_metrics list is the correct answer when the page has no work-criteria.
- DO NOT mark as "error" just because a page is dense.
- DO mark as "error" only for stub / paywall / interstitial / 404 / pure-navigation pages.
- Output strictly conforming JSON. No prose outside the JSON. No markdown fences.
"""
    return SHARED_BASE_VERBOSE + "\n" + task_section + closing
