"""Prompt templates for program generation and library refactoring."""

from typing import List, Optional


def render_program_generation(
    text: str,
    label_value: int,
    text_type: str,
    positive_label: str,
    negative_label: str,
    library_code: Optional[str] = None,
) -> str:
    label_word = positive_label if label_value == 1 else negative_label

    library_section = ""
    if library_code:
        library_section = (
            "\nYou have access to a shared utility library. "
            "You may call any function from it. "
            "The library is already imported as `from lib import *`.\n\n"
            f"```python\n{library_code}\n```\n\n"
            "Use these utilities where helpful, or write new inline logic.\n"
        )

    return f"""\
You are writing a Python function to predict whether a {text_type} is \
{positive_label} or {negative_label}.

The function signature must be:

    def predict(text: str) -> float

Rules:
1. Return a float between 0.0 and 1.0 (1.0 = {positive_label}, \
0.0 = {negative_label}).
2. You may ONLY use Python standard library modules (re, string, \
collections, statistics, math, itertools, functools, textwrap, \
unicodedata, difflib, hashlib). No ML libraries, no network access, \
no file I/O.
3. Focus on VERIFIABLE, STRUCTURAL properties you can compute \
deterministically: text length, word patterns, punctuation, sentence \
structure, formatting, vocabulary sophistication, citation patterns, \
numeric content, etc.
4. The function should capture a GENERALIZABLE hypothesis — not overfit \
to this one example. Think about what structural property of this text \
could predict its label across many similar texts.
5. Include a brief docstring explaining your hypothesis.
{library_section}
Here is the example:

--- TEXT ---
{text[:3000]}
--- END TEXT ---

Label: {label_word} ({label_value})

Write a single `def predict(text: str) -> float` function. \
Wrap it in ```python ... ``` markers. No other text."""


def render_refactoring(
    text_type: str,
    positive_label: str,
    negative_label: str,
    new_programs: List[str],
    existing_library: Optional[str] = None,
) -> str:
    existing_section = ""
    if existing_library:
        existing_section = (
            "Here is the EXISTING shared utility library from previous rounds. "
            "Preserve all functions unless clearly redundant:\n\n"
            f"```python\n{existing_library}\n```\n\n"
        )

    programs_text = "\n\n---\n\n".join(
        f"# Program {i+1}\n{p}" for i, p in enumerate(new_programs)
    )

    return f"""\
You are maintaining a shared Python utility library for text analysis.

These utilities help predict whether a {text_type} is {positive_label} \
or {negative_label} based on verifiable/structural text properties.

{existing_section}\
Here are {len(new_programs)} NEW prediction functions to integrate:

{programs_text}

Your task: identify useful patterns in the new functions and update \
the library. Output ONLY the changes — do NOT repeat unchanged functions.

Output format (JSON):
```json
{{
  "add": [
    {{
      "name": "function_name",
      "code": "def function_name(text: str) -> int:\\n    \\"\\"\\"One-line docstring.\\"\\"\\"\\n    ..."
    }}
  ],
  "modify": [
    {{
      "name": "existing_function_to_update",
      "code": "def existing_function_to_update(text: str) -> int:\\n    ..."
    }}
  ],
  "delete": ["redundant_function_name"],
  "reasoning": "Brief explanation of changes"
}}
```

Rules:
- Only standard library imports (re, string, collections, statistics, math).
- Each function: SHORT (under 15 lines), 1-line docstring, type hints.
- Only ADD functions that capture genuinely new patterns not in the library.
- Only MODIFY if the new version is strictly better.
- DELETE only if truly redundant with a better replacement.
- If the new programs don't add anything new, return empty add/modify/delete.

Return ONLY the JSON wrapped in ```json ... ``` markers."""


SYSTEM_PROMPT_GENERATION = """\
You are an expert Python programmer specializing in text analysis. \
You write clean, efficient functions that extract structural and \
statistical properties from text. You never use machine learning \
libraries — only Python's standard library."""


# =========================================================================
# Approach 1: Full-example agnostic program (with optional z guidance)
# =========================================================================

def render_approach1_generation(
    text: str,
    label_value: int,
    text_type: str,
    positive_label: str,
    negative_label: str,
    features_for: Optional[List[str]] = None,
    features_against: Optional[List[str]] = None,
    library_code: Optional[str] = None,
) -> str:
    """Approach 1: Write a full program that classifies this example.

    Optionally guided by STaR z₊/z₋ rationales, but free to go beyond them.
    """
    label_word = positive_label if label_value == 1 else negative_label

    library_section = ""
    if library_code:
        library_section = (
            "\nYou have access to a shared utility library "
            "(already imported as `from lib import *`):\n"
            f"```python\n{library_code}\n```\n\n"
        )

    guidance_section = ""
    if features_for or features_against:
        guidance_section = "\nAn expert identified these aspects of this text:\n"
        if features_for:
            guidance_section += f"\nArguing FOR {positive_label}:\n"
            guidance_section += "\n".join(f"  - {f}" for f in features_for)
        if features_against:
            guidance_section += f"\nArguing AGAINST (i.e. for {negative_label}):\n"
            guidance_section += "\n".join(f"  - {f}" for f in features_against)
        guidance_section += (
            "\n\nUse these as guidance, but also look for OTHER verifiable "
            "patterns the expert may not have mentioned. Your program should "
            "be a comprehensive structural analysis, not limited to these points.\n"
        )

    return f"""\
You are writing a Python function to analyze and classify a {text_type} \
as {positive_label} or {negative_label}.

The function signature must be:

    def predict(text: str) -> float

Rules:
1. Return a float between 0.0 and 1.0 (1.0 = {positive_label}, \
0.0 = {negative_label}).
2. You may ONLY use Python standard library modules (re, string, \
collections, statistics, math, itertools, functools, textwrap, \
unicodedata, difflib, hashlib). No ML libraries, no network access, \
no file I/O.
3. Write a NON-TRIVIAL program that analyzes multiple structural \
properties of the text. Don't just check one thing — combine several \
verifiable signals into a composite score.
4. Focus on VERIFIABLE properties: lengths, counts, patterns, ratios, \
formatting, vocabulary, structure, numeric content, etc.
5. The function should GENERALIZE — not overfit to this specific text.
6. Be SPECIFIC to this example. Study the text carefully. What concrete \
pattern in THIS text distinguishes it from the opposite label? Don't \
write generic "check vocabulary diversity" — write targeted checks \
based on what you actually observe.
7. Include a docstring explaining your analysis strategy.

After the function, add a comment block listing any EXTERNAL TOOLS or \
DATA SOURCES that would help verify this text but aren't available to \
you as stdlib-only code. For example:
# TOOLS_NEEDED:
# - semantic_scholar_api: check if cited papers actually exist
# - claim_verification_db: verify factual claims against known facts
# - statistical_recomputation: re-run reported statistical tests
# - code_execution: run and test any code mentioned in the paper
# If no external tools would help, write: # TOOLS_NEEDED: none
{library_section}{guidance_section}
Here is the example:

--- TEXT ---
{text[:4000]}
--- END TEXT ---

Label: {label_word} ({label_value})

Write a single `def predict(text: str) -> float` function followed by \
a TOOLS_NEEDED comment. Wrap everything in ```python ... ``` markers."""


# =========================================================================
# Approach 2: Per-z program synthesis
# =========================================================================

def render_approach2_generation(
    text: str,
    label_value: int,
    rationale: str,
    polarity: str,
    text_type: str,
    positive_label: str,
    negative_label: str,
    library_code: Optional[str] = None,
) -> str:
    """Approach 2: Write a program that checks ONE specific z₊/z₋.

    Gets the rationale text AND the example text it was extracted from,
    so the LLM can see what concrete textual patterns correspond to this z.
    """
    label_word = positive_label if label_value == 1 else negative_label
    polarity_desc = (
        f"FOR {positive_label}" if polarity == "for"
        else f"AGAINST (i.e. for {negative_label})"
    )

    library_section = ""
    if library_code:
        library_section = (
            "\nYou have access to a shared utility library "
            "(already imported as `from lib import *`):\n"
            f"```python\n{library_code}\n```\n\n"
        )

    return f"""\
An expert evaluating a {text_type} identified this specific criterion:

    "{rationale}"

This criterion argues {polarity_desc}. The expert identified it in the \
following text (label: {label_word}):

--- TEXT ---
{text[:3000]}
--- END TEXT ---

Write a Python function that checks whether this criterion holds for \
ANY text of this type — not just this example.

    def check(text: str) -> float

Rules:
1. Return a float between 0.0 and 1.0 (1.0 = this criterion clearly \
applies, 0.0 = clearly does not apply).
2. You may ONLY use Python standard library modules (re, string, \
collections, statistics, math, itertools, functools, textwrap, \
unicodedata, difflib, hashlib).
3. Study the example text to understand what CONCRETE textual patterns \
correspond to this criterion. What would you look for in other texts?
4. If this criterion genuinely cannot be checked programmatically \
(requires reading comprehension or subjective judgment), write a \
function that returns 0.5 and add a docstring starting with \
"THICK:" explaining why it can't be programmatized.
5. If it CAN be checked, write a real check and add a docstring \
starting with "THIN:" explaining what structural signal you're using.

After the function, add a TOOLS_NEEDED comment listing any external \
tools or data sources that would help check this criterion but aren't \
available as stdlib-only code. Examples:
# TOOLS_NEEDED:
# - semantic_scholar_api: verify cited papers exist
# - claim_verification_db: check factual claims
# - code_execution_sandbox: test code snippets from paper
# If no external tools would help: # TOOLS_NEEDED: none
{library_section}
Write a single `def check(text: str) -> float` function + TOOLS_NEEDED. \
Wrap everything in ```python ... ``` markers."""


# =========================================================================
# Approach 2: Merge refactoring (merge programs targeting same/similar z's)
# =========================================================================

def render_approach2_merge_refactoring(
    text_type: str,
    positive_label: str,
    negative_label: str,
    program_groups: List[dict],
    existing_library: Optional[str] = None,
) -> str:
    """Refactor programs that target the same or similar z concepts.

    program_groups is a list of:
      {"rationale": str, "programs": [source_code, ...]}

    The refactoring should merge programs within each group AND across
    similar groups.
    """
    existing_section = ""
    if existing_library:
        existing_section = (
            "Here is the EXISTING shared utility library:\n"
            f"```python\n{existing_library}\n```\n\n"
        )

    groups_text = ""
    for i, group in enumerate(program_groups):
        groups_text += f"\n### Criterion: \"{group['rationale']}\"\n"
        groups_text += f"({len(group['programs'])} program variants)\n\n"
        for j, prog in enumerate(group["programs"][:5]):  # cap at 5 per group
            groups_text += f"```python\n# Variant {j+1}\n{prog}\n```\n\n"

    return f"""\
You are refactoring programs that check evaluation criteria for a \
{text_type} ({positive_label} vs {negative_label}).

Below are groups of programs. Within each group, multiple programs were \
independently written to check the SAME criterion on different examples. \
Across groups, some criteria may be related or overlapping.

{existing_section}\
{groups_text}

Your tasks:

1. **Within each group**: Can the variants be merged into ONE unified \
function? If they share core logic, merge them. If they capture \
genuinely different aspects of the criterion, keep them separate and \
note why.

2. **Across groups**: Extract shared utility functions used by multiple \
criteria. Are any groups checking the same underlying concept from \
different angles? Note this.

3. **Classify each criterion**: Based on whether the variants merged \
cleanly, label each criterion as:
   - THIN: variants merged into one clean function (verifiable)
   - PARTIALLY_THIN: some variants merged, others couldn't (mixed)
   - THICK: variants are too different to merge — the criterion requires \
context-specific judgment

Return TWO things:

**Part 1**: The updated library (```python ... ```)

**Part 2**: A merge report (```json ... ```):
```json
[
  {{
    "criterion": "the rationale text",
    "classification": "THIN|PARTIALLY_THIN|THICK",
    "merged_function": "function_name or null",
    "reasoning": "why it did/didn't merge",
    "n_variants_merged": 3,
    "n_variants_total": 5
  }}
]
```"""

SYSTEM_PROMPT_REFACTORING = """\
You are an expert software engineer specializing in code refactoring \
and library design. You identify common patterns across functions and \
extract them into clean, reusable utilities."""


# =========================================================================
# V2: Extraction-aware program generation
# =========================================================================

def render_program_generation_v2(
    text: str,
    label_value: int,
    text_type: str,
    positive_label: str,
    negative_label: str,
    library_code: Optional[str] = None,
    extraction_schema: Optional[str] = None,
) -> str:
    """V2 prompt: programs declare their input schema.

    Programs specify what structured fields they need from the text,
    how each field should be extracted (regex, program, or llm), and
    the check logic operating on extracted fields.
    """
    label_word = positive_label if label_value == 1 else negative_label

    library_section = ""
    if library_code:
        library_section = (
            "\nYou have access to a shared utility library:\n"
            f"```python\n{library_code}\n```\n\n"
        )

    schema_section = ""
    if extraction_schema:
        schema_section = (
            "\nThese input fields have already been defined and can be reused:\n"
            f"```json\n{extraction_schema}\n```\n"
            "You can reference any of these existing fields, or define new ones.\n\n"
        )

    return f"""\
You are building a verifiable check to predict whether a {text_type} is \
{positive_label} or {negative_label}.

Your check has TWO parts:

**Part 1 — Input Schema**: What structured information do you need to \
extract from the text? For each field, specify:
- "name": field name (snake_case)
- "type": Python type (int, float, bool, str, list)
- "extraction": how to extract it:
  - "regex" — a regex pattern that extracts the value directly
  - "program" — a Python stdlib function that computes it from raw text
  - "llm" — requires LLM to read and understand the text (use this \
when regex/program can't reliably extract it)
- "description": what this field represents
- For "regex": include "pattern" field
- For "program": include "code" field (a function `extract(text: str) -> value`)
- For "llm": include "prompt" field (question to ask about the text)

**Part 2 — Check Logic**: A Python function operating on the extracted \
fields:

    def predict(extracted: dict) -> float

Rules:
1. Return float between 0.0 and 1.0 (1.0 = {positive_label}).
2. The predict function may ONLY use extracted fields + stdlib. No raw \
text access.
3. Prefer "regex" or "program" extraction over "llm" when possible — \
the goal is to maximize what's programmatically verifiable.
4. Include a docstring explaining your hypothesis.
{library_section}{schema_section}
Here is the example:

--- TEXT ---
{text[:3000]}
--- END TEXT ---

Label: {label_word} ({label_value})

Output as JSON:
```json
{{
  "hypothesis": "Brief description of what you're checking and why",
  "inputs": [
    {{
      "name": "field_name",
      "type": "int",
      "extraction": "regex",
      "description": "what this field is",
      "pattern": "regex_pattern_here"
    }},
    {{
      "name": "another_field",
      "type": "bool",
      "extraction": "llm",
      "description": "what this field is",
      "prompt": "Question to ask about the text"
    }}
  ],
  "predict_code": "def predict(extracted: dict) -> float:\\n    ..."
}}
```

Return ONLY the JSON wrapped in ```json ... ``` markers."""


def render_schema_refactoring(
    text_type: str,
    positive_label: str,
    negative_label: str,
    all_inputs: List[dict],
    existing_schema: Optional[List[dict]] = None,
) -> str:
    """Refactor accumulated input field definitions into a shared schema.

    Merges duplicate fields, thins LLM extractions to regex/program where
    possible, and produces a canonical extraction schema.
    """
    existing_section = ""
    if existing_schema:
        import json
        existing_section = (
            "Here is the EXISTING shared extraction schema:\n"
            f"```json\n{json.dumps(existing_schema, indent=2)}\n```\n\n"
        )

    import json
    inputs_text = json.dumps(all_inputs, indent=2)

    return f"""\
You are building a shared extraction schema for verifiable checks on \
a {text_type} ({positive_label} vs {negative_label}).

{existing_section}\
Here are input fields defined by individual programs:

```json
{inputs_text}
```

Your task:
1. **Deduplicate**: merge fields that extract the same information \
(e.g., "num_citations" and "citation_count" are the same field).
2. **Thin where possible**: if a field is marked "llm" but could \
actually be extracted via "regex" or "program", downgrade it. The goal \
is to minimize LLM extraction — use it only when truly necessary.
3. **Standardize**: give each field a clear canonical name, type, and \
description.
4. **Preserve**: keep all fields that are used by at least one program. \
Don't remove useful fields.

For each field, output:
- "name", "type", "extraction" ("regex", "program", or "llm")
- "description"
- For regex: "pattern"
- For program: "code" (extract function)
- For llm: "prompt" (extraction question)

Return the consolidated schema as a JSON list wrapped in \
```json ... ``` markers."""
