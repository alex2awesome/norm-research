"""Shared prompt for the rubric-pair sameness judge (graded 0-3 scale).

Used by both the OpenRouter calibration runner (calib_judge.py) and the sk3
vLLM full run (sk3_judge_pairs.py). Iterate JUDGE_VERSION / SYSTEM here.

A graded score (not binary) so the resulting affinity matrix is real-valued
and agglomeratively clusterable.
"""
from __future__ import annotations

import json
import re

JUDGE_VERSION = "v6"

SYSTEM = """You compare two evaluation criteria ("rubrics") used to judge work in the same domain. Score how much they are the SAME criterion: 0, 1, or 2.

What matters: would the two rubrics give the same verdict on real work? Surface wording is irrelevant.

2 = THE SAME criterion -- genuinely equivalent. They would give the SAME verdict on essentially ALL realistic work; they differ only in wording. Reserve 2 for true equivalence: a pure rephrasing, or one spelling out a detail already entailed by the other. If you can describe realistic work that passes one rubric and fails the other, it is NOT a 2.

1 = RELATED but a DIFFERENT criterion -- do NOT merge. They share a topic or a core idea, but they are not the same metric. Any of these makes it a 1: one adds a genuine extra requirement; one is meaningfully broader or narrower; one is a specific sub-case of the other; they check different sub-properties. IMPORTANT: when one rubric explicitly names something extra to check that the other never mentions (e.g. "braces AND indentation" vs. just "braces"), that named extra is a genuine extra requirement -- score 1. Do NOT rationalize it away as "implied by" or "a part of" the other rubric.

0 = UNRELATED. Different properties. Both being {domain} rubrics does NOT make them related.

Examples:
- "Note possible legal consequences" / "Specify the potential legal consequences" -> 2 (pure rephrasing).
- "Monitor power" / "Monitor power and hold it accountable" -> 2 (holding-accountable is entailed by monitoring power).
- "Consistent spacing around operators" / "consistent spacing around operators and keywords" -> 2 (no realistic code is scored differently).
- "The budget should be realistic" / "the budget should be itemized AND realistic" -> 1 (itemization is a genuine extra requirement).
- "Brace placement and indentation should be consistent" / "Braces should be placed consistently" -> 1 (the first bundles indentation as an extra named check).
- "Check return values and handle errors" / "Handle errors properly" -> 1 (checking return values is a named extra requirement, not merely "implied").
- "Imports grouped in a specific order" / "imports sorted alphabetically and grouped" -> 1 (alphabetical is an added requirement).
- "Use consistent fonts" / "use standard fonts" -> 1 (consistency and standard-adherence are different criteria).
- "Corroborate allegations" / "corroborate survivor accounts" -> 1 (survivor accounts is a narrower sub-case).
- "Apply a correction" / "state the correction policy" -> 1 (doing vs. disclosing).
- "Use the active voice" / "the plot should be compelling" -> 0 (unrelated).

Respond with ONE JSON object: {"reasoning": "<1 sentence>", "score": 0|1|2}"""


def build_user(task: str, canonical_a: str, canonical_b: str) -> str:
    return (f'Domain: {task.replace("-", " ")}\n\n'
            f'RUBRIC A: {canonical_a}\n'
            f'RUBRIC B: {canonical_b}')


def salvage(raw: str):
    if not raw:
        return None
    s = re.sub(r"\s*```$", "", re.sub(r"^```(?:json)?\s*", "", raw.strip()))
    for cand in (s, s[s.find("{"):s.rfind("}") + 1] if "{" in s else ""):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None
